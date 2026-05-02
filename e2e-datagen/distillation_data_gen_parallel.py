from pathlib import Path
import asyncio
import heapq
import json
import os
import sys
from typing import Tuple
from PIL import Image
import re

import matplotlib.pyplot as plt

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

PLAN_A_STAR_DIR = Path(__file__).resolve().parents[1] / "plan-a-star"
if str(PLAN_A_STAR_DIR) not in sys.path:
    sys.path.insert(0, str(PLAN_A_STAR_DIR))

from graph_io import Graph, Node
from plan_visualization import visualize_plan_to_file
from prompts import COMBINED_PROMPT, TASK_ALIGNMENT_PROMPT, TASK_TO_GO_PROMPT

#from omnivla.uncertainty_heuristic import infer_action_V_V, infer_action_V_VL, infer_action_V_L, setup_omnivla, calculate_action_distance

load_dotenv(".env")


def plot_explored_plans_heuristic_frequency(
    explored_plans,
    out_path="explored_plans_heuristic_frequency.png",
    *,
    bins="auto",
    title="Explored plans: heuristic score frequency",
):
    """Histogram of ``heuristic`` values from each entry in ``explored_plans``."""
    scores = [ep["heuristic"] for ep in explored_plans]
    if not scores:
        print("plot_explored_plans_heuristic_frequency: no explored plans to plot")
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=bins, edgecolor="black", linewidth=0.4, alpha=0.88, color="steelblue")
    ax.set_xlabel("Heuristic score")
    ax.set_ylabel("Frequency (count)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(out_path).resolve()


class Planner:
    def __init__(self, graph):
        self.graph = graph
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        #self.omnivla_inference = setup_omnivla()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

        self.gemini_model = "gemini-3-flash-preview"
        self.gemini_config = genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_level="low")
        )

    def _record_usage(self, response):
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = (
            getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "response_token_count", 0)
            or 0
        )
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def print_token_usage(self):
        n = self.total_requests
        avg_in = self.total_input_tokens / n if n else 0.0
        avg_out = self.total_output_tokens / n if n else 0.0
        print("=" * 50)
        print("Gemini token usage")
        print(f"  Requests:            {n}")
        print(f"  Total input tokens:  {self.total_input_tokens}")
        print(f"  Total output tokens: {self.total_output_tokens}")
        print(f"  Total tokens:        {self.total_input_tokens + self.total_output_tokens}")
        print(f"  Avg input/request:   {avg_in:.1f}")
        print(f"  Avg output/request:  {avg_out:.1f}")
        print(f"  Model:               {self.gemini_model}")
        if self.gemini_model == "gemini-3-flash-preview":
            input_cost = self.total_input_tokens * 10**-6 * 0.5
            output_cost = self.total_output_tokens * 10**-6 * 3.0
            total_cost = input_cost + output_cost
            print(f"  Input Cost:            ${input_cost}")
            print(f"  Output Cost:           ${output_cost}")
            print(f"  Total Cost:            ${total_cost}")
        print("=" * 50)

    async def plan(self, start_node, start_modality, start_landmark_idx, end_node=None, task=None, end_info=None, separate_landmarks=True):
        """Returns ``(goal_path, explored_plans, reasoning_cache)``.

        ``explored_plans`` is a list of ``{"path", "heuristic", "reasoning"}``
        entries, one per intermediate plan explored during A* search.
        ``reasoning_cache`` maps ``cache_key`` (the tuple of ``(node_id, modality,
        landmark_idx)`` states along a path) to the raw LLM reasoning string
        produced while grading that plan.

        If ``separate_landmarks`` is False, landmarks at each node are not expanded
        into separate states; instead a single state per (node, modality) is used
        that carries every landmark together in the context.
        """
        include_all_landmarks = not separate_landmarks
        effective_start_li = start_landmark_idx if separate_landmarks else None
        context = [self._load_modality(start_node, start_modality, effective_start_li,
                                        include_all_landmarks=include_all_landmarks)]
        _heuristic_cache = {}
        reasoning_cache = {}
        explored_plans = []

        start_cache_key = ((start_node.node_id, start_modality, effective_start_li),)
        start_h = self.calculate_heuristic(
            [], start_node, start_modality, end_node, task, end_info
        )
        _heuristic_cache[start_cache_key] = start_h
        reasoning_cache[start_cache_key] = "(start node; not graded)"
        start_path = [(start_node, start_modality, effective_start_li)]
        explored_plans.append({
            "path": start_path,
            "heuristic": start_h,
            "reasoning": reasoning_cache[start_cache_key],
        })
        print("Heuristic for node", start_node.node_id, "modality", start_modality,
              "landmark_idx", None, "is", start_h)

        # heap entries: (neg_h, tie_counter, node_id, modality, landmark_idx,
        #                path_key, path, context)
        counter = 0
        heap = [(-start_h, counter, start_node.node_id, start_modality, None,
                 start_cache_key, start_path, context)]
        visited = set()

        while heap:
            neg_h, _, _, cur_modality, cur_li, path_key, path, ctx = heapq.heappop(heap)
            cur_node = path[-1][0]
            state = (cur_node.node_id, cur_modality, cur_li)

            if state in visited:
                continue
            visited.add(state)

            candidates = []
            uncached_indices = []
            for neighbor in cur_node.connections:
                for neighbor_mod in neighbor.modalities:
                    landmarks = neighbor.subnodes[neighbor_mod].get("landmarks", [])
                    if separate_landmarks:
                        landmark_indices = range(len(landmarks)) if landmarks else [None]
                    else:
                        landmark_indices = [None]

                    for li in landmark_indices:
                        if (neighbor.node_id, neighbor_mod, li) in visited:
                            continue
                        cache_key = path_key + ((neighbor.node_id, neighbor_mod, li),)
                        ctx_item = self._load_modality(neighbor, neighbor_mod, landmark_idx=li,
                                                       include_all_landmarks=include_all_landmarks)
                        candidates.append((neighbor, neighbor_mod, li, cache_key, ctx_item))
                        if cache_key not in _heuristic_cache:
                            uncached_indices.append(len(candidates) - 1)

            if uncached_indices:
                plans_to_grade = [ctx[:] + [candidates[i][4]] for i in uncached_indices]
                grade_results = await self._batch_combined_grades(plans_to_grade, task)
                alpha, beta = 0.1, -0.1
                for j, idx in enumerate(uncached_indices):
                    neighbor, mod, li, cache_key, _ = candidates[idx]
                    to_go, alignment, reasoning = grade_results[j]
                    print("To go: ", to_go, " for node ", neighbor.node_id)
                    print("Alignment: ", alignment, " for node ", neighbor.node_id)
                    h = alignment * alpha + to_go * beta
                    _heuristic_cache[cache_key] = h
                    reasoning_cache[cache_key] = reasoning
                    print("Heuristic for node", neighbor.node_id, "modality", mod,
                          "landmark_idx", li, "is", h)

            goal_path = None
            for neighbor, mod, li, cache_key, ctx_item in candidates:
                h = _heuristic_cache[cache_key]
                new_path = path + [(neighbor, mod, li)]
                new_path_key = path_key + ((neighbor.node_id, mod, li),)
                explored_plans.append({
                    "path": new_path,
                    "heuristic": h,
                    "reasoning": reasoning_cache.get(cache_key, ""),
                })
                if h > 9.5:
                    goal_path = new_path
                counter += 1
                heapq.heappush(heap, (
                    -h, counter, neighbor.node_id, mod, li,
                    new_path_key, new_path, ctx + [ctx_item]
                ))

            if goal_path is not None:
                return goal_path, explored_plans, reasoning_cache

        return None, explored_plans, reasoning_cache
    
    def _load_modality(self, node, modality, landmark_idx=None, include_all_landmarks=False):
        node_info = {k: v for k, v in node.subnodes[modality].items() if k != "id"}
        if "image" in node_info.keys():
            with open(node_info["image"], 'rb') as f:
                image_bytes = f.read()
            node_info["image"] = image_bytes
        if "landmarks" in node_info:
            if landmark_idx is not None:
                node_info["landmarks"] = ["Go to the " + node_info["landmarks"][landmark_idx]]
            elif include_all_landmarks:
                node_info["landmarks"] = [f"\nOption {i+1}: Go to the " + lm for i, lm in enumerate(node_info["landmarks"])]
        return node_info
    
    def calculate_heuristic(self, context, new_node, modality, end_node, task, end_info,
                            landmark_idx=None, alpha=0.1, beta=-0.1):
        context_new = context[:]
        new_ctx = self._load_modality(new_node, modality, landmark_idx=landmark_idx)
        context_new.append(new_ctx)

        grounding_heuristic = 0

        if end_info is not None:
            if "image" in context[-1].keys():
                if "V" in end_info.keys() and "L" in end_info.keys():
                    actions_end = infer_action_V_VL(self.omnivla_inference, context[-1]["image"], end_info["image"], end_info["text"])
                elif "V" in end_info.keys():
                    actions_end = infer_action_V_V(self.omnivla_inference, context[-1]["image"], end_info["image"])
                elif "L" in end_info.keys():
                    actions_end = infer_action_V_L(self.omnivla_inference, context[-1]["image"], end_info["text"])
                
                if modality == "V":
                    actions_next = infer_action_V_V(self.omnivla_inference, context[-1]["image"], context_new[-1]["image"])
                elif modality == "L":
                    actions_next = infer_action_V_L(self.omnivla_inference, context[-1]["image"], context_new[-1]["text"])
                elif modality == "VL":
                    actions_next = infer_action_V_VL(self.omnivla_inference, context[-1]["image"], context_new[-1]["image"], context_new[-1]["text"])
                
                grounding_heuristic = calculate_action_distance(actions_end, actions_next)

        to_go, alignment, _ = self.combined_grade(context_new, task)
        print("To go: ", to_go, " for node ", new_node.node_id)
        print("Alignment: ", alignment, " for node ", new_node.node_id)

        return alignment * alpha + to_go * beta


    def _build_combined_contents(self, plan, task):
        contents = [COMBINED_PROMPT, "Task:"]
        for item in task:
            if isinstance(item, str):
                contents.append(item)
            else:
                contents.append(genai.types.Part.from_bytes(data=item, mime_type="image/jpeg"))

        contents.append("Plan:")
        for i, step in enumerate(plan):
            contents.append(f"Step {i + 1}:")
            if "image" in step:
                contents.append(genai.types.Part.from_bytes(data=step["image"], mime_type="image/jpeg"))
            if "landmarks" in step:
                landmarks_text = "Options:\n" + "\n".join(step["landmarks"])
                contents.append(landmarks_text)
        return contents

    def _parse_llm_scores(self, llm_response):
        # The regex looks for two groups of digits separated by space, at the very end of the text
        pattern = r"(\d+)\s+(\d+)\s*$"
        
        # re.search scans the string for the pattern
        match = re.search(pattern, llm_response.strip())
        
        if match:
            # Extract the two captured groups and convert them to integers
            task_to_go = int(match.group(1))
            alignment = int(match.group(2))
            return task_to_go, alignment
        else:
            # Fallback/Error handling if the LLM completely failed to output numbers
            print("Warning: Could not parse integers from LLM response.")
            print(f"Raw response: {llm_response}")
            # Return default penalty scores so your A* planner doesn't crash
            return 100, 0


    # Retriable Gemini errors: transient server / rate-limit issues.
    _RETRIABLE_STATUS = {429, 500, 502, 503, 504}
    _MAX_API_RETRIES = 8
    _MAX_PARSE_RETRIES = 3

    @classmethod
    def _is_retriable_api_error(cls, exc: BaseException) -> bool:
        if isinstance(exc, genai_errors.ServerError):
            return True
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in cls._RETRIABLE_STATUS:
            return True
        # Transport-level hiccups (aiohttp / httpx / socket timeouts, resets).
        return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        # 2, 4, 8, 16, ... capped at 60s.
        return min(60.0, 2.0 ** (attempt + 1))

    def combined_grade(self, plan, task) -> Tuple[int, int, str]:
        contents = self._build_combined_contents(plan, task)
        import time
        last_exc = None
        for attempt in range(self._MAX_API_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=self.gemini_config,
                )
                break
            except Exception as e:
                last_exc = e
                if not self._is_retriable_api_error(e) or attempt == self._MAX_API_RETRIES - 1:
                    raise
                delay = self._backoff_seconds(attempt)
                print(f"[combined_grade] transient Gemini error ({type(e).__name__}: {e}); "
                      f"retry {attempt + 1}/{self._MAX_API_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
        self._record_usage(response)
        reasoning = response.text.strip()
        to_go, alignment = self._parse_llm_scores(reasoning)
        return to_go, alignment, reasoning

    async def _async_combined_grade(self, plan, task) -> Tuple[int, int, str]:
        contents = self._build_combined_contents(plan, task)
        response = None

        for attempt in range(self._MAX_API_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=self.gemini_config,
                )
                break
            except Exception as e:
                if not self._is_retriable_api_error(e) or attempt == self._MAX_API_RETRIES - 1:
                    print(f"[_async_combined_grade] giving up after {attempt + 1} attempts: "
                          f"{type(e).__name__}: {e}")
                    raise
                delay = self._backoff_seconds(attempt)
                print(f"[_async_combined_grade] transient Gemini error ({type(e).__name__}: {e}); "
                      f"retry {attempt + 1}/{self._MAX_API_RETRIES} in {delay:.1f}s")
                await asyncio.sleep(delay)

        self._record_usage(response)

        for parse_attempt in range(self._MAX_PARSE_RETRIES):
            try:
                reasoning = response.text.strip()
                to_go, alignment = self._parse_llm_scores(reasoning)
                return to_go, alignment, reasoning
            except Exception as e:
                print(f"[_async_combined_grade] parse error ({type(e).__name__}: {e}); "
                      f"attempt {parse_attempt + 1}/{self._MAX_PARSE_RETRIES}; "
                      f"raw: {getattr(response, 'text', None)!r}")
                if parse_attempt == self._MAX_PARSE_RETRIES - 1:
                    # Give up: return default penalty scores so A* doesn't crash.
                    return 100, 0, getattr(response, "text", "") or ""
                # Re-sample once from the model before giving up.
                try:
                    response = await self.client.aio.models.generate_content(
                        model=self.gemini_model,
                        contents=contents,
                        config=self.gemini_config,
                    )
                    self._record_usage(response)
                except Exception as api_e:
                    print(f"[_async_combined_grade] re-sample failed: "
                          f"{type(api_e).__name__}: {api_e}")
                    return 100, 0, ""

    async def _batch_combined_grades(self, plans, task):
        coros = [self._async_combined_grade(plan, task) for plan in plans]
        return await asyncio.gather(*coros)


if __name__ == "__main__":
    my_graph = Graph.deserialize("graph_vint_loose.json")
    planner = Planner(my_graph)
    start_node = my_graph.nodes[45]
    start_modality = "V"
    start_landmark_idx = None
    end_node = my_graph.nodes[6]
    task = "First, go down the hall past the scooter. Then, pass the pallet and lockers. Next, continue past the wooden doors. Finally, stop at the ladder before the double doors."
    #task = "go to the ladder"
    end_info = {
        "text": "Go to the ladder"
    }
    plan, explored_plans, reasoning_cache = asyncio.run(
        planner.plan(start_node, start_modality, start_landmark_idx, end_node, task, end_info=None, separate_landmarks=False)
    )

    print(f"Explored {len(explored_plans)} intermediate plans")
    if plan is not None:
        print(f"Goal plan found with {len(plan)} steps")
    else:
        print("No goal plan found")


    if plan is not None:
        print("=" * 50)
        print("Reasoning trace for selected plan")
        print("=" * 50)
        cache_key = ()
        for step_idx, (node, modality, li) in enumerate(plan):
            cache_key = cache_key + ((node.node_id, modality, li),)
            reasoning = reasoning_cache.get(cache_key, "(no reasoning cached)")
            print(f"\n--- Step {step_idx}: node={node.node_id} modality={modality} "
                  f"landmark_idx={li} ---")
            print(reasoning)
        print("=" * 50)

    hist_path = plot_explored_plans_heuristic_frequency(explored_plans)
    if hist_path is not None:
        print(f"Heuristic frequency chart saved to {hist_path}")

    planner.print_token_usage()

    #print(explored_plans[-1])

    #best_key = max(explored_plans, key=lambda k: explored_plans[k]["heuristic"])
    #best_entry = explored_plans[best_key]
    #best_plan = plan if plan is not None else best_entry["path"]
    #print(f"Best heuristic: {best_entry['heuristic']}")

    plan_data = {
        "task": task,
        "start_node_id": start_node.node_id,
        "start_modality": start_modality,
        "end_node_id": end_node.node_id,
        "steps": [
            {
                "node_id": node.node_id,
                "modality": modality,
                "landmark_idx": li,
                "frame_id": node.info.get("id", node.node_id),
                "image": node.info.get("image"),
                "landmarks": node.info.get("landmarks"),
            }
            for node, modality, li in plan
        ] if plan else [],
    }

    with open("plan.json", "w") as f:
        json.dump(plan_data, f, indent=2)
    print("Plan saved to plan.json")

    figure_path = visualize_plan_to_file(plan, "plan.png", task=task)
    print(f"Best plan figure saved to {figure_path}")
