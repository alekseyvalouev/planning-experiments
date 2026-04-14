from pathlib import Path
import asyncio
import heapq
import json
import os
from typing import Tuple
from PIL import Image

from dotenv import load_dotenv
from google import genai

from graph import Graph, Node
from plan_visualization import visualize_plan_to_file
from prompts import COMBINED_PROMPT, TASK_ALIGNMENT_PROMPT, TASK_TO_GO_PROMPT

import sys
sys.path.append("..")

#from omnivla.uncertainty_heuristic import infer_action_V_V, infer_action_V_VL, infer_action_V_L, setup_omnivla, calculate_action_distance

load_dotenv(".env")


class Planner:
    def __init__(self, graph):
        self.graph = graph
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        #self.omnivla_inference = setup_omnivla()

    def plan(self, start_node, start_modality, end_node, task, end_info=None):
        """Returns a dict mapping path_key -> {"path": [...], "heuristic": float}
        for every intermediate plan explored during A* search."""
        context = [self._load_modality(start_node, start_modality)]
        _heuristic_cache = {}
        explored_plans = []

        start_cache_key = ((start_node.node_id, start_modality, None),)
        start_h = self.calculate_heuristic(
            [], start_node, start_modality, end_node, task, end_info
        )
        _heuristic_cache[start_cache_key] = start_h
        start_path = [(start_node, start_modality, None)]
        explored_plans.append({"path": start_path, "heuristic": start_h})
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
                    landmark_indices = range(len(landmarks)) if landmarks else [None]

                    for li in landmark_indices:
                        if (neighbor.node_id, neighbor_mod, li) in visited:
                            continue
                        cache_key = path_key + ((neighbor.node_id, neighbor_mod, li),)
                        ctx_item = self._load_modality(neighbor, neighbor_mod, landmark_idx=li)
                        candidates.append((neighbor, neighbor_mod, li, cache_key, ctx_item))
                        if cache_key not in _heuristic_cache:
                            uncached_indices.append(len(candidates) - 1)

            if uncached_indices:
                plans_to_grade = [ctx[:] + [candidates[i][4]] for i in uncached_indices]
                grade_results = asyncio.run(
                    self._batch_combined_grades(plans_to_grade, task)
                )
                alpha, beta = 0.1, -0.1
                for j, idx in enumerate(uncached_indices):
                    neighbor, mod, li, cache_key, _ = candidates[idx]
                    to_go, alignment = grade_results[j]
                    print("To go: ", to_go, " for node ", neighbor.node_id)
                    print("Alignment: ", alignment, " for node ", neighbor.node_id)
                    h = alignment * alpha + to_go * beta
                    _heuristic_cache[cache_key] = h
                    print("Heuristic for node", neighbor.node_id, "modality", mod,
                          "landmark_idx", li, "is", h)

            goal_path = None
            for neighbor, mod, li, cache_key, ctx_item in candidates:
                h = _heuristic_cache[cache_key]
                new_path = path + [(neighbor, mod, li)]
                new_path_key = path_key + ((neighbor.node_id, mod, li),)
                explored_plans.append({"path": new_path, "heuristic": h})
                if h > 9.5:
                    goal_path = new_path
                counter += 1
                heapq.heappush(heap, (
                    -h, counter, neighbor.node_id, mod, li,
                    new_path_key, new_path, ctx + [ctx_item]
                ))

            if goal_path is not None:
                return goal_path, explored_plans

        return None, explored_plans
    
    def _load_modality(self, node, modality, landmark_idx=None):
        node_info = {k: v for k, v in node.subnodes[modality].items() if k != "id"}
        if "image" in node_info.keys():
            with open(node_info["image"], 'rb') as f:
                image_bytes = f.read()
            node_info["image"] = image_bytes
        if landmark_idx is not None and "landmarks" in node_info:
            node_info["landmarks"] = ["Go to the " + node_info["landmarks"][landmark_idx]]
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

        to_go, alignment = self.combined_grade(context_new, task)
        print("To go: ", to_go, " for node ", new_node.node_id)
        print("Alignment: ", alignment, " for node ", new_node.node_id)

        return alignment * alpha + to_go * beta

    def get_distance(self, node, other):
        return abs(node.info["id"] - other.info["id"])

    def grade_task_to_go(self, plan, task) -> int:
        contents = []
        
        contents.append(TASK_TO_GO_PROMPT)

        contents.append("Task:")
        for item in task:
            if isinstance(item, str):
                contents.append(item)
            else:
                contents.append(genai.types.Part.from_bytes(data=item, mime_type="image/jpeg"))

        contents.append("Plan (MUST BE TEMPORALLY CONSISTENT WITH THE TASK. NO OUT OF ORDER STEPS.):")
        for i, step in enumerate(plan):
            contents.append(f"Step {i + 1}:")
            if "image" in step:
                contents.append(genai.types.Part.from_bytes(data=step["image"], mime_type="image/jpeg"))
            if "landmarks" in step:
                landmarks_text = " ".join(step["landmarks"])
                contents.append(f"{landmarks_text}")


        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            #model="gemini-3-flash-preview",
            contents=contents,
        )
        return int(response.text.strip())

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
                landmarks_text = " ".join(step["landmarks"])
                contents.append(landmarks_text)
        return contents

    def combined_grade(self, plan, task) -> Tuple[int, int]:
        contents = self._build_combined_contents(plan, task)
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
        )
        return tuple(int(x) for x in response.text.strip().split(" "))

    async def _async_combined_grade(self, plan, task) -> Tuple[int, int]:
        contents = self._build_combined_contents(plan, task)
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents=contents,
            )
        except Exception as e:
            print("Error generating content: ", e)
            return await self._async_combined_grade(plan, task)

        try:
            return tuple(int(x) for x in response.text.strip().split(" "))
        except Exception as e:
            print("Error parsing response: ", response.text)
            return await self._async_combined_grade(plan, task)

    async def _batch_combined_grades(self, plans, task):
        coros = [self._async_combined_grade(plan, task) for plan in plans]
        return await asyncio.gather(*coros)

    def grade_alignment(self, plan, task) -> int:
        contents = []
        
        contents.append(TASK_ALIGNMENT_PROMPT)

        contents.append("Task:")
        for item in task:
            if isinstance(item, str):
                contents.append(item)
            else:
                contents.append(genai.types.Part.from_bytes(data=item, mime_type="image/jpeg"))

        contents.append("Plan (MUST BE TEMPORALLY CONSISTENT WITH THE TASK. NO OUT OF ORDER STEPS.):")
        for i, step in enumerate(plan):
            contents.append(f"Step {i + 1}:")
            if "image" in step:
                contents.append(genai.types.Part.from_bytes(data=step["image"], mime_type="image/jpeg"))
            if "landmarks" in step:
                landmarks_text = ", ".join(step["landmarks"])
                contents.append(f"{landmarks_text}")


        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            #model="gemini-3-flash-preview",
            contents=contents,
        )
        return int(response.text.strip())


if __name__ == "__main__":
    my_graph = Graph.deserialize("graph_no_drop.json")
    planner = Planner(my_graph)
    start_node = my_graph.nodes[45]
    start_modality = "V"
    end_node = my_graph.nodes[6]
    task = "First, go down the hall past the scooter. Then, pass the pallet and lockers. Next, continue past the wooden doors. Finally, stop at the ladder before the double doors."
    end_info = {
        "text": "Go to the ladder"
    }
    plan, explored_plans = planner.plan(start_node, start_modality, end_node, task, end_info=None)

    print(f"Explored {len(explored_plans)} intermediate plans")
    if plan is not None:
        print(f"Goal plan found with {len(plan)} steps")
    else:
        print("No goal plan found")
    
    print(explored_plans[-1])

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
