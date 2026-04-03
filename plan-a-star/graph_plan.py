from pathlib import Path
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

load_dotenv(".env")


class Planner:
    def __init__(self, graph):
        self.graph = graph
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def plan(self, start_node, start_modality, end_node, task):
        context = [self._load_modality(start_node, start_modality)]
        _heuristic_cache = {}

        def cached_heuristic(path_key, ctx, node, modality):
            # path_key uniquely identifies the context; append new node to form full cache key
            cache_key = path_key + ((node.node_id, modality),)
            if cache_key not in _heuristic_cache:
                _heuristic_cache[cache_key] = self.calculate_heuristic(ctx, node, modality, end_node, task)
            print("Heuristic for node", node.node_id, "with modality", modality, "is", _heuristic_cache[cache_key])
            return _heuristic_cache[cache_key]

        start_path_key = ()
        start_h = cached_heuristic(start_path_key, [], start_node, start_modality)

        # heap entries: (neg_heuristic, tie_counter, node_id, modality, path_key, path, context)
        # neg_heuristic: we maximize heuristic by minimizing its negation
        # tie_counter: stable ordering when scores are equal (avoids comparing Node objects)
        counter = 0
        heap = [(-start_h, counter, start_node.node_id, start_modality,
                 ((start_node.node_id, start_modality),),
                 [(start_node, start_modality)], context)]
        visited = set()

        while heap:
            neg_h, _, _, cur_modality, path_key, path, ctx = heapq.heappop(heap)
            cur_node = path[-1][0]
            state = (cur_node.node_id, cur_modality)

            if state in visited:
                continue
            visited.add(state)

            for conn_mod, neighbor, neighbor_mod in cur_node.connections:
                if conn_mod != cur_modality:
                    continue
                if (neighbor.node_id, neighbor_mod) in visited:
                    continue

                new_ctx = ctx + [self._load_modality(neighbor, neighbor_mod)]
                h = cached_heuristic(path_key, ctx, neighbor, neighbor_mod)
                new_path_key = path_key + ((neighbor.node_id, neighbor_mod),)
                if h > 9.5:
                    return path + [(neighbor, neighbor_mod)]
                counter += 1
                heapq.heappush(heap, (-h, counter, neighbor.node_id, neighbor_mod,
                                      new_path_key, path + [(neighbor, neighbor_mod)], new_ctx))

        return None  # no path found
    
    def _load_modality(self, node, modality):
        node_info = {k: v for k, v in node.subnodes[modality].items() if k != "id"}
        # load the image
        if "image" in node_info.keys():
            with open(node_info["image"], 'rb') as f:
                image_bytes = f.read()
            node_info["image"] = image_bytes
        return node_info
    
    def calculate_heuristic(self, context, new_node, modality, end_node, task, alpha=0.1, beta=-0.1):
        context_new = context[:]
        context_new.append(self._load_modality(new_node, modality))

        #alignment = self.grade_alignment(context_new, task)
        #distance = self.get_distance(new_node, end_node)
        #return alignment * alpha - distance * beta

        #if "to_go" not in new_node.cache.keys():
        #    new_node.cache["to_go"] = self.grade_task_to_go(context_new, task)

        #to_go = new_node.cache["to_go"]
        #print("To go: ", to_go, " for node ", new_node.node_id)
        to_go, alignment = self.combined_grade(context_new, task)
        print("To go: ", to_go, " for node ", new_node.node_id)
        print("Alignment: ", alignment, " for node ", new_node.node_id)

        return alignment * alpha + to_go * beta
        #if "alignment" not in new_node.cache.keys():
        #    new_node.cache["alignment"] = alignment

        #return alignment * alpha
        #return to_go * beta

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
                landmarks_text = ", ".join(step["landmarks"])
                contents.append(f"Landmarks: {landmarks_text}")


        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            #model="gemini-3-flash-preview",
            contents=contents,
        )
        return int(response.text.strip())
    
    def combined_grade(self, plan, task) -> Tuple[int, int]:
        contents = []
        
        contents.append(COMBINED_PROMPT)

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
                contents.append(f"Landmarks: {landmarks_text}")


        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            #model="gemini-3-flash-preview",
            contents=contents,
        )
        return tuple(int(x) for x in response.text.strip().split(" "))

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
                contents.append(f"Landmarks: {landmarks_text}")


        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            #model="gemini-3-flash-preview",
            contents=contents,
        )
        return int(response.text.strip())


if __name__ == "__main__":
    my_graph = Graph.deserialize("graph.json")
    planner = Planner(my_graph)
    start_node = my_graph.nodes[0]
    start_modality = "V"
    end_node = my_graph.nodes[6]
    task = "First go to the kitchen. Then go to the area with two dark doors."
    plan = planner.plan(start_node, start_modality, end_node, task)

    print(plan)

    plan_data = {
        "task": task,
        "start_node_id": start_node.node_id,
        "start_modality": start_modality,
        "end_node_id": end_node.node_id,
        "steps": [
            {
                "node_id": node.node_id,
                "modality": modality,
                "frame_id": node.info.get("id", node.node_id),
                "image": node.info.get("image"),
                "landmarks": node.info.get("landmarks"),
            }
            for node, modality in plan
        ] if plan else [],
    }

    with open("plan.json", "w") as f:
        json.dump(plan_data, f, indent=2)
    print("Plan saved to plan.json")

    figure_path = visualize_plan_to_file(plan, "plan.png", task=task)
    print(f"Plan figure saved to {figure_path}")
