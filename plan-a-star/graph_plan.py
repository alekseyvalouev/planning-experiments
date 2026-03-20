from pathlib import Path
import heapq
import os
from PIL import Image

from dotenv import load_dotenv
from google import genai

from graph import Graph, Node
from prompts import TASK_ALIGNMENT_PROMPT

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

            if cur_node.node_id == end_node.node_id:
                return path

            for conn_mod, neighbor, neighbor_mod in cur_node.connections:
                if conn_mod != cur_modality:
                    continue
                if (neighbor.node_id, neighbor_mod) in visited:
                    continue

                new_ctx = ctx + [self._load_modality(neighbor, neighbor_mod)]
                h = cached_heuristic(path_key, ctx, neighbor, neighbor_mod)
                new_path_key = path_key + ((neighbor.node_id, neighbor_mod),)
                counter += 1
                heapq.heappush(heap, (-h, counter, neighbor.node_id, neighbor_mod,
                                      new_path_key, path + [(neighbor, neighbor_mod)], new_ctx))

        return None  # no path found
    
    def _load_modality(self, node, modality):
        node_info = {k: v for k, v in node.subnodes[modality].items() if k != "id"}
        # load the image
        if "image" in node_info.keys():
            image = node_info["image"]
            image = Image.open(image)
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image = image.tobytes()
            node_info["image"] = image
        return node_info
    
    def calculate_heuristic(self, context, new_node, modality, end_node, task, alpha=1, beta=0.2):
        context_new = context[:]
        context_new.append(self._load_modality(new_node, modality))

        alignment = self.grade_alignment(context_new, task)
        distance = self.get_distance(new_node, end_node)
        return alignment * alpha - distance * beta

    def get_distance(self, node, other):
        return abs(node.info["id"] - other.info["id"])

    def grade_alignment(self, plan, task) -> int:
        contents = [TASK_ALIGNMENT_PROMPT]

        contents.append("Task:")
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
                landmarks_text = ", ".join(step["landmarks"])
                contents.append(f"Landmarks: {landmarks_text}")

        contents.append("Respond with a single integer from 1 to 10 rating how well the plan aligns with the task.")

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        return int(response.text.strip())


if __name__ == "__main__":
    my_graph = Graph.deserialize("graph.json")
