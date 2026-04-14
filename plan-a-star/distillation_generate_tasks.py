# The idea for generating tasks is to use the graph, pick a random node and modality+idx,
# then pick another random node and modality+idx. Then we do BFS between the two 
# to get the context. Select random modalities along the way. Then we generate a task
# by prompting gemini. 

from dotenv import load_dotenv
from google import genai

from graph import Graph, Node
from plan_visualization import visualize_node_sequence_to_file

from prompts import TASK_LABELING_PROMPT

import random
import os
import sys
sys.path.append("..")

load_dotenv(".env")

class TaskBuilder:
    def __init__(self, graph_file):
        self.graph = Graph.deserialize(graph_file)
        self._random_prune(prune_fraction=0.4)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def generate_task(self):
        path = self._generate_path()
        if path is None:
            print("No path found.")
            return None, None
        task = self._label_task(path)
        return path, task
    
    def _random_prune(self, prune_fraction):
        # randomly prune 20% of connections in the graph
        for node in self.graph.nodes:
            node.connections = random.sample(node.connections, int(len(node.connections) * (1 - prune_fraction)))

    def _generate_path(self):
        start_node = random.choice(self.graph.nodes)
        intermediate_node = random.choice(self.graph.nodes)
        end_node = random.choice(self.graph.nodes)
        if len(set([start_node, intermediate_node, end_node])) != 3:
            print("Planned nodes are not unique. Generating new task.")
            return self.generate_task()
        # give them all the plausible info for the plan
        path_1 = self._bfs(start_node, intermediate_node)
        path_2 = self._bfs(intermediate_node, end_node)
        # dist = self._dfs(start_node, end_node, 4)
        if path_1 is None or path_2 is None:
            print("No path found between start and end node. Generating new task.")
            return self.generate_task()
        full_path = path_1[:-1] + path_2
        print(full_path)
        print(len(full_path))
        return full_path
    
    def _label_task(self, path):
        task = ""
        contents = []
        
        contents.append(TASK_LABELING_PROMPT)

        contents.append("Context:")
        for i, step in enumerate(path):
            contents.append(f"Step {i + 1}:")
            if "V" in step.modalities:
                with open(step.subnodes["V"]["image"], 'rb') as f:
                    image_bytes = f.read()
                    contents.append(genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            if "L" in step.modalities:
                landmarks_str = ""
                for j, landmark in enumerate(step.subnodes["L"]["landmarks"]):
                    landmarks_str += f"{j + 1}. {landmark}\n"
                contents.append(f"Landmarks:\n{landmarks_str}")

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
        )
        return response.text.split("Final Draft: ")[1].strip()

    def _bfs(self, start_node, end_node):
        visited = set()
        queue = [(start_node, [start_node])]
        while queue:
            node, path = queue.pop(0)
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            if node == end_node:
                return path
            for neighbor in node.connections:
                if neighbor.node_id not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _dfs(self, start_node, end_node, path_length):
        """Depth-first search from start to end. Branches whose path would exceed
        ``path_length`` nodes (inclusive of start) are not explored.

        Returns a list of ``Node`` from start to end, or ``None`` if no path exists
        within the bound.
        """
        if path_length < 1:
            return None
        if start_node == end_node:
            return [start_node]

        def dfs(node, path, on_path_ids):
            if node == end_node:
                return list(path)
            if len(path) >= path_length:
                return None
            for neighbor in node.connections:
                if neighbor.node_id in on_path_ids:
                    continue
                if len(path) + 1 > path_length:
                    continue
                on_path_ids.add(neighbor.node_id)
                path.append(neighbor)
                result = dfs(neighbor, path, on_path_ids)
                if result is not None:
                    return result
                path.pop()
                on_path_ids.remove(neighbor.node_id)
            return None

        return dfs(start_node, [start_node], {start_node.node_id})

    def visualize_sequence(
        self,
        sequence,
        out_path="path_visualization.png",
        *,
        task=None,
        show=False,
        dpi=120,
        max_cols=8,
    ):
        """Save a matplotlib figure of the node path (images + landmark text). See
        :func:`plan_visualization.visualize_node_sequence_to_file`."""
        return visualize_node_sequence_to_file(
            sequence,
            out_path,
            task=task,
            dpi=dpi,
            max_cols=max_cols,
            show=show,
        )

if __name__ == "__main__":
    builder = TaskBuilder("graph_no_drop.json")
    path, task = builder.generate_task()
    if path:
        builder.visualize_sequence(
            path,
            out_path="path_visualization.png",
            task=task,
            show=False,
        )
    print(task)
