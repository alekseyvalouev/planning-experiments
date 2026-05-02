# build a graph of the environment.
# run with env nomad_train

import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, "/home/alekseyvalouev/goalnav/visualnav-transformer/train")

from vint_train.models.vint.vint import ViNT

CHECKPOINT  = "/home/alekseyvalouev/goalnav/vint.pth"
BATCH_SIZE  = 16

class Graph:
    def __init__(self, scenes, annotation_folder, sparsification_steps=4, drop_modality_p=0.0, dummy=False, name="graph.json"):
        self.checkpoint = CHECKPOINT
        self.dummy = dummy
        self.scenes = scenes
        self.annotation_folder = annotation_folder
        self.sparsification_steps = sparsification_steps
        self.drop_modality_p = drop_modality_p
        self.structured_data = []
        self.image_cache = {}
        self.vint_context = 5
        self.device = torch.device(os.environ.get("VINT_DEVICE", "cuda:1"))
        self._load_vint()
        # we need to build a graph from the scenes.
        self._load_scenes() # data looks like an array of dictionaries, each dictionary contains the information for a single node.
        self.reachability_threshold = 10
        self._build_graph()
        self.serialize(name)
    
    def _load_vint(self):
        try:
            state_dict = torch.load(self.checkpoint, weights_only=False)
        except TypeError:
            # Older PyTorch versions do not expose weights_only.
            state_dict = torch.load(self.checkpoint)
        #self.model = ViNT(context_size=self.vint_context, len_traj_pred=1, learn_angle=False)
        self.model = state_dict["model"]
        # Checkpoint pickled the full module under an older PyTorch that did not
        # set `activation_relu_or_gelu` on TransformerEncoderLayer. Newer PyTorch
        # forward() reads this attr, so backfill it. The decoder uses GELU -> 2.
        for m in self.model.modules():
            if isinstance(m, torch.nn.TransformerEncoderLayer) and not hasattr(m, "activation_relu_or_gelu"):
                act = getattr(m, "activation", None)
                if isinstance(act, torch.nn.ReLU) or act is torch.nn.functional.relu:
                    m.activation_relu_or_gelu = 1
                elif isinstance(act, torch.nn.GELU) or act is torch.nn.functional.gelu:
                    m.activation_relu_or_gelu = 2
                else:
                    m.activation_relu_or_gelu = 0
        self.model.eval()
        self.model.to(self.device)

    def _load_scenes(self):
        # load the structured data from the scenes.
        self.annotations = {}
        for annotation_file in os.listdir(self.annotation_folder):
            annotated_scene = annotation_file.replace("_landmarks.json", "")
            if annotated_scene not in self.scenes:
                continue
            with open(os.path.join(self.annotation_folder, annotation_file), "r") as f:
                loaded = json.load(f)
                self.annotations[annotated_scene] = loaded["landmarks"]

        self.data_path = os.environ.get("VINT_DATA_ROOT", "/hdd/sacson")
        for scene in self.scenes:
            folder = os.path.join(self.data_path, scene)
            # we're in the folder. get the highest #'ed image. 
            images = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder)))
            images.sort(key=lambda x: int(x.split(".")[0]))
            highest_image = images[-1]
            max_idx = int(highest_image.split(".")[0])


            for img in range(self.vint_context, max_idx, self.sparsification_steps):
                curr_image = os.path.join(folder, images[img])
                context = [os.path.join(folder, images[i]) for i in range(img - self.vint_context, img+1)]
                curr_annotation = self.annotations[scene][images[img]]
                self.structured_data.append({"id" : int(images[img].split(".")[0]), "vint_context": context, "image": curr_image, "landmarks": curr_annotation})
        
        return self.structured_data

    def _build_graph(self):
        # Connectivity is at the node (parent) level: if ANY subnode pair between two
        # nodes is connected, the nodes are connected.  Once a directed pair (A→B) is
        # found connected we skip all remaining subnode comparisons for that pair.
        self.nodes = [(info["vint_context"], Node(i, info)) for i, info in enumerate(self.structured_data)]

        pairs = []
        prompt_data_list = []
        for i, (context, node) in enumerate(self.nodes):
            for j, (other_context, other_node) in enumerate(self.nodes):
                if node.node_id == other_node.node_id:
                    continue
                pairs.append((node, other_node))
                prompt_data_list.append((context, other_node.subnodes["V"]["image"]))

        connected = set()
        pos = 0
        pbar = tqdm(total=len(pairs), desc="Building graph")

        while pos < len(pairs):
            batch_pairs = []
            batch_prompts = []

            while pos < len(pairs) and len(batch_pairs) < BATCH_SIZE:
                node, other_node = pairs[pos]
                pd = prompt_data_list[pos]
                pos += 1
                pbar.update(1)

                if (node.node_id, other_node.node_id) in connected:
                    continue

                batch_pairs.append((node, other_node))
                batch_prompts.append(pd)

            if not batch_pairs:
                continue

            results = self.ask(batch_prompts, dummy=self.dummy)

            for (node, other_node), response in zip(batch_pairs, results):
                if response <= self.reachability_threshold and (node.node_id, other_node.node_id) not in connected:
                    connected.add((node.node_id, other_node.node_id))
                    node.add_connection(other_node)

        pbar.close()
    
    def serialize(self, path):
        data = {
            "scenes": self.scenes,
            "sparsification_steps": self.sparsification_steps,
            "drop_modality_p": self.drop_modality_p,
            "nodes": [
                {
                    "node_id": node[1].node_id,
                    "info": node[1].info,
                    "modalities": node[1].modalities,
                    "subnodes": node[1].subnodes,
                    "connections": [
                        {"other_node_id": other.node_id}
                        for other in node[1].connections
                    ],
                }
                for node in self.nodes
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def deserialize(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        instance = cls.__new__(cls)
        instance.scenes = data["scenes"]
        instance.sparsification_steps = data["sparsification_steps"]
        instance.drop_modality_p = data["drop_modality_p"]
        instance.structured_data = [n["info"] for n in data["nodes"]]
        instance.nodes = [Node(n["node_id"], n["info"]) for n in data["nodes"]]
        node_by_id = {node.node_id: node for node in instance.nodes}
        for n, node in zip(data["nodes"], instance.nodes):
            node.subnodes = n.get("subnodes", node.subnodes)
            for conn in n["connections"]:
                node.add_connection(node_by_id[conn["other_node_id"]])
        return instance

    def _load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")

        # Resize so that the smallest dimension is 224 while preserving aspect ratio
        width, height = img.size
        scale = 224.0 / min(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # Center crop to 224x224
        left = (new_width - 224) // 2
        top = (new_height - 224) // 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        return arr

    @torch.no_grad()
    def ask(self, prompt_data_list: list, dummy=False) -> list:
        """
        Run batched inference over a list of prompt dicts.
        Each dict has keys 'image' (list of np.ndarray) and 'prefix' (str).
        Returns a list of decoded response strings in the same order.
        """
        if dummy:
            return ["1" if random.random() < 0.01 else "0" for _ in prompt_data_list]
        
        contexts = []
        end_images = []

        for context, end_image in prompt_data_list:
            context_images = []
            for image in context:
                if image not in self.image_cache:
                    self.image_cache[image] = self._load_image(image)
                context_images.append(torch.tensor(self.image_cache[image]).to(self.device))
            contexts.append(torch.cat(context_images, dim=0))
            
            if end_image not in self.image_cache:
                self.image_cache[end_image] = self._load_image(end_image)
            end_image = torch.tensor(self.image_cache[end_image]).to(self.device)
            end_images.append(end_image)
        contexts = torch.stack(contexts)
        end_images = torch.stack(end_images)
        out, _ = self.model(contexts, end_images)

        return out.squeeze(-1).cpu().numpy().tolist()

class Node:
    def __init__(self, node_id, info):
        # info is a dictionary with the following keys {"image" : image, "landmarks" : text}
        self.node_id = node_id
        self.info = info
        self.modalities = []
        self.subnodes = {}
        self.cache = {} # to store arbitrary info
        if "image" in info.keys():
            self.modalities.append("V")
            self.subnodes["V"] = {"image": info["image"]}
        if "landmarks" in info.keys():
            self.modalities.append("L")
            self.subnodes["L"] = {"landmarks": info["landmarks"]}
        if "image" in info.keys() and "landmarks" in info.keys():
            self.modalities.append("VL")
            self.subnodes["VL"] = {"image": info["image"], "landmarks": info["landmarks"]}
        self.connections = []
    
    def add_connection(self, other_node):
        self.connections.append(other_node)


if __name__ == "__main__":
    scenes = [
        'Feb-15-2023-cory1_00000004_6', 
        'Feb-16-2023-cory1-intloss_00000023_0', 
        'Feb-15-2023-cory1_00000006_5', 
        'Feb-15-2023-cory1_00000000_0', 
        'Feb-16-2023-cory1-intloss_00000021_1', 
        'Feb-15-2023-cory1_00000006_4'
    ]
    graph = Graph(scenes=scenes, drop_modality_p=0.0, annotation_folder="/home/alekseyvalouev/goalnav/language-annotations-other-new", dummy=False, name='graph_vint_tight.json')

    #annotation_folder = "/home/alekseyvalouev/goalnav/language-annotations-train-new"
    #scenes = [
    #    f.replace("_landmarks.json", "")
    #    for f in os.listdir(annotation_folder)
    #    if f.endswith("_landmarks.json")
    #]
    #graph = Graph(
    #    scenes=scenes,
    #    drop_modality_p=0.0,
    #    annotation_folder=annotation_folder,
    #    dummy=False,
    #    name="graph_vint_train.json",
    #)
