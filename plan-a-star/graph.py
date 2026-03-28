# build a graph of the environment.

import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MODEL_ID    = "google/paligemma2-3b-pt-224"
CHECKPOINT  = "/home/alekseyvalouev/goalnav/language-distance/binary-reachability-paligemma/checkpoint-3900"
BATCH_SIZE  = 64

class Graph:
    def __init__(self, scenes, annotation_folder, sparsification_steps=4, drop_modality_p=0.5, dummy=False):
        self.model_id = MODEL_ID
        self.checkpoint = CHECKPOINT
        self.dummy = dummy
        self.scenes = scenes
        self.annotation_folder = annotation_folder
        self.sparsification_steps = sparsification_steps
        self.drop_modality_p = drop_modality_p
        self.structured_data = []
        self._load_paligemma()
        # we need to build a graph from the scenes.
        self._load_scenes() # data looks like an array of dictionaries, each dictionary contains the information for a single node.
        self._build_graph()
        self.serialize("graph.json")
    
    def _load_paligemma(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        self.model = PeftModel.from_pretrained(base_model, self.checkpoint)
        self.model.eval()

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

        self.data_path = "/hdd/sacson"
        for scene in self.scenes:
            folder = os.path.join(self.data_path, scene)
            # we're in the folder. get the highest #'ed image. 
            images = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder)))
            images.sort(key=lambda x: int(x.split(".")[0]))
            highest_image = images[-1]
            max_idx = int(highest_image.split(".")[0])

            for img in range(0, max_idx, self.sparsification_steps):
                curr_image = os.path.join(folder, images[img])
                curr_annotation = self.annotations[scene][images[img]]
                x = random.random()
                if x < self.drop_modality_p:
                    y = random.random()
                    if y < self.drop_modality_p:
                        self.structured_data.append({"id" : int(images[img].split(".")[0]), "image": curr_image})
                    else:
                        self.structured_data.append({"id" : int(images[img].split(".")[0]), "landmarks": curr_annotation})
                else:
                    self.structured_data.append({"id" : int(images[img].split(".")[0]), "image": curr_image, "landmarks": curr_annotation})
        
        return self.structured_data

    def _build_graph(self):
        # first, create nodes of each point in the environment, each node has up to 3 subnodes, one for each
        # modality combination ("V", "L", "VL").
        # then iterate through each subnode. for each subnode, iterate through all remaining subnodes 
        # and assess connectivity. if the two subnodes are connected, add an edge to the graph between the
        # subnodes. Each node should contain information about edges, including which subnodes are connected.
        # the cost of each edge is the same. we do not care about this information. EDGES ARE DIRECTED. COMPARISONS
        # SHOULD BE MADE BOTH WAYS. 
        self.nodes = [Node(i, info) for i, info in enumerate(self.structured_data)]

        # build flat list of (node, modality) subnodes for pairwise comparison
        subnodes = [
            (node, modality)
            for node in self.nodes
            for modality in node.modalities
        ]

        # collect all valid pairs and their prepared prompt data upfront
        pairs = []
        prompt_data_list = []
        for i, (node, modality) in enumerate(subnodes):
            for j, (other_node, other_modality) in enumerate(subnodes):
                if i == j:
                    continue
                pd = self._prepare_prompt_data(node, modality, other_node, other_modality)
                if pd is not None:
                    pairs.append((node, modality, other_node, other_modality))
                    prompt_data_list.append(pd)

        # run inference in batches
        results = []
        for start in tqdm(range(0, len(prompt_data_list), BATCH_SIZE), desc="Building graph"):
            batch = prompt_data_list[start : start + BATCH_SIZE]
            results.extend(self.ask(batch, dummy=self.dummy))

        for (node, modality, other_node, other_modality), response in zip(pairs, results):
            if response == "1":
                node.add_connection(modality, other_node, other_modality)
    
    def serialize(self, path):
        data = {
            "scenes": self.scenes,
            "sparsification_steps": self.sparsification_steps,
            "drop_modality_p": self.drop_modality_p,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "info": node.info,
                    "modalities": node.modalities,
                    "subnodes": node.subnodes,
                    "connections": [
                        {"modality": m, "other_node_id": other.node_id, "other_modality": om}
                        for m, other, om in node.connections
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
                node.add_connection(conn["modality"], node_by_id[conn["other_node_id"]], conn["other_modality"])
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

        return np.array(img)

    def _prepare_prompt_data(self, node, modality, other, other_modality):
        """Build the prompt dict for a single node pair, or return None if no images are present."""
        images = []
        start_prompt = ""
        end_prompt = ""

        if "V" in modality:
            images.append(self._load_image(node.info["image"]))
            start_prompt += "Starting image: <image>"
        if "L" in modality:
            start_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(node.info["landmarks"])])
            start_prompt += f"Starting landmarks: {start_landmarks_str}"

        if "V" in other_modality:
            images.append(self._load_image(other.info["image"]))
            end_prompt += "Ending image: <image>"
        if "L" in other_modality:
            end_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(other.info["landmarks"])])
            end_prompt += f"Ending landmarks: {end_landmarks_str}"

        if len(images) == 0:
            return None

        prompt = f"answer en {start_prompt} {end_prompt} What is the temporal distance?\n"
        return {"image": images, "prefix": prompt}

    def ask(self, prompt_data_list: list, dummy=False) -> list:
        """
        Run batched inference over a list of prompt dicts.
        Each dict has keys 'image' (list of np.ndarray) and 'prefix' (str).
        Returns a list of decoded response strings in the same order.
        """
        if dummy:
            return ["1" if random.random() < 0.01 else "0" for _ in prompt_data_list]

        # group by image count so pixel_values can be stacked within each sub-batch
        groups = defaultdict(list)
        for idx, pd in enumerate(prompt_data_list):
            groups[len(pd["image"])].append((idx, pd))

        results = [""] * len(prompt_data_list)

        for n_images, items in groups.items():
            indices, pds = zip(*items)
            texts  = [pd["prefix"] for pd in pds]
            images = [
                [Image.fromarray(img.astype(np.uint8)) for img in pd["image"]]
                for pd in pds
            ]
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            decoded = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
            for orig_idx, response in zip(indices, decoded):
                results[orig_idx] = response.strip()

        return results

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
        self.connections = [] # triple (modality, other_node, other_modality)
    
    def add_connection(self, modality, other_node, other_modality):
        self.connections.append((modality, other_node, other_modality))


if __name__ == "__main__":
    scenes = ['Feb-15-2023-cory1_00000004_6', 
    'Feb-16-2023-cory1-intloss_00000023_0', 
    'Feb-15-2023-cory1_00000006_5', 
    'Feb-15-2023-cory1_00000000_0', 
    'Feb-16-2023-cory1-intloss_00000021_1', 
    'Feb-15-2023-cory1_00000006_4'
    ]
    graph = Graph(scenes=scenes, annotation_folder="/home/alekseyvalouev/goalnav/language-annotations", dummy=False)
    