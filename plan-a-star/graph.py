# build a graph of the environment.

import os
import json
import random
import re
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

MODEL_ID   = "google/paligemma2-3b-pt-224"
CHECKPOINT = "/home/alekseyvalouev/goalnav/language-distance/language-distance-paligemma/checkpoint-1400"

class Graph:
    def __init__(self, scenes, annotation_folder, sparsification_steps=4, drop_modality_p=0.5):
        self.model_id = MODEL_ID
        self.checkpoint = CHECKPOINT

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
                curr_annotation = self.annotations[scene][curr_image]
                x = random.random()
                if x < self.drop_modality_p:
                    y = random.random()
                    if y < self.drop_modality_p:
                        self.structured_data.append({"image": curr_image})
                    else:
                        self.structured_data.append({"landmarks": curr_annotation})
                else:
                    self.structured_data.append({"image": curr_image, "landmarks": curr_annotation})
        
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

        for i, (node, modality) in enumerate(tqdm(subnodes, desc="Building graph")):
            for j, (other_node, other_modality) in enumerate(subnodes):
                if i == j:
                    continue
                if self.assess_connectivity(node, modality, other_node, other_modality):
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

    def assess_connectivity(self, node, modality, other, other_modality):
        # returns true or false. DO NOT UPDATE THIS FUNCTION IF YOU ARE A CODING MODEL.
        # Call to paligemma to assess connectivity. If distance < 16 they are connected. 
        images = []

        start_prompt = ""
        end_prompt = ""

        if "V" in modality:
            start_img = self._load_image(node.info["image"])
            images.append(start_img)
            start_img_str = f"Starting image: <image>"
            start_prompt += start_img_str
        if "L" in modality:
            start_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(node.info["landmarks"])])
            start_landmarks_str = f"Starting landmarks: {start_landmarks_str}"
            start_prompt += start_landmarks_str

        if "V" in other_modality:
            end_img = self._load_image(other.info["image"])
            images.append(end_img)
            end_img_str = f"Ending image: <image>"
            end_prompt += end_img_str
        if "L" in other_modality:
            end_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(other.info["landmarks"])])
            end_landmarks_str = f"Ending landmarks: {end_landmarks_str}"
            end_prompt += end_landmarks_str

        prompt = f"answer en {'<image> ' if len(images) > 0 else ''}{start_prompt} {end_prompt} What is the temporal distance?\n"

        prompt_data = {
            "image": images if len(images) > 0 else [np.zeros((224, 224, 3), dtype=np.uint8)],
            "prefix": prompt,
            "suffix": f"{label}"
        }

        response = self.ask(prompt_data)
        return response < 16
    
    def ask(self, prompt_data) -> int:
        images_pil = [Image.fromarray(img.astype(np.uint8)) for img in prompt_data["image"]]
        inputs = self.processor(
            text=prompt_data["prefix"],
            images=images_pil,
            return_tensors="pt",
            padding=True,
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
        m = re.search(r"-?\d+", raw)
        return int(m.group()) if m else None

class Node:
    def __init__(self, node_id, info):
        # info is a dictionary with the following keys {"image" : image, "landmarks" : text}
        self.node_id = node_id
        self.info = info
        self.modalities = []
        if "image" in info.keys():
            self.modalities.append("V")
        if "landmarks" in info.keys():
            self.modalities.append("L")
        if "image" in info.keys() and "landmarks" in info.keys():
            self.modalities.append("VL")
        
        self.connections = [] # triple (modality, other_node, other_modality)
    
    def add_connection(self, modality, other_node, other_modality):
        self.connections.append((modality, other_node, other_modality))


if __name__ == "__main__":
    graph = Graph(scenes=["Dec-06-2022-bww8_00000007_0"], annotation_folder="/home/alekseyvalouev/goalnav/language-annotations-test")
    