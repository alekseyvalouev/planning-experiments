# run with env siglip-adapter
from pathlib import Path
import json
import math
import os
import textwrap
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
sys.path.append("../plan-a-star")
sys.path.append("../../modality-fusion/")

from siglip_adapter import Embedder, SplitEmbedder
from graph import Graph, Node
from plan_visualization import visualize_plan_to_file


class RetrievalPlanner:
    def __init__(self, graph, checkpoint):
        self.graph = graph
        self._init_embedder(checkpoint)
        self.images, self.texts, self.embeddings = self._build_embeddings(graph)
        print(self.embeddings.shape)
    
    def _init_embedder(self, checkpoint):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedder = SplitEmbedder(device=device, skip_connection="skip" in checkpoint, use_pos_embedding="pe" in checkpoint)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        self.temperature = torch.tensor(checkpoint["temperature"], device=device)
        self.bias = torch.tensor(checkpoint["bias"], device=device)
        """
        fusion_dict = {}
        siglip_dict = {}
        for key in state_dict.keys():
            if "fusion_layer" in key:
                fusion_dict[key.replace("fusion_layer.", "")] = state_dict[key]
            if "siglip_adapter" in key:
                siglip_dict[key.replace("siglip_adapter.model.", "")] = state_dict[key]
        #print(state_dict.keys())
        self.embedder.fusion_layer.load_state_dict(fusion_dict)
        self.embedder.fusion_layer.eval()
        self.embedder.fusion_layer.to(device)
        self.embedder.siglip_adapter.model.load_state_dict(siglip_dict)
        self.embedder.siglip_adapter.model.eval()
        self.embedder.siglip_adapter.model.to(device)
        self.embedder.device = device
        self.embedder.siglip_adapter.device = device
        self.embedder.eval()
        """
        self.embedder.load_state_dict(state_dict)
        self.embedder.eval()
        self.embedder.to(device)

    def plan(self, start_image, task, max_context):
        default_text = "x"*64
        default_image = torch.zeros((224, 224, 3), dtype=torch.uint8)

        reference_embedding = self.embedder(default_text, default_image, prefix_text=task, prefix_image=start_image)
        indices = []
        reference_embedding = F.normalize(reference_embedding, dim=-1)
        self.embeddings = F.normalize(self.embeddings, dim=-1)
        distances = reference_embedding @ self.embeddings.T #* torch.exp(self.temperature) + self.bias
        print(distances[0])
        indices = distances[0].argsort().detach().cpu().numpy()[-max_context:]
        for i in indices:
            print(distances[0][i])

        return [(self.images[i], self.texts[i]) for i in indices]

    def plan_recursive(self, start_image, task, max_context):
        default_text = ["x"*64]
        default_image = torch.zeros((1, 224, 224, 3), dtype=torch.uint8, device=self.embedder.device)
        start_image = torch.tensor(start_image).unsqueeze(0).to(self.embedder.device)
        print(start_image.shape)
        task = [task]

        reference_embedding = self.embedder(default_text, start_image, prefix_text=task, prefix_image=start_image)
        print("BUILT REFERENCE EMBEDDING.")
        indices = []
        self.embeddings = F.normalize(self.embeddings, dim=-1)
        for i in range(max_context):
            reference_embedding = F.normalize(reference_embedding, dim=-1)
            distances = reference_embedding @ self.embeddings.T
            #print("distances:")
            #print(distances)
            index = distances[0].argsort().detach().cpu().numpy()
            print("Best")
            for i in index[-5:]:
                print(distances[0][i])
            print("======")
            print("Worst")
            for i in index[:5]:
                print(distances[0][i])
            start_id = 1
            max_index = None
            for i in range(len(index)):
                if index[-i] not in indices:
                    max_index = index[-i]
                    break
            #max_index = index[-1]

            indices.append(max_index)
            #indices.append(index[-2])
            #indices.append(index[-3])
            new_text = self._process_full_landmarks(self.texts[max_index])
            reference_embedding = self.embedder([new_text], torch.tensor(self.images[max_index], device=self.embedder.device).unsqueeze(0), prefix_text=task, prefix_image=start_image)

        return indices, [(self.images[i], self.texts[i], ", ".join(self.graph.nodes[self.ids[i]].subnodes["VL"]["landmarks"]), self.graph.nodes[self.ids[i]].subnodes["VL"]["image"]) for i in indices]
    
    def _paired_dot_products(self, pairs, task=None, start_image=None, out_path: str | Path = "paired_dot_products.png", dpi: int = 120):
        # self.ids is the list of node ids in the order they are in the embedding matrix
        resolved_indices = []
        dot_products = []
        if task is not None:
            task = [task]
        for pair in pairs:
            image_0 = self.graph.nodes[pair[0]].subnodes["VL"]["image"]
            image_1 = self.graph.nodes[pair[1]].subnodes["VL"]["image"]
            text_0 = self.graph.nodes[pair[0]].subnodes["VL"]["landmarks"]
            text_1 = self.graph.nodes[pair[1]].subnodes["VL"]["landmarks"]
            text_0 = [self._process_full_landmarks(text_0)]
            text_1 = [self._process_full_landmarks(text_1)]
            default_text = ["x"*64]
            image_0 = torch.tensor(self._load_image(image_0), device=self.embedder.device).unsqueeze(0)
            image_1 = torch.tensor(self._load_image(image_1), device=self.embedder.device).unsqueeze(0)
            start_embedding = self.embedder(default_text, image_0, prefix_text=None, prefix_image=None)
            end_embedding = self.embedder(default_text, image_1, prefix_text=None, prefix_image=None)
            start_embedding = F.normalize(start_embedding, dim=-1)
            end_embedding = F.normalize(end_embedding, dim=-1)
            dot_products.append((start_embedding @ end_embedding.T).item())
            resolved_indices.append((pair[0], pair[1]))

        for i, dot_product in enumerate(dot_products):
            print(f"({pairs[i][0]},{pairs[i][1]}) -> {dot_product}")

        n = len(pairs)
        fig, axes = plt.subplots(
            n, 3,
            figsize=(7.0, 2.6 * n),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.0, 0.6, 1.0]},
        )

        for row, (pair, (idx_a, idx_b), dp) in enumerate(zip(pairs, resolved_indices, dot_products)):
            for col, (node_id, emb_idx) in enumerate(zip(pair, (idx_a, idx_b))):
                ax_col = 0 if col == 0 else 2
                ax = axes[row][ax_col]
                ax.set_axis_off()
                if emb_idx is None:
                    ax.text(0.5, 0.5, f"id {node_id} not found",
                            ha="center", va="center", fontsize=9, transform=ax.transAxes)
                    continue
                image = self.images[emb_idx]
                if image.max() == 0:
                    ax.set_facecolor("#f0f0f0")
                    caption = self.texts[emb_idx]
                    if isinstance(caption, (list, tuple)):
                        caption = " ".join(caption)
                    ax.text(0.5, 0.5, textwrap.fill(str(caption), 28),
                            ha="center", va="center", fontsize=8, transform=ax.transAxes)
                else:
                    ax.imshow(image, aspect="equal")
                ax.set_title(f"id {node_id}", fontsize=9, pad=4)

            mid_ax = axes[row][1]
            mid_ax.set_axis_off()
            mid_ax.text(
                0.5, 0.5,
                f"({pair[0]}, {pair[1]})\ndot = {dp:.4f}",
                ha="center", va="center", fontsize=10,
                transform=mid_ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.95),
            )

        fig.tight_layout()
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved paired dot-product visualization to {out_path.resolve()}")
        return out_path.resolve()
    
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
    
    def _process_full_landmarks(self, landmarks):
        landmarks = " ".join(landmarks)
        landmarks = landmarks.replace("one", "")
        landmarks = landmarks.replace("two", "")
        landmarks = landmarks.replace("three", "")
        landmarks = landmarks.replace("four", "")
        landmarks = landmarks.replace(" on the ", " ")
        landmarks = landmarks.replace(" a ", " ")
        landmarks = landmarks.replace(" an ", " ")
        landmarks = landmarks.replace(" the ", " ")
        landmarks = landmarks.replace("several", " ")
        return landmarks
    
    def _build_embeddings(self, graph):
        # indexed embeddings by node id
        images = []
        texts = []
        self.ids = []

        for node in graph.nodes:
            image = None
            text = None
            for modality in node.modalities:
                if modality == "V":
                    image = self._load_image(node.subnodes[modality]["image"])
                    images.append(image)
                    texts.append("x"*64)
                    #pass
                elif modality == "L":
                    text = node.subnodes[modality]["landmarks"]
                    text = self._process_full_landmarks(text)
                    texts.append(text)
                    images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    #for landmark in text:
                    #    texts.append("Go to the " + landmark)
                    #    images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                elif modality == "VL":
                    text = node.subnodes[modality]["landmarks"]
                    text = self._process_full_landmarks(text)
                    image = self._load_image(node.subnodes[modality]["image"])
                    texts.append(text)
                    images.append(image)
                    #for landmark in text:
                    #    images.append(image)
                    #    texts.append("Go to the " + landmark)
                self.ids.append(node.node_id)
        embeddings = torch.tensor([], device=self.embedder.device)
        BATCH_SIZE = 64
        for i in range(0, len(images), BATCH_SIZE):
            batch_images = torch.tensor(np.array(images[i:i+BATCH_SIZE]), device=self.embedder.device)
            batch_texts = texts[i:i+BATCH_SIZE]
            with torch.no_grad():
                batch_embeddings = self.embedder(batch_texts, batch_images)
                embeddings = torch.cat([embeddings, batch_embeddings], dim=0)
        return images, texts, embeddings
    
def visualize_plan(
    steps: list[tuple[np.ndarray, str]],
    out_path: str | Path,
    *,
    task: str | None = None,
    dpi: int = 120,
    max_cols: int = 6,
) -> Path:
    """Render retrieval plan steps to a PNG.

    Each element of *steps* is an ``(image, text)`` pair returned by
    :meth:`RetrievalPlanner.plan`.
    """
    out_path = Path(out_path)

    if not steps:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_axis_off()
        msg = "No plan steps returned."
        if task:
            msg += f"\n\nTask: {textwrap.fill(task, 70)}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path.resolve()

    n = len(steps)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig_w = 2.6 * ncols + 0.8
    fig_h = 3.2 * nrows + (1.2 if task else 0.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    if task:
        fig.suptitle(textwrap.fill(task, 100), fontsize=12, y=0.98)

    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= n:
            continue

        print("Step: ",idx)

        image, text, landmarks, image_path = steps[idx]
        ax.set_title(f"Step {idx + 1}", fontsize=9, pad=6)

        is_blank = image.max() == 0
        if not is_blank:
            ax.imshow(image, aspect="equal")
            print("Image Path: ", image_path)

        caption = text.strip().replace("  ", "\n")
        is_placeholder = len(set(caption)) <= 1

        if is_blank and not is_placeholder:
            print("Landmarks: ", landmarks)
            ax.text(
                0.5, 0.5,
                textwrap.fill(caption, 36),
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_facecolor("#f0f0f0")
        elif not is_blank and not is_placeholder:
            print("Landmarks: ", landmarks)
            ax.text(
                0.5, -0.06,
                textwrap.fill(caption, 44),
                ha="center", va="top", fontsize=7,
                transform=ax.transAxes, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.93),
            )
        elif is_blank:
            ax.text(
                0.5, 0.5, "(no data)",
                ha="center", va="center", fontsize=9, color="#999",
                transform=ax.transAxes,
            )
            ax.set_facecolor("#fafafa")

        print("======")

    plt.tight_layout(rect=(0, 0, 1, 0.93 if task else 1))
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path.resolve()


if __name__ == "__main__":
    #my_graph = Graph.deserialize("../plan-a-star/graph_vint_loose.json")
    my_graph = Graph.deserialize("../plan-a-star/graph_no_drop.json")
    #my_graph = Graph.deserialize("../plan-a-star/graph_vint_train.json")
    # best so far:
    #ckpt = "/home/alekseyvalouev/goalnav/modality-fusion/checkpoints/recursive_frozen_size64_d05/epoch_99.pt"
    #ckpt = "/home/alekseyvalouev/goalnav/modality-fusion/checkpoints/recursive_frozen_size512_d05/epoch_99.pt"
    ckpt = "/home/alekseyvalouev/goalnav/modality-fusion/checkpoints/recursive_frozen_attention_skip_pe_task_old_data/epoch_49.pt"
    #ckpt = "/home/alekseyvalouev/goalnav/modality-fusion/checkpoints/recursive_frozen_attention_fixedsquash/epoch_69.pt"
    planner = RetrievalPlanner(my_graph, ckpt)
    #task = "Go down the hallway. First, pass the double doors and the small white object. Next, pass the grey lockers and the boards. Then, go past the trash can and the electric scooter. Next, pass the yellow wet floor sign. Finally, stop by the staircase and the framed pictures."
    #task = "Go to the end of the hall. Then go to the area with the desks. Next, go to the yellow divider. Then go to the kitchen."
    #task = "Go to the scooter"
    #task = "Go to the scooter. Then go to the lockers."
    task = "Go to the ladder. Then go to the end of the hallway. Then go to the lockers. Then go to the scooter. "
    start_image = my_graph.nodes[2].subnodes["V"]["image"]
    indices, steps = planner.plan_recursive(planner._load_image(start_image), task, 10)
    steps = [(planner._load_image(start_image), "Start", "Start", start_image)] + steps
    out = visualize_plan(steps, "plan_output.png", task=task)
    pairs = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (1, 18), (1, 20), (1, 30), (2, 18), (2, 20), (2, 30)]
    planner._paired_dot_products(pairs, task=task, start_image=planner._load_image(start_image))
    print(f"Saved to {out}")
    print("Indices:")
    print([planner.ids[i] for i in indices])