# run with env nomad_train
import sys
sys.path.append("../plan-a-star")

from vint_graph import Graph

import os

sys.path.insert(0, "/home/alekseyvalouev/goalnav/visualnav-transformer/train")

from vint_train.models.vint.vint import ViNT
import torch

from PIL import Image
import numpy as np

from itertools import groupby


def _load_vint(device):
    state_dict = torch.load("/home/alekseyvalouev/goalnav/vint.pth", weights_only=False)
    #self.model = ViNT(context_size=self.vint_context, len_traj_pred=1, learn_angle=False)
    model = state_dict["model"]
    # Checkpoint pickled the full module under an older PyTorch that did not
    # set `activation_relu_or_gelu` on TransformerEncoderLayer. Newer PyTorch
    # forward() reads this attr, so backfill it. The decoder uses GELU -> 2.
    for m in model.modules():
        if isinstance(m, torch.nn.TransformerEncoderLayer) and not hasattr(m, "activation_relu_or_gelu"):
            act = getattr(m, "activation", None)
            if isinstance(act, torch.nn.ReLU) or act is torch.nn.functional.relu:
                m.activation_relu_or_gelu = 1
            elif isinstance(act, torch.nn.GELU) or act is torch.nn.functional.gelu:
                m.activation_relu_or_gelu = 2
            else:
                m.activation_relu_or_gelu = 0
    model.eval()
    model.to(device)
    return model

def _load_image(image_path):
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

def _build_inputs(start_image, end_image, device):
    folder = os.path.dirname(start_image)
    start_image_idx = int(start_image.split("/")[-1].split(".")[0])
    if start_image_idx < 5:
        return None, None
    context = [os.path.join(folder, f"{i}.jpg") for i in range(start_image_idx - 5, start_image_idx+1)]

    context = torch.cat([torch.tensor(_load_image(image)).to(device) for image in context], dim=0)
    end_image = torch.tensor(_load_image(end_image)).to(device)

    return context.unsqueeze(0), end_image.unsqueeze(0)

def _check_connectivity_node(vint_model, start_node, end_node):
    context, end_image = _build_inputs(start_node.subnodes["V"]["image"], end_node.subnodes["V"]["image"], device)
    if context is None or end_image is None:
        return 100
    out, _ = vint_model(context, end_image)
    return out.item()

if __name__ == "__main__":
    #indices = [75, 75, 143, 143, 144, 85, 117, 117, 144, 12]
    indices = [125, 173, 119, 39, 33, 124, 48, 75, 34, 78]
    my_graph = Graph.deserialize("../plan-a-star/graph_no_drop.json")
    new_indices = [key for key, group in groupby(indices)]
    print(new_indices)

    device = torch.device("cuda:1")
    vint_model = _load_vint(device)

    vint_ids = []
    for i in range(len(new_indices) - 1):
        start_node = my_graph.nodes[new_indices[i]]
        end_node = my_graph.nodes[new_indices[i+1]]
        connectivity = _check_connectivity_node(vint_model, start_node, end_node)
        vint_ids.append(connectivity)
    print(vint_ids)