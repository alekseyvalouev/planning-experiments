from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

PLAN_A_STAR_DIR = Path(__file__).resolve().parents[1] / "plan-a-star"
if str(PLAN_A_STAR_DIR) not in sys.path:
    sys.path.insert(0, str(PLAN_A_STAR_DIR))


def build_vint_graph(
    *,
    data_root: str | os.PathLike[str],
    scenes: Iterable[str],
    annotation_folder: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    sparsification_steps: int = 4,
    drop_modality_p: float = 0.0,
    dummy: bool = False,
    checkpoint: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    resume: bool = True,
) -> Path:
    import vint_graph
    from vint_graph import Graph

    output_path = Path(output_path)
    if resume and output_path.exists():
        print(f"[graph] reusing existing graph: {output_path}")
        return output_path

    if checkpoint is not None:
        vint_graph.CHECKPOINT = checkpoint
    if batch_size is not None:
        vint_graph.BATCH_SIZE = batch_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    old_device = os.environ.get("VINT_DEVICE")
    old_data_root = os.environ.get("VINT_DATA_ROOT")
    if device is not None:
        os.environ["VINT_DEVICE"] = device
    os.environ["VINT_DATA_ROOT"] = str(data_root)
    try:
        Graph(
            scenes=list(scenes),
            annotation_folder=str(annotation_folder),
            sparsification_steps=sparsification_steps,
            drop_modality_p=drop_modality_p,
            dummy=dummy,
            name=str(output_path),
        )
    finally:
        if old_device is None:
            os.environ.pop("VINT_DEVICE", None)
        else:
            os.environ["VINT_DEVICE"] = old_device
        if old_data_root is None:
            os.environ.pop("VINT_DATA_ROOT", None)
        else:
            os.environ["VINT_DATA_ROOT"] = old_data_root
    return output_path


def run_vint_smoke_test(
    *,
    data_root: str | os.PathLike[str],
    scenes: Iterable[str],
    checkpoint: str | None = None,
    device: str | None = None,
    context_size: int = 5,
) -> dict:
    import torch
    import vint_graph
    from vint_graph import Graph

    if checkpoint is not None:
        vint_graph.CHECKPOINT = checkpoint

    graph = Graph.__new__(Graph)
    graph.checkpoint = checkpoint or vint_graph.CHECKPOINT
    graph.image_cache = {}
    graph.vint_context = context_size
    if device == "auto":
        graph.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        graph.device = torch.device(device or "cuda:1")
    graph._load_vint()

    data_root = Path(data_root)
    for scene in scenes:
        scene_dir = data_root / scene
        if not scene_dir.is_dir():
            continue
        images = sorted(scene_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        if len(images) <= context_size + 1:
            continue
        context = [str(path) for path in images[: context_size + 1]]
        goal = str(images[context_size + 1])
        with torch.no_grad():
            score = graph.ask([(context, goal)], dummy=False)[0]
        return {
            "scene": scene,
            "context_frames": len(context),
            "goal_frame": Path(goal).name,
            "device": str(graph.device),
            "score": float(score),
        }

    raise FileNotFoundError(
        f"No scene under {data_root} had at least {context_size + 2} jpg frames"
    )
