"""Matplotlib visualization for planner paths: list of (Node, modality) steps or plan JSON."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from PIL import Image


def _modality_includes_language(modality: str) -> bool:
    """True for language-only (L) or vision+language (VL) steps."""
    return "L" in modality


def _coerce_plan_input(plan, task: str | None) -> tuple[list, str | None]:
    """
    Normalize to a list of (node_like, modality, landmark_idx) where node_like has
    .node_id and .info.

    Accepts:
    - Sequence of (Node, modality, landmark_idx) from the planner
    - dict in exported plan.json shape (keys: task, steps, ...)
    - str | Path to a JSON file with that dict shape
    """
    if isinstance(plan, (str, Path)):
        with Path(plan).open(encoding="utf-8") as f:
            plan = json.load(f)

    if isinstance(plan, dict):
        json_task = plan.get("task")
        effective_task = task if task is not None else json_task
        steps_out: list = []
        for step in plan.get("steps") or []:
            node_id = step["node_id"]
            modality = step["modality"]
            landmark_idx = step.get("landmark_idx")
            info = {
                "id": step.get("frame_id", node_id),
                "image": step.get("image"),
                "landmarks": step.get("landmarks"),
            }
            node = SimpleNamespace(node_id=node_id, info=info)
            steps_out.append((node, modality, landmark_idx))
        return steps_out, effective_task

    return plan, task


def visualize_node_sequence_to_file(
    sequence,
    out_path: str | Path,
    *,
    task: str | None = None,
    dpi: int = 120,
    max_cols: int = 8,
    show: bool = False,
) -> Path:
    """
    Render a generated graph path (a list of :class:`Node`) to a PNG.

    Chooses a display modality per node (``VL`` > ``V`` > ``L``) so images and
    landmark text match :func:`visualize_plan_to_file` behavior.

    If ``show`` is True, opens an interactive matplotlib window after saving
    (requires a display backend).
    """
    if not sequence:
        out_path = Path(out_path)
        _render_empty_plan(out_path, task=task, dpi=dpi)
        if show:
            plt.figure(figsize=(6, 3))
            plt.imshow(plt.imread(out_path))
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        return out_path.resolve()

    plan: list = []
    for node in sequence:
        modalities = getattr(node, "modalities", None) or []
        if "VL" in modalities:
            modality = "VL"
        elif "V" in modalities:
            modality = "V"
        else:
            modality = "L"
        plan.append((node, modality, None))

    out = visualize_plan_to_file(plan, out_path, task=task, dpi=dpi, max_cols=max_cols)
    if show:
        plt.figure(figsize=(min(18, 2.2 * len(sequence) + 2), 9))
        plt.imshow(plt.imread(out))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return out


def visualize_plan_to_file(
    plan,
    out_path: str | Path,
    *,
    task: str | None = None,
    dpi: int = 120,
    max_cols: int = 8,
) -> Path:
    """
    Render `plan` to a PNG (or other format by extension).

    `plan` may be:
    - a sequence of (Node, modality) from the planner
    - a dict matching exported plan.json (``task``, ``steps`` with ``node_id``, ``modality``,
      ``frame_id``, ``image``, ``landmarks``)
    - a path to such a JSON file

    If ``task`` is None and ``plan`` is JSON (dict or file), the JSON ``task`` field is used.

    Returns the resolved output path.
    """
    out_path = Path(out_path)
    plan, task = _coerce_plan_input(plan, task)
    if not plan:
        _render_empty_plan(out_path, task=task, dpi=dpi)
        return out_path.resolve()

    n = len(plan)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig_w = 2.4 * ncols + 1.0
    fig_h = 3.0 * nrows + (1.2 if task else 0.6)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    if task:
        fig.suptitle(
            textwrap.fill(task, width=100),
            fontsize=11,
            y=0.98,
            wrap=True,
        )

    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= n:
            continue

        node, modality, landmark_idx = plan[idx]
        info = node.info
        frame_id = info.get("id", node.node_id)
        title = f"Step {idx + 1}  ·  node {node.node_id}  ·  {modality}  ·  frame {frame_id}"
        if landmark_idx is not None:
            title += f"  ·  lm {landmark_idx}"
        ax.set_title(title, fontsize=9, pad=6)

        image_path = info.get("image")
        landmarks = info.get("landmarks")
        if landmark_idx is not None and landmarks:
            landmarks = [landmarks[landmark_idx]]

        plotted = False
        if image_path and isinstance(image_path, str) and Path(image_path).is_file():
            try:
                img = Image.open(image_path).convert("RGB")
                ax.imshow(img, aspect="equal")
                plotted = True
            except OSError:
                plotted = False

        if not plotted and landmarks:
            body = "\n".join(f"• {lm}" for lm in landmarks)
            ax.text(
                0.5,
                0.5,
                textwrap.fill(body, width=36),
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_facecolor("#f4f4f4")
        elif plotted and _modality_includes_language(modality) and landmarks:
            body = "\n".join(f"• go to the {lm}" for lm in landmarks)
            ax.text(
                0.5,
                -0.08,
                textwrap.fill(body, width=44),
                ha="center",
                va="top",
                fontsize=7,
                transform=ax.transAxes,
                wrap=True,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#bbb",
                    "alpha": 0.93,
                },
            )
        elif not plotted:
            ax.text(
                0.5,
                0.5,
                "(no image / landmarks)",
                ha="center",
                va="center",
                fontsize=9,
                color="#666",
                transform=ax.transAxes,
            )
            ax.set_facecolor("#fafafa")

    plt.tight_layout(rect=(0, 0, 1, 0.94 if task else 1))
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path.resolve()


def _render_empty_plan(out_path: Path, task: str | None, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()
    msg = "No plan (planner returned None or empty path)."
    if task:
        msg = f"{msg}\n\nTask:\n{textwrap.fill(task, width=70)}"
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, wrap=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    visualize_plan_to_file(base / "plan.json", base / "plan.png")