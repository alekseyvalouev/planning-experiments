from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review SACSoN trajectories and write accepted scene names as YAML list entries."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/hdd/sacson"))
    parser.add_argument(
        "--filter",
        default="cory1",
        help="Substring or glob used to match trajectory directory names.",
    )
    parser.add_argument("--output", type=Path, default=Path(f"location_scenes"))
    parser.add_argument("--gif-dir", type=Path, default=Path("trajectory_vids"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--accept-all", action="store_true")
    args = parser.parse_args()
    args.output = args.output / Path(f"{args.filter}.txt")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.min_length <= 0:
        parser.error("--min-length must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def filter_to_glob(filter_text: str) -> str:
    return filter_text if any(char in filter_text for char in "*?[]") else f"*{filter_text}*"


def discover_trajectories(data_root: Path, filter_text: str) -> list[Path]:
    pattern = filter_to_glob(filter_text)
    traj_paths = [
        path
        for path in data_root.glob(pattern)
        if path.is_dir() and (path / "traj_data.pkl").is_file()
    ]
    return sorted(traj_paths, key=lambda path: path.name)


def load_positions(traj_path: Path) -> np.ndarray:
    with (traj_path / "traj_data.pkl").open("rb") as f:
        traj = np.load(f, allow_pickle=True)
    positions = np.asarray(traj["position"])
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] < 2:
        raise ValueError(f"{traj_path} has invalid position data with shape {positions.shape}")
    return positions


def trajectory_length(traj_path: Path) -> int:
    return load_positions(traj_path).shape[0]


def filter_by_min_length(traj_paths: list[Path], min_length: int) -> list[Path]:
    long_enough = []
    for traj_path in traj_paths:
        try:
            length = trajectory_length(traj_path)
        except Exception as exc:
            print(f"Skipping {traj_path.name}: failed to read trajectory length ({exc})")
            continue
        if length >= min_length:
            long_enough.append(traj_path)
        else:
            print(f"Skipping {traj_path.name}: length {length} < {min_length}")
    return long_enough


def sampled_step_indices(num_steps: int, stride: int) -> np.ndarray:
    step_indices = np.arange(0, num_steps, stride)
    if step_indices[-1] != num_steps - 1:
        step_indices = np.append(step_indices, num_steps - 1)
    return step_indices


def read_frame(traj_path: Path, idx: int) -> np.ndarray:
    img_path = traj_path / f"{idx}.jpg"
    if not img_path.exists():
        return np.zeros((64, 64, 3), dtype=np.uint8)
    with Image.open(img_path) as img:
        return np.array(img.convert("RGB"))


def make_animation(
    traj_path: Path,
    positions: np.ndarray,
    step_indices: np.ndarray,
    *,
    title: str,
    fps: float,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    num_steps = positions.shape[0]
    interval_ms = 1000 / fps

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title(title)
    xy_ax, img_ax = axes

    xy_ax.plot(positions[:, 0], positions[:, 1], color="gray", alpha=0.7, label="Trajectory Path")
    (current_loc_marker,) = xy_ax.plot([], [], "ro", label="Robot Position")
    xy_ax.set_xlabel("X Position")
    xy_ax.set_ylabel("Y Position")
    xy_ax.set_title("XY Trajectory with Robot Location")
    xy_ax.legend()
    xy_ax.set_aspect("equal")
    xy_ax.grid(True)

    img_disp = img_ax.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
    img_ax.axis("off")
    img_ax.set_title("Robot Front Camera")

    fig.suptitle(f"{title}\na/y accept, A accept all remaining, d/n deny, p previous, q/esc quit")

    def init():
        current_loc_marker.set_data([], [])
        img_disp.set_data(np.zeros((64, 64, 3), dtype=np.uint8))
        return current_loc_marker, img_disp

    def animate(frame_i: int):
        idx = int(step_indices[frame_i])
        x, y = positions[idx, :2]
        current_loc_marker.set_data([x], [y])
        img_disp.set_data(read_frame(traj_path, idx))
        xy_ax.set_title(f"XY Trajectory (step {idx}/{num_steps - 1})")
        img_ax.set_title(f"Robot Front Camera (step {idx})")
        return current_loc_marker, img_disp

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(step_indices),
        interval=interval_ms,
        blit=True,
        init_func=init,
        repeat=True,
    )
    fig.tight_layout()
    return fig, ani


def save_yaml_list(output_path: Path, scenes: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(f"- {scene}\n" for scene in scenes)
    output_path.write_text(contents)


def review_trajectory(
    traj_path: Path,
    gif_path: Path,
    *,
    review_i: int,
    total: int,
    stride: int,
    fps: float,
) -> str:
    scene = traj_path.name
    positions = load_positions(traj_path)
    step_indices = sampled_step_indices(positions.shape[0], stride)
    title = f"{review_i + 1}/{total}: {scene}"
    fig, ani = make_animation(traj_path, positions, step_indices, title=title, fps=fps)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(gif_path, writer="pillow", fps=fps)
    print(f"Saved GIF: {gif_path}")

    decision = {"action": "quit"}

    def on_key(event):
        key = event.key
        if key in {"a", "y"}:
            decision["action"] = "accept"
            plt.close(fig)
        elif key == "A":
            decision["action"] = "accept_all"
            plt.close(fig)
        elif key in {"d", "n"}:
            decision["action"] = "deny"
            plt.close(fig)
        elif key == "p":
            decision["action"] = "previous"
            plt.close(fig)
        elif key in {"q", "escape"}:
            decision["action"] = "quit"
            plt.close(fig)

    print(f"Reviewing {scene}: a/y accept, A accept all remaining, d/n deny, p previous, q/esc quit")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
        print(f"(Agg backend — no GUI; see GIF: {gif_path})")
        while True:
            raw = input("Decision: ").strip()
            if not raw:
                continue
            if raw == "A":
                decision["action"] = "accept_all"
                break
            ch = raw[0].lower()
            if ch in {"a", "y"}:
                decision["action"] = "accept"
                break
            if ch in {"d", "n"}:
                decision["action"] = "deny"
                break
            if ch == "p":
                decision["action"] = "previous"
                break
            if ch == "q":
                decision["action"] = "quit"
                break
    else:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        plt.close(fig)
    return decision["action"]


def main():
    args = parse_args()
    traj_paths = discover_trajectories(args.data_root, args.filter)
    traj_paths = filter_by_min_length(traj_paths, args.min_length)
    # shuffle the trajectories
    random.shuffle(traj_paths)
    if args.limit is not None:
        traj_paths = traj_paths[: args.limit]
    if not traj_paths:
        raise SystemExit(
            f"No trajectories found in {args.data_root} matching {filter_to_glob(args.filter)!r} "
            f"with length >= {args.min_length}"
        )

    accepted: set[str] = set()
    ordered_accepted: list[str] = []
    i = 0
    save_yaml_list(args.output, ordered_accepted)
    print(f"Found {len(traj_paths)} trajectories. Writing accepted scenes to {args.output}")

    if args.accept_all:
        for remaining_traj_path in traj_paths:
            remaining_scene = remaining_traj_path.name
            if remaining_scene not in accepted:
                accepted.add(remaining_scene)
                ordered_accepted.append(remaining_scene)
        save_yaml_list(args.output, ordered_accepted)
        print(f"Accepted all trajectories.")
        return

    while i < len(traj_paths):
        traj_path = traj_paths[i]
        scene = traj_path.name
        gif_path = args.gif_dir / f"{scene}.gif"
        action = review_trajectory(
            traj_path,
            gif_path,
            review_i=i,
            total=len(traj_paths),
            stride=args.stride,
            fps=args.fps,
        )

        if action == "accept":
            if scene not in accepted:
                accepted.add(scene)
                ordered_accepted.append(scene)
            save_yaml_list(args.output, ordered_accepted)
            i += 1
        elif action == "accept_all":
            for remaining_traj_path in traj_paths[i:]:
                remaining_scene = remaining_traj_path.name
                if remaining_scene not in accepted:
                    accepted.add(remaining_scene)
                    ordered_accepted.append(remaining_scene)
            save_yaml_list(args.output, ordered_accepted)
            print(f"Accepted all remaining trajectories from {scene}.")
            break
        elif action == "deny":
            if scene in accepted:
                accepted.remove(scene)
                ordered_accepted = [accepted_scene for accepted_scene in ordered_accepted if accepted_scene != scene]
            save_yaml_list(args.output, ordered_accepted)
            i += 1
        elif action == "previous":
            i = max(0, i - 1)
        else:
            save_yaml_list(args.output, ordered_accepted)
            print("Review stopped early.")
            break

    print(f"Accepted {len(ordered_accepted)} trajectories. Wrote {args.output}")

if __name__ == "__main__":
    main()