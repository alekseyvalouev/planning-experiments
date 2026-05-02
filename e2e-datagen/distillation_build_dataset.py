from __future__ import annotations

import asyncio
import glob
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PLAN_A_STAR_DIR = Path(__file__).resolve().parents[1] / "plan-a-star"
if str(PLAN_A_STAR_DIR) not in sys.path:
    sys.path.insert(0, str(PLAN_A_STAR_DIR))

from graph_io import Graph
from distillation_generate_tasks import TaskBuilder
from distillation_data_gen_parallel import Planner


class DatasetBuilder:
    """Generate task/planner examples and persist them as resumable checkpoints.

    Final TFDS assembly intentionally lives outside this module. This class is
    safe to run per location because every checkpoint filename includes both the
    split and location.
    """

    CHECKPOINT_SCHEMA_VERSION = 2

    def __init__(
        self,
        graph_file: str | os.PathLike[str],
        *,
        out_dir: str | os.PathLike[str],
        split: str,
        location: str,
        plan_count: int,
        task_kind: str = "long",
        single_step: bool = False,
        prune_fraction: float = 0.3,
        seed: int = 42,
        cost_log_file: str | os.PathLike[str] | None = None,
    ):
        if task_kind not in {"long", "short"}:
            raise ValueError("task_kind must be 'long' or 'short'")
        self.graph_file = str(graph_file)
        self.graph = Graph.deserialize(self.graph_file)
        self.task_builder = TaskBuilder(
            self.graph_file,
            prune_fraction=prune_fraction,
            seed=seed,
        )
        self.planner = Planner(self.graph)
        self.data: list[dict[str, Any]] = []
        self.out_dir = str(out_dir)
        self.split = split
        self.location = location
        self.task_kind = task_kind
        self.plan_count = plan_count
        self.single_step = single_step
        self.checkpoint_dir = os.path.join(self.out_dir, ".checkpoints", split)
        self.cost_log_file = str(
            cost_log_file or os.path.join(self.out_dir, "plan_generation_cost.jsonl")
        )

    def _generate_task(self):
        path, task = self.task_builder.generate_task(single_step=self.single_step)
        if path is None:
            print("No path found. Generating new task.")
            return self._generate_task()
        print(f"Generated task: {task}")
        print("-" * 100)
        return path, task

    async def _generate_plan(self, task, start_node, start_modality, start_landmark_idx):
        plan, explored_plans, _ = await self.planner.plan(
            start_node,
            start_modality,
            start_landmark_idx,
            None,
            task,
            end_info=None,
            separate_landmarks=False,
        )
        if plan is None:
            print("No goal plan found; keeping explored plans for checkpoint parsing")
        else:
            print(f"Found optimal plan with length {len(plan)}")
        print(f"Explored plans: {explored_plans}")
        print("-" * 100)
        return explored_plans

    async def single_iteration(self):
        path, task = self._generate_task()
        start_node = path[0]
        explored_plans = await self._generate_plan(task, start_node, "V", None)
        self._parse_explored_plans(explored_plans, task)
        print(f"Parsed {len(self.data)} entries")
        print(
            "Running token usage: "
            f"in={self.planner.total_input_tokens} "
            f"out={self.planner.total_output_tokens} "
            f"total={self.planner.total_input_tokens + self.planner.total_output_tokens} "
            f"(over {self.planner.total_requests} requests)"
        )
        print("-" * 100)

    def _checkpoint_path(self, idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.location}_{self.task_kind}_{idx:05d}.pkl")

    def _save_checkpoint(self, idx: int, new_entries: list[dict[str, Any]]):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        payload = {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "split": self.split,
            "location": self.location,
            "task_kind": self.task_kind,
            "single_step": self.single_step,
            "idx": idx,
            "entries": new_entries,
            "tokens_in": self.planner.total_input_tokens,
            "tokens_out": self.planner.total_output_tokens,
            "requests": self.planner.total_requests,
            "graph_file": self.graph_file,
        }
        final_path = self._checkpoint_path(idx)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{self.location}_{self.task_kind}_{idx:05d}.",
            suffix=".pkl.tmp",
            dir=self.checkpoint_dir,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, final_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        print(f"[checkpoint] wrote {final_path} ({len(new_entries)} new entries)")

    def _estimate_plan_generation_cost(self) -> dict[str, float]:
        model = getattr(self.planner, "gemini_model", "")
        input_cost = 0.0
        output_cost = 0.0
        if model == "gemini-3-flash-preview":
            input_cost = self.planner.total_input_tokens * 10**-6 * 0.5
            output_cost = self.planner.total_output_tokens * 10**-6 * 3.0
        return {
            "input_usd": input_cost,
            "output_usd": output_cost,
            "total_usd": input_cost + output_cost,
        }

    def _write_cost_snapshot(self, idx: int, *, status: str, entries: int):
        os.makedirs(os.path.dirname(self.cost_log_file), exist_ok=True)
        costs = self._estimate_plan_generation_cost()
        payload = {
            "timestamp": time.time(),
            "split": self.split,
            "location": self.location,
            "task_kind": self.task_kind,
            "single_step": self.single_step,
            "idx": idx,
            "status": status,
            "entries": entries,
            "model": getattr(self.planner, "gemini_model", ""),
            "tokens_in": self.planner.total_input_tokens,
            "tokens_out": self.planner.total_output_tokens,
            "requests": self.planner.total_requests,
            **costs,
        }
        with open(self.cost_log_file, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _load_checkpoints(self):
        if not os.path.isdir(self.checkpoint_dir):
            return [], 0, None

        pattern = os.path.join(self.checkpoint_dir, f"{self.location}_{self.task_kind}_*.pkl")
        paths = sorted(glob.glob(pattern))
        entries: list[dict[str, Any]] = []
        token_snapshot = None
        expected_idx = 0

        for path in paths:
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print(
                    f"[checkpoint] could not read {path}: {type(e).__name__}: {e}. "
                    f"Stopping resume at idx={expected_idx}."
                )
                break

            if payload.get("schema_version") != self.CHECKPOINT_SCHEMA_VERSION:
                print(
                    f"[checkpoint] schema mismatch in {path} "
                    f"(got {payload.get('schema_version')!r}, "
                    f"expected {self.CHECKPOINT_SCHEMA_VERSION}). "
                    f"Stopping resume at idx={expected_idx}."
                )
                break

            if (
                payload.get("split") != self.split
                or payload.get("location") != self.location
                or payload.get("task_kind") != self.task_kind
                or payload.get("idx") != expected_idx
            ):
                print(
                    f"[checkpoint] gap or mismatch in {path}; "
                    f"expected split={self.split} location={self.location} "
                    f"task_kind={self.task_kind} idx={expected_idx}."
                )
                break

            entries.extend(payload.get("entries", []))
            token_snapshot = {
                "tokens_in": int(payload.get("tokens_in", 0)),
                "tokens_out": int(payload.get("tokens_out", 0)),
                "requests": int(payload.get("requests", 0)),
            }
            expected_idx += 1
            if expected_idx >= self.plan_count:
                break

        if entries:
            print(
                f"[checkpoint] resumed {self.split}/{self.location}: "
                f"task_kind={self.task_kind} loaded {expected_idx} iterations, {len(entries)} entries"
            )
        return entries, expected_idx, token_snapshot

    async def _safe_single_iteration(self, i: int) -> bool:
        try:
            await self.single_iteration()
            return True
        except Exception as e:
            import traceback

            print(
                f"[checkpoints] {self.split}/{self.location} iteration "
                f"{i + 1}/{self.plan_count} ({self.task_kind}) failed with "
                f"{type(e).__name__}: {e}. Skipping."
            )
            traceback.print_exc()
            return False

    async def run_checkpoints(self) -> int:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.data = []
        resumed_entries, next_idx, token_snapshot = self._load_checkpoints()
        self.data.extend(resumed_entries)
        if token_snapshot is not None:
            self.planner.total_input_tokens = token_snapshot["tokens_in"]
            self.planner.total_output_tokens = token_snapshot["tokens_out"]
            self.planner.total_requests = token_snapshot["requests"]

        for i in range(next_idx, self.plan_count):
            before_len = len(self.data)
            ok = await self._safe_single_iteration(i)
            new_entries = self.data[before_len:] if ok else []
            try:
                self._save_checkpoint(i, new_entries)
            except Exception as e:
                print(
                    f"[checkpoint] failed to write {self.split}/{self.location} "
                    f"idx={i}: {type(e).__name__}: {e}"
                )
            self._write_cost_snapshot(
                i,
                status="ok" if ok else "failed",
                entries=len(new_entries),
            )

        self.planner.print_token_usage()
        return self.planner.total_requests

    def show_examples(self):
        for i, entry in enumerate(self.data):
            if i > 3:
                break
            print(f"Query: {entry['query']}")
            print(f"Result: {entry['result']}")
            print(f"Heuristic: {entry['heuristic']}")
            print("-" * 100)

    def load_rgb(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _load_full_entry(self, query, result, heuristic):
        task = query["task"]
        start_image = (
            self.load_rgb(query["start_node"]["image"])
            if "image" in query["start_node"]
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        start_landmark_idx = query["start_landmark_idx"]
        start_text = (
            "Go to the " + query["start_node"]["landmarks"][start_landmark_idx]
            if start_landmark_idx is not None
            else "x" * 64
        )

        prev_image = (
            self.load_rgb(query["prev_node"]["image"])
            if "image" in query["prev_node"]
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        prev_landmark_idx = query["prev_landmark_idx"]
        prev_text = (
            "Go to the " + query["prev_node"]["landmarks"][prev_landmark_idx]
            if prev_landmark_idx is not None
            else "x" * 64
        )

        curr_image = (
            self.load_rgb(result["node"]["image"])
            if "image" in result["node"]
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        curr_landmark_idx = result["landmark_idx"]
        curr_text = (
            "Go to the " + result["node"]["landmarks"][curr_landmark_idx]
            if curr_landmark_idx is not None
            else "x" * 64
        )

        return {
            "task": task,
            "start_node_id": query["start_id"],
            "start_image": start_image,
            "start_text": start_text,
            "prev_node_id": query["prev_id"],
            "prev_image": prev_image,
            "prev_text": prev_text,
            "curr_node_id": result["node_id"],
            "curr_image": curr_image,
            "curr_text": curr_text,
            "heuristic": heuristic,
        }

    def _parse_explored_plans(self, explored_plans, task):
        for entry in explored_plans:
            plan = entry["path"]
            h = entry["heuristic"]
            if len(plan) < 2:
                continue
            for i in range(1, len(plan)):
                node, modality, landmark_idx = plan[i]
                prev_node, prev_modality, prev_landmark_idx = plan[i - 1]
                start_node, start_modality, start_landmark_idx = plan[0]
                query = {
                    "task": task,
                    "prev_node": prev_node.subnodes[prev_modality],
                    "prev_landmark_idx": prev_landmark_idx,
                    "start_node": start_node.subnodes[start_modality],
                    "start_landmark_idx": start_landmark_idx,
                    "prev_id": prev_node.node_id,
                    "start_id": start_node.node_id,
                }
                result = {
                    "node": node.subnodes[modality],
                    "landmark_idx": landmark_idx,
                    "node_id": node.node_id,
                }

                for existing_entry in self.data:
                    if existing_entry["query"] == query and existing_entry["result"] == result:
                        if existing_entry["heuristic"] < h:
                            existing_entry["heuristic"] = h
                            existing_entry["loaded"]["heuristic"] = h
                        break
                else:
                    loaded = self._load_full_entry(query, result, h)
                    self.data.append(
                        {
                            "query": query,
                            "result": result,
                            "heuristic": h,
                            "loaded": loaded,
                        }
                    )


if __name__ == "__main__":
    builder = DatasetBuilder(
        "graph_vint_loose.json",
        out_dir="/hdd/aleksey_cache/distill-a-star-dataset-v3",
        split="train",
        location="default",
        task_kind="long",
        plan_count=50,
        single_step=False,
    )
    asyncio.run(builder.run_checkpoints())
