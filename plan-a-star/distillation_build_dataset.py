# Once we have tasks, plans, and heuristics, we can build a dataset.
# Two options for building the dataset:
# 1. Use task + starting obs as query ex. (T, [o1, o2, o3], H). Q = o1 + Task, R = o3, Similarity = H
#   We call this non-recursive
# 2. Use task + second-last obs as query ex. (T, [o1, o2, o3], H). Q = o2 + Task, R = o3, Similarity = H
#   We call this recursive 
# Plan length must be at least 2. 
# TFDS dataset format

from graph import Graph
from distillation_generate_tasks import TaskBuilder
from distillation_data_gen_parallel import Planner

from PIL import Image
import numpy as np

import asyncio
import glob
import os
import pickle
import tempfile

from typing import Dict, List, Any
import numpy as np
import tensorflow_datasets as tfds


class TaskSequenceDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *args,
        examples_by_split: Dict[str, List[Dict[str, Any]]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.examples_by_split = examples_by_split or {}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "task": tfds.features.Text(),
                "start_node_id": tfds.features.Scalar(dtype=np.int32),
                "start_image": tfds.features.Image(),
                "start_text": tfds.features.Text(),
                "prev_node_id": tfds.features.Scalar(dtype=np.int32),
                "prev_image": tfds.features.Image(),
                "prev_text": tfds.features.Text(),
                "curr_node_id": tfds.features.Scalar(dtype=np.int32),
                "curr_image": tfds.features.Image(),
                "curr_text": tfds.features.Text(),
                "heuristic": tfds.features.Scalar(dtype=np.float32),
            }),
            homepage="https://alekseyvalouev.github.io",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            split_name: self._generate_examples(examples)
            for split_name, examples in self.examples_by_split.items()
        }

    def _generate_examples(self, examples: List[Dict[str, Any]]):
        for idx, ex in enumerate(examples):
            yield str(idx), {
                "task": ex["task"],
                "start_node_id": ex["start_node_id"],
                "start_image": self._normalize_image(ex["start_image"]),
                "start_text": ex["start_text"],
                "prev_node_id": ex["prev_node_id"],
                "prev_image": self._normalize_image(ex["prev_image"]),
                "prev_text": ex["prev_text"],
                "curr_node_id": ex["curr_node_id"],
                "curr_image": self._normalize_image(ex["curr_image"]),
                "curr_text": ex["curr_text"],
                "heuristic": np.float32(ex["heuristic"]),
            }

    @staticmethod
    def _normalize_image(img: Any) -> np.ndarray:
        """
        Convert image-like input to uint8 HxWxC numpy array.

        Expected input:
          - np.ndarray already shaped [H, W, C]
          - optionally grayscale [H, W], which will be expanded to [H, W, 1]
        """
        arr = np.asarray(img)

        if arr.ndim == 2:
            arr = arr[..., None]

        if arr.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {arr.shape}")

        if arr.dtype != np.uint8:
            # If floats are in [0, 1], scale them up.
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (255.0 * arr).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        return arr

class DatasetBuilder:
    # Bump this if the on-disk entry schema / pickle format changes so that
    # stale checkpoints aren't silently reused.
    CHECKPOINT_SCHEMA_VERSION = 2
    SHORT_TASK_FRACTIONS = {
        "train": 0.5,
        "val": 0.5,
    }

    def __init__(self, graph_file, out_dir="/hdd/aleksey_cache/distill-a-star-dataset-v2", train_plans=9, val_plans=1):
        self.graph_file = graph_file
        self.graph = Graph.deserialize(graph_file)
        self.task_builder = TaskBuilder(graph_file)
        self.planner = Planner(self.graph)
        self.data = []
        self.train_plans = train_plans
        self.val_plans = val_plans
        self.out_dir = out_dir
        self.checkpoint_dir = os.path.join(self.out_dir, ".checkpoints")
    
    def _generate_task(self, single_step=False):
        path, task = self.task_builder.generate_task(single_step=single_step)
        if path is None:
            print("No path found. Generating new task.")
            return self._generate_task(single_step=single_step)
        print(f"Generated task: {task}")
        print("-" * 100)
        return path, task
    
    async def _generate_plan(self, task, start_node, start_modality, start_landmark_idx):
        plan, explored_plans, reasoning_cache = await self.planner.plan(start_node, start_modality, start_landmark_idx, None, task, end_info=None, separate_landmarks=False)
        print(f"Found optimal plan with length {len(plan)}")
        print(f"Explored plans: {explored_plans}")
        print("-" * 100)
        return explored_plans
    
    async def single_iteration(self, short):
        path, task = self._generate_task(single_step=short)
        start_node = path[0]
        start_modality = "V"
        start_landmark_idx = None
        explored_plans = await self._generate_plan(task, start_node, start_modality, start_landmark_idx)
        self._parse_explored_plans(explored_plans, task)
        print(f"Parsed {len(self.data)} entries")
        print(f"Running token usage: "
              f"in={self.planner.total_input_tokens} "
              f"out={self.planner.total_output_tokens} "
              f"total={self.planner.total_input_tokens + self.planner.total_output_tokens} "
              f"(over {self.planner.total_requests} requests)")
        print("-" * 100)

    # ----- Checkpointing ------------------------------------------------------
    #
    # Each completed iteration writes one pickle file to
    #   {out_dir}/.checkpoints/{split}_{idx:05d}.pkl
    # containing a dict:
    #   {
    #     "schema_version": int,
    #     "split": "train" | "val",
    #     "idx": int,                      # 0-based iteration index
    #     "entries": List[dict],           # entries appended during this iteration
    #     "tokens_in": int,                # cumulative planner counters AFTER this iter
    #     "tokens_out": int,
    #     "requests": int,
    #     "short": bool,
    #   }
    # Writes are atomic (tmp + os.replace) so a crash mid-write can't leave a
    # corrupt checkpoint.

    def _checkpoint_path(self, split: str, idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"{split}_{idx:05d}.pkl")

    def _save_checkpoint(self, split: str, idx: int, new_entries: List[Dict[str, Any]], short: bool):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        payload = {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "split": split,
            "idx": idx,
            "entries": new_entries,
            "tokens_in": self.planner.total_input_tokens,
            "tokens_out": self.planner.total_output_tokens,
            "requests": self.planner.total_requests,
            "short": short,
        }
        final_path = self._checkpoint_path(split, idx)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{split}_{idx:05d}.", suffix=".pkl.tmp", dir=self.checkpoint_dir
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

    def _load_split_checkpoints(self, split: str, expected_total: int):
        """Load all existing checkpoints for ``split`` in index order.

        Returns ``(entries, next_idx, token_snapshot)`` where:
          - ``entries`` is the combined list of entries from every loaded
            checkpoint (to rehydrate ``self.data``),
          - ``next_idx`` is the index of the first not-yet-completed iteration,
          - ``token_snapshot`` is the planner counters at the time of the last
            loaded checkpoint (used to restore in-memory counters).
        """
        if not os.path.isdir(self.checkpoint_dir):
            return [], 0, None

        pattern = os.path.join(self.checkpoint_dir, f"{split}_*.pkl")
        paths = sorted(glob.glob(pattern))
        if not paths:
            return [], 0, None

        entries: List[Dict[str, Any]] = []
        token_snapshot = None
        expected_idx = 0
        for path in paths:
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print(f"[checkpoint] could not read {path}: {type(e).__name__}: {e}. "
                      f"Stopping resume at idx={expected_idx}.")
                break

            if payload.get("schema_version") != self.CHECKPOINT_SCHEMA_VERSION:
                print(f"[checkpoint] schema mismatch in {path} "
                      f"(got {payload.get('schema_version')!r}, "
                      f"expected {self.CHECKPOINT_SCHEMA_VERSION}). "
                      f"Stopping resume at idx={expected_idx}.")
                break

            if payload.get("split") != split or payload.get("idx") != expected_idx:
                print(f"[checkpoint] gap or mismatch in {path} "
                      f"(expected {split} idx={expected_idx}, "
                      f"got split={payload.get('split')!r} idx={payload.get('idx')!r}). "
                      f"Stopping resume.")
                break

            entries.extend(payload.get("entries", []))
            token_snapshot = {
                "tokens_in": int(payload.get("tokens_in", 0)),
                "tokens_out": int(payload.get("tokens_out", 0)),
                "requests": int(payload.get("requests", 0)),
            }
            expected_idx += 1
            if expected_idx >= expected_total:
                break

        if entries:
            print(f"[checkpoint] resumed {split}: loaded {expected_idx} iterations, "
                  f"{len(entries)} entries")
        return entries, expected_idx, token_snapshot

    def _short_task_count(self, split: str, total: int) -> int:
        fraction = self.SHORT_TASK_FRACTIONS.get(split, 0.0)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"SHORT_TASK_FRACTIONS[{split!r}] must be between 0.0 and 1.0, "
                f"got {fraction!r}"
            )
        return round(total * fraction)

    def _is_short_task(self, split: str, idx: int, total: int) -> bool:
        return idx < self._short_task_count(split, total)

    async def _safe_single_iteration(self, label: str, i: int, total: int, short: bool) -> bool:
        try:
            await self.single_iteration(short)
            return True
        except Exception as e:
            import traceback
            print(f"[build_tfds] {label} iteration {i + 1}/{total} failed with "
                  f"{type(e).__name__}: {e}. Skipping.")
            traceback.print_exc()
            return False

    async def _run_split(self, split: str, total: int) -> int:
        """Run ``total`` iterations for ``split``, resuming from checkpoints.

        Populates ``self.data`` with all entries (resumed + newly generated)
        and returns the planner's cumulative-request count at the end of the
        split (used by callers to compute per-split token deltas).
        """
        # Caller is expected to have already reset ``self.data`` to [].
        resumed_entries, next_idx, token_snapshot = self._load_split_checkpoints(split, total)
        if resumed_entries:
            self.data.extend(resumed_entries)
        if token_snapshot is not None:
            # Restore planner counters so in-memory stats pick up where we left
            # off. Any *new* iterations in this run will add on top.
            self.planner.total_input_tokens = token_snapshot["tokens_in"]
            self.planner.total_output_tokens = token_snapshot["tokens_out"]
            self.planner.total_requests = token_snapshot["requests"]

        for i in range(next_idx, total):
            before_len = len(self.data)
            short = self._is_short_task(split, i, total)
            ok = await self._safe_single_iteration(split, i, total, short)
            new_entries = self.data[before_len:] if ok else []
            # Always write a checkpoint (even on failure, with empty entries)
            # so we don't re-attempt the same failing iteration forever. If you
            # would rather retry on next run, comment out the failure branch.
            try:
                self._save_checkpoint(split, i, new_entries, short)
            except Exception as e:
                print(f"[checkpoint] failed to write {split} idx={i}: "
                      f"{type(e).__name__}: {e}")

        return self.planner.total_requests

    async def build_tfds(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.data = []
        await self._run_split("train", self.train_plans)
        train_data = [entry["loaded"] for entry in self.data]
        train_tokens_in = self.planner.total_input_tokens
        train_tokens_out = self.planner.total_output_tokens
        train_requests = self.planner.total_requests
        print("=" * 50)
        print(f"Train token usage: in={train_tokens_in} out={train_tokens_out} "
              f"total={train_tokens_in + train_tokens_out} "
              f"(over {train_requests} requests)")
        print("=" * 50)

        self.data = []
        await self._run_split("val", self.val_plans)
        val_data = [entry["loaded"] for entry in self.data]
        val_tokens_in = self.planner.total_input_tokens - train_tokens_in
        val_tokens_out = self.planner.total_output_tokens - train_tokens_out
        val_requests = self.planner.total_requests - train_requests
        print("=" * 50)
        print(f"Val token usage:   in={val_tokens_in} out={val_tokens_out} "
              f"total={val_tokens_in + val_tokens_out} "
              f"(over {val_requests} requests)")
        print("=" * 50)
        self.planner.print_token_usage()
        d = {
            "train": train_data,
            "validation": val_data,
        }
        builder = TaskSequenceDataset(
            data_dir=self.out_dir,
            examples_by_split=d,
            #file_format=tfds.core.FileFormat.ARRAY_RECORD,
        )
        builder.download_and_prepare()
        return builder
    
    def show_examples(self):
        for i, entry in enumerate(self.data):
            if i > 3:
                break
            query = entry["query"]
            result = entry["result"]
            heuristic = entry["heuristic"]
            print(f"Query: {query}")
            print(f"Result: {result}")
            print(f"Heuristic: {heuristic}")
            print("-" * 100)
    

    def load_rgb(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _load_full_entry(self, query, result, heuristic):
        task = query["task"]
        start_image = self.load_rgb(query["start_node"]["image"]) if "image" in query["start_node"] else np.zeros((224, 224, 3), dtype=np.uint8)
        start_landmark_idx = query["start_landmark_idx"]
        start_text = "Go to the " + query["start_node"]["landmarks"][start_landmark_idx] if start_landmark_idx is not None else "x"*64

        prev_image = self.load_rgb(query["prev_node"]["image"]) if "image" in query["prev_node"] else np.zeros((224, 224, 3), dtype=np.uint8)
        prev_landmark_idx = query["prev_landmark_idx"]
        prev_text = "Go to the " + query["prev_node"]["landmarks"][prev_landmark_idx] if prev_landmark_idx is not None else "x"*64

        curr_image = self.load_rgb(result["node"]["image"]) if "image" in result["node"] else np.zeros((224, 224, 3), dtype=np.uint8)
        curr_landmark_idx = result["landmark_idx"]
        curr_text = "Go to the " + result["node"]["landmarks"][curr_landmark_idx] if curr_landmark_idx is not None else "x"*64

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
        # sort explored_plans by heuristic
        for entry in explored_plans:
            plan = entry["path"]
            h = entry["heuristic"]
            plan_length = len(plan)
            if plan_length < 2:
                continue
            for i in range(1, plan_length):
                node, modality, landmark_idx = plan[i]
                prev_node, prev_modality, prev_landmark_idx = plan[i-1]
                start_node, start_modality, start_landmark_idx = plan[0]
                query = {"task": task, 
                            "prev_node": prev_node.subnodes[prev_modality], "prev_landmark_idx": prev_landmark_idx,
                            "start_node": start_node.subnodes[start_modality], "start_landmark_idx": start_landmark_idx,
                            "prev_id": prev_node.node_id, "start_id": start_node.node_id}

                result = {"node": node.subnodes[modality], "landmark_idx": landmark_idx, "node_id": node.node_id}

                for existing_entry in self.data:
                    if existing_entry["query"] == query and existing_entry["result"] == result:
                        if existing_entry["heuristic"] < h:
                            existing_entry["heuristic"] = h
                        break
                else:
                    loaded = self._load_full_entry(query, result, h)
                    full_entry = {"query": query, "result": result, "heuristic": h, "loaded" : loaded}
                    self.data.append(full_entry)

if __name__ == "__main__":
    builder = DatasetBuilder("graph_vint_tight.json", out_dir="/hdd/aleksey_cache/distill-a-star-dataset-v4", train_plans=50, val_plans=5)
    data = asyncio.run(builder.build_tfds())
    ds = data.as_dataset(split="train")