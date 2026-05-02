from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path
from typing import Any

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
        examples_by_split: dict[str, list[dict[str, Any]]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.examples_by_split = examples_by_split or {}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
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
                }
            ),
            homepage="https://alekseyvalouev.github.io",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            split_name: self._generate_examples(examples)
            for split_name, examples in self.examples_by_split.items()
        }

    def _generate_examples(self, examples: list[dict[str, Any]]):
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
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {arr.shape}")
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (255.0 * arr).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr


def load_checkpoint_examples(
    checkpoint_root: str | os.PathLike[str],
    *,
    splits: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    checkpoint_root = str(checkpoint_root)
    split_names = splits or [
        Path(path).name
        for path in sorted(glob.glob(os.path.join(checkpoint_root, "*")))
        if os.path.isdir(path)
    ]
    examples_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in split_names:
        paths = sorted(glob.glob(os.path.join(checkpoint_root, split, "*.pkl")))
        examples: list[dict[str, Any]] = []
        for path in paths:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("split") != split:
                print(f"[assemble] skipping {path}: split mismatch")
                continue
            entries = payload.get("entries", [])
            examples.extend(entry["loaded"] for entry in entries if "loaded" in entry)
        examples_by_split[split] = examples
        print(f"[assemble] loaded {len(examples)} examples for split {split}")
    return examples_by_split


def assemble_tfds_from_checkpoints(
    checkpoint_root: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    *,
    splits: list[str] | None = None,
) -> TaskSequenceDataset:
    examples_by_split = load_checkpoint_examples(checkpoint_root, splits=splits)
    builder = TaskSequenceDataset(data_dir=str(data_dir), examples_by_split=examples_by_split)
    builder.download_and_prepare()
    return builder
