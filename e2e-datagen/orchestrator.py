# Run `prepare` in the ViNT-capable environment, then run `generate` in the
# normal Gemini/TFDS environment.

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import random
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from vertex_annotation import (
    VertexBatchConfig,
    reconstruct_batch_results,
    submit_label_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LocationSpec:
    split: str
    name: str
    root: Path
    scenes: list[str]
    labels_dir: Path
    long_tasks_per_location: int
    short_tasks_per_location: int


@dataclass(frozen=True)
class OrchestratorConfig:
    dataset_name: str
    version: str
    random_seed: int
    data_root: Path
    output_root: Path
    vertex: VertexBatchConfig
    graph: dict[str, Any]
    task_generation: dict[str, Any]
    locations: list[LocationSpec]

    @property
    def graph_root(self) -> Path:
        return self.output_root / "graphs"

    @property
    def checkpoint_root(self) -> Path:
        return self.output_root / ".checkpoints"

    @property
    def tfds_dir(self) -> Path:
        return self.output_root / "tfds"

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "manifests"


def _resolve_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"dataset config must contain mapping `{key}`")
    return value


def _read_scene_file(path: Path) -> list[str]:
    with path.open("r") as f:
        scenes = []
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            if item.startswith("- "):
                item = item[2:].strip()
            scenes.append(item)
    return scenes


def load_config(config_file: str | Path = "dataset_config.yaml") -> OrchestratorConfig:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    for key in ("dataset_name", "version", "random_seed", "data_root", "output_root", "splits"):
        if key not in raw:
            raise ValueError(f"dataset config missing required key `{key}`")

    data_root = _resolve_path(raw["data_root"])
    output_root = _resolve_path(raw["output_root"])

    gcs = _require_mapping(raw, "gcs")
    vertex_raw = _require_mapping(raw, "vertex")
    vertex = VertexBatchConfig(
        project_id=gcs["project_id"],
        location=gcs.get("location", "global"),
        bucket_name=gcs["bucket_name"],
        staging_prefix=gcs["staging_prefix"],
        model_id=vertex_raw["model_id"],
        local_data_path=data_root,
        max_upload_workers=int(vertex_raw.get("max_upload_workers", 16)),
        skip_existing_uploads=bool(vertex_raw.get("skip_existing_uploads", True)),
    )

    splits_raw = _require_mapping(raw, "splits")
    locations: list[LocationSpec] = []
    for split_name, split_cfg in splits_raw.items():
        if not isinstance(split_cfg, dict):
            raise ValueError(f"split `{split_name}` must be a mapping")
        labels_dir = _resolve_path(split_cfg["labels_dir"])
        if "long_tasks_per_location" in split_cfg or "short_tasks_per_location" in split_cfg:
            long_tasks_per_location = int(split_cfg.get("long_tasks_per_location", 0))
            short_tasks_per_location = int(split_cfg.get("short_tasks_per_location", 0))
        else:
            # Backward-compatible default for older configs.
            long_tasks_per_location = int(split_cfg.get("plans_per_location", 0))
            short_tasks_per_location = 0
        if long_tasks_per_location < 0 or short_tasks_per_location < 0:
            raise ValueError(f"split `{split_name}` task counts must be non-negative")
        locs = split_cfg.get("locations")
        if not isinstance(locs, dict) or not locs:
            raise ValueError(f"split `{split_name}` must define non-empty `locations`")
        for location_name, location_cfg in locs.items():
            if not isinstance(location_cfg, dict):
                raise ValueError(f"location `{split_name}/{location_name}` must be a mapping")
            scenes_file = location_cfg.get("scenes")
            if not isinstance(scenes_file, str):
                raise ValueError(
                    f"location `{split_name}/{location_name}` must define `scenes` "
                    "as a text file path"
                )
            scenes_path = _resolve_path(scenes_file, root=config_path.parent)
            scenes = _read_scene_file(scenes_path)
            location_root = (
                _resolve_path(location_cfg["root"])
                if location_cfg.get("root") is not None
                else data_root
            )
            locations.append(
                LocationSpec(
                    split=split_name,
                    name=location_name,
                    root=location_root,
                    scenes=list(scenes),
                    labels_dir=labels_dir,
                    long_tasks_per_location=long_tasks_per_location,
                    short_tasks_per_location=short_tasks_per_location,
                )
            )

    return OrchestratorConfig(
        dataset_name=raw["dataset_name"],
        version=str(raw["version"]),
        random_seed=int(raw["random_seed"]),
        data_root=data_root,
        output_root=output_root,
        vertex=vertex,
        graph=dict(raw.get("graph", {})),
        task_generation=dict(raw.get("task_generation", {})),
        locations=locations,
    )


class Orchestrator:
    def __init__(self, config_file: str | Path = "dataset_config.yaml"):
        self.config = load_config(config_file)
        random.seed(self.config.random_seed)
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def selected_locations(
        self,
        *,
        splits: set[str] | None = None,
        locations: set[str] | None = None,
    ) -> list[LocationSpec]:
        selected = []
        for loc in self.config.locations:
            if splits is not None and loc.split not in splits:
                continue
            if locations is not None and loc.name not in locations:
                continue
            selected.append(loc)
        if not selected:
            raise ValueError("no locations matched the provided filters")
        return selected

    def graph_path(self, loc: LocationSpec) -> Path:
        return self.config.graph_root / loc.split / f"{loc.name}.json"

    def _scene_output_dirs(self, locs: list[LocationSpec]) -> dict[str, Path]:
        scene_to_dir: dict[str, Path] = {}
        for loc in locs:
            loc.labels_dir.mkdir(parents=True, exist_ok=True)
            for scene in loc.scenes:
                scene_to_dir[scene] = loc.labels_dir
        return scene_to_dir

    def _missing_label_files(self, loc: LocationSpec) -> list[Path]:
        return [
            loc.labels_dir / f"{scene}_landmarks.json"
            for scene in loc.scenes
            if not (loc.labels_dir / f"{scene}_landmarks.json").exists()
        ]

    def require_labels(self, loc: LocationSpec):
        missing = self._missing_label_files(loc)
        if missing:
            preview = "\n".join(f"  {path}" for path in missing[:10])
            raise FileNotFoundError(
                f"{loc.split}/{loc.name} is missing {len(missing)} label files:\n{preview}\n"
                "Run `labels-submit` and `labels-reconstruct` before graph generation."
            )

    def submit_labels(self, locs: list[LocationSpec]):
        jobs = []
        self.config.manifest_dir.mkdir(parents=True, exist_ok=True)
        roots = sorted({loc.root for loc in locs})
        for root in roots:
            scenes = sorted({scene for loc in locs if loc.root == root for scene in loc.scenes})
            if not scenes:
                continue
            root_slug = str(root).strip("/").replace("/", "_") or "root"
            manifest_path = self.config.manifest_dir / f"batch_input_{root_slug}.jsonl"
            vertex_config = replace(self.config.vertex, local_data_path=root)
            job = submit_label_batch(
                vertex_config,
                scenes,
                manifest_local_path=manifest_path,
                job_display_name=f"{self.config.dataset_name}_{root_slug}_landmarks",
            )
            print(f"[labels] submitted Vertex batch job for {root}: {job.name}")
            jobs.append(job)
        return jobs

    def reconstruct_labels(self, locs: list[LocationSpec]) -> list[Path]:
        return reconstruct_batch_results(self.config.vertex, self._scene_output_dirs(locs))

    async def build_graphs(
        self,
        locs: list[LocationSpec],
        *,
        max_workers: int = 1,
        resume: bool = True,
    ) -> list[Path]:
        semaphore = asyncio.Semaphore(max_workers)
        graph_cfg = self.config.graph

        async def build_one(loc: LocationSpec) -> Path:
            self.require_labels(loc)
            from graph_builder import build_vint_graph

            async with semaphore:
                return await asyncio.to_thread(
                    build_vint_graph,
                    data_root=loc.root,
                    scenes=loc.scenes,
                    annotation_folder=loc.labels_dir,
                    output_path=self.graph_path(loc),
                    sparsification_steps=int(graph_cfg.get("sparsification_steps", 4)),
                    drop_modality_p=float(graph_cfg.get("drop_modality_p", 0.0)),
                    dummy=bool(graph_cfg.get("dummy", False)),
                    checkpoint=graph_cfg.get("checkpoint"),
                    batch_size=graph_cfg.get("batch_size"),
                    device=graph_cfg.get("device"),
                    resume=resume,
                )

        return await asyncio.gather(*(build_one(loc) for loc in locs))

    async def run_checkpoints(
        self,
        locs: list[LocationSpec],
        *,
        max_workers: int = 1,
    ) -> list[int]:
        semaphore = asyncio.Semaphore(max_workers)
        task_cfg = self.config.task_generation

        async def run_one(loc: LocationSpec) -> int:
            from distillation_build_dataset import DatasetBuilder

            graph_file = self.graph_path(loc)
            if not graph_file.exists():
                raise FileNotFoundError(f"missing graph for {loc.split}/{loc.name}: {graph_file}")
            async with semaphore:
                total_requests = 0
                for task_kind, count, single_step in (
                    ("long", loc.long_tasks_per_location, False),
                    ("short", loc.short_tasks_per_location, True),
                ):
                    if count <= 0:
                        continue
                    builder = DatasetBuilder(
                        graph_file,
                        out_dir=self.config.output_root,
                        split=loc.split,
                        location=loc.name,
                        task_kind=task_kind,
                        plan_count=count,
                        single_step=single_step,
                        prune_fraction=float(task_cfg.get("builder_prune_fraction", 0.3)),
                        seed=self.config.random_seed,
                    )
                    total_requests += await builder.run_checkpoints()
                return total_requests

        return await asyncio.gather(*(run_one(loc) for loc in locs))

    def assemble(self, *, splits: list[str] | None = None):
        from assemble_dataset import assemble_tfds_from_checkpoints

        return assemble_tfds_from_checkpoints(
            self.config.checkpoint_root,
            self.config.tfds_dir,
            splits=splits,
        )

    async def run_all(
        self,
        locs: list[LocationSpec],
        *,
        max_location_workers: int = 1,
        resume: bool = True,
    ):
        missing_labels = [path for loc in locs for path in self._missing_label_files(loc)]
        if missing_labels:
            self.submit_labels(locs)
            print(
                "[all] label files are missing, so a Vertex batch job was submitted. "
                "Re-run `labels-reconstruct` after the batch completes, then run `all` again."
            )
            return None
        await self.build_graphs(locs, max_workers=max_location_workers, resume=resume)
        await self.run_checkpoints(locs, max_workers=max_location_workers)
        return self.assemble(splits=sorted({loc.split for loc in locs}))

    async def prepare(
        self,
        locs: list[LocationSpec],
        *,
        max_workers: int = 1,
        resume: bool = True,
    ) -> list[Path]:
        self.reconstruct_labels(locs)
        return await self.build_graphs(locs, max_workers=max_workers, resume=resume)

    async def generate(
        self,
        locs: list[LocationSpec],
        *,
        max_workers: int = 1,
    ) -> list[int]:
        return await self.run_checkpoints(locs, max_workers=max_workers)

    def dry_run(self, locs: list[LocationSpec]) -> bool:
        print("[dry-run] config loaded")
        print(f"  dataset: {self.config.dataset_name} v{self.config.version}")
        print(f"  data_root: {self.config.data_root}")
        print(f"  output_root: {self.config.output_root}")
        print(f"  locations: {len(locs)}")

        for loc in locs:
            missing_scene_dirs = [
                scene for scene in loc.scenes if not (loc.root / scene).is_dir()
            ]
            missing_labels = self._missing_label_files(loc)
            print(
                f"[dry-run] {loc.split}/{loc.name}: "
                f"{len(loc.scenes)} scenes, "
                f"long={loc.long_tasks_per_location}, "
                f"short={loc.short_tasks_per_location}"
            )
            print(f"  root: {loc.root}")
            print(f"  labels_dir: {loc.labels_dir}")
            print(f"  graph_path: {self.graph_path(loc)}")
            print(f"  missing_scene_dirs: {len(missing_scene_dirs)}")
            print(f"  missing_label_files: {len(missing_labels)}")
            if missing_scene_dirs:
                print(f"    first missing scene: {missing_scene_dirs[0]}")
            if missing_labels:
                print(f"    first missing label: {missing_labels[0]}")

        ok = True
        required_specs = [
            ("yaml", "PyYAML"),
            ("tqdm", "tqdm"),
            ("PIL", "Pillow"),
            ("numpy", "numpy"),
            ("google.cloud.storage", "google-cloud-storage"),
            ("google.cloud.aiplatform", "google-cloud-aiplatform"),
            ("google.genai", "google-genai"),
            ("torch", "torch"),
            ("tensorflow_datasets", "tensorflow-datasets"),
        ]
        for module_name, package_name in required_specs:
            found = importlib.util.find_spec(module_name) is not None
            ok = ok and found
            print(f"[dry-run] dependency {package_name}: {'ok' if found else 'missing'}")

        import_checks = [
            "vertex_annotation",
            "assemble_dataset",
            "graph_builder",
            "graph_io",
            "distillation_generate_tasks",
            "distillation_data_gen_parallel",
            "distillation_build_dataset",
        ]
        for module_name in import_checks:
            result = subprocess.run(
                [sys.executable, "-c", f"import {module_name}"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                ok = False
                detail = (result.stderr or result.stdout).strip().splitlines()
                message = detail[-1] if detail else f"exit code {result.returncode}"
                print(f"[dry-run] import {module_name}: failed ({message})")
            else:
                print(f"[dry-run] import {module_name}: ok")

        smoke_loc = next((loc for loc in locs if loc.scenes), None)
        graph_cfg = self.config.graph
        if smoke_loc is None:
            print("[dry-run] vint smoke: skipped (no scenes configured)")
        else:
            smoke_code = (
                "import json\n"
                "from graph_builder import run_vint_smoke_test\n"
                "result = run_vint_smoke_test(\n"
                f"    data_root={str(smoke_loc.root)!r},\n"
                f"    scenes={smoke_loc.scenes!r},\n"
                f"    checkpoint={graph_cfg.get('checkpoint')!r},\n"
                f"    device={graph_cfg.get('device')!r},\n"
                ")\n"
                "print(json.dumps(result, sort_keys=True))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", smoke_code],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                ok = False
                detail = (result.stderr or result.stdout).strip().splitlines()
                message = detail[-1] if detail else f"exit code {result.returncode}"
                print(f"[dry-run] vint smoke: failed ({message})")
            else:
                try:
                    smoke_result = json.loads(result.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    smoke_result = {"raw": result.stdout.strip()}
                print(f"[dry-run] vint smoke: ok {smoke_result}")

        if ok:
            print("[dry-run] all dependency and import checks passed")
        else:
            print("[dry-run] one or more dependency/import checks failed")
        return ok


def _csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


async def _amain():
    parser = argparse.ArgumentParser(description="Run the e2e distillation data pipeline.")
    parser.add_argument(
        "command",
        choices=[
            "dry-run",
            "prepare",
            "generate",
            "labels-submit",
            "labels-reconstruct",
            "graphs",
            "checkpoints",
            "assemble",
            "all",
        ],
    )
    parser.add_argument("--config", default="dataset_config.yaml")
    parser.add_argument("--splits", help="Comma-separated split names to include.")
    parser.add_argument("--locations", help="Comma-separated location names to include.")
    parser.add_argument("--max-location-workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    orchestrator = Orchestrator(args.config)
    selected = orchestrator.selected_locations(
        splits=_csv_set(args.splits),
        locations=_csv_set(args.locations),
    )

    if args.command == "dry-run":
        ok = orchestrator.dry_run(selected)
        if not ok:
            raise SystemExit(1)
    elif args.command == "prepare":
        await orchestrator.prepare(
            selected,
            max_workers=args.max_location_workers,
            resume=not args.no_resume,
        )
    elif args.command == "generate":
        await orchestrator.generate(selected, max_workers=args.max_location_workers)
    elif args.command == "labels-submit":
        orchestrator.submit_labels(selected)
    elif args.command == "labels-reconstruct":
        orchestrator.reconstruct_labels(selected)
    elif args.command == "graphs":
        await orchestrator.build_graphs(
            selected,
            max_workers=args.max_location_workers,
            resume=not args.no_resume,
        )
    elif args.command == "checkpoints":
        await orchestrator.run_checkpoints(selected, max_workers=args.max_location_workers)
    elif args.command == "assemble":
        orchestrator.assemble(splits=sorted({loc.split for loc in selected}))
    elif args.command == "all":
        await orchestrator.run_all(
            selected,
            max_location_workers=args.max_location_workers,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    asyncio.run(_amain())
