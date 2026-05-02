from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from tqdm import tqdm


ANNOTATE_LANDMARKS_PROMPT = """
You are an AI tasked with spatial reasoning and environment mapping. Analyze the image to identify "Navigational Landmarks"--prominent, static features that define the layout or provide orientation.

### 1. STRICT PROHIBITION (DO NOT VIOLATE)
- DO NOT list people, animals, or moving vehicles.
- DO NOT describe the actions, clothing, or presence of humans.
- If a person is obscuring a landmark, describe the landmark as "partially obscured" rather than mentioning the person.
- If the image contains ONLY people and no static landmarks, return: {"landmarks": []}.

### 2. CRITICAL NAVIGATION RULES
- ALL EXITS: You MUST identify every doorway, archway, stairwell, or elevator.
- JUNCTIONS: If at a hallway junction or corner, you MUST list every visible pathway as a separate direction of travel.
- CONSISTENCY: Use standardized, noun-heavy names for objects (e.g., "one silver trash can") so they remain consistent across sequential images.

### 3. DESCRIPTOR FORMAT
Every landmark must include: [Quantity] + [Description] + [Distance] + [Relative Position].
- Distances: "nearby," "at a mid-distance," or "far."
- Positions: "on the left," "on the right," "straight ahead," or "extending to the [direction]."

### 4. EXAMPLE OUTPUT
{
  "landmarks": [
    "a hallway junction extending to the left",
    "a hallway junction extending forward to the far end",
    "two open doorways nearby on the right side",
    "one wooden desk at a mid-distance on the left",
    "a set of double glass exits far at the end of the forward path"
  ]
}
"""


LANDMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["landmarks"],
}


@dataclass(frozen=True)
class VertexBatchConfig:
    project_id: str
    location: str
    bucket_name: str
    staging_prefix: str
    model_id: str
    local_data_path: Path
    max_upload_workers: int = 16
    skip_existing_uploads: bool = True

    @property
    def output_prefix(self) -> str:
        return f"{self.staging_prefix.rstrip('/')}/output/"


def upload_to_gcs(
    local_path: str | os.PathLike[str],
    gcs_path: str,
    *,
    project_id: str,
    bucket_name: str,
    skip_existing: bool = False,
) -> str:
    from google.cloud import storage

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    if skip_existing and blob.exists(client):
        return f"gs://{bucket_name}/{gcs_path}"
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{gcs_path}"


def _iter_scene_images(data_root: Path, scenes: Iterable[str]) -> list[tuple[str, Path]]:
    images: list[tuple[str, Path]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        if not scene_dir.is_dir():
            print(f"[vertex] skipping missing scene directory: {scene_dir}")
            continue
        scene_images = sorted(scene_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        images.extend((scene, path) for path in scene_images)
    return images


def _build_request(gcs_uri: str) -> dict:
    return {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": ANNOTATE_LANDMARKS_PROMPT},
                        {"file_data": {"mime_type": "image/jpeg", "file_uri": gcs_uri}},
                    ],
                }
            ],
            "generation_config": {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": LANDMARK_SCHEMA,
            },
        }
    }


def create_batch_input(
    config: VertexBatchConfig,
    scenes: Iterable[str],
    *,
    manifest_local_path: str | os.PathLike[str] = "batch_input.jsonl",
    manifest_gcs_path: str | None = None,
) -> str:
    image_refs = _iter_scene_images(config.local_data_path, scenes)
    upload_specs = [
        (
            idx,
            img_path,
            f"{config.staging_prefix.rstrip('/')}/images/{scene}/{img_path.name}",
        )
        for idx, (scene, img_path) in enumerate(image_refs)
    ]
    uploaded: list[str | None] = [None] * len(upload_specs)
    failures: list[tuple[Path, BaseException]] = []

    def worker(spec: tuple[int, Path, str]) -> tuple[int, str]:
        idx, img_path, gcs_path = spec
        uri = upload_to_gcs(
            img_path,
            gcs_path,
            project_id=config.project_id,
            bucket_name=config.bucket_name,
            skip_existing=config.skip_existing_uploads,
        )
        return idx, uri

    with ThreadPoolExecutor(max_workers=config.max_upload_workers) as pool:
        futures = {pool.submit(worker, spec): spec[1] for spec in upload_specs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading images"):
            img_path = futures[future]
            try:
                idx, uri = future.result()
                uploaded[idx] = uri
            except Exception as exc:
                failures.append((img_path, exc))

    if failures:
        details = "\n".join(f"  {path}: {type(exc).__name__}: {exc}" for path, exc in failures[:10])
        raise RuntimeError(f"{len(failures)} image uploads failed:\n{details}")

    manifest_path = Path(manifest_local_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        for uri in uploaded:
            if uri is None:
                continue
            f.write(json.dumps(_build_request(uri)) + "\n")

    manifest_gcs_path = manifest_gcs_path or f"{config.staging_prefix.rstrip('/')}/input/manifest.jsonl"
    return upload_to_gcs(
        manifest_path,
        manifest_gcs_path,
        project_id=config.project_id,
        bucket_name=config.bucket_name,
        skip_existing=False,
    )


def submit_batch_job(
    config: VertexBatchConfig,
    manifest_gcs_uri: str,
    *,
    job_display_name: str = "gemini_landmark_mass_annotation",
):
    from google.cloud import aiplatform

    aiplatform.init(project=config.project_id, location=config.location)
    return aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=f"publishers/google/models/{config.model_id}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=[manifest_gcs_uri],
        gcs_destination_prefix=f"gs://{config.bucket_name}/{config.output_prefix}",
    )


def submit_label_batch(
    config: VertexBatchConfig,
    scenes: Iterable[str],
    *,
    manifest_local_path: str | os.PathLike[str] = "batch_input.jsonl",
    job_display_name: str = "gemini_landmark_mass_annotation",
):
    manifest_gcs_uri = create_batch_input(
        config,
        scenes,
        manifest_local_path=manifest_local_path,
    )
    return submit_batch_job(config, manifest_gcs_uri, job_display_name=job_display_name)


def _extract_prediction(data: Mapping) -> tuple[str, str, list[str]]:
    parts = data["request"]["contents"][0]["parts"]
    image_uri = next(
        part["file_data"]["file_uri"]
        for part in parts
        if part.get("file_data") is not None
    )
    uri_parts = image_uri.split("/")
    image_name = uri_parts[-1]
    scene_name = uri_parts[-2]
    prediction_text = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
    prediction_json = json.loads(prediction_text)
    return scene_name, image_name, prediction_json.get("landmarks", [])


def reconstruct_batch_results(
    config: VertexBatchConfig,
    scene_to_output_dir: Mapping[str, str | os.PathLike[str]],
) -> list[Path]:
    from google.cloud import storage

    client = storage.Client(project=config.project_id)
    bucket = client.bucket(config.bucket_name)
    scene_data = defaultdict(lambda: {"landmarks": {}})

    for blob in bucket.list_blobs(prefix=config.output_prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        content = blob.download_as_text()
        for line in content.splitlines():
            if not line:
                continue
            try:
                scene_name, image_name, landmarks = _extract_prediction(json.loads(line))
            except (KeyError, IndexError, StopIteration, json.JSONDecodeError) as exc:
                print(f"[vertex] skipping malformed prediction line: {exc}")
                continue
            scene_data[scene_name]["landmarks"][image_name] = landmarks

    written: list[Path] = []
    for scene_name, content in scene_data.items():
        out_dir = scene_to_output_dir.get(scene_name)
        if out_dir is None:
            print(f"[vertex] no output directory configured for scene {scene_name}; skipped")
            continue
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        output_file = out_path / f"{scene_name}_landmarks.json"
        with output_file.open("w") as f:
            json.dump(content, f, indent=2)
        written.append(output_file)
        print(f"[vertex] reconstructed {output_file} ({len(content['landmarks'])} images)")
    return written
