from __future__ import annotations

import argparse
from pathlib import Path

from vertex_annotation import VertexBatchConfig, submit_label_batch


PROJECT_ID = "cat-gen-lang-proj"
LOCATION = "global"
BUCKET_NAME = "landmark-annotations"
MODEL_ID = "gemini-3.1-pro-preview"
LOCAL_DATA_PATH = "/hdd/sacson"
GCS_STAGING_PATH = "landmark_batch_job"


DEFAULT_SCENES = [
    "Dec-06-2022-bww8_00000030_10",
    "Feb-09-2023-bww8-intloss_00000022_1",
    "Jan-17-2023-bww8_00000001_0",
    "Dec-06-2022-bww8_00000037_0",
    "Feb-09-2023-bww8-intloss_00000031_4",
    "Jan-17-2023-bww8_00000001_5",
    "Dec-07-2022-bww8_00000000_12",
    "Feb-09-2023-bww8-intloss_00000042_1",
    "Jan-17-2023-bww8_00000002_10",
    "Dec-12-2022-bww8_00000036_0",
    "Feb-13-2023-bww8-intloss_00000009_3",
    "Nov-17-2022-bww8_00000009_2",
    "Feb-09-2023-bww8-intloss_00000000_0",
    "Jan-12-2023-bww8_00000008_2",
    "Nov-17-2022-bww8_00000012_0",
    "Jan-12-2023-bww8_00000009_29",
    "Dec-06-2022-bww8_00000007_0",
    "Feb-09-2023-bww8-intloss_00000042_9",
    "Jan-12-2023-bww8_00000007_22",
    "Feb-03-2023-bww8-intloss_00000013_1",
    "Feb-14-2023-bww8-intloss_00000008_25",
]


def main():
    parser = argparse.ArgumentParser(description="Submit a Vertex landmark batch job.")
    parser.add_argument("--manifest", default="batch_input.jsonl")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config = VertexBatchConfig(
        project_id=PROJECT_ID,
        location=LOCATION,
        bucket_name=BUCKET_NAME,
        staging_prefix=GCS_STAGING_PATH,
        model_id=MODEL_ID,
        local_data_path=Path(LOCAL_DATA_PATH),
        max_upload_workers=args.workers,
    )
    job = submit_label_batch(config, DEFAULT_SCENES, manifest_local_path=args.manifest)
    print(f"Job submitted! Job ID: {job.name}")
    print(f"View progress in Console: https://console.cloud.google.com/vertex-ai/batch-predictions?project={PROJECT_ID}")


if __name__ == "__main__":
    main()