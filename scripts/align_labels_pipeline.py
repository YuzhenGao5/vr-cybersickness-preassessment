#!/usr/bin/env python3
"""Align session-level Pirate_Life features with frozen binary MS_rating labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_frozen_threshold(config_path: Path) -> float:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    threshold = float(config["threshold"]["value"])
    if threshold != 2.0 or config.get("frozen") is not True:
        raise ValueError("Label mapping must remain frozen at max(MS_rating) <= 2 vs > 2")
    return threshold


def read_session_label(csv_path: Path, chunk_size: int) -> tuple[int, float | None]:
    non_null = 0
    maximum: float | None = None
    for chunk in pd.read_csv(
        csv_path,
        usecols=["MS_rating"],
        chunksize=chunk_size,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        values = pd.to_numeric(chunk["MS_rating"], errors="coerce").dropna()
        non_null += int(values.size)
        if not values.empty:
            chunk_max = float(values.max())
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    return non_null, maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("features/pipeline_session_features.csv"))
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/vrnet/sampled_csv/Pirate_Life"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/label_mapping_pipeline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("features/pipeline_training_matrix.csv"))
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    for path in (args.features, args.config):
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    threshold = load_frozen_threshold(args.config)
    features = pd.read_csv(args.features, encoding="utf-8-sig")
    expected_sessions = set(features["session_id"].astype(str))
    csv_paths = sorted(args.raw_dir.glob("*/data_file.csv"))
    found_sessions = {path.parent.name for path in csv_paths}
    if len(csv_paths) != 11 or found_sessions != expected_sessions:
        raise ValueError(
            f"Raw/features session mismatch: raw={len(csv_paths)}, features={len(expected_sessions)}, "
            f"missing_raw={sorted(expected_sessions-found_sessions)}, extra_raw={sorted(found_sessions-expected_sessions)}"
        )

    labels: list[dict[str, object]] = []
    for csv_path in csv_paths:
        non_null, maximum = read_session_label(csv_path, args.chunk_size)
        if maximum is None:
            label_name = pd.NA
            label_binary = pd.NA
        elif maximum <= threshold:
            label_name = "low"
            label_binary = 0
        else:
            label_name = "medium"
            label_binary = 1
        labels.append(
            {
                "session_id": csv_path.parent.name,
                "ms_rating_non_null": non_null,
                "ms_rating_max": maximum,
                "fms_label": label_name,
                "fms_label_binary": label_binary,
            }
        )
        print(
            f"{csv_path.parent.name}: non_null={non_null}, max={maximum}, label={label_name}",
            flush=True,
        )

    label_frame = pd.DataFrame(labels)
    merged = features.merge(label_frame, on="session_id", how="left", validate="one_to_one")
    if merged["fms_label_binary"].isna().any():
        missing_labels = merged.loc[merged["fms_label_binary"].isna(), "session_id"].tolist()
        raise ValueError(f"Sessions with no usable MS_rating label: {missing_labels}")
    merged["fms_label_binary"] = merged["fms_label_binary"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Wrote {len(merged)} labelled sessions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
