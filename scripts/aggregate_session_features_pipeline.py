#!/usr/bin/env python3
"""Aggregate Pirate_Life frame-level matrix features to one row per session."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AGGREGATIONS = {
    "fov_horizontal_mean": ("fov_horizontal", "mean"),
    "fov_horizontal_max": ("fov_horizontal", "max"),
    "fov_horizontal_std": ("fov_horizontal", "std"),
    "fov_vertical_mean": ("fov_vertical", "mean"),
    "fov_vertical_max": ("fov_vertical", "max"),
    "fov_vertical_std": ("fov_vertical", "std"),
    "camera_speed_mean": ("camera_speed", "mean"),
    "camera_speed_max": ("camera_speed", "max"),
    "camera_speed_std": ("camera_speed", "std"),
    "camera_rotation_rate_mean": ("camera_rotation_rate", "mean"),
    "camera_rotation_rate_max": ("camera_rotation_rate", "max"),
    "camera_rotation_rate_std": ("camera_rotation_rate", "std"),
    "camera_acceleration_mean": ("camera_acceleration", "mean"),
    "camera_acceleration_max": ("camera_acceleration", "max"),
    "object_count_mean": ("object_count", "mean"),
    "object_count_max": ("object_count", "max"),
    "object_speed_mean": ("mean_object_speed", "mean"),
    "object_speed_max": ("max_object_speed", "max"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_matrix_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/pipeline_session_features.csv"))
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input does not exist: {args.input}")
    frame = pd.read_csv(args.input, low_memory=False)
    required = {"session_id", *(source for source, _ in AGGREGATIONS.values())}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sessions = sorted(frame["session_id"].dropna().astype(str).unique())
    if len(sessions) != 11 or any("Pirate_Life" not in session for session in sessions):
        raise ValueError(f"Expected exactly 11 Pirate_Life sessions, found {len(sessions)}: {sessions}")

    if "camera_jump_flag" in frame.columns:
        jump_flag = frame["camera_jump_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[~jump_flag].copy()

    for column in required.difference({"session_id"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    named_aggs = {
        output_name: pd.NamedAgg(column=source, aggfunc=operation)
        for output_name, (source, operation) in AGGREGATIONS.items()
    }
    aggregated = (
        frame.groupby("session_id", sort=True, observed=True)
        .agg(**named_aggs)
        .reset_index()
    )
    if len(aggregated) != 11:
        raise RuntimeError(f"Aggregation produced {len(aggregated)} rows instead of 11")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Wrote {len(aggregated)} sessions and {len(AGGREGATIONS)} features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
