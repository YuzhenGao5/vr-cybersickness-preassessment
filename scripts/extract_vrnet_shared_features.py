#!/usr/bin/env python3
"""Build 30-second shared visual/HMD feature windows from local Mars Miners."""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feature_extractor import (
    aggregate_to_windows,
    extract_camera_speed,
    extract_depth_change_rate,
    extract_optical_flow,
    extract_path_curvature,
    extract_rotation_axes,
)


def parse_vector(value: object, expected: int) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        return [float("nan")] * expected
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (tuple, list)) and len(parsed) == expected:
            return [float(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        pass
    return [float("nan")] * expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default="data/raw/vrnet/samples_full/11_2_Mars_Miners")
    parser.add_argument("--cache", default="features/mars_miners_full_features.csv")
    parser.add_argument("--output", default="features/vrnet_shared_features.csv")
    parser.add_argument("--window-sec", type=float, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    session_dir, cache_path, output_path = map(resolve, (args.session_dir, args.cache, args.output))
    log_path = root / "results/feature_extractor_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    logger = logging.getLogger("vrnet_shared_features")
    try:
        raw_path = session_dir / "data_file.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Required local session CSV missing: {raw_path}")
        if not cache_path.is_file():
            raise FileNotFoundError(f"Required audited local image-feature cache missing: {cache_path}")
        raw = pd.read_csv(raw_path, encoding="utf-8-sig",
                          usecols=["frame", "timestamp", "head_pos", "head_dir"])
        timestamps = pd.to_numeric(raw["timestamp"], errors="coerce").to_numpy(dtype=float)
        positions = np.asarray([parse_vector(value, 3) for value in raw["head_pos"]], dtype=float)
        quaternions = np.asarray([parse_vector(value, 4) for value in raw["head_dir"]], dtype=float)
        if np.isfinite(positions).sum(axis=0).min() == 0:
            raise ValueError("head_pos exists but contains no complete 3-D pose samples")
        if np.isfinite(quaternions).sum(axis=0).min() == 0:
            raise ValueError("head_dir exists but contains no complete quaternion samples")

        speed = extract_camera_speed(positions, timestamps).rename(
            columns={"camera_speed_source_units_s": "hmd_head_translation_speed_source_units_s"})
        rotation = extract_rotation_axes(quaternions, threshold=0.5)
        curvature = extract_path_curvature(positions)
        pose_sample_features = pd.concat([speed.drop(columns=["timestamp"]), rotation, curvature], axis=1)
        pose_windows = aggregate_to_windows(pose_sample_features, timestamps, window_sec=args.window_sec)

        # The local full-feature pass already measured these exact image features
        # over 30-second windows. Reuse its auditable outputs rather than rerun
        # Farneback or invent depth from RGB.
        flow_windows = extract_optical_flow(session_dir / "video", window_size=30, cached_features=cache_path)
        depth_windows = extract_depth_change_rate(session_dir / "video", window_size=30, cached_features=cache_path)
        flow_columns = [column for column in flow_windows if column == "window_index" or column.startswith("flow_")]
        shared = pose_windows.merge(flow_windows[flow_columns], on="window_index", how="outer", validate="one_to_one")
        shared = shared.merge(depth_windows, on="window_index", how="outer", validate="one_to_one")
        shared["window_id"] = shared["window_index"].map(lambda value: f"window_{int(value):03d}")
        shared["session_id"] = "11_2_Mars_Miners"
        shared = shared.sort_values("window_index").reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shared.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Completed local feature aggregation: input_rows=%d output_windows=%d features=%d image_cache=%s",
                    len(raw), len(shared), len(shared.columns), cache_path)
        logger.info("head_pos/head_dir are HMD telemetry, not scene-camera control signals; speed units remain source-units/s.")
        logger.info("depth_change_rate reused the prior monocular-depth output; RGB images were not treated as depth maps.")
        print(f"input_rows={len(raw)} windows={len(shared)} columns={len(shared.columns)} output={output_path}")
        return 0
    except Exception as exc:
        logger.exception("Feature extraction failed; stopping: %s", exc)
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
