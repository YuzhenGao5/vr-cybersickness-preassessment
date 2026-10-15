#!/usr/bin/env python3
"""Align sparse MS_rating events to the nearest 30-second feature window."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def class_for(value: float, classes: dict) -> str:
    for label in ("low", "medium", "high"):
        limits = classes[label]
        low, high = limits.get("min_inclusive"), limits.get("max_inclusive")
        if (low is None or value >= float(low)) and (high is None or value <= float(high)):
            return label
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default="data/raw/vrnet/samples_full/11_2_Mars_Miners")
    parser.add_argument("--features", default="features/vrnet_shared_features.csv")
    parser.add_argument("--mapping", default="configs/label_mapping_vrnet.yaml")
    parser.add_argument("--output", default="features/vrnet_training_matrix.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    session_dir, feature_path, mapping_path, output_path = map(resolve,
        (args.session_dir, args.features, args.mapping, args.output))
    logger = logging.getLogger("vrnet_feature_extractor")
    try:
        if not (session_dir / "data_file.csv").is_file():
            raise FileNotFoundError(f"Required label CSV missing: {session_dir / 'data_file.csv'}")
        if not feature_path.is_file():
            raise FileNotFoundError(f"Shared feature matrix missing: {feature_path}")
        mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8-sig"))
        thresholds = mapping["classes"]
        raw = pd.read_csv(session_dir / "data_file.csv", encoding="utf-8-sig", usecols=["timestamp", "MS_rating"])
        raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw["MS_rating"] = pd.to_numeric(raw["MS_rating"], errors="coerce")
        events = raw.loc[raw["timestamp"].notna() & raw["MS_rating"].notna()].copy().sort_values("timestamp")
        events = events.reset_index(drop=True)
        if events.empty:
            raise ValueError("No non-empty MS_rating events found in local source CSV")
        event_times = events["timestamp"].to_numpy(dtype=float)
        origin = float(pd.to_numeric(raw["timestamp"], errors="coerce").min())
        feature_frame = pd.read_csv(feature_path, encoding="utf-8-sig").sort_values("window_index").reset_index(drop=True)
        starts = pd.to_numeric(feature_frame["window_start_sec"], errors="coerce").to_numpy(dtype=float)
        ends = pd.to_numeric(feature_frame["window_end_sec"], errors="coerce").to_numpy(dtype=float)
        centers = (starts + ends) / 2.0
        event_secs = (event_times - origin) / 1000.0
        assigned = []
        for center in centers:
            nearest = int(np.argmin(np.abs(event_secs - center)))
            event = events.iloc[nearest]
            assigned.append({
                "fms_raw": float(event["MS_rating"]),
                "fms_label": class_for(float(event["MS_rating"]), thresholds),
                "label_source": "nearest_sparse_event_to_window_center",
                "fms_event_index": nearest,
                "fms_event_time_sec": float(event_secs[nearest]),
                "fms_event_offset_from_window_center_sec": float(event_secs[nearest] - center),
            })
        output = pd.concat([feature_frame, pd.DataFrame(assigned)], axis=1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Aligned %d sparse source ratings to %d windows by nearest event to window center; source events=%d",
                    len(output), len(output), len(events))
        print(f"windows={len(output)} labeled={int(output['fms_label'].isin(['low','medium','high']).sum())} source_events={len(events)} labels={output['fms_label'].value_counts().to_dict()}")
        return 0
    except Exception as exc:
        logger.exception("FMS alignment failed; stopping: %s", exc)
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
