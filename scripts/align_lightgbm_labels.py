#!/usr/bin/env python3
"""Align video-only LightGBM features with explicit multi-label clip targets.

Unknown labels remain unknown and are never converted to the negative class.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ID_COLUMNS = ["clip_id", "session_id", "game_name"]
LABEL_COLUMNS = [
    "fast_camera_rotation",
    "high_camera_acceleration",
    "multi_axis_rotation",
    "wide_fov",
    "high_object_density",
    "high_object_speed",
    "high_optical_flow",
]
FORBIDDEN_OUTCOME = re.compile(r"(^|_)(ms_rating|fms|ssq)($|_)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("features/lightgbm_features.csv")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("features/clip_risk_labels_v3.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("features/lightgbm_training_matrix.csv")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("results/lightgbm_label_alignment_log.txt")
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, expected: Iterable[str], source: Path) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalize_label(value: Any, label: str, row_number: int) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"", "unknown", "na", "n/a", "nan", "none", "null", "tbd"}:
        return "unknown"
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid label {value!r} for {label} at CSV row {row_number}; "
            "expected 0, 1, or unknown"
        ) from exc
    if numeric == 0.0:
        return 0
    if numeric == 1.0:
        return 1
    raise ValueError(
        f"Invalid label {value!r} for {label} at CSV row {row_number}; "
        "expected 0, 1, or unknown"
    )


def assert_unique(frame: pd.DataFrame, source: Path) -> None:
    if frame[ID_COLUMNS].isna().any().any():
        raise ValueError(f"{source} has missing identifier values")
    duplicate = frame.duplicated(ID_COLUMNS, keep=False)
    if duplicate.any():
        raise ValueError(
            f"{source} has duplicate IDs:\n"
            + frame.loc[duplicate, ID_COLUMNS].to_string(index=False)
        )


def main() -> None:
    args = parse_args()
    for path in (args.features, args.labels):
        if not path.exists():
            raise FileNotFoundError(path)
    features = pd.read_csv(args.features, encoding="utf-8-sig", low_memory=False)
    labels = pd.read_csv(args.labels, encoding="utf-8-sig", low_memory=False)
    require_columns(features, ID_COLUMNS, args.features)
    require_columns(labels, [*ID_COLUMNS, *LABEL_COLUMNS], args.labels)
    assert_unique(features, args.features)
    assert_unique(labels, args.labels)

    forbidden = [column for column in features.columns if FORBIDDEN_OUTCOME.search(column)]
    if forbidden:
        raise ValueError(
            "Feature table contains forbidden symptom outcome columns; refusing alignment: "
            f"{forbidden}"
        )

    normalized = labels[ID_COLUMNS].copy()
    for label in LABEL_COLUMNS:
        normalized[label] = [
            normalize_label(value, label, row_number + 2)
            for row_number, value in enumerate(labels[label].tolist())
        ]

    feature_keys = features[ID_COLUMNS].apply(tuple, axis=1)
    label_keys = normalized[ID_COLUMNS].apply(tuple, axis=1)
    feature_only = sorted(set(feature_keys) - set(label_keys), key=str)
    label_only = sorted(set(label_keys) - set(feature_keys), key=str)

    merged = features.merge(
        normalized,
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    merged["label_source"] = "matrix_rule_generated_proxy"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")

    log_lines = [
        "LightGBM clip-label alignment log",
        f"features={args.features}",
        f"labels={args.labels}",
        f"output={args.output}",
        f"feature_rows={len(features)} label_rows={len(labels)} matched_rows={len(merged)}",
        f"feature_only_ids={len(feature_only)}",
        f"label_only_ids={len(label_only)}",
        "unknown policy=preserved as literal unknown; never converted to 0",
        "label source=matrix_rule_generated_proxy; not MS/FMS/SSQ",
        "training input policy=downstream script uses an explicit video-only whitelist",
        "depth/metric camera/object input policy=not present in the video feature whitelist",
        "",
    ]
    if feature_only:
        log_lines.append(f"feature_only_examples={feature_only[:20]}")
    if label_only:
        log_lines.append(f"label_only_examples={label_only[:20]}")
    for label in LABEL_COLUMNS:
        counts = merged[label].astype(str).value_counts(dropna=False)
        log_lines.append(
            f"{label}: 0={int(counts.get('0', 0))} 1={int(counts.get('1', 0))} "
            f"unknown={int(counts.get('unknown', 0))}"
        )
    if "extraction_status" in merged.columns:
        status_counts = merged["extraction_status"].fillna("missing").value_counts().to_dict()
        log_lines.append(f"extraction_status={status_counts}")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(merged)} matched rows to {args.output}; "
        f"feature-only={len(feature_only)}, label-only={len(label_only)}"
    )


if __name__ == "__main__":
    main()
