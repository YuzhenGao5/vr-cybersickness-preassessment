#!/usr/bin/env python3
"""Aggregate cleaned v5 frame features to session-level model features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_session_features_pipeline import AGGREGATIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_matrix_features_v5.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/v5_new_session_features.csv"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    frame = pd.read_csv(args.input, low_memory=False)
    required = {"session_id", "game_name", *(source for source, _ in AGGREGATIONS.values())}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "camera_jump_flag" in frame.columns:
        jump = frame["camera_jump_flag"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[~jump].copy()
    mapping = frame[["session_id", "game_name"]].drop_duplicates()
    if mapping["session_id"].duplicated().any():
        raise ValueError("A session maps to multiple games")
    for source, _ in AGGREGATIONS.values():
        frame[source] = pd.to_numeric(frame[source], errors="coerce")
    named = {
        output: pd.NamedAgg(column=source, aggfunc=operation)
        for output, (source, operation) in AGGREGATIONS.items()
    }
    aggregated = frame.groupby("session_id", sort=True).agg(**named).reset_index()
    aggregated = mapping.merge(aggregated, on="session_id", how="inner", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Wrote {len(aggregated)} sessions and {len(AGGREGATIONS)} features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
