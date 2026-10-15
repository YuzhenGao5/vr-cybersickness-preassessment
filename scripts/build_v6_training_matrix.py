#!/usr/bin/env python3
"""Build the frozen-label v6 session training matrix from local features and ratings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


LABEL_COLUMNS = ["ms_rating_non_null", "ms_rating_max", "fms_label", "fms_label_binary"]


def read_rating(path: Path, chunk_size: int = 256) -> tuple[int, float | None]:
    count = 0
    maximum: float | None = None
    for chunk in pd.read_csv(
        path, usecols=["MS_rating"], chunksize=chunk_size,
        encoding="utf-8-sig", low_memory=False,
    ):
        values = pd.to_numeric(chunk["MS_rating"], errors="coerce").dropna()
        count += int(values.size)
        if not values.empty:
            chunk_max = float(values.max())
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    return count, maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("features/v6_session_features.csv"))
    parser.add_argument("--v5-matrix", type=Path, default=Path("features/vrnet_training_matrix_v5.csv"))
    parser.add_argument("--selection", type=Path, default=Path("results/session_expansion_selection_v6.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/vrnet/v6_sessions"))
    parser.add_argument("--config", type=Path, default=Path("configs/label_mapping_vrnet_binary_v6.yaml"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_training_matrix_v6.csv"))
    args = parser.parse_args()
    for path in (args.features, args.v5_matrix, args.selection, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    threshold = float(config["threshold"]["value"])
    if threshold != 2.0 or config.get("frozen") is not True:
        raise ValueError("v6 label mapping must be frozen at max(MS_rating) threshold 2")

    features = pd.read_csv(args.features, encoding="utf-8-sig")
    previous = pd.read_csv(args.v5_matrix, encoding="utf-8-sig")
    old_labels = previous[["session_id", *LABEL_COLUMNS]].copy()
    selection = pd.read_csv(args.selection, encoding="utf-8-sig")
    successful = selection[selection["download_status"].astype(str).str.startswith("success")].copy()
    if len(successful) != 40:
        raise ValueError(f"expected 40 successful new sessions, found {len(successful)}")
    new_labels: list[dict[str, object]] = []
    for _, row in successful.iterrows():
        game = str(row["game_name"])
        session = str(row["session_name"])
        source = args.raw_dir / game / session / "data_file.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        count, maximum = read_rating(source)
        if maximum is None:
            raise ValueError(f"no usable MS_rating in {source}")
        binary = int(maximum > threshold)
        new_labels.append(
            {
                "session_id": session,
                "ms_rating_non_null": count,
                "ms_rating_max": maximum,
                "fms_label": "medium" if binary else "low",
                "fms_label_binary": binary,
            }
        )
        print(f"{session}: non_null={count}, max={maximum:g}, label={'medium' if binary else 'low'}", flush=True)
    labels = pd.concat([old_labels, pd.DataFrame(new_labels)], ignore_index=True)
    if labels["session_id"].duplicated().any() or len(labels) != 61:
        raise ValueError(f"expected 61 unique labels, got {len(labels)}")
    combined = features.merge(labels, on="session_id", how="inner", validate="one_to_one")
    if len(combined) != 61 or combined["game_name"].nunique() != 11:
        raise ValueError(
            f"expected 61 sessions/11 games, got {len(combined)}/{combined['game_name'].nunique()}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    counts = combined.groupby("game_name")["session_id"].nunique().to_dict()
    distribution = combined["fms_label"].value_counts().to_dict()
    print(f"Wrote {len(combined)} sessions; games={counts}; labels={distribution} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
