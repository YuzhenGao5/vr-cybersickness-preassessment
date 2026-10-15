#!/usr/bin/env python3
"""Combine cleaned Pirate_Life and ten new-game features with frozen labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


LABEL_COLUMNS = ["ms_rating_non_null", "ms_rating_max", "fms_label", "fms_label_binary"]


def read_rating(csv_path: Path, chunk_size: int = 256) -> tuple[int, float | None]:
    count = 0
    maximum: float | None = None
    for chunk in pd.read_csv(
        csv_path,
        usecols=["MS_rating"],
        chunksize=chunk_size,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        values = pd.to_numeric(chunk["MS_rating"], errors="coerce").dropna()
        count += int(values.size)
        if not values.empty:
            value = float(values.max())
            maximum = value if maximum is None else max(maximum, value)
    return count, maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pirate-features", type=Path, default=Path("features/pirate_life_features_clean.csv"))
    parser.add_argument("--pirate-labels", type=Path, default=Path("features/pipeline_training_matrix.csv"))
    parser.add_argument("--new-features", type=Path, default=Path("features/v5_new_session_features.csv"))
    parser.add_argument("--selection", type=Path, default=Path("results/selected_sessions_v5.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/vrnet/v5_sessions"))
    parser.add_argument("--config", type=Path, default=Path("configs/label_mapping_vrnet_binary_v5.yaml"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_training_matrix_v5.csv"))
    args = parser.parse_args()

    for path in (args.pirate_features, args.pirate_labels, args.new_features, args.selection, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    threshold = float(config["threshold"]["value"])
    if threshold != 2.0 or config.get("frozen") is not True:
        raise ValueError("v5 label mapping must remain frozen at threshold 2")

    pirate_features = pd.read_csv(args.pirate_features, encoding="utf-8-sig")
    old_matrix = pd.read_csv(args.pirate_labels, encoding="utf-8-sig")
    pirate_labels = old_matrix[["session_id", *LABEL_COLUMNS]].copy()
    pirate = pirate_features.merge(pirate_labels, on="session_id", how="inner", validate="one_to_one")
    pirate.insert(1, "game_name", "Pirate_Life")
    if len(pirate) != 11:
        raise ValueError(f"Expected 11 cleaned Pirate_Life sessions, found {len(pirate)}")

    new_features = pd.read_csv(args.new_features, encoding="utf-8-sig")
    selection = pd.read_csv(args.selection, encoding="utf-8-sig")
    successful = selection[selection["download_status"].astype(str).str.startswith("success")]
    labels: list[dict[str, object]] = []
    for _, row in successful.iterrows():
        game = str(row["game_name"])
        session = str(row["session_name"])
        source = args.raw_dir / game / session / "data_file.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        count, maximum = read_rating(source)
        if maximum is None:
            raise ValueError(f"No usable MS_rating in {source}")
        binary = 0 if maximum <= threshold else 1
        labels.append(
            {
                "session_id": session,
                "ms_rating_non_null": count,
                "ms_rating_max": maximum,
                "fms_label": "low" if binary == 0 else "medium",
                "fms_label_binary": binary,
            }
        )
        print(f"{session}: non_null={count}, max={maximum}, label={'low' if binary == 0 else 'medium'}", flush=True)
    new_labels = pd.DataFrame(labels)
    new = new_features.merge(new_labels, on="session_id", how="inner", validate="one_to_one")
    if len(new) != len(successful):
        raise ValueError(f"Expected {len(successful)} new labelled sessions, produced {len(new)}")

    feature_columns = [
        column for column in pirate_features.columns if column != "session_id"
    ]
    if set(feature_columns) != set(column for column in new_features.columns if column not in {"session_id", "game_name"}):
        raise ValueError("Pirate and new-game feature schemas differ")
    columns = ["session_id", "game_name", *feature_columns, *LABEL_COLUMNS]
    combined = pd.concat([pirate[columns], new[columns]], ignore_index=True)
    if combined["session_id"].duplicated().any():
        raise ValueError("Duplicate session IDs in combined training matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    distribution = combined["fms_label"].value_counts().to_dict()
    print(
        f"Wrote {len(combined)} sessions from {combined['game_name'].nunique()} games; "
        f"features={len(feature_columns)} labels={distribution} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
