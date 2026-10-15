#!/usr/bin/env python3
"""Z-score session features independently within each VR.net game."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features/vrnet_training_matrix_v7_clean.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/feature_set_v7.yaml"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_training_matrix_v7_normalized.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = list(config["feature_sets"]["full"])
    normalized = data.copy()
    zero_variance: list[tuple[str, str]] = []
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce")
        means = values.groupby(data["game_name"]).transform("mean")
        stds = values.groupby(data["game_name"]).transform("std")
        for game in sorted(data.loc[stds.fillna(0).eq(0), "game_name"].astype(str).unique()):
            zero_variance.append((game, feature))
        z = (values - means) / stds
        z = z.where(stds.notna() & stds.ne(0), 0.0)
        normalized[feature] = z.replace([np.inf, -np.inf], np.nan)
    if not normalized[["session_id", "game_name", "fms_label_binary"]].equals(
        data[["session_id", "game_name", "fms_label_binary"]]
    ):
        raise RuntimeError("normalization changed identifiers or labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Wrote {len(normalized)} sessions and {len(features)} normalized features -> {args.output}")
    print(f"game-feature zero-variance cells set to 0: {len(zero_variance)}")
    if zero_variance:
        print(zero_variance)
    print("WARNING: within-game normalization uses the distribution of each held-out game and is transductive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
