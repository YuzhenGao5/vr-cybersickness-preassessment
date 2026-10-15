#!/usr/bin/env python3
"""Combine cleaned Pirate_Life, v5, and newly parsed v6 frame features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pirate", type=Path, default=Path("features/vrnet_matrix_features_clean.csv"))
    parser.add_argument("--v5", type=Path, default=Path("features/vrnet_matrix_features_v5.csv"))
    parser.add_argument("--v6-new", type=Path, default=Path("features/vrnet_matrix_features_v6_new_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_matrix_features_v6.csv"))
    args = parser.parse_args()
    frames: list[pd.DataFrame] = []
    for path, game in ((args.pirate, "Pirate_Life"), (args.v5, None), (args.v6_new, None)):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        if game is not None:
            if "game_name" in frame.columns:
                frame["game_name"] = game
            else:
                frame.insert(1, "game_name", game)
        if "game_name" not in frame.columns:
            raise ValueError(f"{path} has no game_name column")
        frames.append(frame)
    expected = set(frames[0].columns)
    for frame in frames[1:]:
        if set(frame.columns) != expected:
            raise ValueError("frame feature schemas differ")
    combined = pd.concat([frame[list(frames[0].columns)] for frame in frames], ignore_index=True)
    mapping = combined[["session_id", "game_name"]].drop_duplicates()
    if mapping["session_id"].duplicated().any():
        raise ValueError("one or more session IDs map to multiple games")
    if mapping["session_id"].nunique() != 61 or mapping["game_name"].nunique() != 11:
        raise ValueError(
            f"expected 61 sessions/11 games; got {mapping['session_id'].nunique()}/"
            f"{mapping['game_name'].nunique()}"
        )
    expected_counts = {"Pirate_Life": 11, **{game: 5 for game in mapping["game_name"].unique() if game != "Pirate_Life"}}
    actual_counts = mapping.groupby("game_name")["session_id"].nunique().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"unexpected session counts: {actual_counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    jumps = int(combined["camera_jump_flag"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    print(
        f"Wrote {len(combined)} frame rows, 61 sessions, 11 games, {jumps} marked jumps -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
