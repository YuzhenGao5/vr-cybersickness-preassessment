#!/usr/bin/env python3
"""Parse the ten selected VR.net v5 session CSVs and combine frame features."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from parse_vrnet_matrices import OUTPUT_COLUMNS, parse_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("results/selected_sessions_v5.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/vrnet/v5_sessions"))
    parser.add_argument("--output", type=Path, default=Path("features/vrnet_matrix_features_v5_raw.csv"))
    parser.add_argument(
        "--per-session-dir",
        type=Path,
        default=Path("features/vrnet_matrix_features_v5_by_session"),
    )
    parser.add_argument("--log", type=Path, default=Path("results/matrix_parsing_log_v5.txt"))
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    selection = pd.read_csv(args.selection, encoding="utf-8-sig")
    selected = selection[selection["download_status"].astype(str).str.startswith("success")].copy()
    if selected.empty:
        raise ValueError("No successfully downloaded sessions in selection file")
    args.per_session_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "VR.net v5 selected-session matrix parsing log",
        f"Selected successful sessions: {len(selected)}",
        "No network access, model training, or Git operations are performed.",
        "",
    ]
    outputs: list[tuple[Path, str]] = []
    for _, row in selected.iterrows():
        game = str(row["game_name"])
        session = str(row["session_name"])
        source = args.raw_dir / game / session / "data_file.csv"
        destination = args.per_session_dir / f"{session}.csv"
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            summary = parse_session(source, destination, args.chunk_size)
            outputs.append((destination, game))
            message = (
                f"[{session}] SUCCESS game={game} rows={summary.get('rows', 0)} "
                f"fov={summary.get('fov_rows', 0)} camera_speed={summary.get('camera_speed_rows', 0)} "
                f"object_speed={summary.get('object_speed_rows', 0)} elapsed={summary.get('elapsed_sec', 0):.1f}s"
            )
            print(message, flush=True)
            log_lines.append(message)
            for key, value in sorted(summary.items()):
                log_lines.append(f"{key}: {value}")
            log_lines.append("")
        except Exception as exc:
            message = f"[{session}] FAILED game={game}: {type(exc).__name__}: {exc}"
            print(message, flush=True)
            log_lines.extend([message, traceback.format_exc(), ""])
        args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not outputs:
        raise RuntimeError("All selected sessions failed matrix parsing")
    combined_frames = []
    for path, game in outputs:
        parsed = pd.read_csv(path, low_memory=False)
        parsed.insert(1, "game_name", game)
        combined_frames.append(parsed)
    combined = pd.concat(combined_frames, ignore_index=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.12g")
    log_lines.extend(
        [
            "[COMBINED] SUCCESS",
            f"successful_sessions: {len(outputs)}",
            f"failed_sessions: {len(selected)-len(outputs)}",
            f"combined_rows: {len(combined)}",
            f"combined_output: {args.output}",
        ]
    )
    args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Combined {len(combined)} rows from {len(outputs)} sessions -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
