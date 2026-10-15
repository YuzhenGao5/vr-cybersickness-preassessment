#!/usr/bin/env python3
"""Prepare leakage-resistant 6-DoF behaviour-cloning windows from EyeNavGS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exploration_policy import (
    FEATURE_NAMES,
    normalize_quaternions,
    quaternion_conjugate,
    quaternion_to_rotation_vector,
    relative_quaternions,
    rotate_vectors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "eyenavgS"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_REPORT = REPO_ROOT / "results" / "exploration_data_report.json"
CANONICAL_SCENES = (
    "alameda", "berlin", "bicycle", "drjohnson", "london", "nyc",
    "playroom", "room", "stump", "train", "treehill", "truck",
)
POSE_COLUMNS = (
    "ViewIndex", "PositionX", "PositionY", "PositionZ",
    "QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW",
)
TIME_COLUMNS = ("Timestamp", "timestep")
SPLIT_NAMES = ("train", "val", "test")
SOURCE_URLS = (
    "https://github.com/symmru/EyeNavGS_Rutgers_Dataset",
    "https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def site_for(path: Path) -> str:
    lowered = {part.casefold() for part in path.parts}
    if "rutgers" in lowered:
        return "Rutgers"
    if "nthu" in lowered:
        return "NTHU"
    return "unknown"


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        return next(csv.reader(stream))


def discover_traces(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(data_dir.rglob("*.csv")):
        relative = path.relative_to(data_dir).as_posix()
        columns = read_header(path)
        if not path.stem.casefold().startswith("user"):
            excluded.append({"path": relative, "reason": "metadata or non-trace CSV"})
            continue
        missing = set(POSE_COLUMNS) - set(columns)
        if missing:
            raise ValueError(f"Trace {relative} is missing pose columns: {sorted(missing)}")
        time_matches = [column for column in TIME_COLUMNS if column in columns]
        if len(time_matches) != 1:
            raise ValueError(f"Trace {relative} has no unambiguous time column")
        scene = path.parent.name.casefold()
        if scene not in CANONICAL_SCENES:
            excluded.append({"path": relative, "reason": f"non-canonical scene '{scene}'"})
            continue
        participant_match = re.match(r"^(user[^_]+)_", path.stem, re.IGNORECASE)
        if not participant_match:
            raise ValueError(f"Cannot parse participant ID from {relative}")
        site = site_for(path.relative_to(data_dir))
        if site == "unknown":
            raise ValueError(f"Cannot determine collection site for {relative}")
        participant = participant_match.group(1).casefold()
        records.append(
            {
                "path": path,
                "relative_path": relative,
                "site": site,
                "scene": scene,
                "participant": participant,
                "participant_group": f"{site}:{participant}",
                "trajectory_id": f"{site}:{participant}:{scene}",
                "time_column": time_matches[0],
            }
        )
    return records, excluded


def load_scene_scales(data_dir: Path) -> dict[tuple[str, str], float]:
    scales: dict[tuple[str, str], float] = {}
    settings_files = sorted(data_dir.rglob("scene_setting.csv"))
    for path in settings_files:
        site = site_for(path.relative_to(data_dir))
        frame = pd.read_csv(path)
        name_column = "Scene_Name" if "Scene_Name" in frame.columns else "Dataset_Name"
        if name_column not in frame.columns or "Scale" not in frame.columns:
            raise ValueError(f"Unsupported scene settings schema: {path}")
        for _, row in frame.iterrows():
            scene = str(row[name_column]).strip().casefold()
            scale = float(row["Scale"])
            if scene in CANONICAL_SCENES and math.isfinite(scale) and scale > 0:
                scales[(site, scene)] = scale
    return scales


def participant_splits(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[str]] = defaultdict(list)
    for group in sorted({record["participant_group"] for record in records}):
        by_site[group.split(":", 1)[0]].append(group)
    assignment: dict[str, str] = {}
    for site, groups in sorted(by_site.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        count = len(groups)
        if count < 3:
            raise ValueError(f"Need at least three participants at {site}, got {count}")
        train_count = max(1, int(round(count * 0.70)))
        val_count = max(1, int(round(count * 0.15)))
        if train_count + val_count >= count:
            train_count = count - 2
            val_count = 1
        for group in groups[:train_count]:
            assignment[group] = "train"
        for group in groups[train_count : train_count + val_count]:
            assignment[group] = "val"
        for group in groups[train_count + val_count :]:
            assignment[group] = "test"
    return assignment


def pair_stereo_rows(frame: pd.DataFrame, time_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    numeric = frame[[*POSE_COLUMNS, time_column]].apply(pd.to_numeric, errors="coerce")
    pair_count = len(numeric) // 2
    if pair_count < 2:
        raise ValueError("trace has fewer than two stereo frames")
    first = numeric.iloc[: pair_count * 2 : 2].reset_index(drop=True)
    second = numeric.iloc[1 : pair_count * 2 : 2].reset_index(drop=True)
    first_view = first["ViewIndex"].to_numpy(dtype=np.float64)
    second_view = second["ViewIndex"].to_numpy(dtype=np.float64)
    valid_view = (
        np.isin(first_view, (0.0, 1.0))
        & np.isin(second_view, (0.0, 1.0))
        & (first_view != second_view)
    )
    position_columns = ["PositionX", "PositionY", "PositionZ"]
    quaternion_columns = ["QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"]
    first_position = first[position_columns].to_numpy(dtype=np.float64)
    second_position = second[position_columns].to_numpy(dtype=np.float64)
    first_q = first[quaternion_columns].to_numpy(dtype=np.float64)
    second_q = second[quaternion_columns].to_numpy(dtype=np.float64)
    first_time = first[time_column].to_numpy(dtype=np.float64)
    second_time = second[time_column].to_numpy(dtype=np.float64)
    finite = (
        np.isfinite(first_position).all(axis=1)
        & np.isfinite(second_position).all(axis=1)
        & np.isfinite(first_q).all(axis=1)
        & np.isfinite(second_q).all(axis=1)
        & np.isfinite(first_time)
        & np.isfinite(second_time)
    )
    valid = valid_view & finite
    if int(valid.sum()) < 2:
        raise ValueError("trace has fewer than two valid stereo pairs")

    first_position = first_position[valid]
    second_position = second_position[valid]
    first_q = normalize_quaternions(first_q[valid])
    second_q = normalize_quaternions(second_q[valid])
    sign = np.where(np.sum(first_q * second_q, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    second_q = second_q * sign
    centre_position = (first_position + second_position) * 0.5
    head_q = normalize_quaternions(first_q + second_q)
    timestamps_ms = (first_time[valid] + second_time[valid]) * 0.5
    stats = {
        "raw_rows": int(len(frame)),
        "candidate_pairs": int(pair_count),
        "valid_pairs": int(valid.sum()),
        "dropped_rows": int(len(frame) - pair_count * 2),
        "invalid_pairs": int(pair_count - valid.sum()),
    }
    return centre_position, head_q, timestamps_ms, stats


def trace_windows(
    record: dict[str, Any],
    scale: float,
    history: int,
    max_windows: int,
    min_dt: float,
    max_dt: float,
    max_linear_speed: float,
    max_angular_speed: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(record["path"], usecols=[*POSE_COLUMNS, record["time_column"]])
    positions, quaternions, timestamps_ms, pair_stats = pair_stereo_rows(
        frame, record["time_column"]
    )
    dt = np.diff(timestamps_ms) / 1000.0
    world_displacement_metres = np.diff(positions, axis=0) / scale
    current_q = quaternions[:-1]
    local_displacement = rotate_vectors(
        quaternion_conjugate(current_q), world_displacement_metres
    )
    rotation_vectors = quaternion_to_rotation_vector(
        relative_quaternions(current_q, quaternions[1:])
    )
    safe_dt = np.where(dt > 0.0, dt, 1.0)
    linear_velocity = local_displacement / safe_dt[:, None]
    angular_velocity = rotation_vectors / safe_dt[:, None]
    motions = np.concatenate((linear_velocity, angular_velocity), axis=1)
    linear_speed = np.linalg.norm(linear_velocity, axis=1)
    angular_speed = np.linalg.norm(angular_velocity, axis=1)
    valid_motion = (
        np.isfinite(motions).all(axis=1)
        & np.isfinite(dt)
        & (dt >= min_dt)
        & (dt <= max_dt)
        & (linear_speed <= max_linear_speed)
        & (angular_speed <= max_angular_speed)
    )
    if len(motions) <= history:
        raise ValueError("trace is shorter than the history window")
    valid_run = np.convolve(
        valid_motion.astype(np.int16), np.ones(history + 1, dtype=np.int16), mode="valid"
    ) == (history + 1)
    target_indices = np.flatnonzero(valid_run) + history
    available = len(target_indices)
    if max_windows > 0 and available > max_windows:
        target_indices = np.sort(rng.choice(target_indices, size=max_windows, replace=False))
    if not len(target_indices):
        raise ValueError("trace has no valid motion windows after filtering")
    x = np.stack([motions[index - history : index] for index in target_indices]).astype(np.float32)
    y = motions[target_indices].astype(np.float32)
    target_dt = dt[target_indices].astype(np.float32)
    stats = {
        **pair_stats,
        "scale_applied": float(scale),
        "motion_steps": int(len(motions)),
        "valid_motion_steps": int(valid_motion.sum()),
        "timestamp_resets_or_duplicates": int(np.sum(dt <= 0.0)),
        "available_windows": int(available),
        "selected_windows": int(len(target_indices)),
    }
    return x, y, target_dt, stats


def create_synthetic_records(
    history: int, seed: int, max_windows: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Explicit fallback with independent episodes; never labelled as EyeNavGS."""

    rng = np.random.default_rng(seed)
    split_data: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    participants = [f"synthetic:user{index:02d}" for index in range(30)]
    assignment = {}
    for index, participant in enumerate(participants):
        assignment[participant] = "train" if index < 21 else "val" if index < 25 else "test"
    for participant_index, participant in enumerate(participants):
        for episode in range(4):
            length = 900
            dt = np.full(length, 0.02, dtype=np.float64)
            phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
            frequency = rng.uniform(0.3, 1.2, size=3)
            time = np.arange(length) * dt[0]
            linear = np.column_stack(
                (
                    0.08 * np.sin(frequency[0] * time + phase[0]),
                    0.02 * np.sin(frequency[1] * time + phase[1]),
                    0.18 + 0.05 * np.sin(frequency[2] * time + phase[2]),
                )
            )
            angular = np.column_stack(
                (
                    0.15 * np.sin(0.7 * time + phase[0]),
                    0.25 * np.sin(0.4 * time + phase[1]),
                    0.55 * np.sin(0.25 * time + phase[2]),
                )
            )
            motion = np.concatenate((linear, angular), axis=1)
            targets = np.arange(history, length)
            if max_windows > 0 and len(targets) > max_windows:
                targets = np.sort(rng.choice(targets, max_windows, replace=False))
            split_data[assignment[participant]].append(
                {
                    "trajectory_id": f"{participant}:episode{episode}",
                    "participant_group": participant,
                    "scene": f"synthetic_episode_{episode}",
                    "site": "synthetic",
                    "x": np.stack([motion[i - history : i] for i in targets]).astype(np.float32),
                    "y": motion[targets].astype(np.float32),
                    "dt": dt[targets].astype(np.float32),
                    "stats": {"selected_windows": int(len(targets))},
                }
            )
    return split_data, {
        "reason": "EyeNavGS directory was unavailable or contained no trace CSV files",
        "participant_assignment": assignment,
    }


def combine_and_save(split_name: str, entries: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if not entries:
        raise ValueError(f"Split {split_name} has no trajectories")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    trajectory_ids: list[str] = []
    for trajectory_index, entry in enumerate(entries):
        trajectory_ids.append(entry["trajectory_id"])
        x_parts.append(entry["x"])
        y_parts.append(entry["y"])
        dt_parts.append(entry["dt"])
        trajectory_parts.append(
            np.full(len(entry["x"]), trajectory_index, dtype=np.int32)
        )
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    dt = np.concatenate(dt_parts, axis=0)
    trajectory_index = np.concatenate(trajectory_parts, axis=0)
    output_path = output_dir / f"exploration_{split_name}.npz"
    atomic_npz(
        output_path,
        X=x,
        y=y,
        dt_seconds=dt,
        trajectory_index=trajectory_index,
        trajectory_ids=np.asarray(trajectory_ids),
    )
    return {
        "path": str(output_path.resolve()),
        "samples": int(len(x)),
        "trajectories": len(entries),
        "participants": len({entry["participant_group"] for entry in entries}),
        "scenes": sorted({entry["scene"] for entry in entries}),
        "shape_X": list(x.shape),
        "shape_y": list(y.shape),
        "median_dt_seconds": float(np.median(dt)),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    use_synthetic = args.source == "synthetic"
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    if not use_synthetic and args.data_dir.is_dir():
        records, excluded = discover_traces(args.data_dir)
    if args.source == "eyenavgs" and not records:
        raise FileNotFoundError("No canonical EyeNavGS trace CSVs were found")
    if args.source == "auto" and not records:
        use_synthetic = True

    if use_synthetic:
        split_entries, fallback = create_synthetic_records(
            args.history, args.seed, args.max_windows_per_trajectory
        )
        assignment = fallback["participant_assignment"]
        processing_errors: list[dict[str, str]] = []
        trace_stats: list[dict[str, Any]] = []
        source_type = "synthetic"
        synthetic_reason = fallback["reason"] if args.source == "auto" else "explicit --source synthetic"
    else:
        source_type = "EyeNavGS"
        synthetic_reason = None
        scales = load_scene_scales(args.data_dir)
        assignment = participant_splits(records, args.seed)
        split_entries = {name: [] for name in SPLIT_NAMES}
        processing_errors = []
        trace_stats = []
        for index, record in enumerate(records, start=1):
            scale = scales.get((record["site"], record["scene"]))
            if scale is None:
                raise ValueError(f"Missing scene scale for {record['site']}/{record['scene']}")
            path_seed = int.from_bytes(
                hashlib.sha256(record["relative_path"].encode("utf-8")).digest()[:8], "little"
            ) ^ args.seed
            trace_rng = np.random.default_rng(path_seed)
            try:
                x, y, dt, stats = trace_windows(
                    record,
                    scale,
                    args.history,
                    args.max_windows_per_trajectory,
                    args.min_dt,
                    args.max_dt,
                    args.max_linear_speed,
                    args.max_angular_speed,
                    trace_rng,
                )
            except Exception as exc:
                processing_errors.append(
                    {"path": record["relative_path"], "reason": str(exc)}
                )
                continue
            split_name = assignment[record["participant_group"]]
            split_entries[split_name].append(
                {**record, "x": x, "y": y, "dt": dt, "stats": stats}
            )
            trace_stats.append(
                {
                    "path": record["relative_path"],
                    "split": split_name,
                    "site": record["site"],
                    "scene": record["scene"],
                    "participant_group": record["participant_group"],
                    **stats,
                }
            )
            if index % 50 == 0:
                print(f"  prepared: {index}/{len(records)} trajectories", flush=True)
        if processing_errors:
            raise RuntimeError(
                f"{len(processing_errors)} canonical traces failed preprocessing; "
                f"first failure: {processing_errors[0]}"
            )

    split_summaries = {
        name: combine_and_save(name, split_entries[name], args.output_dir)
        for name in SPLIT_NAMES
    }
    participant_manifest = {
        name: sorted(group for group, split in assignment.items() if split == name)
        for name in SPLIT_NAMES
    }
    fingerprint = hashlib.sha256()
    for record in records:
        fingerprint.update(record["relative_path"].encode("utf-8"))
        fingerprint.update(str(record["path"].stat().st_size).encode("ascii"))

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "source_type": source_type,
        "synthetic": use_synthetic,
        "synthetic_reason": synthetic_reason,
        "official_sources": list(SOURCE_URLS) if not use_synthetic else [],
        "data_root": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": fingerprint.hexdigest() if records else None,
        "seed": args.seed,
        "task": (
            "Predict the next head-centred local linear velocity (m/s) and local angular "
            "velocity (rad/s) from a history of 6-DoF velocities."
        ),
        "scope_warning": (
            "This is short-horizon motion behaviour cloning. It has no visual observation, goal, "
            "collision sensor or risk-zone input and is not an autonomous navigation or avoidance policy."
        ),
        "stereo_processing": (
            "Adjacent ViewIndex 0/1 rows are paired; eye positions are averaged as a head-centre "
            "estimate, head quaternions are sign-aligned and averaged."
        ),
        "coordinate_processing": (
            "World displacement is divided by the official per-site scene Scale to undo uniform "
            "scene scaling, then rotated into the current head-local frame. Constant scene rotation "
            "and translation cancel for the relative local motion used here."
        ),
        "history_length": args.history,
        "feature_names": list(FEATURE_NAMES),
        "input_shape": [args.history, len(FEATURE_NAMES)],
        "action_shape": [len(FEATURE_NAMES)],
        "filtering": {
            "min_dt_seconds": args.min_dt,
            "max_dt_seconds": args.max_dt,
            "max_linear_speed_mps": args.max_linear_speed,
            "max_angular_speed_radps": args.max_angular_speed,
            "max_windows_per_trajectory": args.max_windows_per_trajectory,
            "time_reset_policy": "nonpositive steps split windows; no sample may cross a reset",
        },
        "split_policy": (
            "Participants are split before window extraction, independently stratified by site "
            "with approximately 70/15/15 percent train/validation/test."
        ),
        "participant_manifest": participant_manifest,
        "splits": split_summaries,
        "canonical_trace_count": len(records) if records else None,
        "processed_trace_count": sum(len(values) for values in split_entries.values()),
        "excluded_files": excluded,
        "processing_errors": processing_errors,
        "trace_quality_summary": {
            "timestamp_resets_or_duplicates": int(
                sum(item.get("timestamp_resets_or_duplicates", 0) for item in trace_stats)
            ),
            "candidate_stereo_pairs": int(sum(item.get("candidate_pairs", 0) for item in trace_stats)),
            "valid_stereo_pairs": int(sum(item.get("valid_pairs", 0) for item in trace_stats)),
            "available_windows_before_cap": int(
                sum(item.get("available_windows", 0) for item in trace_stats)
            ),
            "selected_windows": int(sum(item.get("selected_windows", 0) for item in trace_stats)),
        },
        "protocol_scope": (
            "Independent exploration-motion demo; not a cybersickness-risk model or a result under "
            "the frozen PROTOCOL.md evaluation endpoints."
        ),
    }
    atomic_json(args.report, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source", choices=("auto", "eyenavgs", "synthetic"), default="auto")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=384)
    parser.add_argument("--min-dt", type=float, default=0.004)
    parser.add_argument("--max-dt", type=float, default=0.100)
    parser.add_argument("--max-linear-speed", type=float, default=4.0)
    parser.add_argument("--max-angular-speed", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.history < 2 or args.max_windows_per_trajectory < 0:
        print("history must be >=2 and max windows must be >=0", file=sys.stderr)
        return 2
    try:
        report = prepare(args)
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "generated_at_utc": utc_now(),
            "status": "failed",
            "reason": str(exc),
            "source_requested": args.source,
            "synthetic": None,
        }
        atomic_json(args.report, failure)
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    print("=" * 60)
    print("Exploration motion data prepared")
    print("=" * 60)
    print(f"Source : {report['source_type']} (synthetic={report['synthetic']})")
    for split, summary in report["splits"].items():
        print(
            f"{split:5s}  samples={summary['samples']:7d}  "
            f"participants={summary['participants']:2d}  trajectories={summary['trajectories']:3d}"
        )
    print(f"Report : {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
