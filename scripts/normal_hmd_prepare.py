#!/usr/bin/env python3
"""Prepare continuous fixed-rate BodyRoot/HeadLocal sequences for Normal HMD v2.

This pipeline requires the real local EyeNavGS traces and reuses the frozen
participant split from ``results/exploration_data_report.json``.  It has no
synthetic fallback: a successful report therefore proves which real source and
which participants were used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from normal_hmd_common import (
    COORDINATE_SYSTEM,
    STATE_FEATURE_NAMES,
    align_quaternion_signs,
    atomic_json,
    atomic_npz,
    derive_body_head_state,
    normalize_quaternions,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    relative_quaternions,
    resample_pose,
    rotate_vectors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "eyenavgS"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_SPLIT_REPORT = REPO_ROOT / "results" / "exploration_data_report.json"
DEFAULT_REPORT = REPO_ROOT / "results" / "normal_hmd_v2_data_report.json"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def site_for(path: Path) -> str:
    lowered = {part.casefold() for part in path.parts}
    if "rutgers" in lowered or any("rutgers" in part for part in lowered):
        return "Rutgers"
    if "nthu" in lowered or any("nthu" in part for part in lowered):
        return "NTHU"
    return "unknown"


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream))


def discover_traces(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(data_dir.rglob("*.csv")):
        relative_path = path.relative_to(data_dir).as_posix()
        columns = read_header(path)
        if not path.stem.casefold().startswith("user"):
            excluded.append({"path": relative_path, "reason": "metadata/non-trace CSV"})
            continue
        missing = sorted(set(POSE_COLUMNS) - set(columns))
        if missing:
            raise ValueError(f"{relative_path} is missing pose columns {missing}")
        time_columns = [name for name in TIME_COLUMNS if name in columns]
        if len(time_columns) != 1:
            raise ValueError(f"{relative_path} has no unambiguous time column")
        scene = path.parent.name.casefold()
        if scene not in CANONICAL_SCENES:
            excluded.append({"path": relative_path, "reason": f"non-canonical scene {scene}"})
            continue
        match = re.match(r"^(user[^_]+)_", path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"cannot parse participant from {relative_path}")
        site = site_for(path.relative_to(data_dir))
        if site == "unknown":
            raise ValueError(f"cannot determine collection site for {relative_path}")
        participant = match.group(1).casefold()
        records.append(
            {
                "path": path,
                "relative_path": relative_path,
                "site": site,
                "scene": scene,
                "participant": participant,
                "participant_group": f"{site}:{participant}",
                "trajectory_id": f"{site}:{participant}:{scene}",
                "time_column": time_columns[0],
            }
        )
    return records, excluded


def load_scene_settings(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    settings: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(data_dir.rglob("scene_setting.csv")):
        site = site_for(path.relative_to(data_dir))
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                scene = str(row.get("Scene_Name") or row.get("Dataset_Name") or "").casefold()
                if scene not in CANONICAL_SCENES:
                    continue
                quaternion = np.fromstring(str(row["Quaternion"]), sep=",", dtype=np.float64)
                if quaternion.shape != (4,):
                    raise ValueError(f"bad scene quaternion in {path}: {row['Quaternion']!r}")
                scale = float(row["Scale"])
                settings[(site, scene)] = {
                    "quaternion_xyzw": normalize_quaternions(quaternion).tolist(),
                    "render_geometry_scale": scale,
                    "source": str(path.resolve()),
                }
    return settings


def load_frozen_assignment(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("synthetic") is not False or report.get("source_type") != "EyeNavGS":
        raise ValueError("frozen participant report is not a real EyeNavGS report")
    manifest = report.get("participant_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("frozen participant report has no participant_manifest")
    assignment: dict[str, str] = {}
    for split in SPLIT_NAMES:
        participants = manifest.get(split)
        if not isinstance(participants, list) or not participants:
            raise ValueError(f"frozen participant split {split!r} is empty")
        for participant in participants:
            if participant in assignment:
                raise ValueError(f"participant appears in multiple splits: {participant}")
            assignment[str(participant)] = split
    return assignment, report


def pair_stereo_rows(
    frame: pd.DataFrame, time_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    numeric = frame[[*POSE_COLUMNS, time_column]].apply(pd.to_numeric, errors="coerce")
    pair_count = len(numeric) // 2
    if pair_count < 2:
        raise ValueError("fewer than two candidate stereo frames")
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
        raise ValueError("fewer than two valid stereo pairs")
    first_position = first_position[valid]
    second_position = second_position[valid]
    first_q = normalize_quaternions(first_q[valid])
    second_q = normalize_quaternions(second_q[valid])
    second_q *= np.where(np.sum(first_q * second_q, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    centre_position = (first_position + second_position) * 0.5
    head_q = normalize_quaternions(first_q + second_q)
    timestamps_seconds = (first_time[valid] + second_time[valid]) * 0.0005
    return centre_position, head_q, timestamps_seconds, {
        "raw_rows": int(len(frame)),
        "candidate_pairs": int(pair_count),
        "valid_pairs": int(valid.sum()),
        "invalid_pairs": int(pair_count - valid.sum()),
        "unpaired_rows": int(len(frame) - pair_count * 2),
    }


def monotonic_runs(timestamps: np.ndarray, max_gap_seconds: float) -> list[slice]:
    breaks = np.flatnonzero(
        (np.diff(timestamps) <= 0.0) | (np.diff(timestamps) > max_gap_seconds)
    ) + 1
    boundaries = np.concatenate(([0], breaks, [len(timestamps)]))
    return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= 2]


def valid_motion_mask(
    positions: np.ndarray,
    quaternions: np.ndarray,
    dt_seconds: float,
    max_linear_speed_mps: float,
    max_angular_speed_radps: float,
) -> np.ndarray:
    linear_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt_seconds
    relative_q = relative_quaternions(quaternions[:-1], quaternions[1:])
    angular_speed = np.linalg.norm(quaternion_to_rotation_vector(relative_q), axis=1) / dt_seconds
    valid_transition = (
        np.isfinite(linear_speed)
        & np.isfinite(angular_speed)
        & (linear_speed <= max_linear_speed_mps)
        & (angular_speed <= max_angular_speed_radps)
    )
    return np.concatenate(([True], valid_transition))


def contiguous_true_runs(mask: np.ndarray, minimum_length: int = 3) -> list[slice]:
    """Return clean pose runs, excluding the arrival frame of every bad jump.

    Body/head proxy filters must be reset after a teleport or tracking spike.
    Merely masking the one bad transition after filtering leaves seconds of
    low-pass recovery contamination in otherwise-valid training windows.
    """

    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        slice(int(start), int(end))
        for start, end in zip(starts, ends)
        if int(end - start) >= minimum_length
    ]


def select_window_starts(
    state: np.ndarray,
    valid_pose: np.ndarray,
    history: int,
    rollout: int,
    stride: int,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    length = history + rollout
    if len(state) < length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, len(state) - length + 1, stride, dtype=np.int64)
    keep = np.array([bool(valid_pose[start : start + length].all()) for start in starts])
    starts = starts[keep]
    if maximum > 0 and len(starts) > maximum:
        starts = np.sort(rng.choice(starts, size=maximum, replace=False))
    return starts


def process_trace(
    record: dict[str, Any],
    scene_setting: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = pd.read_csv(record["path"], usecols=[*POSE_COLUMNS, record["time_column"]])
    positions, quaternions, timestamps, pair_stats = pair_stereo_rows(
        frame, record["time_column"]
    )
    alignment_q = np.asarray(scene_setting["quaternion_xyzw"], dtype=np.float64)
    aligned_positions = rotate_vectors(
        np.broadcast_to(alignment_q, (len(positions), 4)), positions
    )
    aligned_quaternions = align_quaternion_signs(
        quaternion_multiply(np.broadcast_to(alignment_q, (len(quaternions), 4)), quaternions)
    )

    entries: list[dict[str, Any]] = []
    skipped_short_runs = 0
    total_resampled_frames = 0
    total_candidate_windows = 0
    total_clean_motion_runs = 0
    split_runs = monotonic_runs(timestamps, args.max_source_gap_seconds)
    for run_index, run in enumerate(split_runs):
        try:
            fixed_time, fixed_position, fixed_q = resample_pose(
                timestamps[run],
                aligned_positions[run],
                aligned_quaternions[run],
                args.dt_seconds,
            )
        except ValueError:
            skipped_short_runs += 1
            continue
        total_resampled_frames += len(fixed_time)
        valid_pose = valid_motion_mask(
            fixed_position,
            fixed_q,
            args.dt_seconds,
            args.max_linear_speed_mps,
            args.max_angular_speed_radps,
        )
        clean_runs = contiguous_true_runs(valid_pose)
        total_clean_motion_runs += len(clean_runs)
        if not clean_runs:
            skipped_short_runs += 1
            continue
        for clean_index, clean_run in enumerate(clean_runs):
            clean_position = fixed_position[clean_run]
            clean_q = fixed_q[clean_run]
            state, proxy_stats = derive_body_head_state(
                clean_position,
                clean_q,
                args.dt_seconds,
                args.root_position_tau_seconds,
                args.root_height_tau_seconds,
                args.root_yaw_tau_seconds,
                args.movement_heading_threshold_mps,
                args.calibration_seconds,
            )
            path_seed = int.from_bytes(
                hashlib.sha256(
                    f"{record['relative_path']}:{run_index}:{clean_index}:{args.seed}".encode("utf-8")
                ).digest()[:8],
                "little",
            )
            starts = select_window_starts(
                state,
                np.ones(len(state), dtype=bool),
                args.history,
                args.rollout,
                args.stride,
                args.max_segments_per_trajectory,
                np.random.default_rng(path_seed),
            )
            total_candidate_windows += len(starts)
            if not len(starts):
                skipped_short_runs += 1
                continue
            source_start = int(clean_run.start or 0)
            entries.append(
                {
                    **record,
                    "trajectory_id": (
                        f"{record['trajectory_id']}:run{run_index}:clean{clean_index}"
                    ),
                    "run_index": run_index,
                    "clean_run_index": clean_index,
                    "X": np.stack([state[start : start + args.history] for start in starts]),
                    "Y": np.stack(
                        [state[start + args.history : start + args.history + args.rollout] for start in starts]
                    ),
                    "start_index": (starts + source_start).astype(np.int32),
                    "reference_eye_height_m": float(args.nominal_eye_height_m),
                    "height_zero_aligned_units": proxy_stats["height_zero_aligned_units"],
                    "resampled_frames": len(state),
                }
            )
    return entries, {
        **pair_stats,
        "monotonic_runs": len(split_runs),
        "clean_motion_runs": total_clean_motion_runs,
        "skipped_short_runs": skipped_short_runs,
        "resampled_frames": total_resampled_frames,
        "selected_segments": total_candidate_windows,
        "render_geometry_scale_not_applied_to_hmd_pose": float(
            scene_setting["render_geometry_scale"]
        ),
    }


def combine_split(
    split: str, entries: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    if not entries:
        raise ValueError(f"split {split} has no prepared trajectories")
    trajectory_ids = [entry["trajectory_id"] for entry in entries]
    participant_groups = [entry["participant_group"] for entry in entries]
    sites = [entry["site"] for entry in entries]
    scenes = [entry["scene"] for entry in entries]
    reference_heights = [entry["reference_eye_height_m"] for entry in entries]
    height_zero = [entry["height_zero_aligned_units"] for entry in entries]
    trajectory_index = np.concatenate(
        [np.full(len(entry["X"]), index, dtype=np.int32) for index, entry in enumerate(entries)]
    )
    output_path = output_dir / f"normal_hmd_{split}.npz"
    atomic_npz(
        output_path,
        X=np.concatenate([entry["X"] for entry in entries]).astype(np.float32),
        Y=np.concatenate([entry["Y"] for entry in entries]).astype(np.float32),
        trajectory_index=trajectory_index,
        start_index=np.concatenate([entry["start_index"] for entry in entries]),
        trajectory_ids=np.asarray(trajectory_ids),
        participant_groups=np.asarray(participant_groups),
        sites=np.asarray(sites),
        scenes=np.asarray(scenes),
        reference_eye_height_m=np.asarray(reference_heights, dtype=np.float32),
        height_zero_aligned_units=np.asarray(height_zero, dtype=np.float32),
    )
    sample_count = int(sum(len(entry["X"]) for entry in entries))
    return {
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "segments": sample_count,
        "trajectory_runs": len(entries),
        "participants": len(set(participant_groups)),
        "scenes": sorted(set(scenes)),
        "shape_X": [sample_count, int(entries[0]["X"].shape[1]), len(STATE_FEATURE_NAMES)],
        "shape_Y": [sample_count, int(entries[0]["Y"].shape[1]), len(STATE_FEATURE_NAMES)],
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.history < 2 or args.rollout < 1 or args.stride < 1:
        raise ValueError("history>=2, rollout>=1 and stride>=1 are required")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"EyeNavGS data directory not found: {args.data_dir}")
    assignment, frozen_report = load_frozen_assignment(args.split_report)
    records, excluded = discover_traces(args.data_dir)
    if not records:
        raise FileNotFoundError("no canonical EyeNavGS traces found")
    discovered_participants = {record["participant_group"] for record in records}
    missing_assignment = sorted(discovered_participants - set(assignment))
    if missing_assignment:
        raise ValueError(f"participants missing from frozen split: {missing_assignment}")
    settings = load_scene_settings(args.data_dir)
    missing_settings = sorted(
        {(record["site"], record["scene"]) for record in records} - set(settings)
    )
    if missing_settings:
        raise ValueError(f"missing scene settings: {missing_settings}")

    selected_records: list[dict[str, Any]] = []
    split_seen = Counter()
    for record in records:
        split = assignment[record["participant_group"]]
        if args.max_trajectories_per_split > 0 and split_seen[split] >= args.max_trajectories_per_split:
            continue
        split_seen[split] += 1
        selected_records.append(record)

    split_entries: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    trace_stats: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=1):
        split = assignment[record["participant_group"]]
        entries, stats = process_trace(record, settings[(record["site"], record["scene"])], args)
        split_entries[split].extend(entries)
        trace_stats.append(
            {
                "path": record["relative_path"],
                "split": split,
                "participant_group": record["participant_group"],
                "scene": record["scene"],
                **stats,
            }
        )
        if index % 25 == 0 or index == len(selected_records):
            print(f"prepared {index}/{len(selected_records)} source trajectories", flush=True)

    split_summaries = {
        split: combine_split(split, split_entries[split], args.output_dir)
        for split in SPLIT_NAMES
    }
    fingerprint = hashlib.sha256()
    for record in records:
        fingerprint.update(record["relative_path"].encode("utf-8"))
        fingerprint.update(str(record["path"].stat().st_size).encode("ascii"))
    frozen_manifest_sha = hashlib.sha256(
        json.dumps(frozen_report["participant_manifest"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": "2.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "source_type": "EyeNavGS",
        "synthetic": False,
        "data_root": str(args.data_dir.resolve()),
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "frozen_split_report": str(args.split_report.resolve()),
        "frozen_split_manifest_sha256": frozen_manifest_sha,
        "participant_manifest": frozen_report["participant_manifest"],
        "participant_split_reused_exactly": True,
        "selected_source_trajectories": len(selected_records),
        "canonical_source_trajectories": len(records),
        "smoke_subset": args.max_trajectories_per_split > 0,
        "task": (
            "Multi-step generation of a BodyRoot proxy (planar local motion + yaw) "
            "and bounded HeadLocal pose from continuous HMD pose history."
        ),
        "scope_warning": (
            "EyeNavGS has no pelvis, feet or full-body ground truth. BodyRoot is a "
            "documented low-pass proxy. The model is not scene-aware, collision-aware, "
            "a navigation planner, or a cybersickness model."
        ),
        "fixed_dt_seconds": args.dt_seconds,
        "history_length": args.history,
        "rollout_length": args.rollout,
        "stride": args.stride,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "coordinate_system": COORDINATE_SYSTEM,
        "gravity_alignment": (
            "The official per-site/per-scene XYZW quaternion is left-multiplied onto "
            "both HMD position and orientation before the +Y-up body proxy is derived."
        ),
        "position_scale_policy": (
            "The scene Scale metadata controls render geometry and is not applied to HMD "
            "world-space pose. Position units are treated as metres, consistent with the "
            "existing EyeNavGS pipeline; this assumption remains a dataset limitation."
        ),
        "stereo_processing": (
            "Adjacent left/right records are paired; eye positions are averaged and "
            "sign-aligned eye quaternions are normalized after averaging."
        ),
        "resampling": "linear position interpolation plus shortest-arc quaternion SLERP",
        "body_proxy": {
            "root_dofs": "planar XZ translation plus yaw only; pitch/roll are structurally zero",
            "root_position_tau_seconds": args.root_position_tau_seconds,
            "root_height_detrend_tau_seconds": args.root_height_tau_seconds,
            "root_height_policy": (
                "A causal low-pass height baseline removes slow terrain/floor elevation "
                "from HeadLocal.y; the baseline is not emitted as a vertical BodyRoot DOF."
            ),
            "root_yaw_tau_seconds": args.root_yaw_tau_seconds,
            "movement_heading_threshold_mps": args.movement_heading_threshold_mps,
            "head_height_zero": (
                f"median aligned Y over the first {args.calibration_seconds:g} seconds of each run"
            ),
            "nominal_eye_height_m": args.nominal_eye_height_m,
            "nominal_eye_height_warning": (
                "Engineering display offset only; EyeNavGS scene coordinates do not provide "
                "a reliable participant-specific floor or measured standing height."
            ),
            "head_pose_form": "bounded absolute pose relative to BodyRoot, not integrated velocity",
        },
        "filtering": {
            "max_source_gap_seconds": args.max_source_gap_seconds,
            "max_linear_speed_mps": args.max_linear_speed_mps,
            "max_angular_speed_radps": args.max_angular_speed_radps,
            "max_segments_per_trajectory_run": args.max_segments_per_trajectory,
        },
        "splits": split_summaries,
        "excluded_files": excluded,
        "trace_quality_summary": {
            "raw_rows": int(sum(item["raw_rows"] for item in trace_stats)),
            "valid_stereo_pairs": int(sum(item["valid_pairs"] for item in trace_stats)),
            "resampled_frames": int(sum(item["resampled_frames"] for item in trace_stats)),
            "selected_segments": int(sum(item["selected_segments"] for item in trace_stats)),
            "timestamp_runs": int(sum(item["monotonic_runs"] for item in trace_stats)),
            "clean_motion_runs": int(sum(item["clean_motion_runs"] for item in trace_stats)),
        },
        "trace_stats": trace_stats,
    }
    atomic_json(args.report, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-report", type=Path, default=DEFAULT_SPLIT_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dt-seconds", type=float, default=0.027)
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-segments-per-trajectory", type=int, default=64)
    parser.add_argument("--max-trajectories-per-split", type=int, default=0)
    parser.add_argument("--max-source-gap-seconds", type=float, default=0.2)
    parser.add_argument("--max-linear-speed-mps", type=float, default=4.0)
    parser.add_argument("--max-angular-speed-radps", type=float, default=15.0)
    parser.add_argument("--root-position-tau-seconds", type=float, default=0.25)
    parser.add_argument("--root-height-tau-seconds", type=float, default=0.60)
    parser.add_argument("--root-yaw-tau-seconds", type=float, default=0.75)
    parser.add_argument("--movement-heading-threshold-mps", type=float, default=0.08)
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--nominal-eye-height-m", type=float, default=1.65)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = prepare(args)
    except Exception as exc:
        atomic_json(
            args.report,
            {
                "schema_version": "2.0.0",
                "generated_at_utc": utc_now(),
                "status": "failed",
                "reason": str(exc),
                "synthetic": False,
            },
        )
        print(f"Normal HMD preparation failed: {exc}", file=sys.stderr)
        return 1
    print("Normal HMD v2 data preparation complete")
    for split, summary in report["splits"].items():
        print(f"  {split}: {summary['segments']} segments -> {summary['path']}")
    print(f"  report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
