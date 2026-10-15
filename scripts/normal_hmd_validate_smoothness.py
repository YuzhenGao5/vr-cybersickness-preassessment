#!/usr/bin/env python3
"""Validate Normal HMD HeadLocal temporal smoothness against held-out EyeNavGS.

The validator consumes one or more Unity ``telemetry.csv`` files plus the
held-out Normal HMD test NPZ.  It measures each HeadLocal position/rotation
axis independently.  Position remains in metres and rotation-vector values
remain in radians throughout; no degree conversion is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_NPZ = (
    REPO_ROOT / "data" / "processed" / "normal_hmd_v2" / "normal_hmd_test.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "normal_hmd_smoothness_validation.json"
DEFAULT_NPZ_DT_SECONDS = 0.027

HEAD_FEATURE_NAMES = (
    "head_local_right_m",
    "head_local_up_offset_m",
    "head_local_forward_m",
    "head_local_rotvec_x_rad_rh",
    "head_local_rotvec_y_rad_rh",
    "head_local_rotvec_z_rad_rh",
)
HEAD_AXIS_UNITS = ("m", "m", "m", "rad", "rad", "rad")
FULL_STATE_FEATURE_NAMES = (
    "root_right_velocity_mps",
    "root_forward_velocity_mps",
    "root_yaw_rate_radps_rh",
    *HEAD_FEATURE_NAMES,
)

LAG1_MINIMUM = 0.70
REVERSAL_RATE_MAXIMUM_PER_SECOND = 8.5
MIN_SEQUENCE_FRAMES = 4


@dataclass(frozen=True)
class MotionSequence:
    """One contiguous sequence with no synthetic boundary between its frames."""

    label: str
    time_seconds: np.ndarray
    head_state: np.ndarray


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson r, or None when it is undefined for a constant series."""

    if len(x) < 2:
        return None
    x_centered = x.astype(np.float64, copy=False) - float(np.mean(x))
    y_centered = y.astype(np.float64, copy=False) - float(np.mean(y))
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered))
        * float(np.dot(y_centered, y_centered))
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    result = float(np.dot(x_centered, y_centered) / denominator)
    return float(np.clip(result, -1.0, 1.0))


def _percentile_95_absolute(values: np.ndarray) -> float | None:
    if not len(values):
        return None
    return float(np.percentile(np.abs(values), 95.0))


def compute_metrics(sequences: Sequence[MotionSequence]) -> dict[str, Any]:
    """Compute pooled within-sequence metrics without crossing boundaries."""

    delta_left: list[np.ndarray] = []
    delta_right: list[np.ndarray] = []
    acceleration_chunks: list[np.ndarray] = []
    jerk_chunks: list[np.ndarray] = []
    reversal_counts = np.zeros(len(HEAD_FEATURE_NAMES), dtype=np.int64)
    duration_seconds = 0.0
    frame_count = 0
    interval_count = 0

    for sequence in sequences:
        time_seconds = np.asarray(sequence.time_seconds, dtype=np.float64)
        state = np.asarray(sequence.head_state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != len(HEAD_FEATURE_NAMES):
            raise ValueError(f"{sequence.label}: expected HeadLocal shape (N, 6)")
        if len(state) != len(time_seconds):
            raise ValueError(f"{sequence.label}: time/state length mismatch")
        if len(state) < MIN_SEQUENCE_FRAMES:
            continue
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(time_seconds)):
            raise ValueError(f"{sequence.label}: sequence contains non-finite values")

        interval_dt = np.diff(time_seconds)
        if np.any(interval_dt <= 0.0):
            raise ValueError(f"{sequence.label}: timestamps are not strictly increasing")
        delta = np.diff(state, axis=0)
        velocity = delta / interval_dt[:, None]
        velocity_time = (time_seconds[:-1] + time_seconds[1:]) * 0.5
        acceleration_dt = np.diff(velocity_time)
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        acceleration_time = (velocity_time[:-1] + velocity_time[1:]) * 0.5
        jerk_dt = np.diff(acceleration_time)
        jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]

        previous_delta = delta[:-1]
        next_delta = delta[1:]
        reversals = ((previous_delta < 0.0) & (next_delta > 0.0)) | (
            (previous_delta > 0.0) & (next_delta < 0.0)
        )
        reversal_counts += np.count_nonzero(reversals, axis=0)
        delta_left.append(previous_delta)
        delta_right.append(next_delta)
        acceleration_chunks.append(acceleration)
        jerk_chunks.append(jerk)
        duration_seconds += float(time_seconds[-1] - time_seconds[0])
        frame_count += len(state)
        interval_count += len(delta)

    if not delta_left or duration_seconds <= 0.0:
        raise ValueError("no contiguous sequence is long enough for smoothness metrics")

    pooled_left = np.concatenate(delta_left, axis=0)
    pooled_right = np.concatenate(delta_right, axis=0)
    pooled_acceleration = np.concatenate(acceleration_chunks, axis=0)
    pooled_jerk = np.concatenate(jerk_chunks, axis=0)

    per_axis: dict[str, dict[str, Any]] = {}
    for axis, feature_name in enumerate(HEAD_FEATURE_NAMES):
        per_axis[feature_name] = {
            "state_unit": HEAD_AXIS_UNITS[axis],
            "delta_lag1_pearson": _pearson(
                pooled_left[:, axis], pooled_right[:, axis]
            ),
            "direction_reversal_count": int(reversal_counts[axis]),
            "direction_reversal_rate_per_s": float(
                reversal_counts[axis] / duration_seconds
            ),
            "acceleration_abs_p95": _percentile_95_absolute(
                pooled_acceleration[:, axis]
            ),
            "acceleration_unit": f"{HEAD_AXIS_UNITS[axis]}/s^2",
            "jerk_abs_p95": _percentile_95_absolute(pooled_jerk[:, axis]),
            "jerk_unit": f"{HEAD_AXIS_UNITS[axis]}/s^3",
        }

    return {
        "sequence_count": len(sequences),
        "frame_count": int(frame_count),
        "frame_interval_count": int(interval_count),
        "lag_pair_count": int(len(pooled_left)),
        "acceleration_sample_count": int(len(pooled_acceleration)),
        "jerk_sample_count": int(len(pooled_jerk)),
        "analyzed_duration_s": float(duration_seconds),
        "per_axis": per_axis,
    }


def evaluate_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    per_axis: dict[str, dict[str, Any]] = {}
    for feature_name in HEAD_FEATURE_NAMES:
        axis_metrics = metrics["per_axis"][feature_name]
        lag = axis_metrics["delta_lag1_pearson"]
        reversal_rate = axis_metrics["direction_reversal_rate_per_s"]
        lag_pass = lag is not None and lag >= LAG1_MINIMUM
        reversal_pass = reversal_rate <= REVERSAL_RATE_MAXIMUM_PER_SECOND
        per_axis[feature_name] = {
            "lag1_pass": bool(lag_pass),
            "reversal_rate_pass": bool(reversal_pass),
            "pass": bool(lag_pass and reversal_pass),
        }
    return {
        "thresholds": {
            "delta_lag1_pearson_minimum": LAG1_MINIMUM,
            "direction_reversal_rate_maximum_per_s": (
                REVERSAL_RATE_MAXIMUM_PER_SECOND
            ),
        },
        "per_axis": per_axis,
        "all_axes_pass": all(item["pass"] for item in per_axis.values()),
    }


def _resolve_telemetry_columns(fieldnames: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    available = set(fieldnames)
    schemas = (
        (
            "applied_state_features",
            tuple(f"applied_{feature}" for feature in HEAD_FEATURE_NAMES),
        ),
        (
            "legacy_executed_state_features",
            tuple(f"executed_{feature}" for feature in HEAD_FEATURE_NAMES),
        ),
    )
    for schema_name, columns in schemas:
        if all(column in available for column in columns):
            return schema_name, columns

    expected = [list(columns) for _, columns in schemas]
    applied_columns = sorted(name for name in available if name.startswith("applied_head_local_"))
    raise ValueError(
        "telemetry lacks a complete final HeadLocal output schema; expected one of "
        f"{expected}, found applied HeadLocal columns {applied_columns}"
    )


def _contiguous_slices(
    time_seconds: np.ndarray, steps: np.ndarray | None
) -> tuple[list[slice], np.ndarray, float]:
    if len(time_seconds) < MIN_SEQUENCE_FRAMES:
        raise ValueError(f"telemetry needs at least {MIN_SEQUENCE_FRAMES} rows")
    delta_time = np.diff(time_seconds)
    positive = delta_time[delta_time > 0.0]
    if not len(positive):
        raise ValueError("telemetry has no positive time interval")
    nominal_dt = float(np.median(positive))
    tolerance = max(nominal_dt * 1.5, nominal_dt + 1e-9)
    contiguous = (delta_time > 0.0) & (delta_time <= tolerance)
    if steps is not None:
        contiguous &= np.isclose(np.diff(steps), 1.0, rtol=0.0, atol=1e-6)

    boundary_after = np.flatnonzero(~contiguous)
    starts = np.concatenate(([0], boundary_after + 1))
    stops = np.concatenate((boundary_after + 1, [len(time_seconds)]))
    slices = [
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop - start >= MIN_SEQUENCE_FRAMES
    ]
    return slices, contiguous, nominal_dt


def load_telemetry(path: Path) -> tuple[list[MotionSequence], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header")
        schema_name, head_columns = _resolve_telemetry_columns(reader.fieldnames)
        if "time_s" not in reader.fieldnames:
            raise ValueError(f"{path}: missing time_s column")
        has_step = "step" in reader.fieldnames

        times: list[float] = []
        steps: list[float] = []
        rows: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(_float(row["time_s"]))
                if has_step:
                    steps.append(_float(row["step"]))
                rows.append([_float(row[column]) for column in head_columns])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number}: invalid telemetry row: {exc}") from exc

    time_array = np.asarray(times, dtype=np.float64)
    state_array = np.asarray(rows, dtype=np.float64)
    step_array = np.asarray(steps, dtype=np.float64) if has_step else None
    slices, contiguous_mask, nominal_dt = _contiguous_slices(time_array, step_array)
    sequences = [
        MotionSequence(
            label=f"{path.name}:segment{index}",
            time_seconds=time_array[segment],
            head_state=state_array[segment],
        )
        for index, segment in enumerate(slices)
    ]
    if not sequences:
        raise ValueError(f"{path}: no contiguous segment has enough rows")

    used_rows = sum(len(sequence.time_seconds) for sequence in sequences)
    contiguous_dt = np.diff(time_array)[contiguous_mask]
    source = {
        "path": str(path.resolve()),
        "column_schema": schema_name,
        "head_columns": list(head_columns),
        "csv_rows": len(time_array),
        "analyzed_rows": int(used_rows),
        "contiguous_segments": len(sequences),
        "discontinuities": int(np.count_nonzero(~contiguous_mask)),
        "nominal_dt_s": nominal_dt,
        "contiguous_dt_min_s": float(np.min(contiguous_dt)) if len(contiguous_dt) else None,
        "contiguous_dt_max_s": float(np.max(contiguous_dt)) if len(contiguous_dt) else None,
    }
    return sequences, source


def _decode_feature_names(values: np.ndarray) -> tuple[str, ...]:
    decoded: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _head_feature_indices(
    archive: Any, feature_count: int
) -> tuple[list[int], str]:
    for key in ("state_feature_names", "feature_names"):
        if key in archive.files:
            names = _decode_feature_names(archive[key])
            missing = [name for name in HEAD_FEATURE_NAMES if name not in names]
            if missing:
                raise ValueError(f"NPZ {key} is missing HeadLocal features: {missing}")
            return [names.index(name) for name in HEAD_FEATURE_NAMES], key
    if feature_count != len(FULL_STATE_FEATURE_NAMES):
        raise ValueError(
            "NPZ has no feature-name array and its final dimension is not the "
            f"expected 9-state contract: {feature_count}"
        )
    return list(range(3, 9)), "implicit_normal_hmd_9d_contract"


def _filled_slices(mask: np.ndarray) -> list[slice]:
    changes = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop - start >= MIN_SEQUENCE_FRAMES
    ]


def load_held_out_npz(
    path: Path, fallback_dt_seconds: float
) -> tuple[list[MotionSequence], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not math.isfinite(fallback_dt_seconds) or fallback_dt_seconds <= 0.0:
        raise ValueError("--npz-dt-seconds must be finite and positive")

    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "Y", "trajectory_index", "start_index", "trajectory_ids"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing NPZ arrays: {missing}")

        x = np.asarray(archive["X"])
        y = np.asarray(archive["Y"])
        trajectory_index = np.asarray(archive["trajectory_index"], dtype=np.int64)
        start_index = np.asarray(archive["start_index"], dtype=np.int64)
        trajectory_ids = _decode_feature_names(archive["trajectory_ids"])
        if "dt_seconds" in archive.files:
            dt_values = np.asarray(archive["dt_seconds"], dtype=np.float64).reshape(-1)
            if not len(dt_values) or not np.all(np.isfinite(dt_values)):
                raise ValueError(f"{path}: invalid dt_seconds array")
            dt_seconds = float(np.median(dt_values))
            dt_source = "npz:dt_seconds"
        else:
            dt_seconds = float(fallback_dt_seconds)
            dt_source = "--npz-dt-seconds/default"
        head_indices, feature_schema = _head_feature_indices(archive, x.shape[-1])

    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"{path}: X and Y must both have shape (windows, frames, features)")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"{path}: incompatible X/Y shapes {x.shape} and {y.shape}")
    if len(trajectory_index) != len(x) or len(start_index) != len(x):
        raise ValueError(f"{path}: window metadata length does not match X/Y")
    if np.any(trajectory_index < 0) or np.any(trajectory_index >= len(trajectory_ids)):
        raise ValueError(f"{path}: trajectory_index is out of range")
    if np.any(start_index < 0):
        raise ValueError(f"{path}: start_index contains a negative frame")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{path}: X/Y contain non-finite state values")
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(f"{path}: fixed timestep must be finite and positive")

    window_length = x.shape[1] + y.shape[1]
    sequences: list[MotionSequence] = []
    unique_frame_count = 0
    overlap_observation_count = 0
    uncovered_gap_count = 0

    for trajectory in np.unique(trajectory_index):
        window_rows = np.flatnonzero(trajectory_index == trajectory)
        starts = start_index[window_rows]
        first_frame = int(np.min(starts))
        stop_frame = int(np.max(starts) + window_length)
        reconstructed = np.full(
            (stop_frame - first_frame, len(HEAD_FEATURE_NAMES)), np.nan, dtype=np.float64
        )
        filled = np.zeros(stop_frame - first_frame, dtype=bool)

        for row_index in window_rows[np.argsort(starts, kind="stable")]:
            window = np.concatenate((x[row_index], y[row_index]), axis=0)[
                :, head_indices
            ].astype(np.float64, copy=False)
            offset = int(start_index[row_index] - first_frame)
            destination = slice(offset, offset + window_length)
            overlap = filled[destination]
            if np.any(overlap):
                existing = reconstructed[destination][overlap]
                incoming = window[overlap]
                if not np.allclose(existing, incoming, rtol=0.0, atol=1e-7):
                    maximum_error = float(np.max(np.abs(existing - incoming)))
                    raise ValueError(
                        f"{path}: inconsistent overlapping windows for trajectory "
                        f"{trajectory_ids[int(trajectory)]!r}; max error {maximum_error}"
                    )
                overlap_observation_count += int(np.count_nonzero(overlap))
            reconstructed[destination] = window
            filled[destination] = True

        filled_runs = _filled_slices(filled)
        uncovered_gap_count += max(0, len(filled_runs) - 1)
        unique_frame_count += int(np.count_nonzero(filled))
        for run_index, run in enumerate(filled_runs):
            absolute_start = first_frame + int(run.start or 0)
            frame_numbers = absolute_start + np.arange(run.stop - run.start)
            sequences.append(
                MotionSequence(
                    label=f"{trajectory_ids[int(trajectory)]}:segment{run_index}",
                    time_seconds=frame_numbers.astype(np.float64) * dt_seconds,
                    head_state=reconstructed[run],
                )
            )

    if not sequences:
        raise ValueError(f"{path}: no reconstructed held-out sequence is long enough")

    source = {
        "path": str(path.resolve()),
        "npz_arrays": ["X", "Y", "trajectory_index", "start_index", "trajectory_ids"],
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "window_count": int(len(x)),
        "trajectory_count": int(len(np.unique(trajectory_index))),
        "reconstructed_segment_count": len(sequences),
        "reconstructed_unique_frames": int(unique_frame_count),
        "overlap_observations_verified": int(overlap_observation_count),
        "uncovered_gap_count": int(uncovered_gap_count),
        "fixed_dt_s": dt_seconds,
        "fixed_dt_source": dt_source,
        "feature_schema": feature_schema,
        "head_feature_indices": head_indices,
    }
    return sequences, source


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    telemetry_files: list[dict[str, Any]] = []
    all_telemetry_sequences: list[MotionSequence] = []
    for telemetry_path in args.telemetry:
        sequences, source = load_telemetry(telemetry_path)
        metrics = compute_metrics(sequences)
        gate = evaluate_gate(metrics)
        telemetry_files.append({"source": source, "metrics": metrics, "gate": gate})
        all_telemetry_sequences.extend(sequences)

    aggregate_metrics = compute_metrics(all_telemetry_sequences)
    aggregate_gate = evaluate_gate(aggregate_metrics)
    reference_sequences, reference_source = load_held_out_npz(
        args.test_npz, args.npz_dt_seconds
    )
    reference_metrics = compute_metrics(reference_sequences)

    every_file_pass = all(item["gate"]["all_axes_pass"] for item in telemetry_files)
    overall_pass = bool(aggregate_gate["all_axes_pass"] and every_file_pass)
    return {
        "schema_version": "normal-hmd-smoothness-validation-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if overall_pass else "fail",
        "method": {
            "state": "HeadLocal position (m) and right-handed rotation vector (rad)",
            "delta_lag1_pearson": (
                "Pearson correlation of consecutive within-sequence frame deltas"
            ),
            "direction_reversal_rate_per_s": (
                "strict nonzero delta sign changes divided by analyzed sequence seconds"
            ),
            "acceleration_abs_p95": (
                "95th percentile absolute second derivative; interval-midpoint timing"
            ),
            "jerk_abs_p95": (
                "95th percentile absolute third derivative; interval-midpoint timing"
            ),
            "boundary_policy": (
                "No delta, correlation pair, acceleration, or jerk crosses a CSV "
                "discontinuity, file boundary, trajectory boundary, or uncovered NPZ gap."
            ),
        },
        "gate_thresholds": {
            "delta_lag1_pearson_minimum": LAG1_MINIMUM,
            "direction_reversal_rate_maximum_per_s": REVERSAL_RATE_MAXIMUM_PER_SECOND,
        },
        "unity_telemetry": {
            "files": telemetry_files,
            "aggregate": {
                "metrics": aggregate_metrics,
                "gate": aggregate_gate,
            },
            "every_file_pass": bool(every_file_pass),
        },
        "eyenavgs_held_out_reference": {
            "source": reference_source,
            "metrics": reference_metrics,
        },
        "overall_gate_pass": overall_pass,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-axis HeadLocal temporal smoothness in Unity telemetry and "
            "held-out EyeNavGS Normal HMD windows."
        )
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        nargs="+",
        required=True,
        metavar="CSV",
        help="one or more Unity telemetry.csv files",
    )
    parser.add_argument(
        "--test-npz",
        type=Path,
        default=DEFAULT_TEST_NPZ,
        help=f"held-out EyeNavGS NPZ (default: {DEFAULT_TEST_NPZ})",
    )
    parser.add_argument(
        "--npz-dt-seconds",
        type=float,
        default=DEFAULT_NPZ_DT_SECONDS,
        help=(
            "fixed timestep used when the NPZ has no dt_seconds array "
            f"(default: {DEFAULT_NPZ_DT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="return exit status 1 after writing JSON when any Unity gate fails",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
        write_json(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"smoothness validation error: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {args.output.resolve()} "
        f"(overall_gate_pass={str(report['overall_gate_pass']).lower()})"
    )
    if args.fail_on_gate and not report["overall_gate_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
