#!/usr/bin/env python3
"""Compare horizontal and vertical HMD motion events in EyeNavGS and Unity.

This is an offline diagnostic.  It reconstructs non-duplicated trajectories
from the prepared Normal HMD v2 windows, learns robust event thresholds from
the participant-disjoint training split, then applies the frozen definitions
to the held-out test split and Unity telemetry.

It deliberately keeps four concepts separate:

* lateral sway: HeadLocal X translation velocity;
* vertical bob: HeadLocal Y translation velocity;
* look turn: HeadLocal yaw-component angular velocity;
* nod: HeadLocal pitch-component angular velocity.

The labels are diagnostic pseudo-labels, not ground-truth human annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_NAMES = (
    "root_right_velocity_mps",
    "root_forward_velocity_mps",
    "root_yaw_rate_radps_rh",
    "head_local_right_m",
    "head_local_up_offset_m",
    "head_local_forward_m",
    "head_local_rotvec_x_rad_rh",
    "head_local_rotvec_y_rad_rh",
    "head_local_rotvec_z_rad_rh",
)
SIGNAL_NAMES = (
    "lateral_sway_speed_mps",
    "vertical_bob_speed_mps",
    "look_turn_rate_radps",
    "nod_rate_radps",
)
DEFAULT_DT = 0.027
SMOOTH_WINDOW_FRAMES = 9
SMOOTH_POLYORDER = 2
HIGH_QUANTILE = 0.80
LOW_THRESHOLD_RATIO = 0.50
MIN_EVENT_FRAMES = 5
MERGE_GAP_FRAMES = 4
ONSET_COINCIDENCE_SECONDS = 0.25


@dataclass(frozen=True)
class Sequence:
    label: str
    state: np.ndarray
    dt: float


def decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def filled_slices(mask: np.ndarray, minimum: int = 4) -> list[slice]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops)
        if int(stop - start) >= minimum
    ]


def load_prepared_npz(path: Path, dt: float = DEFAULT_DT) -> list[Sequence]:
    """Reconstruct full-state fragments while removing overlapping windows."""

    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float64)
        y = np.asarray(archive["Y"], dtype=np.float64)
        trajectory_index = np.asarray(archive["trajectory_index"], dtype=np.int64)
        start_index = np.asarray(archive["start_index"], dtype=np.int64)
        trajectory_ids = decode_strings(archive["trajectory_ids"])

    if x.ndim != 3 or y.ndim != 3 or x.shape[2] != len(STATE_NAMES):
        raise ValueError(f"unexpected prepared shapes: X={x.shape}, Y={y.shape}")
    window_length = x.shape[1] + y.shape[1]
    sequences: list[Sequence] = []

    for trajectory in np.unique(trajectory_index):
        rows = np.flatnonzero(trajectory_index == trajectory)
        starts = start_index[rows]
        first = int(np.min(starts))
        stop = int(np.max(starts) + window_length)
        state = np.full((stop - first, len(STATE_NAMES)), np.nan, dtype=np.float64)
        filled = np.zeros(stop - first, dtype=bool)
        for row in rows[np.argsort(starts, kind="stable")]:
            window = np.concatenate((x[row], y[row]), axis=0)
            offset = int(start_index[row] - first)
            destination = slice(offset, offset + window_length)
            if np.any(filled[destination]):
                existing = state[destination][filled[destination]]
                incoming = window[filled[destination]]
                if not np.allclose(existing, incoming, rtol=0.0, atol=1e-7):
                    raise ValueError("inconsistent overlapping prepared windows")
            state[destination] = window
            filled[destination] = True

        for part_index, part in enumerate(filled_slices(filled)):
            sequences.append(
                Sequence(
                    label=f"{trajectory_ids[int(trajectory)]}:segment{part_index}",
                    state=state[part],
                    dt=float(dt),
                )
            )
    return sequences


def load_telemetry(path: Path) -> list[Sequence]:
    prefix = "applied_"
    columns = tuple(f"{prefix}{name}" for name in STATE_NAMES)
    times: list[float] = []
    steps: list[int] = []
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not all(column in reader.fieldnames for column in columns):
            raise ValueError(f"{path}: missing final applied state columns")
        for row in reader:
            times.append(float(row["time_s"]))
            steps.append(int(row["step"]))
            rows.append([float(row[column]) for column in columns])

    time = np.asarray(times, dtype=np.float64)
    step = np.asarray(steps, dtype=np.int64)
    state = np.asarray(rows, dtype=np.float64)
    dt = float(np.median(np.diff(time)))
    contiguous = (
        (np.diff(time) > 0.0)
        & (np.diff(time) <= dt * 1.5)
        & (np.diff(step) == 1)
    )
    boundaries = np.flatnonzero(~contiguous) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(state)]))
    return [
        Sequence(f"{path.parent.name}:segment{i}", state[a:b], dt)
        for i, (a, b) in enumerate(zip(starts, stops))
        if b - a >= 9
    ]


def smooth_state_and_signals(sequence: Sequence) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state = np.asarray(sequence.state, dtype=np.float64)
    window = min(SMOOTH_WINDOW_FRAMES, len(state) if len(state) % 2 else len(state) - 1)
    if window <= SMOOTH_POLYORDER:
        raise ValueError(f"{sequence.label}: sequence too short")
    smooth = savgol_filter(
        state,
        window_length=window,
        polyorder=SMOOTH_POLYORDER,
        axis=0,
        mode="interp",
    )
    derivative = savgol_filter(
        state,
        window_length=window,
        polyorder=SMOOTH_POLYORDER,
        deriv=1,
        delta=sequence.dt,
        axis=0,
        mode="interp",
    )
    signals = {
        "lateral_sway_speed_mps": np.abs(derivative[:, 3]),
        "vertical_bob_speed_mps": np.abs(derivative[:, 4]),
        "look_turn_rate_radps": np.abs(derivative[:, 7]),
        "nod_rate_radps": np.abs(derivative[:, 6]),
        "root_speed_mps": np.hypot(smooth[:, 0], smooth[:, 1]),
        "root_turn_rate_radps": np.abs(smooth[:, 2]),
    }
    return smooth, signals


def learn_thresholds(train_sequences: Iterable[Sequence]) -> dict[str, dict[str, float]]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in SIGNAL_NAMES}
    for sequence in train_sequences:
        _, signals = smooth_state_and_signals(sequence)
        for name in SIGNAL_NAMES:
            chunks[name].append(signals[name])
    thresholds: dict[str, dict[str, float]] = {}
    for name in SIGNAL_NAMES:
        pooled = np.concatenate(chunks[name])
        high = float(np.quantile(pooled, HIGH_QUANTILE))
        thresholds[name] = {"high": high, "low": high * LOW_THRESHOLD_RATIO}
    return thresholds


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def hysteresis_events(
    score: np.ndarray,
    high: float,
    low: float,
    minimum_frames: int = MIN_EVENT_FRAMES,
    merge_gap_frames: int = MERGE_GAP_FRAMES,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    candidates = [
        (int(start), int(stop))
        for start, stop in _runs(score >= low)
        if np.any(score[start:stop] >= high)
    ]
    merged: list[tuple[int, int]] = []
    for start, stop in candidates:
        if merged and start - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    events = [(start, stop) for start, stop in merged if stop - start >= minimum_frames]
    mask = np.zeros(len(score), dtype=bool)
    for start, stop in events:
        mask[start:stop] = True
    return mask, events


def fixed_motion_masks(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    locomotion, _ = hysteresis_events(
        signals["root_speed_mps"], high=0.10, low=0.06,
        minimum_frames=11, merge_gap_frames=7,
    )
    body_turn, _ = hysteresis_events(
        signals["root_turn_rate_radps"], high=math.radians(15.0), low=math.radians(8.0),
        minimum_frames=7, merge_gap_frames=4,
    )
    return {"locomotion": locomotion, "body_turn": body_turn}


def make_labels(
    sequence: Sequence, thresholds: dict[str, dict[str, float]]
) -> dict[str, Any]:
    smooth, signals = smooth_state_and_signals(sequence)
    masks: dict[str, np.ndarray] = {}
    events: dict[str, list[tuple[int, int]]] = {}
    for name in SIGNAL_NAMES:
        event_name = {
            "lateral_sway_speed_mps": "lateral_sway",
            "vertical_bob_speed_mps": "vertical_bob",
            "look_turn_rate_radps": "look_turn",
            "nod_rate_radps": "nod",
        }[name]
        masks[event_name], events[event_name] = hysteresis_events(
            signals[name], thresholds[name]["high"], thresholds[name]["low"]
        )
    masks.update(fixed_motion_masks(signals))
    masks["horizontal_any"] = masks["lateral_sway"] | masks["look_turn"]
    masks["vertical_any"] = masks["vertical_bob"] | masks["nod"]
    # Strict idle is a sustained lack of locomotion, turning, and head events.
    raw_idle = ~(
        masks["locomotion"]
        | masks["body_turn"]
        | masks["horizontal_any"]
        | masks["vertical_any"]
    )
    idle = np.zeros(len(raw_idle), dtype=bool)
    for start, stop in _runs(raw_idle):
        if stop - start >= 19:  # 0.513 s at the frozen 37 Hz timestep.
            idle[start:stop] = True
    masks["idle"] = idle
    return {"smooth": smooth, "signals": signals, "masks": masks, "events": events}


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def pair_statistics(
    labelled: list[dict[str, Any]], a_name: str, b_name: str, dt: float
) -> dict[str, Any]:
    a = np.concatenate([item["masks"][a_name] for item in labelled])
    b = np.concatenate([item["masks"][b_name] for item in labelled])
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    pab = float(np.mean(a & b))
    expected = pa * pb
    denominator = math.sqrt(pa * (1.0 - pa) * pb * (1.0 - pb))
    phi = float((pab - expected) / denominator) if denominator else None

    a_onsets = 0
    b_onsets = 0
    a_near_b = 0
    b_near_a = 0
    tolerance = max(1, int(round(ONSET_COINCIDENCE_SECONDS / dt)))
    for item in labelled:
        a_starts = np.asarray([start for start, _ in _runs(item["masks"][a_name])], dtype=int)
        b_starts = np.asarray([start for start, _ in _runs(item["masks"][b_name])], dtype=int)
        a_onsets += len(a_starts)
        b_onsets += len(b_starts)
        if len(b_starts):
            a_near_b += sum(bool(np.any(np.abs(b_starts - start) <= tolerance)) for start in a_starts)
        if len(a_starts):
            b_near_a += sum(bool(np.any(np.abs(a_starts - start) <= tolerance)) for start in b_starts)

    return {
        "a": a_name,
        "b": b_name,
        "p_a": pa,
        "p_b": pb,
        "p_both": pab,
        "expected_p_both_if_frame_independent": expected,
        "overlap_lift_vs_independence": safe_ratio(pab, expected),
        "p_b_given_a": safe_ratio(float(np.count_nonzero(a & b)), float(np.count_nonzero(a))),
        "p_b_given_not_a": safe_ratio(float(np.count_nonzero((~a) & b)), float(np.count_nonzero(~a))),
        "phi_frame_association": phi,
        "a_onset_count": int(a_onsets),
        "b_onset_count": int(b_onsets),
        "a_onset_near_b_fraction": safe_ratio(a_near_b, a_onsets),
        "b_onset_near_a_fraction": safe_ratio(b_near_a, b_onsets),
        "onset_tolerance_s": ONSET_COINCIDENCE_SECONDS,
    }


def summarize_dataset(
    sequences: list[Sequence], thresholds: dict[str, dict[str, float]]
) -> dict[str, Any]:
    labelled = [make_labels(sequence, thresholds) for sequence in sequences]
    total_frames = sum(len(sequence.state) for sequence in sequences)
    total_seconds = sum((len(sequence.state) - 1) * sequence.dt for sequence in sequences)
    labels = (
        "idle", "locomotion", "body_turn", "lateral_sway", "vertical_bob",
        "look_turn", "nod", "horizontal_any", "vertical_any",
    )
    occupancy = {
        name: float(
            sum(np.count_nonzero(item["masks"][name]) for item in labelled) / total_frames
        )
        for name in labels
    }

    event_summary: dict[str, Any] = {}
    for name in ("lateral_sway", "vertical_bob", "look_turn", "nod"):
        durations: list[float] = []
        amplitudes: list[float] = []
        intervals: list[float] = []
        axis = {"lateral_sway": 3, "vertical_bob": 4, "look_turn": 7, "nod": 6}[name]
        for sequence, item in zip(sequences, labelled):
            starts: list[int] = []
            for start, stop in item["events"][name]:
                durations.append((stop - start) * sequence.dt)
                amplitudes.append(float(np.ptp(item["smooth"][start:stop, axis])))
                starts.append(start)
            intervals.extend(np.diff(starts).astype(float) * sequence.dt)
        event_summary[name] = {
            "count": len(durations),
            "rate_per_min": safe_ratio(len(durations) * 60.0, total_seconds),
            "duration_s_median": float(np.median(durations)) if durations else None,
            "duration_s_p90": float(np.percentile(durations, 90.0)) if durations else None,
            "amplitude_median": float(np.median(amplitudes)) if amplitudes else None,
            "amplitude_p90": float(np.percentile(amplitudes, 90.0)) if amplitudes else None,
            "within_sequence_onset_interval_s_median": (
                float(np.median(intervals)) if intervals else None
            ),
        }

    vertical_during_walk = []
    vertical_stationary = []
    for item in labelled:
        vertical = item["masks"]["vertical_bob"]
        walk = item["masks"]["locomotion"]
        vertical_during_walk.append(vertical[walk])
        vertical_stationary.append(vertical[~walk])
    walk_values = np.concatenate(vertical_during_walk) if vertical_during_walk else np.array([])
    stationary_values = np.concatenate(vertical_stationary) if vertical_stationary else np.array([])

    dt = float(np.median([sequence.dt for sequence in sequences]))
    return {
        "sequence_count": len(sequences),
        "frame_count": total_frames,
        "duration_s": total_seconds,
        "occupancy": occupancy,
        "events": event_summary,
        "pair_statistics": {
            "lateral_vs_vertical_translation": pair_statistics(
                labelled, "lateral_sway", "vertical_bob", dt
            ),
            "look_turn_vs_nod_rotation": pair_statistics(labelled, "look_turn", "nod", dt),
            "horizontal_any_vs_vertical_any": pair_statistics(
                labelled, "horizontal_any", "vertical_any", dt
            ),
        },
        "vertical_bob_conditioning": {
            "p_vertical_bob_given_locomotion": (
                float(np.mean(walk_values)) if len(walk_values) else None
            ),
            "p_vertical_bob_given_not_locomotion": (
                float(np.mean(stationary_values)) if len(stationary_values) else None
            ),
        },
    }


def per_sequence_reference_bands(
    sequences: list[Sequence], thresholds: dict[str, dict[str, float]],
    minimum_seconds: float = 20.0,
) -> dict[str, Any]:
    """Empirical P10-P90 bands for long held-out human fragments.

    These bands are deliberately labelled as proposed diagnostic gates.  They
    were not pre-registered and should not be presented as clinical cutoffs.
    """

    values: dict[str, list[float]] = {
        "horizontal_any_occupancy": [],
        "vertical_any_occupancy": [],
        "vertical_to_horizontal_occupancy_ratio": [],
        "horizontal_vertical_overlap_lift": [],
        "vertical_bob_rate_per_min": [],
        "vertical_bob_event_amplitude_median_m": [],
    }
    used = 0
    for sequence in sequences:
        duration = (len(sequence.state) - 1) * sequence.dt
        if duration < minimum_seconds:
            continue
        item = make_labels(sequence, thresholds)
        horizontal = item["masks"]["horizontal_any"]
        vertical = item["masks"]["vertical_any"]
        p_horizontal = float(np.mean(horizontal))
        p_vertical = float(np.mean(vertical))
        p_both = float(np.mean(horizontal & vertical))
        independent = p_horizontal * p_vertical
        events = item["events"]["vertical_bob"]
        amplitudes = [
            float(np.ptp(item["smooth"][start:stop, 4])) for start, stop in events
        ]
        if p_horizontal <= 0.0 or independent <= 0.0 or not amplitudes:
            continue
        used += 1
        values["horizontal_any_occupancy"].append(p_horizontal)
        values["vertical_any_occupancy"].append(p_vertical)
        values["vertical_to_horizontal_occupancy_ratio"].append(
            p_vertical / p_horizontal
        )
        values["horizontal_vertical_overlap_lift"].append(p_both / independent)
        values["vertical_bob_rate_per_min"].append(len(events) * 60.0 / duration)
        values["vertical_bob_event_amplitude_median_m"].append(
            float(np.median(amplitudes))
        )

    bands: dict[str, Any] = {}
    for name, items in values.items():
        array = np.asarray(items, dtype=np.float64)
        bands[name] = {
            "p10": float(np.percentile(array, 10.0)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90.0)),
        }
    return {
        "status": "proposed_diagnostic_only_not_clinical_or_preregistered",
        "minimum_fragment_seconds": minimum_seconds,
        "fragment_count": used,
        "acceptance_rule": "aggregate demo metric must lie within held-out-human P10-P90",
        "bands": bands,
    }


def evaluate_proposed_gates(
    demo: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    pair = demo["pair_statistics"]["horizontal_any_vs_vertical_any"]
    horizontal = pair["p_a"]
    vertical = pair["p_b"]
    current = {
        "horizontal_any_occupancy": horizontal,
        "vertical_any_occupancy": vertical,
        "vertical_to_horizontal_occupancy_ratio": (
            vertical / horizontal if horizontal else None
        ),
        "horizontal_vertical_overlap_lift": pair["overlap_lift_vs_independence"],
        "vertical_bob_rate_per_min": demo["events"]["vertical_bob"]["rate_per_min"],
        "vertical_bob_event_amplitude_median_m": demo["events"]["vertical_bob"][
            "amplitude_median"
        ],
    }
    checks: dict[str, Any] = {}
    for name, value in current.items():
        band = reference["bands"][name]
        passed = value is not None and band["p10"] <= value <= band["p90"]
        checks[name] = {"value": value, **band, "pass": bool(passed)}
    return {
        "checks": checks,
        "all_pass": all(item["pass"] for item in checks.values()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    human = report["held_out_human"]
    demo = report["unity_demo"]
    hpair = human["pair_statistics"]["horizontal_any_vs_vertical_any"]
    dpair = demo["pair_statistics"]["horizontal_any_vs_vertical_any"]
    hv = human["events"]["vertical_bob"]
    dv = demo["events"]["vertical_bob"]
    gates = report["proposed_human_likeness_gates"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    def num(value: float | None, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Normal HMD horizontal/vertical event analysis",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Frozen pseudo-label definitions",
        "",
        "Head signals are Savitzky-Golay smoothed (9 frames, 243 ms, polynomial order 2). "
        "The high threshold is the 80th percentile of the participant-disjoint EyeNavGS "
        "training split; the sustain threshold is 50% of high. Events require 5 frames "
        "(135 ms) and gaps up to 4 frames (108 ms) are merged. These are diagnostic "
        "pseudo-labels, not manually annotated actions.",
        "",
        "- `look_turn`: absolute HeadLocal yaw-component rate.",
        "- `lateral_sway`: absolute HeadLocal X speed.",
        "- `nod`: absolute HeadLocal pitch-component rate.",
        "- `vertical_bob`: absolute HeadLocal Y speed.",
        "- `locomotion`: BodyRoot planar speed, 0.10 m/s onset and 0.06 m/s sustain.",
        "- `body_turn`: BodyRoot yaw rate, 15 deg/s onset and 8 deg/s sustain.",
        "- `idle`: at least 513 ms with none of locomotion, body turn, horizontal, or vertical head events.",
        "",
        "## Key comparison",
        "",
        "| metric | held-out EyeNavGS human | current Unity demo |",
        "|---|---:|---:|",
        f"| horizontal-active frames | {pct(hpair['p_a'])} | {pct(dpair['p_a'])} |",
        f"| vertical-active frames | {pct(hpair['p_b'])} | {pct(dpair['p_b'])} |",
        f"| simultaneous horizontal + vertical | {pct(hpair['p_both'])} | {pct(dpair['p_both'])} |",
        f"| overlap lift vs independent timing | {num(hpair['overlap_lift_vs_independence'])} | {num(dpair['overlap_lift_vs_independence'])} |",
        f"| vertical event rate/min | {num(hv['rate_per_min'])} | {num(dv['rate_per_min'])} |",
        f"| vertical duration median | {num(hv['duration_s_median'], 3)} s | {num(dv['duration_s_median'], 3)} s |",
        f"| vertical amplitude median | {num(100.0 * hv['amplitude_median'], 2)} cm | {num(100.0 * dv['amplitude_median'], 2)} cm |",
        f"| vertical amplitude p90 | {num(100.0 * hv['amplitude_p90'], 2)} cm | {num(100.0 * dv['amplitude_p90'], 2)} cm |",
        "",
        "## Interpretation guardrails",
        "",
        "The four labels are deliberately separate. Human horizontal and vertical motion "
        "may overlap; an overlap lift below 1 means partial temporal separation, not strict "
        "mutual exclusion. Vertical translation and pitch nodding must also not be conflated.",
        "",
        "## Proposed held-out-human gates",
        "",
        "These are diagnostic P10-P90 bands across held-out fragments lasting at least 20 s, "
        "not clinical cutoffs. A human-like demo should pass all of them in aggregate.",
        "",
        "| metric | demo | human P10 | human P90 | pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, item in gates["checks"].items():
        lines.append(
            f"| `{name}` | {num(item['value'], 4)} | {num(item['p10'], 4)} | "
            f"{num(item['p90'], 4)} | {'yes' if item['pass'] else 'no'} |"
        )
    lines.extend([
        "",
        "Thresholds and all remaining statistics are stored in the sibling JSON report.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-npz",
        type=Path,
        default=REPO_ROOT / "data/processed/normal_hmd_v2/normal_hmd_train.npz",
    )
    parser.add_argument(
        "--test-npz",
        type=Path,
        default=REPO_ROOT / "data/processed/normal_hmd_v2/normal_hmd_test.npz",
    )
    parser.add_argument("--telemetry", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results/normal_hmd_event_axis_analysis.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "results/normal_hmd_event_axis_analysis.md",
    )
    args = parser.parse_args()

    train = load_prepared_npz(args.train_npz)
    test = load_prepared_npz(args.test_npz)
    demo: list[Sequence] = []
    for path in args.telemetry:
        demo.extend(load_telemetry(path))
    thresholds = learn_thresholds(train)

    human_summary = summarize_dataset(test, thresholds)
    demo_summary = summarize_dataset(demo, thresholds)
    reference_bands = per_sequence_reference_bands(test, thresholds)
    report = {
        "schema_version": "normal-hmd-event-axis-analysis-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "method": {
            "dt_seconds": DEFAULT_DT,
            "smoothing": {
                "type": "Savitzky-Golay",
                "window_frames": SMOOTH_WINDOW_FRAMES,
                "window_seconds": SMOOTH_WINDOW_FRAMES * DEFAULT_DT,
                "polynomial_order": SMOOTH_POLYORDER,
            },
            "threshold_calibration_split": str(args.train_npz.resolve()),
            "high_quantile": HIGH_QUANTILE,
            "low_threshold_ratio": LOW_THRESHOLD_RATIO,
            "minimum_event_frames": MIN_EVENT_FRAMES,
            "minimum_event_seconds": MIN_EVENT_FRAMES * DEFAULT_DT,
            "merge_gap_frames": MERGE_GAP_FRAMES,
            "merge_gap_seconds": MERGE_GAP_FRAMES * DEFAULT_DT,
            "thresholds": thresholds,
            "warning": (
                "Offline diagnostic pseudo-labels inferred from HMD pose; EyeNavGS has no "
                "manual nod/bob/walk event annotation and no pelvis/feet ground truth."
            ),
        },
        "sources": {
            "train_npz": str(args.train_npz.resolve()),
            "test_npz": str(args.test_npz.resolve()),
            "telemetry": [str(path.resolve()) for path in args.telemetry],
        },
        "held_out_human": human_summary,
        "unity_demo": demo_summary,
        "held_out_human_reference_bands": reference_bands,
        "proposed_human_likeness_gates": evaluate_proposed_gates(
            demo_summary, reference_bands
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "json": str(args.output_json.resolve()),
        "markdown": str(args.output_md.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
