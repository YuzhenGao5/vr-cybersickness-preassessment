#!/usr/bin/env python3
"""Trajectory-first event labels for the Normal HMD Agent v4.

The frozen v2 files contain overlapping training windows.  Event detection on
those windows independently would give one physical onset several different
durations and amplitudes.  This module therefore reconstructs every unique
continuous trajectory fragment, applies the existing Savitzky--Golay plus
hysteresis definition once, and only then maps labels back to target windows.

Horizontal events are the union of HeadLocal-X and yaw detections.  Vertical
events are the union of HeadLocal-Y and pitch detections.  The two masks are
deliberately independent and may overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from normal_hmd_event_axis_analysis import (
    DEFAULT_DT,
    SIGNAL_NAMES,
    Sequence,
    _runs,
    hysteresis_events,
    learn_thresholds,
    smooth_state_and_signals,
)


ROOT_INDICES = (0, 1, 2)
HORIZONTAL_INDICES = (3, 5, 7, 8)  # x, z, yaw, roll
VERTICAL_INDICES = (4, 6)  # y, pitch
HORIZONTAL_SIGNAL_NAMES = ("lateral_sway_speed_mps", "look_turn_rate_radps")
VERTICAL_SIGNAL_NAMES = ("vertical_bob_speed_mps", "nod_rate_radps")


@dataclass(frozen=True)
class TrajectoryFragment:
    """One gap-free portion of one frozen trajectory, in absolute indices."""

    trajectory_index: int
    trajectory_id: str
    start_index: int
    state: np.ndarray

    @property
    def stop_index(self) -> int:
        return self.start_index + len(self.state)


@dataclass(frozen=True)
class BranchTrajectoryLabels:
    """Event labels aligned to every state sample in a trajectory fragment."""

    active: np.ndarray
    onset: np.ndarray
    eligible: np.ndarray
    parameter_mask: np.ndarray
    parameters: np.ndarray
    component: np.ndarray
    event_id: np.ndarray


@dataclass(frozen=True)
class LabelledFragment:
    fragment: TrajectoryFragment
    smooth_state: np.ndarray
    horizontal: BranchTrajectoryLabels
    vertical: BranchTrajectoryLabels


@dataclass(frozen=True)
class BranchWindowTargets:
    active: np.ndarray
    onset: np.ndarray
    eligible: np.ndarray
    parameter_mask: np.ndarray
    parameters: np.ndarray
    event_id: np.ndarray


@dataclass(frozen=True)
class V4WindowTargets:
    """Labels aligned to ``split['Y']``.

    Parameter layout is:

    * horizontal: absolute x/z/yaw/roll end pose, log duration;
    * vertical: absolute y/pitch end pose, log duration.

    Runtime derives and latches the signed amplitude as ``end - current``.
    Modelling a bounded absolute endpoint prevents an unconditional delta
    process from accumulating into a clipped random walk.
    """

    horizontal: BranchWindowTargets
    vertical: BranchWindowTargets
    horizontal_pose: np.ndarray
    trajectory_index: np.ndarray
    absolute_index: np.ndarray


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def load_split(path_or_directory: Path, split: str | None = None) -> dict[str, np.ndarray]:
    path = (
        Path(path_or_directory) / f"normal_hmd_{split}.npz"
        if split is not None
        else Path(path_or_directory)
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def reconstruct_trajectories(
    split: Mapping[str, np.ndarray], *, overlap_atol: float = 1e-7
) -> list[TrajectoryFragment]:
    """Reconstruct unique, gap-free fragments from overlapping X/Y windows."""

    x = np.asarray(split["X"], dtype=np.float64)
    y = np.asarray(split["Y"], dtype=np.float64)
    trajectory_index = np.asarray(split["trajectory_index"], dtype=np.int64)
    start_index = np.asarray(split["start_index"], dtype=np.int64)
    trajectory_ids = _decode_strings(np.asarray(split["trajectory_ids"]))
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != 9:
        raise ValueError(f"unexpected frozen window shapes X={x.shape}, Y={y.shape}")
    if len(trajectory_index) != len(x) or len(start_index) != len(x):
        raise ValueError("window metadata length does not match X/Y")
    if np.any(trajectory_index < 0) or np.any(trajectory_index >= len(trajectory_ids)):
        raise ValueError("trajectory_index is outside trajectory_ids")

    window_length = x.shape[1] + y.shape[1]
    fragments: list[TrajectoryFragment] = []
    for trajectory in np.unique(trajectory_index):
        rows = np.flatnonzero(trajectory_index == trajectory)
        order = rows[np.argsort(start_index[rows], kind="stable")]
        first = int(start_index[order[0]])
        final = int(np.max(start_index[order]) + window_length)
        state = np.full((final - first, 9), np.nan, dtype=np.float64)
        filled = np.zeros(final - first, dtype=bool)
        for row in order:
            window = np.concatenate((x[row], y[row]), axis=0).astype(np.float64)
            offset = int(start_index[row]) - first
            destination = slice(offset, offset + window_length)
            overlap = filled[destination]
            if np.any(overlap):
                existing = state[destination][overlap]
                incoming = window[overlap]
                if not np.allclose(existing, incoming, rtol=0.0, atol=overlap_atol):
                    maximum = float(np.max(np.abs(existing - incoming)))
                    raise ValueError(
                        f"trajectory {trajectory} has inconsistent overlapping windows "
                        f"(max abs difference {maximum:.3g})"
                    )
            state[destination] = window
            filled[destination] = True

        edges = np.diff(np.concatenate(([0], filled.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for local_start, local_stop in zip(starts, stops):
            if local_stop - local_start < 5:
                continue
            fragments.append(
                TrajectoryFragment(
                    trajectory_index=int(trajectory),
                    trajectory_id=trajectory_ids[int(trajectory)],
                    start_index=first + int(local_start),
                    state=state[int(local_start) : int(local_stop)].copy(),
                )
            )
    if not fragments:
        raise ValueError("no continuous trajectory fragments could be reconstructed")
    return fragments


def learn_event_thresholds(
    fragments: Iterable[TrajectoryFragment], dt_seconds: float = DEFAULT_DT
) -> dict[str, dict[str, float]]:
    sequences = [
        Sequence(
            f"{fragment.trajectory_id}@{fragment.start_index}",
            fragment.state,
            dt_seconds,
        )
        for fragment in fragments
    ]
    thresholds = learn_thresholds(sequences)
    missing = set(SIGNAL_NAMES) - set(thresholds)
    if missing:
        raise ValueError(f"threshold learner omitted signals: {sorted(missing)}")
    return thresholds


def minimum_jerk(progress: np.ndarray | float) -> np.ndarray:
    """Quintic minimum-jerk step with zero first/second endpoint derivative."""

    value = np.asarray(progress, dtype=np.float64)
    value = np.clip(value, 0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def closed_event_primitive(
    phase: np.ndarray | float, peak_phase: np.ndarray | float
) -> np.ndarray:
    """Minimum-jerk excursion from zero to one and exactly back to zero."""

    u, peak = np.broadcast_arrays(
        np.asarray(phase, dtype=np.float64),
        np.clip(np.asarray(peak_phase, dtype=np.float64), 0.2, 0.8),
    )
    rising = minimum_jerk(np.divide(u, peak, out=np.zeros_like(u), where=peak > 0.0))
    falling_fraction = np.divide(
        u - peak,
        1.0 - peak,
        out=np.zeros_like(u),
        where=peak < 1.0,
    )
    falling = 1.0 - minimum_jerk(falling_fraction)
    value = np.where(u <= peak, rising, falling)
    return np.where((u >= 0.0) & (u <= 1.0), value, 0.0)


def _union_events(
    signals: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mask = np.zeros(len(next(iter(signals.values()))), dtype=bool)
    for name in names:
        definition = thresholds[name]
        axis_mask, _ = hysteresis_events(
            np.asarray(signals[name]),
            high=float(definition["high"]),
            low=float(definition["low"]),
        )
        mask |= axis_mask
    return mask, [(int(start), int(stop)) for start, stop in _runs(mask)]


def _event_parameters(
    smooth: np.ndarray,
    start: int,
    stop: int,
    indices: tuple[int, ...],
) -> np.ndarray:
    target = smooth[stop - 1, indices]
    return target


def _label_branch(
    smooth: np.ndarray,
    events: list[tuple[int, int]],
    indices: tuple[int, ...],
    *,
    cooldown_steps: int,
    event_id_offset: int,
) -> tuple[BranchTrajectoryLabels, int]:
    length = len(smooth)
    parameter_dim = len(indices) + 1
    active = np.zeros(length, dtype=bool)
    onset = np.zeros(length, dtype=bool)
    eligible = np.ones(length, dtype=bool)
    parameter_mask = np.zeros(length, dtype=bool)
    parameters = np.zeros((length, parameter_dim), dtype=np.float64)
    component = np.zeros((length, len(indices)), dtype=np.float64)
    event_id = np.full(length, -1, dtype=np.int64)
    next_event_id = event_id_offset

    for start, stop in events:
        if stop <= start:
            continue
        active[start:stop] = True
        eligible[start:stop] = False
        known_onset = start > 0
        complete = known_onset and stop < length
        current_id = next_event_id
        next_event_id += 1
        event_id[start:stop] = current_id
        if known_onset:
            onset[start] = True
            eligible[start] = True
        eligible[stop : min(length, stop + cooldown_steps)] = False

        target_pose = _event_parameters(smooth, start, stop, indices)
        duration = max(stop - start, 2)
        values = np.concatenate((target_pose, [math.log(float(duration))]))
        parameters[start] = values
        if complete:
            parameter_mask[start] = True
            phase = np.linspace(0.0, 1.0, duration, dtype=np.float64)
            baseline = smooth[start - 1, indices]
            amplitudes = target_pose - baseline
            component[start:stop] = (
                minimum_jerk(phase)[:, None]
                * amplitudes[None, :]
            )

    return (
        BranchTrajectoryLabels(
            active=active,
            onset=onset,
            eligible=eligible,
            parameter_mask=parameter_mask,
            parameters=parameters,
            component=component,
            event_id=event_id,
        ),
        next_event_id,
    )


def label_trajectories(
    fragments: Iterable[TrajectoryFragment],
    thresholds: Mapping[str, Mapping[str, float]],
    *,
    dt_seconds: float = DEFAULT_DT,
    cooldown_steps: int = 4,
) -> list[LabelledFragment]:
    """Apply the fixed event definition once to each complete fragment."""

    if cooldown_steps < 0:
        raise ValueError("cooldown_steps cannot be negative")
    result: list[LabelledFragment] = []
    horizontal_id = 0
    vertical_id = 1_000_000_000
    for fragment in fragments:
        sequence = Sequence(
            f"{fragment.trajectory_id}@{fragment.start_index}",
            fragment.state,
            dt_seconds,
        )
        smooth, signals = smooth_state_and_signals(sequence)
        _, horizontal_events = _union_events(
            signals, HORIZONTAL_SIGNAL_NAMES, thresholds
        )
        _, vertical_events = _union_events(signals, VERTICAL_SIGNAL_NAMES, thresholds)
        horizontal, horizontal_id = _label_branch(
            smooth,
            horizontal_events,
            HORIZONTAL_INDICES,
            cooldown_steps=cooldown_steps,
            event_id_offset=horizontal_id,
        )
        vertical, vertical_id = _label_branch(
            smooth,
            vertical_events,
            VERTICAL_INDICES,
            cooldown_steps=cooldown_steps,
            event_id_offset=vertical_id,
        )
        result.append(LabelledFragment(fragment, smooth, horizontal, vertical))
    return result


def _empty_window_branch(count: int, steps: int, parameter_dim: int) -> BranchWindowTargets:
    return BranchWindowTargets(
        active=np.zeros((count, steps), dtype=np.float32),
        onset=np.zeros((count, steps), dtype=np.float32),
        eligible=np.zeros((count, steps), dtype=np.float32),
        parameter_mask=np.zeros((count, steps), dtype=np.float32),
        parameters=np.zeros((count, steps, parameter_dim), dtype=np.float32),
        event_id=np.full((count, steps), -1, dtype=np.int64),
    )


def _copy_branch_slice(
    destination: BranchWindowTargets,
    row: int,
    source: BranchTrajectoryLabels,
    source_slice: slice,
) -> None:
    destination.active[row] = source.active[source_slice]
    destination.onset[row] = source.onset[source_slice]
    destination.eligible[row] = source.eligible[source_slice]
    destination.parameter_mask[row] = source.parameter_mask[source_slice]
    destination.parameters[row] = source.parameters[source_slice]
    destination.event_id[row] = source.event_id[source_slice]


def map_labels_to_windows(
    split: Mapping[str, np.ndarray], labelled: Iterable[LabelledFragment]
) -> V4WindowTargets:
    """Map global labels to Y using trajectory id plus absolute target index."""

    count, history_steps, _ = np.asarray(split["X"]).shape
    target_steps = np.asarray(split["Y"]).shape[1]
    trajectories = np.asarray(split["trajectory_index"], dtype=np.int64)
    starts = np.asarray(split["start_index"], dtype=np.int64)
    by_trajectory: dict[int, list[LabelledFragment]] = {}
    for item in labelled:
        by_trajectory.setdefault(item.fragment.trajectory_index, []).append(item)
    for fragments in by_trajectory.values():
        fragments.sort(key=lambda item: item.fragment.start_index)

    horizontal = _empty_window_branch(count, target_steps, len(HORIZONTAL_INDICES) + 1)
    vertical = _empty_window_branch(count, target_steps, len(VERTICAL_INDICES) + 1)
    horizontal_pose = np.asarray(split["Y"])[
        ..., HORIZONTAL_INDICES
    ].astype(np.float32, copy=True)
    absolute_index = starts[:, None] + history_steps + np.arange(target_steps)[None, :]

    for row in range(count):
        target_start = int(absolute_index[row, 0])
        target_stop = int(absolute_index[row, -1]) + 1
        match = next(
            (
                item
                for item in by_trajectory.get(int(trajectories[row]), [])
                if item.fragment.start_index <= target_start
                and target_stop <= item.fragment.stop_index
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"target window row {row} is not contained in a reconstructed fragment"
            )
        local_start = target_start - match.fragment.start_index
        source_slice = slice(local_start, local_start + target_steps)
        _copy_branch_slice(horizontal, row, match.horizontal, source_slice)
        _copy_branch_slice(vertical, row, match.vertical, source_slice)

    return V4WindowTargets(
        horizontal=horizontal,
        vertical=vertical,
        horizontal_pose=horizontal_pose,
        trajectory_index=trajectories.astype(np.int64, copy=True),
        absolute_index=absolute_index.astype(np.int64),
    )


def assert_onset_parameter_consistency(targets: V4WindowTargets) -> dict[str, int]:
    """Prove duplicate views of an onset have identical event-level labels."""

    summary: dict[str, int] = {}
    for name, branch in (("horizontal", targets.horizontal), ("vertical", targets.vertical)):
        locations = np.argwhere(branch.onset.astype(bool))
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row, step in locations:
            key = (
                int(targets.trajectory_index[row]),
                int(targets.absolute_index[row, step]),
            )
            groups.setdefault(key, []).append((int(row), int(step)))
        duplicate_groups = 0
        for key, entries in groups.items():
            reference_row, reference_step = entries[0]
            reference_parameters = branch.parameters[reference_row, reference_step]
            reference_mask = branch.parameter_mask[reference_row, reference_step]
            reference_id = branch.event_id[reference_row, reference_step]
            for row, step in entries[1:]:
                if not np.array_equal(branch.parameters[row, step], reference_parameters):
                    raise AssertionError(f"{name} onset {key} has inconsistent parameters")
                if branch.parameter_mask[row, step] != reference_mask:
                    raise AssertionError(f"{name} onset {key} has inconsistent censor mask")
                if branch.event_id[row, step] != reference_id:
                    raise AssertionError(f"{name} onset {key} has inconsistent event id")
            duplicate_groups += int(len(entries) > 1)
        summary[f"{name}_physical_onsets"] = len(groups)
        summary[f"{name}_duplicate_onset_groups"] = duplicate_groups
    return summary


def event_summary(labelled: Iterable[LabelledFragment], dt_seconds: float) -> dict[str, dict[str, float]]:
    items = list(labelled)
    total_seconds = sum(len(item.fragment.state) for item in items) * dt_seconds
    report: dict[str, dict[str, float]] = {}
    for name in ("horizontal", "vertical"):
        branches = [getattr(item, name) for item in items]
        onset_count = sum(int(branch.onset.sum()) for branch in branches)
        active_count = sum(int(branch.active.sum()) for branch in branches)
        eligible_count = sum(int(branch.eligible.sum()) for branch in branches)
        report[name] = {
            "event_count": onset_count,
            "events_per_minute": onset_count * 60.0 / max(total_seconds, 1e-12),
            "active_fraction": active_count
            / max(sum(len(branch.active) for branch in branches), 1),
            "eligible_onset_probability": onset_count / max(eligible_count, 1),
            "complete_parameter_events": sum(
                int(branch.parameter_mask.sum()) for branch in branches
            ),
        }
    overlap = sum(
        int(np.sum(item.horizontal.active & item.vertical.active)) for item in items
    )
    total = sum(len(item.fragment.state) for item in items)
    report["overlap"] = {"active_fraction": overlap / max(total, 1)}
    return report
