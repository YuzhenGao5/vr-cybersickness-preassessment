#!/usr/bin/env python3
"""Train the event-conditioned Normal HMD Agent v3 on the frozen v2 NPZ data.

v3 deliberately reuses the participant-disjoint EyeNavGS preparation so it can
be compared with v2 without changing the data split.  It replaces the single
9-D decoder and per-axis HeadLocal noise with a shared recurrent encoder,
an 8-D main-motion decoder, and a learned vertical event decoder.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from normal_hmd_common import STATE_FEATURE_NAMES, atomic_json, sha256_file
from normal_hmd_train import fit_normalization, sampled_quantiles
from normal_hmd_v3_model import (
    MAIN_STATE_INDICES,
    PITCH_INDEX,
    VERTICAL_INDEX,
    NormalHMDV3,
    V3Rollout,
    parameter_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_DATA_REPORT = REPO_ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "normal_hmd_v3.pth"
DEFAULT_METADATA = REPO_ROOT / "models" / "normal_hmd_v3_metadata.json"
DEFAULT_METRICS = REPO_ROOT / "results" / "normal_hmd_v3_training_metrics.json"
DEFAULT_HISTORY = REPO_ROOT / "results" / "normal_hmd_v3_training_history.csv"


@dataclass(frozen=True)
class VerticalTargets:
    velocity_mps: np.ndarray
    rms_velocity_mps: np.ndarray
    amplitude_m: np.ndarray
    pitch_velocity_radps: np.ndarray
    pitch_amplitude_rad: np.ndarray
    log_duration_steps: np.ndarray
    event_active: np.ndarray
    event_onset: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"normal_hmd_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def subset_split(split: dict[str, np.ndarray], maximum: int, seed: int) -> dict[str, np.ndarray]:
    count = len(split["X"])
    if maximum <= 0 or maximum >= count:
        return split
    indices = np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))
    result: dict[str, np.ndarray] = {}
    for name, values in split.items():
        if len(values) == count:
            result[name] = values[indices]
        else:
            result[name] = values
    return result


def raw_axis_velocity_rms(
    split: dict[str, np.ndarray],
    state_index: int,
    dt_seconds: float,
    rms_window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocity, rms = full_axis_velocity_rms(
        split, state_index, dt_seconds, rms_window_steps
    )
    history_steps = split["X"].shape[1]
    target_steps = split["Y"].shape[1]
    first = history_steps - 1
    return velocity[:, first : first + target_steps], rms[:, first : first + target_steps]


def full_axis_velocity_rms(
    split: dict[str, np.ndarray],
    state_index: int,
    dt_seconds: float,
    rms_window_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = split["X"][..., state_index].astype(np.float64)
    target = split["Y"][..., state_index].astype(np.float64)
    combined = np.concatenate((history, target), axis=1)
    velocity = np.diff(combined, axis=1) / dt_seconds
    square = np.square(velocity)
    cumulative = np.concatenate(
        (np.zeros((len(square), 1), dtype=np.float64), np.cumsum(square, axis=1)),
        axis=1,
    )
    indices = np.arange(velocity.shape[1])
    starts = np.maximum(0, indices - rms_window_steps + 1)
    counts = indices - starts + 1
    rms = np.sqrt((cumulative[:, indices + 1] - cumulative[:, starts]) / counts[None, :])
    return velocity, rms


def hysteresis_event_labels(
    rms_velocity: np.ndarray,
    on_threshold: float,
    off_threshold: float,
    minimum_active_steps: int,
) -> np.ndarray:
    if not (on_threshold > off_threshold >= 0.0):
        raise ValueError("vertical event thresholds must satisfy on > off >= 0")
    active = np.zeros(rms_velocity.shape, dtype=bool)
    state = rms_velocity[:, 0] >= on_threshold
    hold = np.where(state, minimum_active_steps, 0)
    active[:, 0] = state
    for step in range(1, rms_velocity.shape[1]):
        starts = (~state) & (rms_velocity[:, step] >= on_threshold)
        remains = state & ((rms_velocity[:, step] >= off_threshold) | (hold > 0))
        state = starts | remains
        hold = np.where(starts, minimum_active_steps, np.maximum(hold - 1, 0))
        active[:, step] = state
    return active


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    sample = values.astype(np.float64).reshape(-1)
    center = float(np.median(sample))
    low, high = np.quantile(sample, (0.16, 0.84))
    scale = max(float((high - low) * 0.5), 1e-6)
    return center, scale


def make_vertical_targets(
    split: dict[str, np.ndarray],
    dt_seconds: float,
    rms_window_steps: int,
    on_threshold: float,
    off_threshold: float,
    pitch_on_threshold: float,
    pitch_off_threshold: float,
    minimum_active_steps: int,
) -> VerticalTargets:
    velocity, rms = raw_axis_velocity_rms(
        split, VERTICAL_INDEX, dt_seconds, rms_window_steps
    )
    pitch_velocity, pitch_rms = raw_axis_velocity_rms(
        split, PITCH_INDEX, dt_seconds, rms_window_steps
    )
    _, full_vertical_rms = full_axis_velocity_rms(
        split, VERTICAL_INDEX, dt_seconds, rms_window_steps
    )
    _, full_pitch_rms = full_axis_velocity_rms(
        split, PITCH_INDEX, dt_seconds, rms_window_steps
    )
    full_active = hysteresis_event_labels(
        full_vertical_rms, on_threshold, off_threshold, minimum_active_steps
    ) | hysteresis_event_labels(
        full_pitch_rms, pitch_on_threshold, pitch_off_threshold, minimum_active_steps
    )
    first_target_transition = split["X"].shape[1] - 1
    target_steps = split["Y"].shape[1]
    event_active = full_active[
        :, first_target_transition : first_target_transition + target_steps
    ]
    onset = np.zeros(event_active.shape, dtype=bool)
    previous_active = (
        full_active[:, first_target_transition - 1]
        if first_target_transition > 0
        else np.zeros(len(full_active), dtype=bool)
    )
    onset[:, 0] = event_active[:, 0] & ~previous_active
    onset[:, 1:] = event_active[:, 1:] & ~event_active[:, :-1]
    event_amplitude = np.zeros_like(velocity, dtype=np.float64)
    pitch_event_amplitude = np.zeros_like(velocity, dtype=np.float64)
    log_duration = np.zeros_like(velocity, dtype=np.float64)
    target_y = split["Y"][..., VERTICAL_INDEX].astype(np.float64)
    target_pitch = split["Y"][..., PITCH_INDEX].astype(np.float64)
    history_last_y = split["X"][:, -1, VERTICAL_INDEX].astype(np.float64)
    history_last_pitch = split["X"][:, -1, PITCH_INDEX].astype(np.float64)
    for row in range(len(event_active)):
        starts = np.flatnonzero(onset[row])
        for start in starts:
            inactive = np.flatnonzero(~event_active[row, start:])
            end = int(start + inactive[0]) if len(inactive) else event_active.shape[1]
            end = max(end, int(start) + 1)
            baseline = history_last_y[row] if start == 0 else target_y[row, start - 1]
            excursion = target_y[row, start:end] - baseline
            peak = int(np.argmax(np.abs(excursion)))
            event_amplitude[row, start] = excursion[peak]
            pitch_baseline = (
                history_last_pitch[row] if start == 0 else target_pitch[row, start - 1]
            )
            pitch_excursion = target_pitch[row, start:end] - pitch_baseline
            pitch_peak = int(np.argmax(np.abs(pitch_excursion)))
            pitch_event_amplitude[row, start] = pitch_excursion[pitch_peak]
            log_duration[row, start] = math.log(max(end - int(start), minimum_active_steps))
    return VerticalTargets(
        velocity_mps=velocity.astype(np.float32),
        rms_velocity_mps=rms.astype(np.float32),
        amplitude_m=event_amplitude.astype(np.float32),
        pitch_velocity_radps=pitch_velocity.astype(np.float32),
        pitch_amplitude_rad=pitch_event_amplitude.astype(np.float32),
        log_duration_steps=log_duration.astype(np.float32),
        event_active=event_active.astype(np.float32),
        event_onset=onset.astype(np.float32),
    )


def make_loader(
    split: dict[str, np.ndarray],
    targets: VerticalTargets,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    velocity_mean: float,
    velocity_std: float,
    amplitude_mean: float,
    amplitude_std: float,
    log_duration_mean: float,
    log_duration_std: float,
    pitch_velocity_mean: float,
    pitch_velocity_std: float,
    pitch_amplitude_mean: float,
    pitch_amplitude_std: float,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    history = (split["X"].astype(np.float32) - state_mean.astype(np.float32)) / state_std.astype(np.float32)
    target = (split["Y"].astype(np.float32) - state_mean.astype(np.float32)) / state_std.astype(np.float32)
    velocity = (targets.velocity_mps - velocity_mean) / velocity_std
    amplitude = (targets.amplitude_m - amplitude_mean) / amplitude_std
    duration = (targets.log_duration_steps - log_duration_mean) / log_duration_std
    pitch_velocity = (targets.pitch_velocity_radps - pitch_velocity_mean) / pitch_velocity_std
    pitch_amplitude = (targets.pitch_amplitude_rad - pitch_amplitude_mean) / pitch_amplitude_std
    # Robust normalization protects training from rare tracking spikes while
    # preserving physical targets for the reconstructed-state loss.
    velocity = np.clip(velocity, -8.0, 8.0).astype(np.float32)
    amplitude = np.clip(amplitude, -8.0, 8.0).astype(np.float32)
    duration = np.clip(duration, -8.0, 8.0).astype(np.float32)
    pitch_velocity = np.clip(pitch_velocity, -8.0, 8.0).astype(np.float32)
    pitch_amplitude = np.clip(pitch_amplitude, -8.0, 8.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(history),
        torch.from_numpy(target),
        torch.from_numpy(targets.event_active),
        torch.from_numpy(targets.event_onset),
        torch.from_numpy(velocity),
        torch.from_numpy(amplitude),
        torch.from_numpy(duration),
        torch.from_numpy(pitch_velocity),
        torch.from_numpy(pitch_amplitude),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def rollout_curriculum(epoch: int, epochs: int, maximum: int) -> tuple[int, list[int]]:
    stages: list[int] = []
    for candidate in (8, 16, 32, maximum):
        value = min(candidate, maximum)
        if value not in stages:
            stages.append(value)
    index = min(len(stages) - 1, ((epoch - 1) * len(stages)) // max(epochs, 1))
    return stages[index], stages


def teacher_forcing_ratio(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    return start + ((epoch - 1) / (epochs - 1)) * (end - start)


def v3_loss(
    prediction: V3Rollout,
    history: torch.Tensor,
    target_state: torch.Tensor,
    target_event: torch.Tensor,
    target_onset: torch.Tensor,
    target_velocity: torch.Tensor,
    target_amplitude: torch.Tensor,
    target_log_duration: torch.Tensor,
    target_pitch_velocity: torch.Tensor,
    target_pitch_amplitude: torch.Tensor,
    positive_weight: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    main_indices = torch.as_tensor(MAIN_STATE_INDICES, device=target_state.device)
    main = F.smooth_l1_loss(
        prediction.state_normalized.index_select(-1, main_indices),
        target_state.index_select(-1, main_indices),
    )
    vertical_state = F.smooth_l1_loss(
        prediction.state_normalized[..., VERTICAL_INDEX], target_state[..., VERTICAL_INDEX]
    )
    pitch_state = F.smooth_l1_loss(
        prediction.state_normalized[..., PITCH_INDEX], target_state[..., PITCH_INDEX]
    )
    gate = F.binary_cross_entropy_with_logits(
        prediction.vertical_event_logit,
        target_event,
        pos_weight=target_event.new_tensor(positive_weight),
    )
    velocity_weight = 0.25 + 0.75 * target_event
    velocity = (
        F.smooth_l1_loss(
            prediction.vertical_velocity_normalized,
            target_velocity,
            reduction="none",
        )
        * velocity_weight
    ).mean()
    onset_denominator = torch.clamp(target_onset.sum(), min=1.0)
    amplitude = (
        F.smooth_l1_loss(
            prediction.vertical_amplitude_normalized,
            target_amplitude,
            reduction="none",
        )
        * target_onset
    ).sum() / onset_denominator
    duration = (
        F.smooth_l1_loss(
            prediction.vertical_log_duration_normalized,
            target_log_duration,
            reduction="none",
        )
        * target_onset
    ).sum() / onset_denominator
    pitch_velocity = (
        F.smooth_l1_loss(
            prediction.pitch_velocity_normalized,
            target_pitch_velocity,
            reduction="none",
        )
        * velocity_weight
    ).mean()
    pitch_amplitude = (
        F.smooth_l1_loss(
            prediction.pitch_amplitude_normalized,
            target_pitch_amplitude,
            reduction="none",
        )
        * target_onset
    ).sum() / onset_denominator

    predicted_with_context = torch.cat((history[:, -2:], prediction.state_normalized), dim=1)
    target_with_context = torch.cat((history[:, -2:], target_state), dim=1)
    delta = F.smooth_l1_loss(
        torch.diff(predicted_with_context, dim=1),
        torch.diff(target_with_context, dim=1),
    )
    acceleration = F.smooth_l1_loss(
        torch.diff(predicted_with_context, n=2, dim=1),
        torch.diff(target_with_context, n=2, dim=1),
    )
    gate_change = prediction.state_normalized.new_zeros(())
    if prediction.vertical_event_logit.shape[1] > 1:
        gate_change = F.smooth_l1_loss(
            torch.diff(torch.sigmoid(prediction.vertical_event_logit), dim=1),
            torch.diff(target_event, dim=1),
        )
    predicted_y = prediction.state_normalized[..., VERTICAL_INDEX]
    previous_y = torch.cat((history[:, -1:, VERTICAL_INDEX], predicted_y[:, :-1]), dim=1)
    predicted_vertical_delta = predicted_y - previous_y
    predicted_pitch = prediction.state_normalized[..., PITCH_INDEX]
    previous_pitch = torch.cat((history[:, -1:, PITCH_INDEX], predicted_pitch[:, :-1]), dim=1)
    predicted_pitch_delta = predicted_pitch - previous_pitch
    endpoint = torch.zeros_like(target_event)
    if target_event.shape[1] > 1:
        endpoint[:, 1:] = target_event[:, :-1] * (1.0 - target_event[:, 1:])
    endpoint_denominator = torch.clamp(endpoint.sum(), min=1.0)
    endpoint_closure = (
        F.smooth_l1_loss(
            predicted_vertical_delta,
            torch.zeros_like(predicted_vertical_delta),
            reduction="none",
        )
        * endpoint
    ).sum() / endpoint_denominator
    pitch_endpoint_closure = (
        F.smooth_l1_loss(
            predicted_pitch_delta,
            torch.zeros_like(predicted_pitch_delta),
            reduction="none",
        )
        * endpoint
    ).sum() / endpoint_denominator
    total = (
        args.main_state_loss_weight * main
        + args.vertical_state_loss_weight * vertical_state
        + args.pitch_state_loss_weight * pitch_state
        + args.event_loss_weight * gate
        + args.velocity_loss_weight * velocity
        + args.pitch_velocity_loss_weight * pitch_velocity
        + args.amplitude_loss_weight * amplitude
        + args.duration_loss_weight * duration
        + args.pitch_amplitude_loss_weight * pitch_amplitude
        + args.delta_loss_weight * delta
        + args.acceleration_loss_weight * acceleration
        + args.gate_change_loss_weight * gate_change
        + args.endpoint_closure_loss_weight * (endpoint_closure + pitch_endpoint_closure)
    )
    parts = {
        "main": float(main.detach()),
        "vertical_state": float(vertical_state.detach()),
        "pitch_state": float(pitch_state.detach()),
        "event": float(gate.detach()),
        "velocity": float(velocity.detach()),
        "pitch_velocity": float(pitch_velocity.detach()),
        "amplitude": float(amplitude.detach()),
        "duration": float(duration.detach()),
        "pitch_amplitude": float(pitch_amplitude.detach()),
        "delta": float(delta.detach()),
        "acceleration": float(acceleration.detach()),
        "gate_change": float(gate_change.detach()),
        "endpoint_closure": float(endpoint_closure.detach()),
        "pitch_endpoint_closure": float(pitch_endpoint_closure.detach()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: NormalHMDV3,
    loader: DataLoader,
    device: torch.device,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    velocity_mean: float,
    velocity_std: float,
    amplitude_mean: float,
    amplitude_std: float,
    pitch_velocity_mean: float,
    pitch_velocity_std: float,
    pitch_amplitude_mean: float,
    pitch_amplitude_std: float,
    positive_weight: float,
    args: argparse.Namespace,
    collect: bool,
) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    item_count = 0
    collected: dict[str, list[np.ndarray]] = {
        "state": [], "target_state": [], "event_probability": [], "target_event": [],
        "velocity": [], "target_velocity": [], "amplitude": [], "target_amplitude": [],
        "pitch_velocity": [], "target_pitch_velocity": [],
        "pitch_amplitude": [], "target_pitch_amplitude": [],
    }
    mean_tensor = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(state_std, dtype=torch.float32, device=device)
    for history, target, event, onset, velocity, amplitude, duration, pitch_velocity, pitch_amplitude in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        event = event.to(device, non_blocking=True)
        onset = onset.to(device, non_blocking=True)
        velocity = velocity.to(device, non_blocking=True)
        amplitude = amplitude.to(device, non_blocking=True)
        duration = duration.to(device, non_blocking=True)
        pitch_velocity = pitch_velocity.to(device, non_blocking=True)
        pitch_amplitude = pitch_amplitude.to(device, non_blocking=True)
        output = model.rollout(history, target.shape[1])
        loss, _ = v3_loss(
            output, history, target, event, onset, velocity, amplitude, duration,
            pitch_velocity, pitch_amplitude,
            positive_weight, args
        )
        total_loss += float(loss) * len(history)
        item_count += len(history)
        if collect:
            collected["state"].append((output.state_normalized * std_tensor + mean_tensor).cpu().numpy())
            collected["target_state"].append((target * std_tensor + mean_tensor).cpu().numpy())
            collected["event_probability"].append(torch.sigmoid(output.vertical_event_logit).cpu().numpy())
            collected["target_event"].append(event.cpu().numpy())
            collected["velocity"].append(
                (output.vertical_velocity_normalized * velocity_std + velocity_mean).cpu().numpy()
            )
            collected["target_velocity"].append((velocity * velocity_std + velocity_mean).cpu().numpy())
            collected["amplitude"].append(
                (output.vertical_amplitude_normalized * amplitude_std + amplitude_mean).cpu().numpy()
            )
            collected["target_amplitude"].append((amplitude * amplitude_std + amplitude_mean).cpu().numpy())
            collected["pitch_velocity"].append(
                (output.pitch_velocity_normalized * pitch_velocity_std + pitch_velocity_mean).cpu().numpy()
            )
            collected["target_pitch_velocity"].append(
                (pitch_velocity * pitch_velocity_std + pitch_velocity_mean).cpu().numpy()
            )
            collected["pitch_amplitude"].append(
                (output.pitch_amplitude_normalized * pitch_amplitude_std + pitch_amplitude_mean).cpu().numpy()
            )
            collected["target_pitch_amplitude"].append(
                (pitch_amplitude * pitch_amplitude_std + pitch_amplitude_mean).cpu().numpy()
            )
    arrays = {
        name: np.concatenate(values, axis=0) if values else np.empty(0)
        for name, values in collected.items()
    }
    return total_loss / max(item_count, 1), arrays


def binary_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predicted = probability >= threshold
    actual = target >= 0.5
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if predicted.ndim >= 2:
        rising = predicted[:, 0].sum() + np.sum(predicted[:, 1:] & ~predicted[:, :-1])
        events_per_window = float(rising / max(predicted.shape[0], 1))
    else:
        events_per_window = float("nan")
    return {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "predicted_active_fraction": float(predicted.mean()),
        "target_active_fraction": float(actual.mean()),
        "threshold": float(threshold),
        "predicted_events_per_window": events_per_window,
    }


def select_event_gate_threshold(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    target_fraction = float(np.mean(target >= 0.5))
    matched_quantile = float(np.quantile(probability, np.clip(1.0 - target_fraction, 0.0, 1.0)))
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 181), [matched_quantile])))
    scored = [(binary_metrics(probability, target, float(value)), float(value)) for value in candidates]
    tolerance = max(0.05, 0.25 * target_fraction)
    plausible = [
        item for item in scored
        if abs(item[0]["predicted_active_fraction"] - target_fraction) <= tolerance
        and 0.0 < item[0]["predicted_active_fraction"] < 1.0
        and item[0]["predicted_events_per_window"] > 0.0
    ]
    pool = plausible if plausible else scored
    best_metrics, best_threshold = max(
        pool,
        key=lambda item: (
            item[0]["f1"],
            -abs(item[0]["predicted_active_fraction"] - item[0]["target_active_fraction"]),
        ),
    )
    return {
        "threshold": best_threshold,
        "frequency_constraint_satisfied": bool(plausible),
        "target_active_fraction_tolerance": tolerance,
        **best_metrics,
    }


def vertical_dynamics(values: np.ndarray, dt_seconds: float) -> dict[str, float | None]:
    y = values[..., VERTICAL_INDEX].reshape(-1, values.shape[-2])
    velocity = np.diff(y, axis=1) / dt_seconds
    acceleration = np.diff(velocity, axis=1) / dt_seconds
    jerk = np.diff(acceleration, axis=1) / dt_seconds
    pair_a = velocity[:, :-1].reshape(-1)
    pair_b = velocity[:, 1:].reshape(-1)
    lag1 = None
    if pair_a.std() > 1e-12 and pair_b.std() > 1e-12:
        lag1 = float(np.corrcoef(pair_a, pair_b)[0, 1])
    reversals = np.sum(velocity[:, 1:] * velocity[:, :-1] < 0.0)
    duration = velocity.shape[0] * max((velocity.shape[1] - 1) * dt_seconds, dt_seconds)
    return {
        "velocity_lag1_correlation": lag1,
        "direction_reversals_per_second": float(reversals / duration),
        "absolute_acceleration_p95_mps2": float(np.quantile(np.abs(acceleration), 0.95)),
        "absolute_jerk_p95_mps3": float(np.quantile(np.abs(jerk), 0.95)),
    }


def evaluation_metrics(
    arrays: dict[str, np.ndarray], dt_seconds: float, event_gate_threshold: float
) -> dict[str, Any]:
    prediction = arrays["state"].astype(np.float64)
    target = arrays["target_state"].astype(np.float64)
    absolute = np.abs(prediction - target)
    return {
        "mae_by_feature": {
            name: float(value)
            for name, value in zip(STATE_FEATURE_NAMES, absolute.mean(axis=(0, 1)))
        },
        "vertical_event": binary_metrics(
            arrays["event_probability"], arrays["target_event"], event_gate_threshold
        ),
        "vertical_velocity_mae_mps": float(
            np.mean(np.abs(arrays["velocity"] - arrays["target_velocity"]))
        ),
        "vertical_amplitude_mae_m": float(
            np.mean(np.abs(arrays["amplitude"] - arrays["target_amplitude"]))
        ),
        "pitch_velocity_mae_radps": float(
            np.mean(np.abs(arrays["pitch_velocity"] - arrays["target_pitch_velocity"]))
        ),
        "pitch_amplitude_mae_rad": float(
            np.mean(np.abs(arrays["pitch_amplitude"] - arrays["target_pitch_amplitude"]))
        ),
        "predicted_vertical_dynamics": vertical_dynamics(prediction, dt_seconds),
        "target_vertical_dynamics": vertical_dynamics(target, dt_seconds),
    }


@torch.no_grad()
def closed_loop_probe(
    model: NormalHMDV3,
    history_physical: np.ndarray,
    steps: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    device: torch.device,
    root_noise_std_normalized: float,
    root_noise_rho: float,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    normalized = (history_physical.astype(np.float64) - state_mean) / state_std
    history = torch.from_numpy(normalized.astype(np.float32))[None].to(device)
    step_output, hidden = model.warm_history(history)
    previous = history[:, -1]
    controller = model.initial_vertical_controller(previous)
    rng = np.random.default_rng(seed)
    root_noise = np.zeros(3, dtype=np.float64)
    states: list[np.ndarray] = []
    gates: list[float] = []
    clipped_steps = 0
    for step in range(steps):
        predicted_normalized, controller = model.compose_state(
            previous, step_output, controller
        )
        predicted = predicted_normalized[0].cpu().numpy().astype(np.float64) * state_std + state_mean
        clipped = np.clip(predicted, clip_low, clip_high)
        clipped_steps += int(np.any(clipped != predicted))
        states.append(clipped.copy())
        gates.append(float(torch.sigmoid(step_output.vertical_event_logit)[0].cpu()))
        if step + 1 == steps:
            break
        feedback = clipped.copy()
        if root_noise_std_normalized > 0.0:
            root_noise = (
                root_noise_rho * root_noise
                + math.sqrt(1.0 - root_noise_rho**2) * rng.standard_normal(3)
            )
            feedback[:3] += root_noise * root_noise_std_normalized * state_std[:3]
            feedback[:3] = np.clip(feedback[:3], clip_low[:3], clip_high[:3])
        # Deployment contract: final safe/applied BodyRoot is fed back, but all
        # six HeadLocal channels are the model output with no stochastic layer.
        feedback_normalized = ((feedback - state_mean) / state_std).astype(np.float32)
        previous = torch.from_numpy(feedback_normalized)[None].to(device)
        step_output, hidden = model.advance(previous, hidden)
    generated = np.asarray(states)
    gate_active = np.asarray(gates) >= model.event_gate_threshold
    event_count = int(gate_active[0]) + int(np.sum(gate_active[1:] & ~gate_active[:-1]))
    delta_rms = float(np.sqrt(np.mean(np.square(np.diff(generated, axis=0)))))
    return {
        "steps": int(len(generated)),
        "finite": bool(np.isfinite(generated).all()),
        "clip_step_fraction": clipped_steps / max(len(generated), 1),
        "state_delta_rms": delta_rms,
        "head_local_y_min_m": float(generated[:, VERTICAL_INDEX].min()),
        "head_local_y_max_m": float(generated[:, VERTICAL_INDEX].max()),
        "head_local_pitch_min_rad": float(generated[:, PITCH_INDEX].min()),
        "head_local_pitch_max_rad": float(generated[:, PITCH_INDEX].max()),
        "event_probability_mean": float(np.mean(gates)),
        "event_gate_threshold": model.event_gate_threshold,
        "predicted_active_fraction": float(gate_active.mean()),
        "event_count": event_count,
        "vertical_dynamics": vertical_dynamics(generated[None], model.fixed_dt_seconds),
        "feedback_contract": (
            "safe final BodyRoot[0:3], including optional root innovation, is recurrent feedback; "
            "generated HeadLocal[3:9] has no per-frame stochastic innovation"
        ),
        "passed": bool(np.isfinite(generated).all() and delta_rms > 1e-6),
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def initialize_from_v2(model: NormalHMDV3, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"v2 warm-start checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    if (
        int(config.get("input_dim", -1)) != 9
        or int(config.get("hidden_dim", -1)) != model.hidden_dim
        or int(config.get("num_layers", -1)) != model.num_layers
    ):
        raise ValueError(
            "v2 warm-start dimensions do not match v3; use hidden_dim=128 and num_layers=2 "
            "for the production checkpoint, or pass --no-warm-start for architecture smoke tests"
        )
    source = checkpoint["state_dict"]
    destination = model.state_dict()
    recurrent_keys = [name for name in destination if name.startswith("lstm.")]
    for name in recurrent_keys:
        if name not in source or source[name].shape != destination[name].shape:
            raise ValueError(f"v2 warm-start tensor mismatch: {name}")
        destination[name].copy_(source[name])
    index = torch.as_tensor(MAIN_STATE_INDICES, dtype=torch.long)
    destination["main_output.weight"].copy_(source["output.weight"].index_select(0, index))
    destination["main_output.bias"].copy_(source["output.bias"].index_select(0, index))
    model.load_state_dict(destination)
    return {
        "used": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "copied": "all LSTM tensors plus v2 output rows for MAIN_STATE_INDICES",
        "new_random_heads": [
            "vertical_event", "vertical_velocity", "vertical_amplitude",
            "pitch_velocity", "pitch_amplitude", "vertical_log_duration",
        ],
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if data_report.get("status") != "success" or data_report.get("synthetic") is not False:
        raise ValueError("v3 requires the successful real EyeNavGS v2 data report")
    dt_seconds = float(data_report["fixed_dt_seconds"])
    full_splits = {name: load_split(args.data_dir, name) for name in ("train", "val", "test")}
    prepared_split_sha256: dict[str, str] = {}
    for name, split in full_splits.items():
        expected = data_report["splits"][name]["sha256"]
        actual = sha256_file(args.data_dir / f"normal_hmd_{name}.npz")
        if actual != expected:
            raise ValueError(f"{name} NPZ hash differs from frozen v2 report")
        prepared_split_sha256[name] = actual
        if split["X"].shape[-1] != 9 or split["Y"].shape[-1] != 9:
            raise ValueError(f"{name} does not use the frozen 9-D state")

    train_raw_velocity, train_raw_rms = raw_axis_velocity_rms(
        full_splits["train"], VERTICAL_INDEX, dt_seconds, args.event_rms_window_steps,
    )
    event_on = float(np.quantile(train_raw_rms, args.event_on_quantile))
    event_off = float(np.quantile(train_raw_rms, args.event_off_quantile))
    train_pitch_velocity, train_pitch_rms = raw_axis_velocity_rms(
        full_splits["train"], PITCH_INDEX, dt_seconds, args.event_rms_window_steps
    )
    pitch_event_on = float(np.quantile(train_pitch_rms, args.event_on_quantile))
    pitch_event_off = float(np.quantile(train_pitch_rms, args.event_off_quantile))
    velocity_mean, velocity_std = robust_center_scale(train_raw_velocity)
    pitch_velocity_mean, pitch_velocity_std = robust_center_scale(train_pitch_velocity)
    full_train_targets = make_vertical_targets(
        full_splits["train"], dt_seconds, args.event_rms_window_steps,
        event_on, event_off,
        pitch_event_on, pitch_event_off, args.minimum_event_steps,
    )
    onset_mask = full_train_targets.event_onset.astype(bool)
    if not np.any(onset_mask):
        raise RuntimeError("vertical event labelling produced no training onsets")
    amplitude_mean, amplitude_std = robust_center_scale(
        full_train_targets.amplitude_m[onset_mask]
    )
    pitch_amplitude_mean, pitch_amplitude_std = robust_center_scale(
        full_train_targets.pitch_amplitude_rad[onset_mask]
    )
    log_duration_mean, log_duration_std = robust_center_scale(
        full_train_targets.log_duration_steps[onset_mask]
    )
    state_mean, state_std = fit_normalization(full_splits["train"]["X"], full_splits["train"]["Y"])
    clip_low, clip_high = sampled_quantiles(
        full_splits["train"]["X"], full_splits["train"]["Y"],
        args.clip_low_quantile, args.clip_high_quantile, args.seed,
    )
    clip_low[3:6] = np.maximum(clip_low[3:6], np.array([-0.35, -0.35, -0.35]))
    clip_high[3:6] = np.minimum(clip_high[3:6], np.array([0.35, 0.35, 0.35]))

    limits = {"train": args.max_train_segments, "val": args.max_val_segments, "test": args.max_test_segments}
    splits = {
        name: subset_split(split, limits[name], args.seed + index * 101)
        for index, (name, split) in enumerate(full_splits.items())
    }
    targets = {
        name: make_vertical_targets(
            split, dt_seconds, args.event_rms_window_steps,
            event_on, event_off, pitch_event_on, pitch_event_off,
            args.minimum_event_steps,
        )
        for name, split in splits.items()
    }
    positive_fraction = float(targets["train"].event_active.mean())
    positive_weight = float(np.clip((1.0 - positive_fraction) / max(positive_fraction, 1e-6), 0.25, 4.0))
    loaders = {
        name: make_loader(
            splits[name], targets[name], state_mean, state_std,
            velocity_mean, velocity_std, amplitude_mean, amplitude_std,
            log_duration_mean, log_duration_std,
            pitch_velocity_mean, pitch_velocity_std,
            pitch_amplitude_mean, pitch_amplitude_std,
            args.batch_size, name == "train", args.workers, device.type == "cuda",
        )
        for name in splits
    }

    model = NormalHMDV3(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fixed_dt_seconds=dt_seconds,
        vertical_state_mean=float(state_mean[VERTICAL_INDEX]),
        vertical_state_std=float(state_std[VERTICAL_INDEX]),
        vertical_velocity_mean=velocity_mean,
        vertical_velocity_std=velocity_std,
        vertical_amplitude_mean=amplitude_mean,
        vertical_amplitude_std=amplitude_std,
        vertical_log_duration_mean=log_duration_mean,
        vertical_log_duration_std=log_duration_std,
        pitch_state_mean=float(state_mean[PITCH_INDEX]),
        pitch_state_std=float(state_std[PITCH_INDEX]),
        pitch_velocity_mean=pitch_velocity_mean,
        pitch_velocity_std=pitch_velocity_std,
        pitch_amplitude_mean=pitch_amplitude_mean,
        pitch_amplitude_std=pitch_amplitude_std,
    ).to(device)
    warm_start = {"used": False, "reason": "disabled by --no-warm-start"}
    if not args.no_warm_start:
        warm_start = initialize_from_v2(model, args.initialize_from_v2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rollout_length = int(splits["train"]["Y"].shape[1])
    _, curriculum = rollout_curriculum(1, args.epochs, rollout_length)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_steps, _ = rollout_curriculum(epoch, args.epochs, rollout_length)
        teacher_ratio = teacher_forcing_ratio(
            epoch, args.epochs, args.teacher_forcing_start, args.teacher_forcing_end
        )
        loss_sum = 0.0
        item_count = 0
        for history, target, event, onset, velocity, amplitude, duration, pitch_velocity, pitch_amplitude in loaders["train"]:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            event = event.to(device, non_blocking=True)
            onset = onset.to(device, non_blocking=True)
            velocity = velocity.to(device, non_blocking=True)
            amplitude = amplitude.to(device, non_blocking=True)
            duration = duration.to(device, non_blocking=True)
            pitch_velocity = pitch_velocity.to(device, non_blocking=True)
            pitch_amplitude = pitch_amplitude.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model.rollout(
                    history, training_steps, teacher=target[:, :training_steps],
                    vertical_gate_teacher=event[:, :training_steps],
                    vertical_onset_teacher=onset[:, :training_steps],
                    teacher_forcing_ratio=teacher_ratio,
                    root_feedback_noise_std=args.root_feedback_noise_std,
                    root_feedback_noise_rho=args.root_feedback_noise_rho,
                )
                loss, _ = v3_loss(
                    output, history, target[:, :training_steps], event[:, :training_steps],
                    onset[:, :training_steps], velocity[:, :training_steps],
                    amplitude[:, :training_steps], duration[:, :training_steps],
                    pitch_velocity[:, :training_steps], pitch_amplitude[:, :training_steps],
                    positive_weight, args,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(history)
            item_count += len(history)
        train_loss = loss_sum / max(item_count, 1)
        val_loss, _ = evaluate(
            model, loaders["val"], device, state_mean, state_std,
            velocity_mean, velocity_std, amplitude_mean, amplitude_std,
            pitch_velocity_mean, pitch_velocity_std,
            pitch_amplitude_mean, pitch_amplitude_std,
            positive_weight, args, collect=False,
        )
        eligible = training_steps == rollout_length
        improved = eligible and val_loss < best_loss - args.min_delta
        history_rows.append({
            "epoch": epoch,
            "training_rollout_steps": training_steps,
            "teacher_forcing_ratio": teacher_ratio,
            "train_loss": train_loss,
            "val_closed_loop_loss": val_loss,
            "eligible_for_checkpoint": eligible,
            "best_so_far": improved,
        })
        print(
            f"epoch {epoch:03d} rollout={training_steps} teacher={teacher_ratio:.3f} "
            f"train={train_loss:.6f} val={val_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        elif eligible:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        # One-epoch smoke tests never reach a later curriculum stage; their
        # single full available rollout is nevertheless a valid code-path test.
        if args.epochs == 1:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_loss = history_rows[-1]["val_closed_loop_loss"]
            best_epoch = 1
        else:
            raise RuntimeError("training produced no full-rollout checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    _, threshold_arrays = evaluate(
        model, loaders["val"], device, state_mean, state_std,
        velocity_mean, velocity_std, amplitude_mean, amplitude_std,
        pitch_velocity_mean, pitch_velocity_std,
        pitch_amplitude_mean, pitch_amplitude_std,
        positive_weight, args, collect=True,
    )
    gate_calibration = select_event_gate_threshold(
        threshold_arrays["event_probability"], threshold_arrays["target_event"]
    )
    model.event_gate_threshold = float(gate_calibration["threshold"])

    evaluations: dict[str, Any] = {}
    for name in ("val", "test"):
        closed_loss, arrays = evaluate(
            model, loaders[name], device, state_mean, state_std,
            velocity_mean, velocity_std, amplitude_mean, amplitude_std,
            pitch_velocity_mean, pitch_velocity_std,
            pitch_amplitude_mean, pitch_amplitude_std,
            positive_weight, args, collect=True,
        )
        evaluations[name] = {
            "closed_loop_loss": closed_loss,
            **evaluation_metrics(arrays, dt_seconds, model.event_gate_threshold),
        }

    probe_steps = int(math.ceil(args.probe_seconds / dt_seconds))
    probe = closed_loop_probe(
        model, splits["test"]["X"][0].astype(np.float64), probe_steps,
        state_mean, state_std, clip_low, clip_high, device,
        args.probe_root_noise_std, args.probe_root_noise_rho, args.seed + 7001,
    )
    model_config = {
        "architecture": "shared_lstm_7d_main_plus_y_pitch_event_heads",
        "input_dim": 9,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "main_output_dim": 7,
        "main_state_indices": list(MAIN_STATE_INDICES),
        "vertical_state_index": VERTICAL_INDEX,
        "pitch_state_index": PITCH_INDEX,
        "vertical_heads": [
            "event_gate_logit", "signed_velocity_auxiliary",
            "signed_event_amplitude", "signed_pitch_velocity_auxiliary",
            "signed_pitch_amplitude", "log_event_duration",
        ],
        "vertical_synthesis": "signed 16*u^2*(1-u)^2 event bump; zero displacement and velocity at endpoints",
        "history_length": int(splits["train"]["X"].shape[1]),
        "rollout_length": rollout_length,
        "stateful_inference": True,
        "event_gate_threshold": model.event_gate_threshold,
    }
    checkpoint = {
        "schema_version": "3.0.0",
        "state_dict": best_state,
        "model_config": model_config,
        "parameter_count": parameter_count(model),
        "normalization": {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "vertical_velocity_mean_mps": velocity_mean,
            "vertical_velocity_std_mps": velocity_std,
            "vertical_amplitude_mean_m": amplitude_mean,
            "vertical_amplitude_std_m": amplitude_std,
            "vertical_log_duration_mean": log_duration_mean,
            "vertical_log_duration_std": log_duration_std,
            "pitch_velocity_mean_radps": pitch_velocity_mean,
            "pitch_velocity_std_radps": pitch_velocity_std,
            "pitch_amplitude_mean_rad": pitch_amplitude_mean,
            "pitch_amplitude_std_rad": pitch_amplitude_std,
        },
        "state_clip_low": clip_low.tolist(),
        "state_clip_high": clip_high.tolist(),
        "dt_seconds": dt_seconds,
        "feature_names": list(STATE_FEATURE_NAMES),
        "vertical_event_definition": {
            "source": "causal RMS of HeadLocal Y velocity",
            "rms_window_steps": args.event_rms_window_steps,
            "on_threshold_mps": event_on,
            "off_threshold_mps": event_off,
            "pitch_on_threshold_radps": pitch_event_on,
            "pitch_off_threshold_radps": pitch_event_off,
            "minimum_active_steps": args.minimum_event_steps,
            "amplitude_label": "signed anchor-to-peak excursion of each complete active event",
            "event_template": "16*u^2*(1-u)^2 signed bump returning to onset baseline",
            "duration_bounds_steps": [4, 74],
            "gate_semantics": "active probability; a low-to-high crossing starts one event and falling low rearms",
            "gate_threshold_selection": "validation F1 grid over 0.10..0.90",
            "event_gate_threshold": model.event_gate_threshold,
        },
        "stochastic_policy": {
            "headlocal": "none; all six HeadLocal channels are generated without per-frame AR noise",
            "bodyroot": "optional root-only innovation may be applied, safety-clipped, and fed back",
            "training_root_feedback_noise_std_normalized": args.root_feedback_noise_std,
            "training_root_feedback_noise_rho": args.root_feedback_noise_rho,
        },
        "root_innovation_model": {
            "dimension": 3,
            "feature_indices": [0, 1, 2],
            "method": "diagonal stationary normalized AR(1) training/deployment contract",
            "ar1_rho": [args.root_feedback_noise_rho] * 3,
            "stationary_std_normalized": [args.root_feedback_noise_std] * 3,
            "stationary_std_physical": (
                state_std[:3] * args.root_feedback_noise_std
            ).tolist(),
            "physical_covariance": np.diag(
                np.square(state_std[:3] * args.root_feedback_noise_std)
            ).tolist(),
            "default_noise_scale": 1.0,
            "feedback": "apply to root only, safety clip, then feed final root back",
        },
        "runtime_feedback_contract": (
            "Feed final safe/applied BodyRoot channels 0:3 and noise-free generated "
            "HeadLocal channels 3:9 into the next recurrent step."
        ),
        "data_source": "EyeNavGS",
        "synthetic": False,
        "dataset_fingerprint_sha256": data_report["dataset_fingerprint_sha256"],
        "data_report_sha256": sha256_file(args.data_report),
        "participant_manifest": data_report["participant_manifest"],
        "prepared_split_sha256": prepared_split_sha256,
        "coordinate_system": data_report["coordinate_system"],
        "body_proxy": data_report["body_proxy"],
        "nominal_eye_height_m": float(data_report["body_proxy"]["nominal_eye_height_m"]),
        "seed": args.seed,
        "warm_start": warm_start,
    }
    atomic_torch_save(args.model, checkpoint)
    model_hash = sha256_file(args.model)
    write_history(args.history_output, history_rows)
    metrics = {
        "schema_version": "3.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "task": "event-conditioned Normal HMD generation",
        "data_source": "EyeNavGS",
        "synthetic": False,
        "smoke_subset": any(value > 0 for value in limits.values()),
        "segments": {name: int(len(split["X"])) for name, split in splits.items()},
        "model_config": model_config,
        "parameter_count": parameter_count(model),
        "warm_start": warm_start,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": use_amp,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "best_val_closed_loop_loss": best_loss,
        "training_elapsed_seconds": time.perf_counter() - started,
        "vertical_event_definition": checkpoint["vertical_event_definition"],
        "train_event_active_fraction": positive_fraction,
        "event_positive_weight": positive_weight,
        "event_gate_calibration": gate_calibration,
        "loss_weights": {
            "main_state": args.main_state_loss_weight,
            "vertical_state": args.vertical_state_loss_weight,
            "pitch_state": args.pitch_state_loss_weight,
            "event": args.event_loss_weight,
            "velocity": args.velocity_loss_weight,
            "pitch_velocity": args.pitch_velocity_loss_weight,
            "amplitude": args.amplitude_loss_weight,
            "pitch_amplitude": args.pitch_amplitude_loss_weight,
            "duration": args.duration_loss_weight,
            "delta": args.delta_loss_weight,
            "acceleration": args.acceleration_loss_weight,
            "gate_change": args.gate_change_loss_weight,
            "endpoint_closure": args.endpoint_closure_loss_weight,
        },
        "stochastic_policy": checkpoint["stochastic_policy"],
        "runtime_feedback_contract": checkpoint["runtime_feedback_contract"],
        "evaluations": evaluations,
        "closed_loop_probe": probe,
        "artifacts": {
            "model": str(args.model.resolve()),
            "model_sha256": model_hash,
            "metadata": str(args.metadata.resolve()),
            "history": str(args.history_output.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    metadata = copy.deepcopy(metrics)
    metadata.pop("evaluations", None)
    metadata.update({
        "model_sha256": model_hash,
        "normalization": checkpoint["normalization"],
        "state_clip_low": checkpoint["state_clip_low"],
        "state_clip_high": checkpoint["state_clip_high"],
        "feature_names": checkpoint["feature_names"],
        "dataset_fingerprint_sha256": checkpoint["dataset_fingerprint_sha256"],
    })
    atomic_json(args.metadata, metadata)
    atomic_json(args.metrics, metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-report", type=Path, default=DEFAULT_DATA_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.05)
    parser.add_argument("--root-feedback-noise-std", type=float, default=0.15)
    parser.add_argument("--root-feedback-noise-rho", type=float, default=0.95)
    parser.add_argument("--event-rms-window-steps", type=int, default=5)
    parser.add_argument("--event-on-quantile", type=float, default=0.65)
    parser.add_argument("--event-off-quantile", type=float, default=0.45)
    parser.add_argument("--minimum-event-steps", type=int, default=4)
    parser.add_argument("--main-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--vertical-state-loss-weight", type=float, default=1.5)
    parser.add_argument("--pitch-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=0.35)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.50)
    parser.add_argument("--pitch-velocity-loss-weight", type=float, default=0.35)
    parser.add_argument("--amplitude-loss-weight", type=float, default=0.15)
    parser.add_argument("--pitch-amplitude-loss-weight", type=float, default=0.15)
    parser.add_argument("--duration-loss-weight", type=float, default=0.10)
    parser.add_argument("--delta-loss-weight", type=float, default=0.15)
    parser.add_argument("--acceleration-loss-weight", type=float, default=0.05)
    parser.add_argument("--gate-change-loss-weight", type=float, default=0.05)
    parser.add_argument("--endpoint-closure-loss-weight", type=float, default=0.20)
    parser.add_argument("--clip-low-quantile", type=float, default=0.001)
    parser.add_argument("--clip-high-quantile", type=float, default=0.999)
    parser.add_argument("--probe-seconds", type=float, default=60.0)
    parser.add_argument("--probe-root-noise-std", type=float, default=0.15)
    parser.add_argument("--probe-root-noise-rho", type=float, default=0.95)
    parser.add_argument("--max-train-segments", type=int, default=0)
    parser.add_argument("--max-val-segments", type=int, default=0)
    parser.add_argument("--max-test-segments", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--initialize-from-v2", type=Path, default=REPO_ROOT / "models" / "normal_hmd_v2.pth"
    )
    parser.add_argument("--no-warm-start", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or args.num_layers < 1:
        raise ValueError("positive epochs, batch size and model dimensions are required")
    if not (0.0 < args.event_off_quantile < args.event_on_quantile < 1.0):
        raise ValueError("event quantiles must satisfy 0 < off < on < 1")
    if args.event_rms_window_steps < 1 or args.minimum_event_steps < 1:
        raise ValueError("vertical event windows must be positive")
    if args.root_feedback_noise_std < 0.0 or args.probe_root_noise_std < 0.0:
        raise ValueError("root noise standard deviations cannot be negative")
    if not (0.0 <= args.root_feedback_noise_rho < 1.0) or not (
        0.0 <= args.probe_root_noise_rho < 1.0
    ):
        raise ValueError("root noise AR coefficients must be in [0, 1)")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        validate_args(args)
        metrics = train(args)
    except Exception as exc:
        atomic_json(args.metrics, {
            "schema_version": "3.0.0",
            "generated_at_utc": utc_now(),
            "status": "failed",
            "reason": str(exc),
            "model_written": False,
        })
        print(f"Normal HMD v3 training failed: {exc}", file=sys.stderr)
        return 1
    event = metrics["evaluations"]["test"]["vertical_event"]
    print("Normal HMD v3 training complete")
    print(f"  device: {metrics['device']}")
    print(f"  test vertical event F1: {event['f1']:.4f}")
    print(f"  head stochastic policy: {metrics['stochastic_policy']['headlocal']}")
    print(f"  model: {metrics['artifacts']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
