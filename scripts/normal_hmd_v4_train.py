#!/usr/bin/env python3
"""Train and smoke-test the trajectory-first Normal HMD Agent v4.

This trainer intentionally consumes the frozen v2 NPZ files.  It reconstructs
unique trajectories before event detection, learns the existing offline
SG+hysteresis thresholds on train only, freezes them for validation/test, and
then maps labels to windows by ``trajectory_index`` plus absolute
``start_index``.  The classification targets are onset hazards, never active
event gates, and BCE is evaluated only at frames where the runtime controller
would be eligible to start an event.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from normal_hmd_common import STATE_FEATURE_NAMES, atomic_json, sha256_file
from normal_hmd_v4_data import (
    DEFAULT_DT,
    HORIZONTAL_INDICES,
    ROOT_INDICES,
    VERTICAL_INDICES,
    V4WindowTargets,
    assert_onset_parameter_consistency,
    event_summary,
    label_trajectories,
    learn_event_thresholds,
    load_split,
    map_labels_to_windows,
    minimum_jerk,
    reconstruct_trajectories,
)
from normal_hmd_v4_model import (
    HORIZONTAL_PARAMETER_DIM,
    VERTICAL_PARAMETER_DIM,
    NormalHMDV4,
    V4Sequence,
    V4Step,
    minimum_jerk_torch,
    parameter_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_DATA_REPORT = REPO_ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_METADATA = REPO_ROOT / "models" / "normal_hmd_v4_metadata.json"
DEFAULT_METRICS = REPO_ROOT / "results" / "normal_hmd_v4_training_metrics.json"
DEFAULT_HISTORY = REPO_ROOT / "results" / "normal_hmd_v4_training_history.json"
SMOKE_DIR = REPO_ROOT / "tmp" / "normal_hmd_v4_smoke"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    low, high = np.quantile(values, (0.16, 0.84), axis=0)
    scale = np.maximum((high - low) * 0.5, 1e-6)
    return center, scale


def stable_correlation(normalized_values: np.ndarray, shrinkage: float = 0.15) -> np.ndarray:
    values = np.asarray(normalized_values, dtype=np.float64)
    dimension = values.shape[1]
    if len(values) < dimension + 2:
        return np.eye(dimension, dtype=np.float64)
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(dimension)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), 1e-12))
    correlation = correlation / diagonal[:, None] / diagonal[None, :]
    return 0.5 * (correlation + correlation.T)


def collect_event_parameters(labelled: Iterable[Any], branch_name: str) -> np.ndarray:
    chunks = []
    for item in labelled:
        branch = getattr(item, branch_name)
        if np.any(branch.parameter_mask):
            chunks.append(branch.parameters[branch.parameter_mask])
    if not chunks:
        raise RuntimeError(f"no complete {branch_name} events for parameter training")
    return np.concatenate(chunks, axis=0)


def collect_pose_values(labelled: Iterable[Any], indices: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(
        [item.smooth_state[:, indices] for item in labelled], axis=0
    ).astype(np.float64)


def fit_state_normalization(split: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        (split["X"].reshape(-1, 9), split["Y"].reshape(-1, 9)), axis=0
    ).astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-6)
    return mean, std


def parameter_statistics(
    values: np.ndarray,
    *,
    minimum_duration_steps: int,
    maximum_duration_steps: int,
) -> dict[str, np.ndarray]:
    center, scale = robust_center_scale(values)
    low, high = np.quantile(values, (0.05, 0.95), axis=0)
    low[-1] = math.log(float(minimum_duration_steps))
    high[-1] = math.log(float(maximum_duration_steps))
    for index in range(len(low)):
        if high[index] - low[index] < 1e-6:
            low[index] -= 5e-4
            high[index] += 5e-4
    normalized = (values - center) / scale
    return {
        "center": center,
        "scale": scale,
        "low": low,
        "high": high,
        "correlation": stable_correlation(normalized),
    }


def window_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= count:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def make_dataset(
    split: dict[str, np.ndarray],
    targets: V4WindowTargets,
    indices: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    horizontal_stats: dict[str, np.ndarray],
    vertical_stats: dict[str, np.ndarray],
) -> TensorDataset:
    history = (
        split["X"][indices].astype(np.float32) - state_mean.astype(np.float32)
    ) / state_std.astype(np.float32)
    target = (
        split["Y"][indices].astype(np.float32) - state_mean.astype(np.float32)
    ) / state_std.astype(np.float32)
    horizontal_pose = (
        targets.horizontal_pose[indices]
        - state_mean[np.asarray(HORIZONTAL_INDICES)].astype(np.float32)
    ) / state_std[np.asarray(HORIZONTAL_INDICES)].astype(np.float32)
    horizontal_parameters = (
        targets.horizontal.parameters[indices]
        - horizontal_stats["center"].astype(np.float32)
    ) / horizontal_stats["scale"].astype(np.float32)
    vertical_parameters = (
        targets.vertical.parameters[indices]
        - vertical_stats["center"].astype(np.float32)
    ) / vertical_stats["scale"].astype(np.float32)
    return TensorDataset(
        torch.from_numpy(history.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(horizontal_pose.astype(np.float32)),
        torch.from_numpy(targets.horizontal.onset[indices]),
        torch.from_numpy(targets.horizontal.eligible[indices]),
        torch.from_numpy(targets.horizontal.parameter_mask[indices]),
        torch.from_numpy(horizontal_parameters.astype(np.float32)),
        torch.from_numpy(targets.vertical.onset[indices]),
        torch.from_numpy(targets.vertical.eligible[indices]),
        torch.from_numpy(targets.vertical.parameter_mask[indices]),
        torch.from_numpy(vertical_parameters.astype(np.float32)),
    )


def make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def eligible_positive_weight(targets: V4WindowTargets, branch_name: str) -> float:
    branch = getattr(targets, branch_name)
    eligible = branch.eligible.astype(bool)
    positives = float(np.sum(branch.onset[eligible]))
    negatives = float(np.sum(eligible) - positives)
    if positives <= 0.0:
        raise RuntimeError(f"{branch_name} has no eligible onset labels")
    # Square-root weighting aids rare-onset learning without pretending that
    # the weighted sigmoid is a calibrated runtime probability.  A validation
    # eligible-time bias calibration is fitted after training.
    return float(np.sqrt(max(negatives / positives, 1.0)))


def masked_hazard_bce(
    logits: torch.Tensor,
    onset: torch.Tensor,
    eligible: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        onset,
        pos_weight=logits.new_tensor(positive_weight),
        reduction="none",
    )
    mask = eligible.to(loss.dtype)
    return torch.sum(loss * mask) / torch.clamp(mask.sum(), min=1.0)


def correlated_gaussian_nll(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cholesky: torch.Tensor,
) -> torch.Tensor:
    selected = mask.to(torch.bool)
    if not torch.any(selected):
        return mean.sum() * 0.0
    residual = (target[selected] - mean[selected]) / torch.exp(logstd[selected])
    whitened = torch.linalg.solve_triangular(
        cholesky,
        residual.T,
        upper=False,
    ).T
    return torch.mean(0.5 * torch.sum(whitened.square(), dim=-1) + torch.sum(logstd[selected], dim=-1))


def loss_bundle(
    output: V4Sequence,
    target: torch.Tensor,
    horizontal_pose: torch.Tensor,
    horizontal_onset: torch.Tensor,
    horizontal_eligible: torch.Tensor,
    horizontal_parameter_mask: torch.Tensor,
    horizontal_parameters: torch.Tensor,
    vertical_onset: torch.Tensor,
    vertical_eligible: torch.Tensor,
    vertical_parameter_mask: torch.Tensor,
    vertical_parameters: torch.Tensor,
    model: NormalHMDV4,
    horizontal_positive_weight: float,
    vertical_positive_weight: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    root = F.smooth_l1_loss(
        output.root_normalized,
        target[..., list(ROOT_INDICES)],
    )
    horizontal_continuous = F.smooth_l1_loss(
        output.horizontal_pose_normalized,
        horizontal_pose,
    )
    horizontal_hazard = masked_hazard_bce(
        output.horizontal_hazard_logit,
        horizontal_onset,
        horizontal_eligible,
        horizontal_positive_weight,
    )
    vertical_hazard = masked_hazard_bce(
        output.vertical_hazard_logit,
        vertical_onset,
        vertical_eligible,
        vertical_positive_weight,
    )
    horizontal_parameters_loss = correlated_gaussian_nll(
        output.horizontal_parameter_mean_normalized,
        output.horizontal_parameter_logstd,
        horizontal_parameters,
        horizontal_parameter_mask,
        model.horizontal_parameter_cholesky,
    )
    vertical_parameters_loss = correlated_gaussian_nll(
        output.vertical_parameter_mean_normalized,
        output.vertical_parameter_logstd,
        vertical_parameters,
        vertical_parameter_mask,
        model.vertical_parameter_cholesky,
    )
    total = (
        args.root_loss_weight * root
        + args.horizontal_continuous_loss_weight * horizontal_continuous
        + args.horizontal_hazard_loss_weight * horizontal_hazard
        + args.vertical_hazard_loss_weight * vertical_hazard
        + args.horizontal_parameter_loss_weight * horizontal_parameters_loss
        + args.vertical_parameter_loss_weight * vertical_parameters_loss
    )
    return total, {
        "root": float(root.detach()),
        "horizontal_continuous": float(horizontal_continuous.detach()),
        "horizontal_hazard": float(horizontal_hazard.detach()),
        "vertical_hazard": float(vertical_hazard.detach()),
        "horizontal_parameters": float(horizontal_parameters_loss.detach()),
        "vertical_parameters": float(vertical_parameters_loss.detach()),
    }


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.no_grad()
def evaluate(
    model: NormalHMDV4,
    loader: DataLoader,
    device: torch.device,
    horizontal_positive_weight: float,
    vertical_positive_weight: float,
    args: argparse.Namespace,
    *,
    collect_hazards: bool = False,
) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    total = 0.0
    count = 0
    collected: dict[str, list[np.ndarray]] = {
        "horizontal_logit": [],
        "horizontal_onset": [],
        "horizontal_eligible": [],
        "vertical_logit": [],
        "vertical_onset": [],
        "vertical_eligible": [],
    }
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        (
            history, target, horizontal_pose,
            horizontal_onset, horizontal_eligible, horizontal_parameter_mask,
            horizontal_parameters, vertical_onset, vertical_eligible,
            vertical_parameter_mask, vertical_parameters,
        ) = batch
        output = model.teacher_forced(history, target)
        loss, _ = loss_bundle(
            output, target, horizontal_pose,
            horizontal_onset, horizontal_eligible, horizontal_parameter_mask,
            horizontal_parameters, vertical_onset, vertical_eligible,
            vertical_parameter_mask, vertical_parameters,
            model, horizontal_positive_weight, vertical_positive_weight, args,
        )
        total += float(loss) * len(history)
        count += len(history)
        if collect_hazards:
            for name, tensor in (
                ("horizontal_logit", output.horizontal_hazard_logit),
                ("horizontal_onset", horizontal_onset),
                ("horizontal_eligible", horizontal_eligible),
                ("vertical_logit", output.vertical_hazard_logit),
                ("vertical_onset", vertical_onset),
                ("vertical_eligible", vertical_eligible),
            ):
                collected[name].append(tensor.detach().cpu().numpy())
    arrays = {
        name: np.concatenate(chunks, axis=0) if chunks else np.empty(0)
        for name, chunks in collected.items()
    }
    return total / max(count, 1), arrays


def calibrate_eligible_hazard_bias(
    logits: np.ndarray, target: np.ndarray, eligible: np.ndarray
) -> dict[str, float]:
    mask = eligible.astype(bool)
    selected_logits = np.asarray(logits, dtype=np.float64)[mask]
    selected_target = np.asarray(target, dtype=np.float64)[mask]
    if len(selected_logits) == 0:
        raise RuntimeError("hazard calibration has no eligible frames")
    target_rate = float(np.mean(selected_target))
    target_rate = float(np.clip(target_rate, 1e-8, 1.0 - 1e-8))
    low, high = -30.0, 30.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        probability = 1.0 / (1.0 + np.exp(-np.clip(selected_logits + middle, -60.0, 60.0)))
        if float(np.mean(probability)) < target_rate:
            low = middle
        else:
            high = middle
    bias = 0.5 * (low + high)
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(selected_logits + bias, -60.0, 60.0)))
    return {
        "bias": float(bias),
        "eligible_target_rate": target_rate,
        "eligible_probability_mean": float(np.mean(calibrated)),
        "eligible_brier": float(np.mean(np.square(calibrated - selected_target))),
        "eligible_frames": int(len(selected_logits)),
        "onsets": int(np.sum(selected_target)),
    }


def _synthetic_split() -> dict[str, np.ndarray]:
    length = 128
    history_steps = 16
    target_steps = 24
    state = np.zeros((length, 9), dtype=np.float64)

    def transition(index: int, start: int, stop: int, delta: float) -> None:
        phase = np.linspace(0.0, 1.0, stop - start)
        state[start:stop, index] += delta * minimum_jerk(phase)
        state[stop:, index] += delta

    # Left-censored horizontal activity, one complete H event, one overlapping
    # V event, and a right-censored H event.
    transition(3, 0, 12, 0.08)
    transition(7, 0, 12, 0.30)
    transition(3, 25, 45, 0.12)
    transition(7, 25, 45, 0.50)
    transition(4, 35, 55, 0.045)
    transition(6, 35, 55, -0.32)
    phase = minimum_jerk(np.linspace(0.0, 0.65, length - 112))
    state[112:, 3] += 0.10 * phase
    state[112:, 7] += -0.40 * phase

    window_length = history_steps + target_steps
    starts = list(range(0, length - window_length + 1, 8))
    if starts[-1] != length - window_length:
        starts.append(length - window_length)
    x = np.stack([state[start : start + history_steps] for start in starts]).astype(np.float32)
    y = np.stack(
        [state[start + history_steps : start + window_length] for start in starts]
    ).astype(np.float32)
    return {
        "X": x,
        "Y": y,
        "trajectory_index": np.zeros(len(starts), dtype=np.int32),
        "start_index": np.asarray(starts, dtype=np.int32),
        "trajectory_ids": np.asarray(["synthetic:overlap"], dtype="U32"),
    }


def _test_model() -> NormalHMDV4:
    horizontal_dim = HORIZONTAL_PARAMETER_DIM
    vertical_dim = VERTICAL_PARAMETER_DIM
    return NormalHMDV4(
        hidden_dim=8,
        state_mean=np.zeros(9),
        state_std=np.ones(9),
        horizontal_parameter_center=np.zeros(horizontal_dim),
        horizontal_parameter_scale=np.ones(horizontal_dim),
        vertical_parameter_center=np.zeros(vertical_dim),
        vertical_parameter_scale=np.ones(vertical_dim),
        horizontal_parameter_low=[-1.0] * (horizontal_dim - 1) + [math.log(5.0)],
        horizontal_parameter_high=[1.0] * (horizontal_dim - 1) + [math.log(30.0)],
        vertical_parameter_low=[-1.0] * (vertical_dim - 1) + [math.log(5.0)],
        vertical_parameter_high=[1.0] * (vertical_dim - 1) + [math.log(30.0)],
        horizontal_pose_low=[-2.0] * len(HORIZONTAL_INDICES),
        horizontal_pose_high=[2.0] * len(HORIZONTAL_INDICES),
        vertical_pose_low=[-2.0] * len(VERTICAL_INDICES),
        vertical_pose_high=[2.0] * len(VERTICAL_INDICES),
        cooldown_steps=3,
    )


def run_self_checks() -> dict[str, Any]:
    # Analytic endpoint closure: position reaches endpoints and the first two
    # derivatives of S(u)=10u^3-15u^4+6u^5 vanish at both ends.
    u = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    value = minimum_jerk_torch(u)
    first = torch.autograd.grad(value.sum(), u, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), u)[0]
    if not torch.allclose(value, torch.tensor([0.0, 1.0], dtype=value.dtype), atol=1e-12):
        raise AssertionError("minimum-jerk primitive does not reach both endpoints")
    if not torch.allclose(first, torch.zeros_like(first), atol=1e-12):
        raise AssertionError("minimum-jerk endpoint velocity is not zero")
    if not torch.allclose(second, torch.zeros_like(second), atol=1e-12):
        raise AssertionError("minimum-jerk endpoint acceleration is not zero")

    split = _synthetic_split()
    fragments = reconstruct_trajectories(split)
    thresholds = {
        "lateral_sway_speed_mps": {"high": 0.08, "low": 0.04},
        "vertical_bob_speed_mps": {"high": 0.025, "low": 0.0125},
        "look_turn_rate_radps": {"high": 0.30, "low": 0.15},
        "nod_rate_radps": {"high": 0.15, "low": 0.075},
    }
    labelled = label_trajectories(
        fragments, thresholds, dt_seconds=DEFAULT_DT, cooldown_steps=4
    )
    targets = map_labels_to_windows(split, labelled)
    consistency = assert_onset_parameter_consistency(targets)
    if consistency["horizontal_duplicate_onset_groups"] < 1:
        raise AssertionError("synthetic windows did not duplicate a horizontal onset")
    if consistency["vertical_duplicate_onset_groups"] < 1:
        raise AssertionError("synthetic windows did not duplicate a vertical onset")
    overlap_frames = sum(
        int(np.sum(item.horizontal.active & item.vertical.active)) for item in labelled
    )
    if overlap_frames < 1:
        raise AssertionError("horizontal and vertical labels were made mutually exclusive")
    boundary_events = 0
    for item in labelled:
        for branch in (item.horizontal, item.vertical):
            for event_id in np.unique(branch.event_id[branch.event_id >= 0]):
                locations = np.flatnonzero(branch.event_id == event_id)
                if locations[0] == 0 or locations[-1] == len(branch.active) - 1:
                    boundary_events += 1
                    if np.any(branch.parameter_mask[locations]):
                        raise AssertionError("boundary-censored event supervises parameters")
    if boundary_events < 1:
        raise AssertionError("synthetic censorship fixture produced no boundary event")

    # Exercise both controllers at the same physical frame.  Change neural
    # parameter proposals during the event and verify the latched targets do
    # not change; after completion the pose must remain at the target.
    model = _test_model().eval()
    previous = torch.zeros((1, 9))
    previous[:, list(HORIZONTAL_INDICES)] = torch.tensor(
        [[0.30, -0.20, 0.40, -0.10]]
    )
    previous[:, list(VERTICAL_INDICES)] = torch.tensor([[0.50, -0.30]])
    initial_pose = previous.clone()
    runtime = model.initial_runtime_state(previous)
    generator = model.seeded_generator(previous.device, 20261012)
    horizontal_parameters = torch.tensor(
        [[0.10, -0.04, 0.25, 0.03, math.log(7.0)]], dtype=torch.float32
    )
    vertical_parameters = torch.tensor(
        [[0.025, -0.12, math.log(7.0)]], dtype=torch.float32
    )
    triggers: list[tuple[bool, bool]] = []
    states: list[torch.Tensor] = []
    latched_targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(12):
        step = V4Step(
            root_normalized=torch.zeros((1, 3)),
            horizontal_pose_normalized=torch.full((1, 4), 99.0),
            horizontal_hazard_logit=torch.full((1,), -100.0),
            horizontal_parameter_mean_normalized=torch.full((1, HORIZONTAL_PARAMETER_DIM), float(index)),
            horizontal_parameter_logstd=torch.full((1, HORIZONTAL_PARAMETER_DIM), -5.0),
            vertical_hazard_logit=torch.full((1,), -100.0),
            vertical_parameter_mean_normalized=torch.full((1, VERTICAL_PARAMETER_DIM), -float(index)),
            vertical_parameter_logstd=torch.full((1, VERTICAL_PARAMETER_DIM), -5.0),
        )
        state, runtime, diagnostic = model.runtime_step(
            previous,
            step,
            runtime,
            generator,
            horizontal_trigger_override=torch.tensor([index == 0]),
            vertical_trigger_override=torch.tensor([index == 0]),
            horizontal_parameter_override_physical=(
                horizontal_parameters if index == 0 else None
            ),
            vertical_parameter_override_physical=(
                vertical_parameters if index == 0 else None
            ),
        )
        triggers.append(
            (bool(diagnostic.horizontal_trigger[0]), bool(diagnostic.vertical_trigger[0]))
        )
        states.append(state.clone())
        latched_targets.append(
            (
                runtime.horizontal.target_pose_physical.clone(),
                runtime.vertical.target_pose_physical.clone(),
            )
        )
        previous = state
    if triggers[0] != (True, True):
        raise AssertionError("independent H/V controllers could not trigger simultaneously")
    if any(pair != (False, False) for pair in triggers[1:]):
        raise AssertionError("controller retriggered during an event/cooldown")
    for horizontal_target, vertical_target in latched_targets[1:]:
        if not torch.equal(horizontal_target, latched_targets[0][0]) or not torch.equal(
            vertical_target, latched_targets[0][1]
        ):
            raise AssertionError("event parameter was not latched for its full duration")
    expected = torch.zeros((1, 9))
    expected[:, list(HORIZONTAL_INDICES)] = horizontal_parameters[:, :-1]
    expected[:, list(VERTICAL_INDICES)] = vertical_parameters[:, :-1]
    expected_horizontal_amplitude = (
        horizontal_parameters[:, :-1]
        - initial_pose[:, list(HORIZONTAL_INDICES)]
    )
    expected_vertical_amplitude = (
        vertical_parameters[:, :-1]
        - initial_pose[:, list(VERTICAL_INDICES)]
    )
    if not torch.allclose(
        runtime.horizontal.amplitude_physical,
        expected_horizontal_amplitude,
        atol=1e-7,
    ) or not torch.allclose(
        runtime.vertical.amplitude_physical,
        expected_vertical_amplitude,
        atol=1e-7,
    ):
        raise AssertionError("latched signed amplitude is not absolute target minus start")
    if not torch.allclose(states[6], expected, atol=1e-6):
        raise AssertionError("holding transition did not reach sampled absolute target")
    if not all(torch.equal(state, states[6]) for state in states[7:10]):
        raise AssertionError("completed event target was not held during cooldown")

    return {
        "status": "passed",
        "primitive_endpoint_value": value.detach().tolist(),
        "primitive_endpoint_velocity": first.detach().tolist(),
        "primitive_endpoint_acceleration": second.detach().tolist(),
        "label_consistency": consistency,
        "horizontal_vertical_overlap_frames": overlap_frames,
        "boundary_censored_events_checked": boundary_events,
        "simultaneous_trigger": True,
        "latched_parameter_and_hold": True,
        "absolute_endpoint_from_nonzero_start": True,
        "signed_amplitude_equals_endpoint_minus_start": True,
        "headlocal_per_frame_noise": False,
    }


@torch.no_grad()
def closed_loop_probe(
    model: NormalHMDV4,
    history_physical: np.ndarray,
    *,
    seconds: float,
    dt_seconds: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run a seeded rollout and report boundary dwell and 10-second drift."""

    steps = max(int(math.ceil(seconds / dt_seconds)), 1)
    history = torch.from_numpy(
        (
            (history_physical.astype(np.float32) - model.state_mean.cpu().numpy())
            / model.state_std.cpu().numpy()
        )[None]
    ).to(device)
    normalized, diagnostics = model.runtime_rollout(
        history, steps, seed=seed, root_noise_std_normalized=0.0
    )
    physical = (
        normalized * model.state_std[None, None] + model.state_mean[None, None]
    )[0].cpu().numpy()
    head_indices = tuple(HORIZONTAL_INDICES) + tuple(VERTICAL_INDICES)
    horizontal_bounds = {
        index: (
            float(model.horizontal_pose_low[offset]),
            float(model.horizontal_pose_high[offset]),
        )
        for offset, index in enumerate(HORIZONTAL_INDICES)
    }
    vertical_bounds = {
        index: (
            float(model.vertical_pose_low[offset]),
            float(model.vertical_pose_high[offset]),
        )
        for offset, index in enumerate(VERTICAL_INDICES)
    }
    bounds = {**horizontal_bounds, **vertical_bounds}
    boundary: dict[str, dict[str, float]] = {}
    for index in head_indices:
        low, high = bounds[index]
        margin = 0.05 * (high - low)
        values = physical[:, index]
        boundary[STATE_FEATURE_NAMES[index]] = {
            "low": low,
            "high": high,
            "near_boundary_fraction": float(
                np.mean((values <= low + margin) | (values >= high - margin))
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    ten_second_steps = min(steps, max(int(round(10.0 / dt_seconds)), 1))
    drift = {
        STATE_FEATURE_NAMES[index]: float(
            np.mean(physical[-ten_second_steps:, index])
            - np.mean(physical[:ten_second_steps, index])
        )
        for index in head_indices
    }
    horizontal_triggers = np.asarray(
        [bool(item.horizontal_trigger[0]) for item in diagnostics]
    )
    vertical_triggers = np.asarray(
        [bool(item.vertical_trigger[0]) for item in diagnostics]
    )
    return {
        "seconds_requested": float(seconds),
        "steps": steps,
        "seconds_generated": steps * dt_seconds,
        "seed": int(seed),
        "headlocal_near_safety_boundary": boundary,
        "headlocal_last_10s_minus_first_10s": drift,
        "horizontal_event_count": int(horizontal_triggers.sum()),
        "vertical_event_count": int(vertical_triggers.sum()),
        "simultaneous_event_count": int(np.sum(horizontal_triggers & vertical_triggers)),
        "headlocal_randomness_contract": (
            "seeded Bernoulli plus one bounded endpoint/duration sample per onset; "
            "no per-frame HeadLocal noise"
        ),
    }


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    self_checks = run_self_checks()

    with args.data_report.open("r", encoding="utf-8") as stream:
        data_report = json.load(stream)
    if data_report.get("status") != "success" or data_report.get("synthetic") is not False:
        raise ValueError("v4 requires the successful, non-synthetic frozen v2 data report")
    prepared_split_sha256: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = args.data_dir / f"normal_hmd_{name}.npz"
        expected = str(data_report.get("splits", {}).get(name, {}).get("sha256", ""))
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"{name} NPZ hash does not match frozen v2 report: "
                f"expected={expected!r}, actual={actual!r}"
            )
        prepared_split_sha256[name] = actual
    dt_seconds = float(data_report.get("fixed_dt_seconds", DEFAULT_DT))
    if not math.isclose(dt_seconds, DEFAULT_DT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"v4 frozen event definition requires dt={DEFAULT_DT}, got {dt_seconds}"
        )
    splits = {name: load_split(args.data_dir, name) for name in ("train", "val", "test")}
    reconstructed = {
        name: reconstruct_trajectories(split) for name, split in splits.items()
    }
    thresholds = learn_event_thresholds(reconstructed["train"], dt_seconds)
    labelled = {
        name: label_trajectories(
            fragments,
            thresholds,
            dt_seconds=dt_seconds,
            cooldown_steps=args.cooldown_steps,
        )
        for name, fragments in reconstructed.items()
    }
    targets = {
        name: map_labels_to_windows(splits[name], labelled[name])
        for name in ("train", "val", "test")
    }
    consistency = {
        name: assert_onset_parameter_consistency(target)
        for name, target in targets.items()
    }

    state_mean, state_std = fit_state_normalization(splits["train"])
    horizontal_events = collect_event_parameters(labelled["train"], "horizontal")
    vertical_events = collect_event_parameters(labelled["train"], "vertical")
    horizontal_stats = parameter_statistics(
        horizontal_events,
        minimum_duration_steps=args.minimum_duration_steps,
        maximum_duration_steps=args.maximum_duration_steps,
    )
    vertical_stats = parameter_statistics(
        vertical_events,
        minimum_duration_steps=args.minimum_duration_steps,
        maximum_duration_steps=args.maximum_duration_steps,
    )
    horizontal_pose = collect_pose_values(labelled["train"], HORIZONTAL_INDICES)
    vertical_pose = collect_pose_values(labelled["train"], VERTICAL_INDICES)
    horizontal_pose_low, horizontal_pose_high = np.quantile(
        horizontal_pose, (0.01, 0.99), axis=0
    )
    vertical_pose_low, vertical_pose_high = np.quantile(
        vertical_pose, (0.01, 0.99), axis=0
    )

    model = NormalHMDV4(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        state_mean=state_mean,
        state_std=state_std,
        horizontal_parameter_center=horizontal_stats["center"],
        horizontal_parameter_scale=horizontal_stats["scale"],
        vertical_parameter_center=vertical_stats["center"],
        vertical_parameter_scale=vertical_stats["scale"],
        horizontal_parameter_low=horizontal_stats["low"],
        horizontal_parameter_high=horizontal_stats["high"],
        vertical_parameter_low=vertical_stats["low"],
        vertical_parameter_high=vertical_stats["high"],
        horizontal_pose_low=horizontal_pose_low,
        horizontal_pose_high=horizontal_pose_high,
        vertical_pose_low=vertical_pose_low,
        vertical_pose_high=vertical_pose_high,
        horizontal_parameter_correlation=horizontal_stats["correlation"],
        vertical_parameter_correlation=vertical_stats["correlation"],
        cooldown_steps=args.cooldown_steps,
        minimum_duration_steps=args.minimum_duration_steps,
        maximum_duration_steps=args.maximum_duration_steps,
    )
    device = choose_device(args.device)
    model.to(device)
    use_amp = bool(args.amp and device.type == "cuda")

    maxima = {
        "train": args.max_train_segments,
        "val": args.max_val_segments,
        "test": args.max_test_segments,
    }
    indices = {
        name: window_indices(len(split["X"]), maxima[name], args.seed + offset)
        for offset, (name, split) in enumerate(splits.items())
    }
    datasets = {
        name: make_dataset(
            splits[name], targets[name], indices[name], state_mean, state_std,
            horizontal_stats, vertical_stats,
        )
        for name in ("train", "val", "test")
    }
    loaders = {
        name: make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    horizontal_positive_weight = eligible_positive_weight(
        targets["train"], "horizontal"
    )
    vertical_positive_weight = eligible_positive_weight(targets["train"], "vertical")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for raw_batch in loaders["train"]:
            batch = move_batch(raw_batch, device)
            (
                history_state, target, horizontal_pose_target,
                horizontal_onset, horizontal_eligible, horizontal_parameter_mask,
                horizontal_parameters, vertical_onset, vertical_eligible,
                vertical_parameter_mask, vertical_parameters,
            ) = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model.teacher_forced(history_state, target)
                loss, _ = loss_bundle(
                    output, target, horizontal_pose_target,
                    horizontal_onset, horizontal_eligible, horizontal_parameter_mask,
                    horizontal_parameters, vertical_onset, vertical_eligible,
                    vertical_parameter_mask, vertical_parameters,
                    model, horizontal_positive_weight, vertical_positive_weight, args,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(history_state)
            count += len(history_state)
        train_loss = loss_sum / max(count, 1)
        val_loss, _ = evaluate(
            model, loaders["val"], device,
            horizontal_positive_weight, vertical_positive_weight, args,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    _, validation_arrays = evaluate(
        model, loaders["val"], device,
        horizontal_positive_weight, vertical_positive_weight, args,
        collect_hazards=True,
    )
    horizontal_calibration = calibrate_eligible_hazard_bias(
        validation_arrays["horizontal_logit"],
        validation_arrays["horizontal_onset"],
        validation_arrays["horizontal_eligible"],
    )
    vertical_calibration = calibrate_eligible_hazard_bias(
        validation_arrays["vertical_logit"],
        validation_arrays["vertical_onset"],
        validation_arrays["vertical_eligible"],
    )
    model.horizontal_hazard_bias.fill_(horizontal_calibration["bias"])
    model.vertical_hazard_bias.fill_(vertical_calibration["bias"])
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    test_loss, _ = evaluate(
        model, loaders["test"], device,
        horizontal_positive_weight, vertical_positive_weight, args,
    )
    probe = closed_loop_probe(
        model,
        splits["val"]["X"][int(indices["val"][0])],
        seconds=args.probe_seconds,
        dt_seconds=dt_seconds,
        seed=args.seed + 7001,
        device=device,
    )
    model_config = {
        "architecture": "independent_root_horizontal_vertical_lstm_holding_events",
        "input_dim": 9,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "root_indices": list(ROOT_INDICES),
        "horizontal_indices": list(HORIZONTAL_INDICES),
        "vertical_indices": list(VERTICAL_INDICES),
        "horizontal_parameter_layout": [
            "absolute_end_x_m", "absolute_end_z_m", "absolute_end_yaw_rad",
            "absolute_end_roll_rad", "log_duration_steps",
        ],
        "vertical_parameter_layout": [
            "absolute_end_y_m", "absolute_end_pitch_rad", "log_duration_steps",
        ],
        "event_synthesis": "holding minimum-jerk S(u)=10u^3-15u^4+6u^5",
        "runtime_trigger": "seeded Bernoulli hazard on eligible frames only",
        "cooldown_steps": args.cooldown_steps,
        "duration_bounds_steps": [
            args.minimum_duration_steps, args.maximum_duration_steps
        ],
        "headlocal_noise": "none; Gaussian endpoint/duration is sampled once per onset; signed amplitude=end-current is latched",
    }
    checkpoint = {
        "schema_version": "4.0.0",
        "state_dict": best_state,
        "model_config": model_config,
        "normalization": {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "horizontal_parameter_center": horizontal_stats["center"].tolist(),
            "horizontal_parameter_scale": horizontal_stats["scale"].tolist(),
            "vertical_parameter_center": vertical_stats["center"].tolist(),
            "vertical_parameter_scale": vertical_stats["scale"].tolist(),
        },
        "bounds": {
            "horizontal_parameter_p05_p95": [
                horizontal_stats["low"].tolist(), horizontal_stats["high"].tolist()
            ],
            "vertical_parameter_p05_p95": [
                vertical_stats["low"].tolist(), vertical_stats["high"].tolist()
            ],
            "horizontal_absolute_pose_p01_p99": [
                horizontal_pose_low.tolist(), horizontal_pose_high.tolist()
            ],
            "vertical_absolute_pose_p01_p99": [
                vertical_pose_low.tolist(), vertical_pose_high.tolist()
            ],
        },
        "parameter_correlation": {
            "horizontal": horizontal_stats["correlation"].tolist(),
            "vertical": vertical_stats["correlation"].tolist(),
        },
        "hazard_calibration": {
            "horizontal": horizontal_calibration,
            "vertical": vertical_calibration,
        },
        "event_definition": {
            "name": "normal_hmd_event_axis_analysis_sg_hysteresis_v1",
            "semantics": "offline non-causal pseudo-label; centered SG has future context",
            "dt_seconds": dt_seconds,
            "savgol_window_frames": 9,
            "savgol_polyorder": 2,
            "high_quantile_train_only": 0.80,
            "low_threshold_ratio": 0.50,
            "minimum_event_frames": 5,
            "merge_gap_frames": 4,
            "thresholds": thresholds,
            "horizontal_union": list(("head_local_x", "yaw")),
            "vertical_union": list(("head_local_y", "pitch")),
            "branches_may_overlap": True,
            "parameter_censorship": "no amplitude/duration supervision if event touches fragment boundary",
        },
        "dataset_fingerprint_sha256": data_report.get("dataset_fingerprint_sha256"),
        "prepared_split_sha256": prepared_split_sha256,
        "data_report_sha256": sha256_file(args.data_report),
        "seed": args.seed,
    }
    write_checkpoint(args.model, checkpoint)
    model_hash = sha256_file(args.model)
    metrics = {
        "schema_version": "4.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "smoke": bool(args.smoke),
        "device": str(device),
        "amp": use_amp,
        "parameter_count": parameter_count(model),
        "epochs_completed": len(history),
        "best_val_loss": best_loss,
        "test_loss": test_loss,
        "segments_used": {name: len(value) for name, value in indices.items()},
        "trajectory_fragments": {
            name: len(value) for name, value in reconstructed.items()
        },
        "prepared_split_sha256": prepared_split_sha256,
        "label_consistency": consistency,
        "event_summary": {
            name: event_summary(value, dt_seconds) for name, value in labelled.items()
        },
        "self_checks": self_checks,
        "hazard_positive_weight": {
            "horizontal": horizontal_positive_weight,
            "vertical": vertical_positive_weight,
        },
        "hazard_calibration": checkpoint["hazard_calibration"],
        "closed_loop_probe": probe,
        "model_config": model_config,
        "training_elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "model": str(args.model.resolve()),
            "model_sha256": model_hash,
            "metadata": str(args.metadata.resolve()),
            "metrics": str(args.metrics.resolve()),
            "history": str(args.history_output.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    metadata = copy.deepcopy(metrics)
    metadata["event_definition"] = checkpoint["event_definition"]
    metadata["bounds"] = checkpoint["bounds"]
    metadata["normalization"] = checkpoint["normalization"]
    metadata["parameter_correlation"] = checkpoint["parameter_correlation"]
    metadata["dataset_fingerprint_sha256"] = checkpoint["dataset_fingerprint_sha256"]
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.history_output, {"epochs": history})
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--cooldown-steps", type=int, default=4)
    parser.add_argument("--minimum-duration-steps", type=int, default=5)
    parser.add_argument("--maximum-duration-steps", type=int, default=148)
    parser.add_argument("--root-loss-weight", type=float, default=1.0)
    parser.add_argument("--horizontal-continuous-loss-weight", type=float, default=0.5)
    parser.add_argument("--horizontal-hazard-loss-weight", type=float, default=0.4)
    parser.add_argument("--vertical-hazard-loss-weight", type=float, default=0.4)
    parser.add_argument("--horizontal-parameter-loss-weight", type=float, default=0.3)
    parser.add_argument("--vertical-parameter-loss-weight", type=float, default=0.3)
    parser.add_argument("--max-train-segments", type=int, default=0)
    parser.add_argument("--max-val-segments", type=int, default=0)
    parser.add_argument("--max-test-segments", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--probe-seconds", type=float, default=60.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    return parser


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.epochs = 1
    args.patience = 1
    args.batch_size = min(args.batch_size, 16)
    args.hidden_dim = min(args.hidden_dim, 24)
    args.num_layers = 1
    args.dropout = 0.0
    args.max_train_segments = args.max_train_segments or 48
    args.max_val_segments = args.max_val_segments or 24
    args.max_test_segments = args.max_test_segments or 24
    args.device = "cpu" if args.device == "auto" else args.device
    if args.model == DEFAULT_MODEL:
        args.model = SMOKE_DIR / "normal_hmd_v4.pth"
    if args.metadata == DEFAULT_METADATA:
        args.metadata = SMOKE_DIR / "metadata.json"
    if args.metrics == DEFAULT_METRICS:
        args.metrics = SMOKE_DIR / "metrics.json"
    if args.history_output == DEFAULT_HISTORY:
        args.history_output = SMOKE_DIR / "history.json"


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, batch size, and hidden dimension must be positive")
    if args.cooldown_steps < 0 or args.minimum_duration_steps < 2:
        raise ValueError("invalid cooldown or minimum event duration")
    if args.maximum_duration_steps < args.minimum_duration_steps:
        raise ValueError("maximum duration must not precede minimum duration")
    if args.probe_seconds <= 0.0:
        raise ValueError("probe seconds must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    apply_smoke_defaults(args)
    try:
        validate_args(args)
        if args.self_test_only:
            result = run_self_checks()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        metrics = train(args)
    except Exception as exc:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.metrics,
            {
                "schema_version": "4.0.0",
                "generated_at_utc": utc_now(),
                "status": "failed",
                "reason": str(exc),
                "model_written": False,
            },
        )
        print(f"Normal HMD v4 training failed: {exc}", file=sys.stderr)
        return 1
    print("Normal HMD v4 training complete")
    print(f"  smoke: {metrics['smoke']}")
    print(f"  self checks: {metrics['self_checks']['status']}")
    print(f"  test loss: {metrics['test_loss']:.6f}")
    print(f"  model: {metrics['artifacts']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
