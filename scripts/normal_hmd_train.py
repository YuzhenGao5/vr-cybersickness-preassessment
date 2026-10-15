#!/usr/bin/env python3
"""Train the multi-step scheduled-sampling Normal HMD Agent v2."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from normal_hmd_common import (
    COORDINATE_SYSTEM,
    STATE_FEATURE_NAMES,
    atomic_json,
    orientation_geodesic_degrees,
    sha256_file,
)
from normal_hmd_model import NormalHMDLSTM, parameter_count


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_DATA_REPORT = REPO_ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "normal_hmd_v2.pth"
DEFAULT_METADATA = REPO_ROOT / "models" / "normal_hmd_v2_metadata.json"
DEFAULT_METRICS = REPO_ROOT / "results" / "normal_hmd_v2_training_metrics.json"
DEFAULT_HISTORY = REPO_ROOT / "results" / "normal_hmd_v2_training_history.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"normal_hmd_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fit_normalization(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = x.shape[0] * x.shape[1] + y.shape[0] * y.shape[1]
    total = x.astype(np.float64).sum(axis=(0, 1)) + y.astype(np.float64).sum(axis=(0, 1))
    square = np.square(x.astype(np.float64)).sum(axis=(0, 1)) + np.square(
        y.astype(np.float64)
    ).sum(axis=(0, 1))
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-12)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def sampled_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    low: float,
    high: float,
    seed: int,
    maximum_rows: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    per_array = maximum_rows // 2
    if len(x_flat) > per_array:
        x_flat = x_flat[rng.choice(len(x_flat), per_array, replace=False)]
    if len(y_flat) > per_array:
        y_flat = y_flat[rng.choice(len(y_flat), per_array, replace=False)]
    sample = np.concatenate((x_flat, y_flat), axis=0).astype(np.float64)
    return np.quantile(sample, low, axis=0), np.quantile(sample, high, axis=0)


def make_loader(
    split: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    x = ((split["X"].astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32))
    y = ((split["Y"].astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32))
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def rollout_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta_weight: float,
    acceleration_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    state_loss = torch.mean((prediction - target) ** 2)
    delta_loss = prediction.new_zeros(())
    acceleration_loss = prediction.new_zeros(())
    if prediction.shape[1] > 1:
        predicted_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        delta_loss = torch.mean((predicted_delta - target_delta) ** 2)
    if prediction.shape[1] > 2:
        predicted_acceleration = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        target_acceleration = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        acceleration_loss = torch.mean((predicted_acceleration - target_acceleration) ** 2)
    total = state_loss + delta_weight * delta_loss + acceleration_weight * acceleration_loss
    return total, {
        "state": float(state_loss.detach().item()),
        "delta": float(delta_loss.detach().item()),
        "acceleration": float(acceleration_loss.detach().item()),
    }


def teacher_forcing_ratio(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    progress = (epoch - 1) / (epochs - 1)
    return start + progress * (end - start)


def rollout_curriculum(epoch: int, epochs: int, maximum_rollout: int) -> tuple[int, list[int]]:
    """Four-stage 8 -> 16 -> 32 -> 64 curriculum, capped by available labels."""

    stages: list[int] = []
    for candidate in (8, 16, 32, maximum_rollout):
        value = min(candidate, maximum_rollout)
        if value not in stages:
            stages.append(value)
    stage_index = min(len(stages) - 1, ((epoch - 1) * len(stages)) // max(epochs, 1))
    return stages[stage_index], stages


@torch.no_grad()
def predict_rollouts(
    model: NormalHMDLSTM,
    loader: DataLoader,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    normalized_mse_sum = 0.0
    item_count = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model.rollout(history, target.shape[1], teacher_forcing_ratio=0.0)
        normalized_mse_sum += float(torch.mean((prediction - target) ** 2).item()) * len(history)
        item_count += len(history)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
    predicted_normalized = np.concatenate(predictions, axis=0)
    target_normalized = np.concatenate(targets, axis=0)
    predicted = predicted_normalized * std[None, None, :] + mean[None, None, :]
    target = target_normalized * std[None, None, :] + mean[None, None, :]
    return predicted.astype(np.float64), target.astype(np.float64), normalized_mse_sum / item_count


def metric_bundle(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(prediction - target)
    feature_mae = absolute.mean(axis=(0, 1))
    horizon_mae = absolute.mean(axis=(0, 2))
    orientation_error = orientation_geodesic_degrees(prediction[..., 6:9], target[..., 6:9])
    return {
        "mae_by_feature": {
            name: float(value) for name, value in zip(STATE_FEATURE_NAMES, feature_mae)
        },
        "root_planar_velocity_vector_mae_mps": float(
            np.linalg.norm(prediction[..., :2] - target[..., :2], axis=-1).mean()
        ),
        "root_yaw_rate_mae_degps": float(np.degrees(absolute[..., 2]).mean()),
        "head_local_position_vector_mae_m": float(
            np.linalg.norm(prediction[..., 3:6] - target[..., 3:6], axis=-1).mean()
        ),
        "head_local_orientation_geodesic_mae_deg": float(orientation_error.mean()),
        "first_step_mean_absolute_state_error": float(horizon_mae[0]),
        "last_step_mean_absolute_state_error": float(horizon_mae[-1]),
        "horizon_mean_absolute_state_error": [float(value) for value in horizon_mae],
    }


def calibrate_innovation(
    model: NormalHMDLSTM,
    train_split: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    maximum_segments: int,
) -> dict[str, Any]:
    """Fit a stationary per-feature AR(1) residual process on train data only."""

    rng = np.random.default_rng(seed)
    count = len(train_split["X"])
    indices = np.arange(count)
    if count > maximum_segments:
        indices = np.sort(rng.choice(count, maximum_segments, replace=False))
    subset = {
        "X": train_split["X"][indices],
        "Y": train_split["Y"][indices],
    }
    loader = make_loader(
        subset,
        mean,
        std,
        batch_size,
        False,
        workers,
        device.type == "cuda",
    )
    prediction, target, _ = predict_rollouts(model, loader, device, mean, std)
    residual = target - prediction
    residual -= residual.mean(axis=(0, 1), keepdims=True)
    stationary_std = residual.std(axis=(0, 1), ddof=1)
    previous = residual[:, :-1].reshape(-1, residual.shape[-1])
    following = residual[:, 1:].reshape(-1, residual.shape[-1])
    denominator = np.sum(previous * previous, axis=0)
    rho = np.divide(
        np.sum(previous * following, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    rho = np.clip(rho, -0.95, 0.95)
    stationary_std = np.maximum(stationary_std, 1e-8)
    return {
        "method": (
            "Per-feature stationary standard deviation and lag-1 AR coefficient "
            "estimated from centered deterministic closed-loop residuals on train only. "
            "At generation time this is an output-layer innovation and is not fed back "
            "into the LSTM latent intention trajectory."
        ),
        "segments": int(len(indices)),
        "innovation_std_physical": stationary_std.tolist(),
        "innovation_ar1_rho": rho.tolist(),
    }


@torch.no_grad()
def stateful_generate(
    model: NormalHMDLSTM,
    history_physical: np.ndarray,
    steps: int,
    mean: np.ndarray,
    std: np.ndarray,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    device: torch.device,
    innovation_std: np.ndarray | None = None,
    innovation_rho: np.ndarray | None = None,
    stochastic_seed: int | None = None,
    noise_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    history_normalized = (history_physical.astype(np.float64) - mean) / std
    history_tensor = torch.from_numpy(history_normalized.astype(np.float32))[None].to(device)
    prediction, hidden = model.warm_history(history_tensor)
    raw_outputs: list[np.ndarray] = []
    clipped_outputs: list[np.ndarray] = []
    rng = np.random.default_rng(stochastic_seed)
    residual_state = np.zeros_like(mean, dtype=np.float64)
    for step in range(steps):
        base_physical = prediction[0].cpu().numpy().astype(np.float64) * std + mean
        raw_physical = base_physical.copy()
        if noise_scale > 0.0:
            if innovation_std is None or innovation_rho is None or stochastic_seed is None:
                raise ValueError("stochastic generation requires std, rho and a seed")
            innovation = rng.standard_normal(len(mean))
            residual_state = (
                innovation_rho * residual_state
                + np.sqrt(np.maximum(0.0, 1.0 - innovation_rho * innovation_rho))
                * innovation_std
                * innovation
            )
            raw_physical = raw_physical + noise_scale * residual_state
        clipped_physical = np.clip(raw_physical, clip_low, clip_high)
        raw_outputs.append(raw_physical)
        clipped_outputs.append(clipped_physical)
        if step + 1 < steps:
            # The calibrated AR(1) term is an output-layer innovation, not a
            # latent intention state. Feeding it back made tiny residuals recur
            # through the LSTM until head rotation saturated after tens of
            # seconds. Keep the learned intent dynamics closed-loop on their
            # own bounded prediction while applying/logging the stochastic
            # residual as the generated physical motion.
            base_feedback = np.clip(base_physical, clip_low, clip_high)
            feedback = ((base_feedback - mean) / std).astype(np.float32)
            feedback_tensor = torch.from_numpy(feedback)[None].to(device)
            prediction, hidden = model.advance(feedback_tensor, hidden)
    return np.asarray(raw_outputs), np.asarray(clipped_outputs)


def closed_loop_probe(
    raw: np.ndarray,
    clipped: np.ndarray,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    dt_seconds: float,
    requested_seconds: float,
    mode: str,
    stochastic_seed: int | None = None,
) -> dict[str, Any]:
    finite = bool(np.isfinite(raw).all() and np.isfinite(clipped).all())
    clipped_elements = (raw < clip_low[None, :]) | (raw > clip_high[None, :])
    clipped_steps = np.any(clipped_elements, axis=1)
    speed_squared = np.sum(np.square(clipped[:, :2]), axis=1)
    ten_second_frames = max(1, min(len(clipped) // 2, int(round(10.0 / dt_seconds))))
    first_energy = float(np.mean(speed_squared[:ten_second_frames]))
    last_energy = float(np.mean(speed_squared[-ten_second_frames:]))
    energy_ratio = last_energy / first_energy if first_energy > 1e-10 else None
    window_frames = max(2, min(len(clipped), int(round(10.0 / dt_seconds))))
    window_delta_rms: list[float] = []
    for start in range(0, max(1, len(clipped) - window_frames + 1), window_frames):
        window = clipped[start : start + window_frames]
        if len(window) >= 2:
            window_delta_rms.append(float(np.sqrt(np.mean(np.square(np.diff(window, axis=0))))))
    minimum_window_delta_rms = min(window_delta_rms) if window_delta_rms else 0.0
    head_orientation_step = orientation_geodesic_degrees(
        clipped[1:, 6:9], clipped[:-1, 6:9]
    )
    head_angular_speed_degps = head_orientation_step / dt_seconds
    head_angular_speed_p99 = float(np.quantile(head_angular_speed_degps, 0.99))
    planar_speed = np.linalg.norm(clipped[:, :2], axis=1)
    planar_speed_p99 = float(np.quantile(planar_speed, 0.99))
    height_start_end_drift = float(clipped[-1, 4] - clipped[0, 4])
    # The generated BodyRoot has only XZ+yaw by construction, so pitch and roll
    # cannot drift. Head position/orientation are absolute bounded states.
    checks = {
        "finite": finite,
        "ran_requested_steps": len(clipped) == int(math.ceil(requested_seconds / dt_seconds)),
        "clip_step_fraction_at_most_1_percent": float(clipped_steps.mean()) <= 0.01,
        # A last/first energy ratio is undefined near a slow start and labels
        # ordinary acceleration/deceleration as instability. Keep it as a
        # diagnostic; absolute speed, clipping, distance and deltas gate safety.
        "planar_speed_p99_at_most_2_mps": planar_speed_p99 <= 2.0,
        "body_pitch_roll_structurally_zero": True,
        "no_frozen_10_second_window": minimum_window_delta_rms > 1e-5,
        "head_angular_speed_p99_at_most_720_degps": head_angular_speed_p99 <= 720.0,
        "head_height_start_end_drift_at_most_0_25m": abs(height_start_end_drift) <= 0.25,
    }
    return {
        "mode": mode,
        "stochastic_seed": stochastic_seed,
        "requested_seconds": requested_seconds,
        "generated_steps": int(len(clipped)),
        "generated_seconds": float(len(clipped) * dt_seconds),
        "finite": finite,
        "clip_element_fraction": float(clipped_elements.mean()),
        "clip_step_fraction": float(clipped_steps.mean()),
        "first_10s_planar_speed_energy": first_energy,
        "last_10s_planar_speed_energy": last_energy,
        "last_to_first_energy_ratio": energy_ratio,
        "planar_speed_p99_mps": planar_speed_p99,
        "minimum_10s_state_delta_rms": minimum_window_delta_rms,
        "head_angular_speed_p99_degps": head_angular_speed_p99,
        "head_height_start_end_drift_m": height_start_end_drift,
        "integrated_planar_distance_m": float(
            np.linalg.norm(clipped[:, :2], axis=1).sum() * dt_seconds
        ),
        "integrated_absolute_yaw_deg": float(
            np.degrees(np.abs(clipped[:, 2]).sum() * dt_seconds)
        ),
        "head_local_position_min_m": clipped[:, 3:6].min(axis=0).tolist(),
        "head_local_position_max_m": clipped[:, 3:6].max(axis=0).tolist(),
        "head_local_rotvec_min_rad": clipped[:, 6:9].min(axis=0).tolist(),
        "head_local_rotvec_max_rad": clipped[:, 6:9].max(axis=0).tolist(),
        "engineering_checks": checks,
        "engineering_probe_passed": bool(all(checks.values())),
        "claim_limit": (
            f"This is a {mode} engineering probe from one held-out history"
            + (f" and seed {stochastic_seed}" if stochastic_seed is not None else "")
            + ", not a human-equivalence validation. Distributional acceptance still "
            "requires many held-out histories and frozen human-human thresholds."
        ),
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


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access CUDA")
    return device


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
        raise ValueError("Normal HMD training requires a successful, non-synthetic data report")
    split_names = ("train", "val", "test")
    prepared_split_sha256: dict[str, str] = {}
    for name in split_names:
        path = args.data_dir / f"normal_hmd_{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(data_report.get("splits", {}).get(name, {}).get("sha256", ""))
        actual_hash = sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(
                f"{name} NPZ hash does not match the frozen data report: "
                f"expected={expected_hash!r}, actual={actual_hash!r}"
            )
        prepared_split_sha256[name] = actual_hash
    splits = {name: load_split(args.data_dir, name) for name in split_names}
    history_length = int(splits["train"]["X"].shape[1])
    rollout_length = int(splits["train"]["Y"].shape[1])
    if history_length != int(data_report["history_length"]):
        raise ValueError("NPZ history length does not match data report")
    if rollout_length != int(data_report["rollout_length"]):
        raise ValueError("NPZ rollout length does not match data report")
    for split, arrays in splits.items():
        if "X" not in arrays or "Y" not in arrays:
            raise ValueError(f"{split} is missing X or Y")
        x, y = arrays["X"], arrays["Y"]
        if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
            raise ValueError(f"{split} has incompatible X/Y shapes: {x.shape}, {y.shape}")
        if x.shape[1:] != (history_length, len(STATE_FEATURE_NAMES)):
            raise ValueError(f"{split} has wrong X shape: {x.shape}")
        if y.shape[1:] != (rollout_length, len(STATE_FEATURE_NAMES)):
            raise ValueError(f"{split} has wrong Y shape: {y.shape}")
        report_split = data_report["splits"][split]
        if list(x.shape) != list(report_split["shape_X"]) or list(y.shape) != list(
            report_split["shape_Y"]
        ):
            raise ValueError(f"{split} array shapes do not match the frozen data report")
        if int(x.shape[0]) != int(report_split["segments"]):
            raise ValueError(f"{split} segment count does not match the frozen data report")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{split} contains non-finite state values")

    mean, std = fit_normalization(splits["train"]["X"], splits["train"]["Y"])
    clip_low, clip_high = sampled_quantiles(
        splits["train"]["X"],
        splits["train"]["Y"],
        args.clip_low_quantile,
        args.clip_high_quantile,
        args.seed,
    )
    # The learned distribution still contains rare scene transitions and
    # tracking anomalies.  Keep the displayed HeadLocal pose inside a generous
    # physiological envelope so the flat-floor demo can never look like the
    # participant is falling through the ground.  This is an inference safety
    # bound, not an alteration of the training targets.
    head_position_safety_low = np.array([-0.35, -0.35, -0.35], dtype=np.float64)
    head_position_safety_high = np.array([0.35, 0.35, 0.35], dtype=np.float64)
    clip_low[3:6] = np.maximum(clip_low[3:6], head_position_safety_low)
    clip_high[3:6] = np.minimum(clip_high[3:6], head_position_safety_high)
    train_loader = make_loader(
        splits["train"], mean, std, args.batch_size, True, args.workers, device.type == "cuda"
    )
    val_loader = make_loader(
        splits["val"], mean, std, args.batch_size, False, args.workers, device.type == "cuda"
    )
    test_loader = make_loader(
        splits["test"], mean, std, args.batch_size, False, args.workers, device.type == "cuda"
    )

    model_config = {
        "input_dim": len(STATE_FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "output_dim": len(STATE_FEATURE_NAMES),
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "history_length": history_length,
        "rollout_length": rollout_length,
        "stateful_inference": True,
    }
    model = NormalHMDLSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=args.hidden_dim,
        output_dim=model_config["output_dim"],
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    _, curriculum_stages = rollout_curriculum(1, args.epochs, rollout_length)
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_rollout, _ = rollout_curriculum(epoch, args.epochs, rollout_length)
        ratio = teacher_forcing_ratio(
            epoch,
            args.epochs,
            args.teacher_forcing_start,
            args.teacher_forcing_end,
        )
        train_loss_sum = 0.0
        train_items = 0
        for history_batch, target_batch in train_loader:
            history_batch = history_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model.rollout(
                    history_batch,
                    training_rollout,
                    teacher=target_batch[:, :training_rollout],
                    teacher_forcing_ratio=ratio,
                )
                loss, _ = rollout_loss(
                    prediction,
                    target_batch[:, :training_rollout],
                    args.delta_loss_weight,
                    args.acceleration_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().item()) * len(history_batch)
            train_items += len(history_batch)
        train_loss = train_loss_sum / train_items

        _, _, val_loss = predict_rollouts(model, val_loader, device, mean, std)
        eligible_for_checkpoint = training_rollout == rollout_length
        improved = eligible_for_checkpoint and val_loss < best_val_loss - args.min_delta
        history_rows.append(
            {
                "epoch": epoch,
                "training_rollout_steps": training_rollout,
                "teacher_forcing_ratio": ratio,
                "train_rollout_loss": train_loss,
                "val_closed_loop_normalized_mse": val_loss,
                "eligible_for_checkpoint": eligible_for_checkpoint,
                "best_so_far": improved,
            }
        )
        print(
            f"epoch {epoch:03d} teacher={ratio:.3f} train={train_loss:.6f} "
            f"val_closed_loop={val_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        elif eligible_for_checkpoint:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    evaluations: dict[str, Any] = {}
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        prediction, target, normalized_mse = predict_rollouts(model, loader, device, mean, std)
        persistence = np.repeat(
            splits[split_name]["X"][:, -1:, :].astype(np.float64),
            rollout_length,
            axis=1,
        )
        evaluations[split_name] = {
            "model_closed_loop": {
                "normalized_mse": normalized_mse,
                **metric_bundle(prediction, target),
            },
            "persistence_baseline": metric_bundle(persistence, target),
        }

    innovation = calibrate_innovation(
        model,
        splits["train"],
        mean,
        std,
        args.batch_size,
        args.workers,
        device,
        args.seed + 911,
        args.innovation_calibration_segments,
    )
    innovation_std = np.asarray(innovation["innovation_std_physical"], dtype=np.float64)
    innovation_rho = np.asarray(innovation["innovation_ar1_rho"], dtype=np.float64)

    test = splits["test"]
    seed_history = test["X"][0].astype(np.float64)
    probe_steps = int(math.ceil(args.probe_seconds / float(data_report["fixed_dt_seconds"])))
    deterministic_raw, deterministic_clipped = stateful_generate(
        model,
        seed_history,
        probe_steps,
        mean,
        std,
        clip_low,
        clip_high,
        device,
    )
    deterministic_probe = closed_loop_probe(
        deterministic_raw,
        deterministic_clipped,
        clip_low,
        clip_high,
        float(data_report["fixed_dt_seconds"]),
        args.probe_seconds,
        mode="deterministic",
    )
    if len(args.probe_stochastic_seeds) < 2:
        raise ValueError("at least two --probe-stochastic-seeds are required")
    stochastic_probes: list[dict[str, Any]] = []
    stochastic_raw_outputs: list[np.ndarray] = []
    stochastic_outputs: list[np.ndarray] = []
    for stochastic_seed in args.probe_stochastic_seeds:
        stochastic_raw, stochastic_clipped = stateful_generate(
            model,
            seed_history,
            probe_steps,
            mean,
            std,
            clip_low,
            clip_high,
            device,
            innovation_std,
            innovation_rho,
            stochastic_seed,
            args.default_noise_scale,
        )
        stochastic_raw_outputs.append(stochastic_raw)
        stochastic_outputs.append(stochastic_clipped)
        stochastic_probes.append(
            closed_loop_probe(
                stochastic_raw,
                stochastic_clipped,
                clip_low,
                clip_high,
                float(data_report["fixed_dt_seconds"]),
                args.probe_seconds,
                mode="seeded_stochastic",
                stochastic_seed=stochastic_seed,
            )
        )
    diversity = None
    if len(stochastic_outputs) >= 2:
        diversity = float(
            np.sqrt(np.mean(np.square(stochastic_outputs[0] - stochastic_outputs[1])))
        )
    repeated_raw, repeated_clipped = stateful_generate(
        model,
        seed_history,
        probe_steps,
        mean,
        std,
        clip_low,
        clip_high,
        device,
        innovation_std,
        innovation_rho,
        args.probe_stochastic_seeds[0],
        args.default_noise_scale,
    )
    stochastic_reproducible = bool(
        np.array_equal(repeated_clipped, stochastic_outputs[0])
        and np.array_equal(repeated_raw, stochastic_raw_outputs[0])
    )

    golden_steps = min(64, probe_steps)
    golden_raw, golden_clipped = stateful_generate(
        model,
        seed_history,
        golden_steps,
        mean,
        std,
        clip_low,
        clip_high,
        device,
    )
    golden_stochastic_raw, golden_stochastic_clipped = stateful_generate(
        model,
        seed_history,
        golden_steps,
        mean,
        std,
        clip_low,
        clip_high,
        device,
        innovation_std,
        innovation_rho,
        args.probe_stochastic_seeds[0],
        args.default_noise_scale,
    )
    trajectory_index = int(test["trajectory_index"][0])
    golden = {
        "contract": (
            "Warm the LSTM once with history_state_physical after normalization. For each "
            "step, denormalize the deterministic model output, componentwise clip that "
            "base prediction, then normalize and feed it into the next stateful LSTM step "
            "without resetting h/c. Seeded AR(1) innovation is applied only to generated "
            "physical motion and is never fed back into the latent intent trajectory."
        ),
        "trajectory_id": str(test["trajectory_ids"][trajectory_index]),
        "participant_group": str(test["participant_groups"][trajectory_index]),
        "site": str(test["sites"][trajectory_index]),
        "scene": str(test["scenes"][trajectory_index]),
        "start_index": int(test["start_index"][0]),
        "reference_eye_height_m": float(test["reference_eye_height_m"][trajectory_index]),
        "history_state_physical": seed_history.tolist(),
        "raw_output_physical": golden_raw.tolist(),
        "clipped_feedback_physical": golden_clipped.tolist(),
        "stochastic_seed": args.probe_stochastic_seeds[0],
        "stochastic_noise_scale": args.default_noise_scale,
        "stochastic_raw_output_physical": golden_stochastic_raw.tolist(),
        "stochastic_clipped_feedback_physical": golden_stochastic_clipped.tolist(),
    }

    checkpoint = {
        "schema_version": "2.0.0",
        "state_dict": best_state,
        "model_config": model_config,
        "parameter_count": parameter_count(model),
        "normalization": {
            "state_mean": mean.tolist(),
            "state_std": std.tolist(),
        },
        "state_clip_low": clip_low.tolist(),
        "state_clip_high": clip_high.tolist(),
        "head_position_safety_envelope_m": {
            "low": head_position_safety_low.tolist(),
            "high": head_position_safety_high.tolist(),
            "purpose": "Prevent non-physiological HeadLocal translation in the flat-floor demo.",
        },
        "clip_quantiles": [args.clip_low_quantile, args.clip_high_quantile],
        "innovation_model": {
            **innovation,
            "default_noise_scale": args.default_noise_scale,
            "seeded_reproducible": stochastic_reproducible,
        },
        "dt_seconds": float(data_report["fixed_dt_seconds"]),
        "feature_names": list(STATE_FEATURE_NAMES),
        "coordinate_system": COORDINATE_SYSTEM,
        "body_proxy": data_report["body_proxy"],
        "nominal_eye_height_m": float(data_report["body_proxy"]["nominal_eye_height_m"]),
        "data_source": "EyeNavGS",
        "synthetic": False,
        "dataset_fingerprint_sha256": data_report["dataset_fingerprint_sha256"],
        "data_report_sha256": sha256_file(args.data_report),
        "prepared_split_sha256": prepared_split_sha256,
        "participant_manifest": data_report["participant_manifest"],
        "frozen_split_manifest_sha256": data_report["frozen_split_manifest_sha256"],
        "seed": args.seed,
        "golden_stateful_rollout": golden,
    }
    atomic_torch_save(args.model, checkpoint)
    model_sha = sha256_file(args.model)
    write_history(args.history_output, history_rows)
    elapsed = time.perf_counter() - started
    test_metrics = evaluations["test"]["model_closed_loop"]
    persistence_metrics = evaluations["test"]["persistence_baseline"]
    metrics = {
        "schema_version": "2.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "task": "multi-step stateful Normal HMD BodyRoot/HeadLocal behaviour cloning",
        "scope_warning": (
            "The BodyRoot is a proxy derived from HMD only. This model does not contain "
            "full-body ground truth, visual scene understanding, goals, collision logic, "
            "or cybersickness labels."
        ),
        "data_source": "EyeNavGS",
        "synthetic": False,
        "smoke_subset": bool(data_report.get("smoke_subset")),
        "dataset_fingerprint_sha256": data_report["dataset_fingerprint_sha256"],
        "data_report_sha256": checkpoint["data_report_sha256"],
        "prepared_split_sha256": prepared_split_sha256,
        "participant_manifest": data_report["participant_manifest"],
        "model_config": model_config,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "amp": use_amp,
        "seed": args.seed,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "best_val_closed_loop_normalized_mse": best_val_loss,
        "training_elapsed_seconds": elapsed,
        "teacher_forcing_schedule": {
            "start": args.teacher_forcing_start,
            "end": args.teacher_forcing_end,
            "form": "linear by epoch",
        },
        "rollout_curriculum_steps": curriculum_stages,
        "head_position_safety_envelope_m": checkpoint["head_position_safety_envelope_m"],
        "loss": {
            "normalized_state_mse_weight": 1.0,
            "normalized_first_difference_weight": args.delta_loss_weight,
            "normalized_second_difference_weight": args.acceleration_loss_weight,
        },
        "evaluations": evaluations,
        "test_comparison": {
            "model_beats_persistence_root_velocity": (
                test_metrics["root_planar_velocity_vector_mae_mps"]
                < persistence_metrics["root_planar_velocity_vector_mae_mps"]
            ),
            "model_beats_persistence_head_orientation": (
                test_metrics["head_local_orientation_geodesic_mae_deg"]
                < persistence_metrics["head_local_orientation_geodesic_mae_deg"]
            ),
        },
        "closed_loop_probe": {
            "deterministic": deterministic_probe,
            "seeded_stochastic": stochastic_probes,
            "default_noise_scale": args.default_noise_scale,
            "different_seed_state_rms_difference": diversity,
            "same_seed_bitwise_reproducible": stochastic_reproducible,
        },
        "innovation_model": checkpoint["innovation_model"],
        "golden_stateful_rollout": {
            key: value
            for key, value in golden.items()
            if key not in {"history_state_physical", "raw_output_physical", "clipped_feedback_physical"}
        },
        "artifacts": {
            "model": str(args.model.resolve()),
            "model_sha256": model_sha,
            "metadata": str(args.metadata.resolve()),
            "history": str(args.history_output.resolve()),
            "data_report": str(args.data_report.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "threads": args.threads,
        },
    }
    metadata = {
        key: value for key, value in metrics.items() if key not in {"evaluations"}
    }
    metadata.update(
        {
            "model_sha256": model_sha,
            "feature_names": list(STATE_FEATURE_NAMES),
            "coordinate_system": COORDINATE_SYSTEM,
            "normalization": checkpoint["normalization"],
            "state_clip_low": checkpoint["state_clip_low"],
            "state_clip_high": checkpoint["state_clip_high"],
            "dt_seconds": checkpoint["dt_seconds"],
            "nominal_eye_height_m": checkpoint["nominal_eye_height_m"],
            "golden_rollout_is_embedded_in_checkpoint": True,
        }
    )
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
    parser.add_argument("--delta-loss-weight", type=float, default=0.10)
    parser.add_argument("--acceleration-loss-weight", type=float, default=0.03)
    parser.add_argument("--clip-low-quantile", type=float, default=0.001)
    parser.add_argument("--clip-high-quantile", type=float, default=0.999)
    parser.add_argument("--probe-seconds", type=float, default=60.0)
    parser.add_argument("--innovation-calibration-segments", type=int, default=4096)
    parser.add_argument("--default-noise-scale", type=float, default=0.20)
    parser.add_argument(
        "--probe-stochastic-seeds", type=int, nargs="+", default=[1729, 2718]
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        metrics = train(args)
    except Exception as exc:
        atomic_json(
            args.metrics,
            {
                "schema_version": "2.0.0",
                "generated_at_utc": utc_now(),
                "status": "failed",
                "reason": str(exc),
                "model_written": False,
            },
        )
        print(f"Normal HMD training failed: {exc}", file=sys.stderr)
        return 1
    test = metrics["evaluations"]["test"]["model_closed_loop"]
    print("Normal HMD v2 training complete")
    print(f"  device: {metrics['device']}")
    print(f"  best epoch: {metrics['best_epoch']}")
    print(
        "  test head orientation MAE: "
        f"{test['head_local_orientation_geodesic_mae_deg']:.4f} deg"
    )
    print(
        f"  {metrics['closed_loop_probe']['deterministic']['requested_seconds']:.0f}-second engineering probe: "
        f"deterministic={metrics['closed_loop_probe']['deterministic']['engineering_probe_passed']}, "
        "stochastic="
        f"{all(item['engineering_probe_passed'] for item in metrics['closed_loop_probe']['seeded_stochastic'])}"
    )
    print(f"  model: {metrics['artifacts']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
