#!/usr/bin/env python3
"""Fine-tune only the Normal HMD v4 BodyRoot branch for smoother walking.

The horizontal and vertical HeadLocal branches are kept bit-for-bit frozen.
The Root branch is initialized from the production v4 checkpoint and trained
with both teacher-forced and short autoregressive objectives.  In addition to
matching velocity, the objective matches the first and second temporal
differences so the generated acceleration and jerk follow the already
low-pass EyeNavGS BodyRoot proxy instead of changing direction frame by frame.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from normal_hmd_common import atomic_json, sha256_file
from normal_hmd_v4_model import NormalHMDV4, ROOT_INDICES, parameter_count


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_DATA = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_MODEL = ROOT / "models" / "normal_hmd_v4_smooth_root.pth"
DEFAULT_REPORT = ROOT / "results" / "normal_hmd_v4_smooth_root_training.json"
DEFAULT_HISTORY = ROOT / "results" / "normal_hmd_v4_smooth_root_history.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(checkpoint: dict[str, Any]) -> NormalHMDV4:
    config = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    bounds = checkpoint["bounds"]
    correlation = checkpoint["parameter_correlation"]
    model = NormalHMDV4(
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        state_mean=normalization["state_mean"],
        state_std=normalization["state_std"],
        horizontal_parameter_center=normalization["horizontal_parameter_center"],
        horizontal_parameter_scale=normalization["horizontal_parameter_scale"],
        vertical_parameter_center=normalization["vertical_parameter_center"],
        vertical_parameter_scale=normalization["vertical_parameter_scale"],
        horizontal_parameter_low=bounds["horizontal_parameter_p05_p95"][0],
        horizontal_parameter_high=bounds["horizontal_parameter_p05_p95"][1],
        vertical_parameter_low=bounds["vertical_parameter_p05_p95"][0],
        vertical_parameter_high=bounds["vertical_parameter_p05_p95"][1],
        horizontal_pose_low=bounds["horizontal_absolute_pose_p01_p99"][0],
        horizontal_pose_high=bounds["horizontal_absolute_pose_p01_p99"][1],
        vertical_pose_low=bounds["vertical_absolute_pose_p01_p99"][0],
        vertical_pose_high=bounds["vertical_absolute_pose_p01_p99"][1],
        horizontal_parameter_correlation=correlation["horizontal"],
        vertical_parameter_correlation=correlation["vertical"],
        cooldown_steps=int(config["cooldown_steps"]),
        minimum_duration_steps=int(config["duration_bounds_steps"][0]),
        maximum_duration_steps=int(config["duration_bounds_steps"][1]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def tensor_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_digest(model: NormalHMDV4) -> str:
    return tensor_digest(
        (name, value)
        for name, value in model.state_dict().items()
        if not name.startswith("root_lstm.") and not name.startswith("root_output.")
    )


def load_windows(
    data_dir: Path,
    split: str,
    mean: np.ndarray,
    std: np.ndarray,
    maximum: int,
    seed: int,
) -> TensorDataset:
    path = data_dir / f"normal_hmd_{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["Y"], dtype=np.float32)
    if maximum > 0 and len(x) > maximum:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(x), size=maximum, replace=False))
        x = x[selected]
        y = y[selected]
    x = ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    y = ((y - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def root_autoregressive(
    model: NormalHMDV4,
    history: torch.Tensor,
    target: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Predict Root closed-loop while teacher-forcing only HeadLocal context."""

    encoded, hidden = model.root_lstm(history)
    prediction = model.root_output(encoded[:, -1])
    predictions = [prediction]
    previous = target[:, 0].clone()
    previous[:, list(ROOT_INDICES)] = prediction.to(previous.dtype)
    for step in range(1, steps):
        encoded, hidden = model.root_lstm(previous[:, None], hidden)
        prediction = model.root_output(encoded[:, 0])
        predictions.append(prediction)
        previous = target[:, step].clone()
        previous[:, list(ROOT_INDICES)] = prediction.to(previous.dtype)
    return torch.stack(predictions, dim=1)


def temporal_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    initial_root: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = F.smooth_l1_loss(prediction, target)
    predicted_with_initial = torch.cat((initial_root[:, None], prediction), dim=1)
    target_with_initial = torch.cat((initial_root[:, None], target), dim=1)
    predicted_delta = torch.diff(predicted_with_initial, dim=1)
    target_delta = torch.diff(target_with_initial, dim=1)
    delta = F.smooth_l1_loss(predicted_delta, target_delta)
    predicted_second = torch.diff(predicted_delta, dim=1)
    target_second = torch.diff(target_delta, dim=1)
    second = F.smooth_l1_loss(predicted_second, target_second)
    return value, delta, second


def loss_bundle(
    model: NormalHMDV4,
    history: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_root = target[..., list(ROOT_INDICES)]
    initial_root = history[:, -1, list(ROOT_INDICES)]

    teacher = model.teacher_forced(history, target).root_normalized
    teacher_value, teacher_delta, teacher_second = temporal_terms(
        teacher, target_root, initial_root
    )

    rollout_steps = min(args.rollout_steps, target.shape[1])
    autoregressive = root_autoregressive(model, history, target, rollout_steps)
    ar_value, ar_delta, ar_second = temporal_terms(
        autoregressive, target_root[:, :rollout_steps], initial_root
    )

    total = (
        args.teacher_value_weight * teacher_value
        + args.teacher_delta_weight * teacher_delta
        + args.teacher_second_weight * teacher_second
        + args.autoregressive_value_weight * ar_value
        + args.autoregressive_delta_weight * ar_delta
        + args.autoregressive_second_weight * ar_second
    )
    terms = {
        "teacher_value": teacher_value,
        "teacher_delta": teacher_delta,
        "teacher_second": teacher_second,
        "autoregressive_value": ar_value,
        "autoregressive_delta": ar_delta,
        "autoregressive_second": ar_second,
    }
    return total, terms


@torch.no_grad()
def evaluate(
    model: NormalHMDV4,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"total": 0.0}
    count = 0
    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        total, terms = loss_bundle(model, history, target, args)
        batch = len(history)
        sums["total"] += float(total) * batch
        for name, value in terms.items():
            sums[name] = sums.get(name, 0.0) + float(value) * batch
        count += batch
    return {name: value / max(count, 1) for name, value in sums.items()}


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parent_hash = sha256_file(args.parent_model)
    checkpoint = torch.load(args.parent_model, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "4.0.0":
        raise ValueError("root smoothing requires a production Normal HMD v4 checkpoint")
    model = build_model(checkpoint)
    before_frozen_hash = frozen_digest(model)
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.root_lstm.parameters():
        parameter.requires_grad_(True)
    for parameter in model.root_output.parameters():
        parameter.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    device = choose_device(args.device)
    model.to(device)
    use_amp = bool(args.amp and device.type == "cuda")

    mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["state_std"], dtype=np.float32)
    datasets = {
        "train": load_windows(
            args.data_dir, "train", mean, std, args.max_train_segments, args.seed + 1
        ),
        "val": load_windows(
            args.data_dir, "val", mean, std, args.max_val_segments, args.seed + 2
        ),
        "test": load_windows(
            args.data_dir, "test", mean, std, args.max_test_segments, args.seed + 3
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    baseline = {
        name: evaluate(model, loaders[name], device, args)
        for name in ("val", "test")
    }
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history_rows: list[dict[str, Any]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {"total": 0.0}
        count = 0
        for history, target in loaders["train"]:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                total, terms = loss_bundle(model, history, target, args)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch = len(history)
            sums["total"] += float(total.detach()) * batch
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch
            count += batch

        train_metrics = {name: value / max(count, 1) for name, value in sums.items()}
        validation = evaluate(model, loaders["val"], device, args)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "val": validation,
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:03d} train={train_metrics['total']:.6f} "
            f"val={validation['total']:.6f} "
            f"val_ar_delta={validation['autoregressive_delta']:.6f} "
            f"val_ar_second={validation['autoregressive_second']:.6f}",
            flush=True,
        )
        if validation["total"] < best_loss - args.min_delta:
            best_loss = validation["total"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("root fine-tuning did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    after_frozen_hash = frozen_digest(model)
    if before_frozen_hash != after_frozen_hash:
        raise AssertionError("HeadLocal branches or frozen buffers changed during Root fine-tuning")
    for name, value in model.state_dict().items():
        if name.startswith("root_lstm.") or name.startswith("root_output."):
            continue
        if not torch.equal(value.detach().cpu(), parent_state[name]):
            raise AssertionError(f"frozen tensor changed: {name}")

    final = {
        name: evaluate(model, loaders[name], device, args)
        for name in ("val", "test")
    }
    output_checkpoint = copy.deepcopy(checkpoint)
    output_checkpoint["state_dict"] = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    output_checkpoint["parent_model_sha256"] = parent_hash
    output_checkpoint["root_smoothing"] = {
        "name": "root_temporal_finetune_v1",
        "initialized_from_parent": True,
        "headlocal_branches_frozen": True,
        "teacher_forcing_and_root_autoregressive_training": True,
        "rollout_steps": args.rollout_steps,
        "loss_weights": {
            "teacher_value": args.teacher_value_weight,
            "teacher_delta": args.teacher_delta_weight,
            "teacher_second": args.teacher_second_weight,
            "autoregressive_value": args.autoregressive_value_weight,
            "autoregressive_delta": args.autoregressive_delta_weight,
            "autoregressive_second": args.autoregressive_second_weight,
        },
    }
    output_checkpoint["model_config"] = copy.deepcopy(checkpoint["model_config"])
    output_checkpoint["model_config"]["root_training_variant"] = (
        "v4_root_temporal_finetune_v1"
    )
    atomic_torch_save(args.model, output_checkpoint)
    model_hash = sha256_file(args.model)

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "parent_model": str(args.parent_model.resolve()),
        "parent_model_sha256": parent_hash,
        "model": str(args.model.resolve()),
        "model_sha256": model_hash,
        "data_dir": str(args.data_dir.resolve()),
        "device": str(device),
        "amp": use_amp,
        "parameter_count_total": parameter_count(model),
        "parameter_count_trainable": trainable_count,
        "headlocal_frozen_digest_before": before_frozen_hash,
        "headlocal_frozen_digest_after": after_frozen_hash,
        "headlocal_frozen_exact": before_frozen_hash == after_frozen_hash,
        "segments": {name: len(dataset) for name, dataset in datasets.items()},
        "epochs_completed": len(history_rows),
        "best_val_total": best_loss,
        "baseline": baseline,
        "final": final,
        "configuration": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "rollout_steps": args.rollout_steps,
            "loss_weights": output_checkpoint["root_smoothing"]["loss_weights"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(args.report, report)
    atomic_json(args.history_output, {"epochs": history_rows})
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--patience", type=int, default=4)
    result.add_argument("--min-delta", type=float, default=1e-6)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--gradient-clip-norm", type=float, default=0.5)
    result.add_argument("--rollout-steps", type=int, default=24)
    result.add_argument("--teacher-value-weight", type=float, default=1.0)
    result.add_argument("--teacher-delta-weight", type=float, default=0.75)
    result.add_argument("--teacher-second-weight", type=float, default=0.25)
    result.add_argument("--autoregressive-value-weight", type=float, default=0.75)
    result.add_argument("--autoregressive-delta-weight", type=float, default=1.0)
    result.add_argument("--autoregressive-second-weight", type=float, default=0.35)
    result.add_argument("--max-train-segments", type=int, default=0)
    result.add_argument("--max-val-segments", type=int, default=0)
    result.add_argument("--max-test-segments", type=int, default=0)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--amp", action="store_true")
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    result.add_argument("--seed", type=int, default=45)
    result.add_argument("--smoke", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.rollout_steps < 2 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("invalid rollout, batch, or epoch count")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.rollout_steps = min(args.rollout_steps, 8)
        args.max_train_segments = 256
        args.max_val_segments = 128
        args.max_test_segments = 128
        args.model = ROOT / "tmp" / "normal_hmd_v4_smooth_root_smoke.pth"
        args.report = ROOT / "tmp" / "normal_hmd_v4_smooth_root_smoke.json"
        args.history_output = ROOT / "tmp" / "normal_hmd_v4_smooth_root_smoke_history.json"
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["model"],
                "epochs": report["epochs_completed"],
                "best_val_total": report["best_val_total"],
                "headlocal_frozen_exact": report["headlocal_frozen_exact"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
