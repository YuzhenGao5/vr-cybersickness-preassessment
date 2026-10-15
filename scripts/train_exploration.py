#!/usr/bin/env python3
"""Train and evaluate the EyeNavGS 6-DoF short-horizon motion policy."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from exploration_policy import (
    FEATURE_NAMES,
    ExplorationLSTM,
    quaternion_to_rotation_vector,
    relative_quaternions,
    rotation_vector_to_quaternion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_DATA_REPORT = REPO_ROOT / "results" / "exploration_data_report.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "exploration_model.pth"
DEFAULT_MODEL_METADATA = REPO_ROOT / "models" / "exploration_model_metadata.json"
DEFAULT_METRICS = REPO_ROOT / "results" / "exploration_training_metrics.json"
DEFAULT_HISTORY = REPO_ROOT / "results" / "exploration_training_history.csv"


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


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_split(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "dt_seconds", "trajectory_index", "trajectory_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
        payload = {name: archive[name] for name in required}
    x = payload["X"]
    y = payload["y"]
    dt = payload["dt_seconds"]
    if x.ndim != 3 or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected X shape in {path}: {x.shape}")
    if y.shape != (len(x), len(FEATURE_NAMES)) or dt.shape != (len(x),):
        raise ValueError(f"Inconsistent arrays in {path}")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(dt).all():
        raise ValueError(f"Non-finite values in {path}")
    return payload


def predict_physical(
    model: nn.Module,
    x: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = ((x[start : start + batch_size] - x_mean) / x_std).astype(np.float32)
            normalized = model(torch.from_numpy(batch)).cpu().numpy()
            predictions.append(normalized * y_std + y_mean)
    return np.concatenate(predictions, axis=0).astype(np.float64)


def metric_bundle(
    prediction: np.ndarray,
    target: np.ndarray,
    dt: np.ndarray,
    trajectory_index: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    linear_error = prediction[:, :3] - target[:, :3]
    angular_error = prediction[:, 3:] - target[:, 3:]
    linear_abs = np.abs(linear_error)
    angular_abs_deg = np.degrees(np.abs(angular_error))
    linear_vector = np.linalg.norm(linear_error, axis=1)
    angular_vector_deg = np.degrees(np.linalg.norm(angular_error, axis=1))

    predicted_delta_q = rotation_vector_to_quaternion(prediction[:, 3:] * dt[:, None])
    target_delta_q = rotation_vector_to_quaternion(target[:, 3:] * dt[:, None])
    delta_error_q = relative_quaternions(predicted_delta_q, target_delta_q)
    geodesic_step_deg = np.degrees(
        np.linalg.norm(quaternion_to_rotation_vector(delta_error_q), axis=1)
    )
    position_step_error_mm = linear_vector * dt * 1000.0

    per_trajectory: list[dict[str, float]] = []
    for trajectory in np.unique(trajectory_index):
        mask = trajectory_index == trajectory
        per_trajectory.append(
            {
                "linear_velocity_vector_mae_mps": float(linear_vector[mask].mean()),
                "angular_velocity_vector_mae_degps": float(angular_vector_deg[mask].mean()),
                "position_step_error_mm": float(position_step_error_mm[mask].mean()),
                "orientation_step_geodesic_error_deg": float(geodesic_step_deg[mask].mean()),
            }
        )

    def trajectory_summary(field: str) -> dict[str, float]:
        values = np.asarray([item[field] for item in per_trajectory], dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    return {
        "samples": int(len(target)),
        "linear_velocity_mae_mps_by_axis": [float(value) for value in linear_abs.mean(axis=0)],
        "linear_velocity_rmse_mps_by_axis": [
            float(value) for value in np.sqrt(np.mean(linear_error**2, axis=0))
        ],
        "linear_velocity_vector_mae_mps": float(linear_vector.mean()),
        "angular_velocity_mae_degps_by_axis": [float(value) for value in angular_abs_deg.mean(axis=0)],
        "angular_velocity_rmse_degps_by_axis": [
            float(value) for value in np.degrees(np.sqrt(np.mean(angular_error**2, axis=0)))
        ],
        "angular_velocity_vector_mae_degps": float(angular_vector_deg.mean()),
        "position_step_error_mm": float(position_step_error_mm.mean()),
        "orientation_step_geodesic_error_deg": float(geodesic_step_deg.mean()),
        "trajectory_count": len(per_trajectory),
        "trajectory_macro": {
            field: trajectory_summary(field)
            for field in (
                "linear_velocity_vector_mae_mps",
                "angular_velocity_vector_mae_degps",
                "position_step_error_mm",
                "orientation_step_geodesic_error_deg",
            )
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    torch.set_num_threads(args.threads)
    if not args.data_report.is_file():
        raise FileNotFoundError(f"Data report is missing: {args.data_report}")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if data_report.get("source_type") not in {"EyeNavGS", "synthetic"}:
        raise ValueError("Data report has no valid source_type")

    splits = {
        name: load_split(args.data_dir / f"exploration_{name}.npz")
        for name in ("train", "val", "test")
    }
    train_x = splits["train"]["X"].astype(np.float64)
    train_y = splits["train"]["y"].astype(np.float64)
    x_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    x_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    x_std = np.maximum(x_std, 1e-6)
    y_std = np.maximum(y_std, 1e-6)

    normalized_train_x = ((splits["train"]["X"] - x_mean) / x_std).astype(np.float32)
    normalized_train_y = ((splits["train"]["y"] - y_mean) / y_std).astype(np.float32)
    normalized_val_x = ((splits["val"]["X"] - x_mean) / x_std).astype(np.float32)
    normalized_val_y = ((splits["val"]["y"] - y_mean) / y_std).astype(np.float32)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized_train_x), torch.from_numpy(normalized_train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized_val_x), torch.from_numpy(normalized_val_y)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_config = {
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "output_dim": len(FEATURE_NAMES),
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "history_length": int(splits["train"]["X"].shape[1]),
    }
    model = ExplorationLSTM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss(reduction="mean")
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for features, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_items = len(features)
            train_loss_sum += float(loss.item()) * batch_items
            train_items += batch_items
        train_loss = train_loss_sum / train_items

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for features, target in val_loader:
                loss = criterion(model(features), target)
                batch_items = len(features)
                val_loss_sum += float(loss.item()) * batch_items
                val_items += batch_items
        val_loss = val_loss_sum / val_items
        improved = val_loss < best_val_loss - args.min_delta
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "val_normalized_mse": val_loss,
                "best_so_far": improved,
            }
        )
        print(
            f"Epoch {epoch:02d}: train_normalized_mse={train_loss:.6f} "
            f"val_normalized_mse={val_loss:.6f}{' *' if improved else ''}",
            flush=True,
        )
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed_seconds = time.perf_counter() - started

    evaluations: dict[str, Any] = {}
    for split_name in ("val", "test"):
        split = splits[split_name]
        target = split["y"].astype(np.float64)
        prediction = predict_physical(
            model,
            split["X"],
            x_mean,
            x_std,
            y_mean,
            y_std,
            args.batch_size,
        )
        zero = np.zeros_like(target)
        persistence = split["X"][:, -1, :].astype(np.float64)
        evaluations[split_name] = {
            "model": metric_bundle(
                prediction, target, split["dt_seconds"], split["trajectory_index"]
            ),
            "zero_velocity_baseline": metric_bundle(
                zero, target, split["dt_seconds"], split["trajectory_index"]
            ),
            "persistence_baseline": metric_bundle(
                persistence, target, split["dt_seconds"], split["trajectory_index"]
            ),
        }

    median_dt = float(np.median(splits["train"]["dt_seconds"]))
    clip_low = np.quantile(splits["train"]["y"], 0.005, axis=0).astype(np.float64)
    clip_high = np.quantile(splits["train"]["y"], 0.995, axis=0).astype(np.float64)
    demo_seed = splits["test"]["X"][0].astype(np.float64)
    checkpoint = {
        "schema_version": "1.0.0",
        "state_dict": best_state,
        "model_config": model_config,
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "median_dt_seconds": median_dt,
        "action_clip_low": clip_low.tolist(),
        "action_clip_high": clip_high.tolist(),
        "demo_seed_velocity": demo_seed.tolist(),
        "data_source": data_report["source_type"],
        "synthetic": bool(data_report.get("synthetic")),
        "dataset_fingerprint_sha256": data_report.get("dataset_fingerprint_sha256"),
        "feature_names": list(FEATURE_NAMES),
        "seed": args.seed,
    }
    atomic_torch_save(args.model, checkpoint)
    model_sha256 = sha256_file(args.model)
    atomic_history_csv(args.history_output, history)

    test_model = evaluations["test"]["model"]
    test_persistence = evaluations["test"]["persistence_baseline"]
    metrics = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "task": "one-step 6-DoF head-motion behaviour cloning",
        "scope_warning": (
            "The model predicts short-horizon motion from motion history only. It is not scene-aware, "
            "goal-conditioned, collision-aware or a cybersickness predictor."
        ),
        "protocol_scope": (
            "Independent demo outside the frozen cybersickness endpoint protocol; no AUC, clinical, "
            "or risk-avoidance claim is made."
        ),
        "data_source": data_report["source_type"],
        "synthetic": bool(data_report.get("synthetic")),
        "dataset_fingerprint_sha256": data_report.get("dataset_fingerprint_sha256"),
        "participant_manifest": data_report.get("participant_manifest"),
        "seed": args.seed,
        "model_config": model_config,
        "normalization_fitted_on": "train split only",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_normalized_mse": best_val_loss,
        "training_elapsed_seconds": elapsed_seconds,
        "evaluations": evaluations,
        "test_comparison": {
            "model_beats_persistence_linear_velocity_mae": (
                test_model["linear_velocity_vector_mae_mps"]
                < test_persistence["linear_velocity_vector_mae_mps"]
            ),
            "model_beats_persistence_orientation_step_error": (
                test_model["orientation_step_geodesic_error_deg"]
                < test_persistence["orientation_step_geodesic_error_deg"]
            ),
        },
        "artifacts": {
            "model": str(args.model.resolve()),
            "model_sha256": model_sha256,
            "metadata": str(args.metadata.resolve()),
            "history": str(args.history_output.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "threads": args.threads,
        },
    }
    metadata = {
        key: value
        for key, value in metrics.items()
        if key not in {"evaluations"}
    }
    metadata.update(
        {
            "model_sha256": model_sha256,
            "feature_names": list(FEATURE_NAMES),
            "units": {"linear_velocity": "m/s", "angular_velocity": "rad/s"},
            "median_dt_seconds": median_dt,
            "normalization": checkpoint["normalization"],
            "action_clip_low": checkpoint["action_clip_low"],
            "action_clip_high": checkpoint["action_clip_high"],
            "demo_seed_origin": "first held-out test window; velocities only, no participant identity",
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
    parser.add_argument("--metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        metrics = train(args)
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "generated_at_utc": utc_now(),
            "status": "failed",
            "reason": str(exc),
            "model_written": False,
        }
        atomic_json(args.metrics, failure)
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    test = metrics["evaluations"]["test"]["model"]
    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Data source                     : {metrics['data_source']}")
    print(f"Best epoch                     : {metrics['best_epoch']}")
    print(f"Linear velocity vector MAE     : {test['linear_velocity_vector_mae_mps']:.6f} m/s")
    print(f"Angular velocity vector MAE    : {test['angular_velocity_vector_mae_degps']:.6f} deg/s")
    print(f"Orientation step geodesic MAE  : {test['orientation_step_geodesic_error_deg']:.6f} deg")
    print(f"Model                          : {args.model.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

