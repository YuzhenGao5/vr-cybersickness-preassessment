#!/usr/bin/env python3
"""Train and evaluate the Habitat-Web six-action decision prior."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from habitat_decision_policy import (
    ACTIONS,
    ACTION_TO_INDEX,
    FEATURE_NAMES,
    build_model,
    export_runtime_json,
    normalize_features,
    sha256_file,
)
from prepare_habitat_decision_data import load_config, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_prepared_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"X", "y", "scene_index", "episode_hash"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"prepared split missing arrays: {missing}")
        result = {name: archive[name] for name in required}
    if result["X"].ndim != 2 or result["X"].shape[1] != len(FEATURE_NAMES):
        raise ValueError("prepared X has the wrong feature dimension")
    if result["y"].shape != (len(result["X"]),):
        raise ValueError("prepared y shape does not match X")
    if not np.all(np.isfinite(result["X"])):
        raise ValueError("prepared X contains non-finite values")
    if np.any(result["y"] >= len(ACTIONS)):
        raise ValueError("prepared y contains an unsupported class")
    return result


def deterministic_subset(
    arrays: Mapping[str, np.ndarray], maximum: int | None, seed: int
) -> dict[str, np.ndarray]:
    count = len(arrays["y"])
    if maximum is None or maximum >= count:
        return {name: np.asarray(value) for name, value in arrays.items()}
    if maximum < 1:
        raise ValueError("subset maximum must be positive")
    indices = np.random.default_rng(seed).choice(count, size=maximum, replace=False)
    indices.sort()
    return {name: np.asarray(value)[indices] for name, value in arrays.items()}


def fit_train_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0, dtype=np.float64)
    std = np.std(features, axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(
    labels: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(ACTIONS)).astype(np.float64)
    if np.any(counts == 0):
        absent = [ACTIONS[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"training split has no examples for classes: {absent}")
    weights = len(labels) / (len(ACTIONS) * counts)
    weights = np.clip(weights, float(minimum), float(maximum))
    weights /= np.mean(weights)
    return weights.astype(np.float32), counts.astype(np.int64)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
    np.add.at(matrix, (labels.astype(np.int64), predictions.astype(np.int64)), 1)
    return matrix


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("ECE bins must be at least two")
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(labels))
    ece = 0.0
    records: list[dict[str, Any]] = []
    for index in range(bins):
        if index + 1 == bins:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(np.sum(mask))
        accuracy = float(np.mean(correct[mask])) if count else None
        average_confidence = float(np.mean(confidence[mask])) if count else None
        if count:
            ece += count / total * abs(float(accuracy) - float(average_confidence))
        records.append(
            {
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": average_confidence,
            }
        )
    return {"ece": float(ece), "bins": records}


def metric_bundle(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels and predictions must be equal non-empty vectors")
    matrix = confusion_matrix(labels, predictions)
    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, action in enumerate(ACTIONS):
        true_positive = int(matrix[index, index])
        false_positive = int(np.sum(matrix[:, index]) - true_positive)
        false_negative = int(np.sum(matrix[index, :]) - true_positive)
        support = int(np.sum(matrix[index, :]))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class[action] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        recalls.append(float(recall))
        f1_values.append(float(f1))
    result: dict[str, Any] = {
        "n": int(len(labels)),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "class_order": list(ACTIONS),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(labels), len(ACTIONS)):
            raise ValueError("probability shape does not match labels/classes")
        result["calibration"] = expected_calibration_error(
            labels, probabilities, bins=ece_bins
        )
        result["negative_log_likelihood"] = float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)))
        )
    return result


def always_move_predictions(features: np.ndarray) -> np.ndarray:
    return np.full(len(features), ACTION_TO_INDEX["MOVE_FORWARD"], dtype=np.int64)


def repeat_previous_predictions(features: np.ndarray) -> np.ndarray:
    start = FEATURE_NAMES.index("previous_action_MOVE_FORWARD")
    previous = features[:, start : start + len(ACTIONS)]
    predictions = np.argmax(previous, axis=1).astype(np.int64)
    has_previous = np.sum(previous, axis=1) > 0.5
    predictions[~has_previous] = ACTION_TO_INDEX["MOVE_FORWARD"]
    return predictions


def geometric_rule_predictions(
    features: np.ndarray, baseline_config: Mapping[str, float]
) -> np.ndarray:
    distance = features[:, FEATURE_NAMES.index("goal_distance_m")]
    direction_right = features[:, FEATURE_NAMES.index("goal_direction_right")]
    direction_up = features[:, FEATURE_NAMES.index("goal_direction_up")]
    stop_distance = float(baseline_config["stop_distance_m"])
    look_threshold = float(baseline_config["look_vertical_direction_threshold"])
    turn_threshold = float(baseline_config["turn_direction_threshold"])
    result = always_move_predictions(features)
    result[direction_right < -turn_threshold] = ACTION_TO_INDEX["TURN_LEFT"]
    result[direction_right > turn_threshold] = ACTION_TO_INDEX["TURN_RIGHT"]
    horizontal_enough = np.abs(direction_right) <= turn_threshold
    result[horizontal_enough & (direction_up > look_threshold)] = ACTION_TO_INDEX["LOOK_UP"]
    result[horizontal_enough & (direction_up < -look_threshold)] = ACTION_TO_INDEX["LOOK_DOWN"]
    result[distance <= stop_distance] = ACTION_TO_INDEX["STOP"]
    return result


def predict_model(
    model: nn.Module,
    normalized_features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    dataset = TensorDataset(torch.from_numpy(normalized_features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (features,) in loader:
            logits_parts.append(model(features.to(device)).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities, float(np.mean(np.max(probabilities, axis=1)))


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def train(
    config: Mapping[str, Any],
    *,
    prepared_dir: Path,
    device_name: str,
    epochs_override: int | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    output_suffix: str = "",
) -> dict[str, Any]:
    seed = int(config["seed"])
    set_determinism(seed)
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_kind") != "public_human":
        raise ValueError("prepared data provenance is not public_human")
    if manifest.get("license") != "CC BY-NC 4.0":
        raise ValueError("prepared data license metadata is missing or unexpected")
    if any(manifest["scene_overlap_checks"].values()):
        raise ValueError("prepared manifest reports scene leakage")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("prepared feature contract differs from training code")
    if manifest.get("action_classes") != list(ACTIONS):
        raise ValueError("prepared action contract differs from training code")
    partial_dataset = bool(
        manifest.get("max_episodes_per_scene") is not None
        or int(manifest.get("selected_scene_count", 0))
        < int(manifest.get("available_scene_count", 0))
    )
    training_scope = "smoke_subset" if partial_dataset else "configured_full_scene_set"

    raw: dict[str, dict[str, np.ndarray]] = {}
    for index, split_name in enumerate(("train", "val", "test")):
        split_path = Path(manifest["splits"][split_name]["path"])
        if sha256_file(split_path) != manifest["splits"][split_name]["sha256"]:
            raise ValueError(f"prepared {split_name} hash mismatch")
        maximum = max_train_samples if split_name == "train" else max_eval_samples
        raw[split_name] = deterministic_subset(
            load_prepared_split(split_path), maximum, seed + 7919 * (index + 1)
        )

    mean, std = fit_train_normalization(raw["train"]["X"])
    normalized = {
        name: normalize_features(values["X"], mean, std).astype(np.float32)
        for name, values in raw.items()
    }
    weights, train_counts = class_weights(
        raw["train"]["y"],
        minimum=float(config["training"]["class_weight_min"]),
        maximum=float(config["training"]["class_weight_max"]),
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model_config = {
        "input_dim": len(FEATURE_NAMES),
        "hidden_dims": list(config["model"]["hidden_dims"]),
        "dropout": float(config["model"]["dropout"]),
        "output_dim": len(ACTIONS),
    }
    model = build_model(model_config).to(device)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(normalized["train"]),
        torch.from_numpy(raw["train"]["y"].astype(np.int64)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        generator=generator,
        drop_last=False,
    )

    epochs = int(epochs_override or config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    best_macro_f1 = -1.0
    best_val_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            count += len(labels)
        val_predictions, val_probabilities, _ = predict_model(
            model,
            normalized["val"],
            device=device,
            batch_size=batch_size,
        )
        val_metrics = metric_bundle(
            raw["val"]["y"],
            val_predictions,
            val_probabilities,
            ece_bins=int(config["training"]["ece_bins"]),
        )
        val_f1 = float(val_metrics["macro_f1"])
        val_nll = float(val_metrics["negative_log_likelihood"])
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": loss_sum / max(1, count),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_f1,
                "val_nll": val_nll,
                "val_ece": val_metrics["calibration"]["ece"],
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_weighted_loss']:.6f} "
            f"val_macro_f1={val_f1:.6f} val_nll={val_nll:.6f}",
            flush=True,
        )
        improved = val_f1 > best_macro_f1 + 1e-8 or (
            abs(val_f1 - best_macro_f1) <= 1e-8 and val_nll < best_val_nll
        )
        if improved:
            best_macro_f1 = val_f1
            best_val_nll = val_nll
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)

    evaluations: dict[str, Any] = {}
    for split_name in ("val", "test"):
        predictions, probabilities, confidence = predict_model(
            model,
            normalized[split_name],
            device=device,
            batch_size=batch_size,
        )
        evaluations[split_name] = {
            "model": metric_bundle(
                raw[split_name]["y"], predictions, probabilities,
                ece_bins=int(config["training"]["ece_bins"]),
            ),
            "baselines": {
                "always_move": metric_bundle(
                    raw[split_name]["y"], always_move_predictions(raw[split_name]["X"])
                ),
                "repeat_previous": metric_bundle(
                    raw[split_name]["y"], repeat_previous_predictions(raw[split_name]["X"])
                ),
                "simple_geometry": metric_bundle(
                    raw[split_name]["y"],
                    geometric_rule_predictions(
                        raw[split_name]["X"], config["geometry_baseline"]
                    ),
                ),
            },
            "mean_max_probability": confidence,
            "natural_distribution": True,
            "class_counts": {
                action: int(np.sum(raw[split_name]["y"] == index))
                for index, action in enumerate(ACTIONS)
            },
        }

    full_dataset_coverage = not partial_dataset
    strongest_test_baseline_macro_f1 = max(
        float(bundle["macro_f1"])
        for bundle in evaluations["test"]["baselines"].values()
    )
    beats_all_test_macro_f1_baselines = (
        float(evaluations["test"]["model"]["macro_f1"])
        > strongest_test_baseline_macro_f1
    )
    # Reading every source episode is a data-coverage property, not a production
    # safety claim.  This offline prior still lacks simulator closed-loop and
    # Unity integration validation, so it must remain non-production even if an
    # offline metric happens to beat a baseline.
    readiness = {
        "full_dataset_coverage": full_dataset_coverage,
        "beats_all_test_macro_f1_baselines": beats_all_test_macro_f1_baselines,
        "strongest_test_baseline_macro_f1": strongest_test_baseline_macro_f1,
        "closed_loop_validation": False,
        "unity_runtime_integration": False,
        "production_model": False,
    }

    output_paths = {
        name: resolve_project_path(path)
        for name, path in config["outputs"].items()
    }
    if output_suffix:
        output_paths = {
            name: path.with_name(f"{path.stem}{output_suffix}{path.suffix}")
            for name, path in output_paths.items()
        }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "source": manifest["source"],
        "source_kind": "public_human",
        "license": "CC BY-NC 4.0",
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "prepared_split_sha256": {
            name: manifest["splits"][name]["sha256"] for name in ("train", "val", "test")
        },
        "split_unit": "scene_id",
        "human_like_general_claim_allowed": False,
        "complete_unity_brain": False,
        "training_scope": training_scope,
        "full_dataset_coverage": full_dataset_coverage,
        "production_model": False,
        "readiness": readiness,
    }
    checkpoint = {
        "schema_version": "habitat-decision-prior-checkpoint-1.0",
        "generated_at_utc": utc_now(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "feature_names": list(FEATURE_NAMES),
        "action_classes": list(ACTIONS),
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "fitted_on": "train only"},
        "class_weights": weights.tolist(),
        "training_class_counts": train_counts.tolist(),
        "seed": seed,
        "best_epoch": best_epoch,
        "provenance": provenance,
        "claims": config["claims"],
    }
    torch.save(checkpoint, output_paths["checkpoint"])

    runtime_metadata = {
        "source_kind": "public_human",
        "license": "CC BY-NC 4.0",
        "scope": config["claims"]["scope"],
        "complete_unity_brain": False,
        "human_like_general_claim_allowed": False,
        "training_scope": training_scope,
        "full_dataset_coverage": full_dataset_coverage,
        "production_model": False,
        "readiness": readiness,
        "prepared_manifest_sha256": provenance["prepared_manifest_sha256"],
        "best_epoch": best_epoch,
    }
    export_runtime_json(
        model=model,
        normalization=checkpoint["normalization"],
        output_path=output_paths["runtime_json"],
        checkpoint_path=output_paths["checkpoint"],
        metadata=runtime_metadata,
        golden_raw_features=raw["test"]["X"][0],
    )

    metrics = {
        "schema_version": "habitat-decision-prior-metrics-1.0",
        "generated_at_utc": utc_now(),
        "best_epoch": best_epoch,
        "device": str(device),
        "torch_version": torch.__version__,
        "sample_counts": {name: int(len(values["y"])) for name, values in raw.items()},
        "evaluations": evaluations,
        "evaluation_distribution": "natural deterministic reservoir; no class balancing at evaluation",
        "readiness": readiness,
        "baseline_definitions": {
            "always_move": "always predicts MOVE_FORWARD",
            "repeat_previous": "repeats the strictly previous action; MOVE_FORWARD if none",
            "simple_geometry": dict(config["geometry_baseline"]),
        },
    }
    output_paths["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_paths["history"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "schema_version": "habitat-decision-prior-metadata-1.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "training_scope": training_scope,
        "full_dataset_coverage": full_dataset_coverage,
        "production_model": False,
        "readiness": readiness,
        "config_sha256": _json_hash(config),
        "provenance": provenance,
        "model_config": model_config,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < epochs,
        "normalization_fitted_on": "train split only",
        "class_weights": weights.tolist(),
        "output_sha256": {
            "checkpoint": sha256_file(output_paths["checkpoint"]),
            "runtime_json": sha256_file(output_paths["runtime_json"]),
            "metrics": sha256_file(output_paths["metrics"]),
            "history": sha256_file(output_paths["history"]),
        },
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "claims": config["claims"],
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "success", "outputs": metadata["outputs"]}, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/habitat_decision_prior_v1.json")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Append a suffix before every configured output extension (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared_dir = Path(args.prepared_dir or config["data"]["prepared_dir"])
    train(
        config,
        prepared_dir=prepared_dir,
        device_name=args.device,
        epochs_override=args.epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
