#!/usr/bin/env python3
"""Predictive evaluation for the frozen Normal HMD Agent v4 checkpoint.

The predictor is evaluated with causal teacher forcing: every prediction at
absolute index t sees observed state only through t-1.  The event labels are
the frozen offline SG/hysteresis pseudo-labels stored in the checkpoint and
are themselves non-causal because the centered smoothing filter sees future
samples.

Overlapping prepared windows can expose the same physical frame repeatedly.
Before any metric is computed, this script keeps exactly one view of each
``(trajectory_index, absolute_index)`` pair: the view with the largest target
step, hence the longest causal context available among duplicate windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from normal_hmd_common import atomic_json, sha256_file
from normal_hmd_v4_data import (
    HORIZONTAL_INDICES,
    VERTICAL_INDICES,
    V4WindowTargets,
    assert_onset_parameter_consistency,
    label_trajectories,
    load_split,
    map_labels_to_windows,
    reconstruct_trajectories,
)
from normal_hmd_v4_model import NormalHMDV4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_DATA_REPORT = REPO_ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_JSON = REPO_ROOT / "results" / "normal_hmd_v4_predictive_evaluation.json"
DEFAULT_MARKDOWN = REPO_ROOT / "results" / "normal_hmd_v4_predictive_evaluation.md"
BRANCHES = ("horizontal", "vertical")
INTERVAL_Z = {"50": 0.6744897501960817, "90": 1.6448536269514722}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def equal_frequency_ece(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float | int]]]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) == 0:
        raise ValueError("ECE requires at least one sample")
    order = np.argsort(probabilities, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    details: list[dict[str, float | int]] = []
    weighted_error = 0.0
    for index, group in enumerate(groups, start=1):
        mean_probability = float(np.mean(probabilities[group]))
        observed_rate = float(np.mean(labels[group]))
        absolute_gap = abs(mean_probability - observed_rate)
        weighted_error += len(group) * absolute_gap
        details.append(
            {
                "bin": index,
                "count": int(len(group)),
                "probability_min": float(np.min(probabilities[group])),
                "probability_max": float(np.max(probabilities[group])),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": float(absolute_gap),
            }
        )
    return float(weighted_error / len(labels)), details


def pr_auc_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(np.sum(labels))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    group_end = np.r_[
        sorted_probabilities[:-1] != sorted_probabilities[1:], True
    ]
    ends = np.flatnonzero(group_end)
    true_positive = np.cumsum(sorted_labels)[ends].astype(np.float64)
    predicted_positive = (ends + 1).astype(np.float64)
    recall = true_positive / positives
    precision = true_positive / predicted_positive
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def auroc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_probabilities[:-1] != sorted_probabilities[1:]) + 1,
        len(labels),
    ]
    positive_rank_sum = 0.0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        average_rank = 0.5 * ((start + 1) + stop)
        positive_rank_sum += average_rank * int(np.sum(sorted_labels[start:stop]))
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(statistic / (positives * negatives))


def classification_counts(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels_bool = np.asarray(labels, dtype=bool)
    predictions = np.asarray(probabilities) >= threshold
    true_positive = int(np.sum(predictions & labels_bool))
    false_positive = int(np.sum(predictions & ~labels_bool))
    false_negative = int(np.sum(~predictions & labels_bool))
    true_negative = int(np.sum(~predictions & ~labels_bool))
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Choose one validation threshold, breaking F1 ties toward higher thresholds."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(np.sum(labels))
    if positives == 0:
        raise ValueError("cannot select an F1 threshold without positive labels")
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    ends = np.flatnonzero(
        np.r_[sorted_probabilities[:-1] != sorted_probabilities[1:], True]
    )
    true_positive = np.cumsum(sorted_labels)[ends].astype(np.float64)
    predicted_positive = (ends + 1).astype(np.float64)
    false_positive = predicted_positive - true_positive
    false_negative = positives - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0.0,
    )
    best = int(np.argmax(f1))
    threshold = float(sorted_probabilities[ends[best]])
    counts = classification_counts(labels, probabilities, threshold)
    return {
        "threshold": threshold,
        "criterion": "maximum unweighted eligible-frame F1 on validation",
        "tie_break": "highest threshold among tied maximum-F1 candidates",
        "diagnostic_only": True,
        "candidate_thresholds": int(len(ends)),
        "validation_counts": counts,
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) == 0:
        raise ValueError("binary metrics require at least one sample")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    log_loss = -float(
        np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )
    brier = float(np.mean(np.square(probabilities - labels)))
    ece, calibration_bins = equal_frequency_ece(labels, probabilities, bins=10)
    positives = int(np.sum(labels))
    return {
        "eligible_frames": int(len(labels)),
        "true_onsets": positives,
        "onset_prevalence": float(np.mean(labels)),
        "unweighted_log_loss": log_loss,
        "brier_score": brier,
        "ece_10_equal_frequency": ece,
        "ece_bins": calibration_bins,
        "pr_auc": pr_auc_average_precision(labels, probabilities),
        "pr_auc_definition": "average precision (step integral of precision-recall curve)",
        "auroc": auroc(labels, probabilities),
        "sum_probability": float(np.sum(probabilities)),
        "sum_probability_over_true_onsets": (
            float(np.sum(probabilities) / positives) if positives > 0 else None
        ),
    }


def match_events_one_to_one(
    true_trajectory: np.ndarray,
    true_index: np.ndarray,
    predicted_trajectory: np.ndarray,
    predicted_index: np.ndarray,
    tolerance: int,
) -> dict[str, float | int]:
    """Maximum-cardinality chronological matching within each trajectory."""

    true_by_trajectory: dict[int, list[int]] = {}
    predicted_by_trajectory: dict[int, list[int]] = {}
    for trajectory, index in zip(true_trajectory, true_index):
        true_by_trajectory.setdefault(int(trajectory), []).append(int(index))
    for trajectory, index in zip(predicted_trajectory, predicted_index):
        predicted_by_trajectory.setdefault(int(trajectory), []).append(int(index))

    matched = 0
    absolute_errors: list[int] = []
    for trajectory in sorted(set(true_by_trajectory) | set(predicted_by_trajectory)):
        truth = sorted(true_by_trajectory.get(trajectory, ()))
        prediction = sorted(predicted_by_trajectory.get(trajectory, ()))
        truth_cursor = 0
        prediction_cursor = 0
        while truth_cursor < len(truth) and prediction_cursor < len(prediction):
            true_value = truth[truth_cursor]
            predicted_value = prediction[prediction_cursor]
            if predicted_value < true_value - tolerance:
                prediction_cursor += 1
            elif true_value < predicted_value - tolerance:
                truth_cursor += 1
            else:
                matched += 1
                absolute_errors.append(abs(predicted_value - true_value))
                truth_cursor += 1
                prediction_cursor += 1

    predicted_count = int(len(predicted_index))
    true_count = int(len(true_index))
    false_positive = predicted_count - matched
    false_negative = true_count - matched
    precision = safe_ratio(matched, predicted_count)
    recall = safe_ratio(matched, true_count)
    return {
        "tolerance_frames": int(tolerance),
        "matched": int(matched),
        "predicted_events": predicted_count,
        "true_events": true_count,
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2.0 * precision * recall, precision + recall),
        "matched_mean_absolute_frame_error": (
            float(np.mean(absolute_errors)) if absolute_errors else None
        ),
    }


def event_metrics(
    deduplicated: Mapping[str, Any], branch_name: str, threshold: float
) -> dict[str, dict[str, float | int]]:
    branch = deduplicated["branches"][branch_name]
    eligible = branch["eligible"]
    truth = branch["onset"].astype(bool)
    predicted = eligible & (branch["probability"] >= threshold)
    true_mask = eligible & truth
    result: dict[str, dict[str, float | int]] = {}
    for tolerance in (0, 1, 3, 5):
        label = "exact" if tolerance == 0 else f"plus_minus_{tolerance}"
        result[label] = match_events_one_to_one(
            deduplicated["trajectory_index"][true_mask],
            deduplicated["absolute_index"][true_mask],
            deduplicated["trajectory_index"][predicted],
            deduplicated["absolute_index"][predicted],
            tolerance,
        )
    return result


def build_model(checkpoint: Mapping[str, Any]) -> NormalHMDV4:
    if str(checkpoint.get("schema_version")) != "4.0.0":
        raise ValueError(f"unsupported checkpoint schema {checkpoint.get('schema_version')!r}")
    config = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    bounds = checkpoint["bounds"]
    correlation = checkpoint["parameter_correlation"]
    duration_bounds = config["duration_bounds_steps"]
    model = NormalHMDV4(
        input_dim=int(config["input_dim"]),
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
        minimum_duration_steps=int(duration_bounds[0]),
        maximum_duration_steps=int(duration_bounds[1]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def infer_windows(
    model: NormalHMDV4,
    split: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    state_mean = model.state_mean.detach().cpu().numpy().astype(np.float32)
    state_std = model.state_std.detach().cpu().numpy().astype(np.float32)
    count = len(split["X"])
    output: dict[str, dict[str, list[np.ndarray]]] = {
        branch: {"logit": [], "mean_normalized": [], "logstd": []}
        for branch in BRANCHES
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            history_numpy = (
                split["X"][start:stop].astype(np.float32) - state_mean
            ) / state_std
            target_numpy = (
                split["Y"][start:stop].astype(np.float32) - state_mean
            ) / state_std
            history = torch.from_numpy(history_numpy).to(device)
            target = torch.from_numpy(target_numpy).to(device)
            prediction = model.teacher_forced(history, target)
            for branch in BRANCHES:
                output[branch]["logit"].append(
                    getattr(prediction, f"{branch}_hazard_logit")
                    .detach()
                    .cpu()
                    .numpy()
                )
                output[branch]["mean_normalized"].append(
                    getattr(prediction, f"{branch}_parameter_mean_normalized")
                    .detach()
                    .cpu()
                    .numpy()
                )
                output[branch]["logstd"].append(
                    getattr(prediction, f"{branch}_parameter_logstd")
                    .detach()
                    .cpu()
                    .numpy()
                )
    merged: dict[str, dict[str, np.ndarray]] = {}
    for branch in BRANCHES:
        bias = float(getattr(model, f"{branch}_hazard_bias").detach().cpu())
        logit = np.concatenate(output[branch]["logit"], axis=0).astype(np.float64)
        merged[branch] = {
            "probability": sigmoid(logit + bias),
            "mean_normalized": np.concatenate(
                output[branch]["mean_normalized"], axis=0
            ).astype(np.float64),
            "logstd": np.concatenate(output[branch]["logstd"], axis=0).astype(
                np.float64
            ),
        }
    return merged


def assert_duplicate_target_consistency(targets: V4WindowTargets) -> None:
    steps = targets.absolute_index.shape[1]
    trajectory = np.repeat(targets.trajectory_index, steps)
    absolute = targets.absolute_index.reshape(-1)
    order = np.lexsort((absolute, trajectory))
    same_key = (trajectory[order][1:] == trajectory[order][:-1]) & (
        absolute[order][1:] == absolute[order][:-1]
    )
    left = order[:-1][same_key]
    right = order[1:][same_key]
    for branch_name in BRANCHES:
        branch = getattr(targets, branch_name)
        for field in ("onset", "eligible", "parameter_mask", "event_id"):
            flattened = np.asarray(getattr(branch, field)).reshape(-1)
            if not np.array_equal(flattened[left], flattened[right]):
                raise AssertionError(
                    f"duplicate {branch_name} targets disagree for field {field}"
                )
        parameters = branch.parameters.reshape(-1, branch.parameters.shape[-1])
        if not np.allclose(parameters[left], parameters[right], rtol=0.0, atol=0.0):
            raise AssertionError(f"duplicate {branch_name} parameters disagree")


def deduplicate_predictions(
    split: Mapping[str, np.ndarray],
    targets: V4WindowTargets,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    assert_duplicate_target_consistency(targets)
    count, steps = targets.absolute_index.shape
    trajectory = np.repeat(targets.trajectory_index, steps)
    absolute = targets.absolute_index.reshape(-1)
    target_step = np.tile(np.arange(steps, dtype=np.int64), count)
    # np.lexsort uses the last key as primary.  Within a duplicate key, the
    # negative target step places the longest-context view first.
    order = np.lexsort((-target_step, absolute, trajectory))
    sorted_trajectory = trajectory[order]
    sorted_absolute = absolute[order]
    first = np.r_[
        True,
        (sorted_trajectory[1:] != sorted_trajectory[:-1])
        | (sorted_absolute[1:] != sorted_absolute[:-1]),
    ]
    keep = order[first]
    selected_steps = target_step[keep]
    result: dict[str, Any] = {
        "trajectory_index": trajectory[keep].astype(np.int64),
        "absolute_index": absolute[keep].astype(np.int64),
        "selected_target_step": selected_steps,
        "branches": {},
        "deduplication": {
            "key": ["trajectory_index", "absolute_index"],
            "selection": "largest target step (longest causal context), stable first on any residual tie",
            "raw_window_target_rows": int(count * steps),
            "unique_physical_frames": int(len(keep)),
            "duplicate_rows_removed": int(count * steps - len(keep)),
            "retained_fraction": float(len(keep) / (count * steps)),
            "selected_target_step_min": int(np.min(selected_steps)),
            "selected_target_step_max": int(np.max(selected_steps)),
            "history_frames_min": int(split["X"].shape[1] + np.min(selected_steps)),
            "history_frames_max": int(split["X"].shape[1] + np.max(selected_steps)),
            "duplicate_label_consistency": "passed",
        },
    }
    for branch_name in BRANCHES:
        branch = getattr(targets, branch_name)
        prediction = predictions[branch_name]
        result["branches"][branch_name] = {
            "onset": branch.onset.reshape(-1)[keep].astype(bool),
            "eligible": branch.eligible.reshape(-1)[keep].astype(bool),
            "parameter_mask": branch.parameter_mask.reshape(-1)[keep].astype(bool),
            "event_id": branch.event_id.reshape(-1)[keep].astype(np.int64),
            "true_parameters": branch.parameters.reshape(
                -1, branch.parameters.shape[-1]
            )[keep].astype(np.float64),
            "probability": prediction["probability"].reshape(-1)[keep],
            "mean_normalized": prediction["mean_normalized"].reshape(
                -1, prediction["mean_normalized"].shape[-1]
            )[keep],
            "logstd": prediction["logstd"].reshape(
                -1, prediction["logstd"].shape[-1]
            )[keep],
        }
    return result


def mark_metrics(
    deduplicated: Mapping[str, Any],
    model: NormalHMDV4,
    branch_name: str,
    parameter_layout: list[str],
    dt_seconds: float,
) -> dict[str, Any]:
    branch = deduplicated["branches"][branch_name]
    selected = branch["onset"] & branch["parameter_mask"]
    if not np.any(selected):
        raise ValueError(f"no complete {branch_name} onset marks after deduplication")
    target = branch["true_parameters"][selected]
    mean_normalized = branch["mean_normalized"][selected]
    logstd = branch["logstd"][selected]
    center = (
        getattr(model, f"{branch_name}_parameter_center")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    scale = (
        getattr(model, f"{branch_name}_parameter_scale")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    cholesky = (
        getattr(model, f"{branch_name}_parameter_cholesky")
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    mean = center + scale * mean_normalized
    standard_deviation = scale * np.exp(logstd)
    standardized_residual = (target - mean) / standard_deviation
    whitened = np.linalg.solve(cholesky, standardized_residual.T).T
    joint_nll = (
        0.5 * target.shape[1] * math.log(2.0 * math.pi)
        + np.sum(np.log(standard_deviation), axis=1)
        + float(np.sum(np.log(np.diag(cholesky))))
        + 0.5 * np.sum(np.square(whitened), axis=1)
    )
    marginal_nll = (
        0.5 * math.log(2.0 * math.pi)
        + np.log(standard_deviation)
        + 0.5 * np.square(standardized_residual)
    )

    endpoints: dict[str, Any] = {}
    for dimension, name in enumerate(parameter_layout[:-1]):
        dimension_result: dict[str, Any] = {
            "mae": float(np.mean(np.abs(mean[:, dimension] - target[:, dimension]))),
            "pearson_r": safe_pearson(mean[:, dimension], target[:, dimension]),
            "marginal_gaussian_nll_mean": float(np.mean(marginal_nll[:, dimension])),
            "true_mean": float(np.mean(target[:, dimension])),
            "predicted_mean": float(np.mean(mean[:, dimension])),
        }
        for level, z_value in INTERVAL_Z.items():
            covered = np.abs(target[:, dimension] - mean[:, dimension]) <= (
                z_value * standard_deviation[:, dimension]
            )
            dimension_result[f"gaussian_{level}_interval_coverage"] = float(
                np.mean(covered)
            )
        endpoints[name] = dimension_result

    duration_dimension = len(parameter_layout) - 1
    true_duration = np.exp(target[:, duration_dimension])
    predicted_duration = np.exp(mean[:, duration_dimension])
    duration: dict[str, Any] = {
        "mae_steps": float(np.mean(np.abs(predicted_duration - true_duration))),
        "mae_seconds": float(
            np.mean(np.abs(predicted_duration - true_duration)) * dt_seconds
        ),
        "true_mean_steps": float(np.mean(true_duration)),
        "predicted_median_mean_steps": float(np.mean(predicted_duration)),
        "marginal_gaussian_nll_mean_log_steps": float(
            np.mean(marginal_nll[:, duration_dimension])
        ),
        "point_estimate": "exp(predicted Gaussian mean in log-duration space), i.e. lognormal median",
    }
    for level, z_value in INTERVAL_Z.items():
        covered = np.abs(
            target[:, duration_dimension] - mean[:, duration_dimension]
        ) <= (z_value * standard_deviation[:, duration_dimension])
        duration[f"gaussian_{level}_interval_coverage"] = float(np.mean(covered))

    coverage: dict[str, float] = {}
    for level, z_value in INTERVAL_Z.items():
        covered = np.abs(target - mean) <= z_value * standard_deviation
        coverage[f"all_parameters_marginal_{level}"] = float(np.mean(covered))
        coverage[f"endpoints_marginal_{level}"] = float(
            np.mean(covered[:, :-1])
        )
    return {
        "complete_true_onsets": int(np.sum(selected)),
        "conditioning": "teacher-forced predictions at true complete onsets; independent of deterministic hazard threshold",
        "parameter_space": "physical endpoints plus log(duration_steps)",
        "endpoint_axes": endpoints,
        "duration": duration,
        "gaussian_interval_coverage": coverage,
        "joint_correlated_gaussian_nll_mean": float(np.mean(joint_nll)),
        "joint_correlated_gaussian_nll_median": float(np.median(joint_nll)),
        "nll_definition": "full multivariate Gaussian NLL in physical endpoint/log-duration coordinates, including normalization constant, scale Jacobian, and fixed correlation determinant",
    }


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_verify_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.data_report.is_file():
        raise FileNotFoundError(args.data_report)
    with args.data_report.open("r", encoding="utf-8") as stream:
        data_report = json.load(stream)
    if data_report.get("status") != "success" or data_report.get("synthetic") is not False:
        raise ValueError("evaluation requires the successful non-synthetic frozen data report")
    actual_report_hash = sha256_file(args.data_report)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if actual_report_hash != checkpoint.get("data_report_sha256"):
        raise ValueError("data report hash does not match the formal v4 checkpoint")
    for split_name in ("val", "test"):
        path = args.data_dir / f"normal_hmd_{split_name}.npz"
        actual = sha256_file(path)
        checkpoint_expected = checkpoint["prepared_split_sha256"][split_name]
        report_expected = data_report["splits"][split_name]["sha256"]
        if actual != checkpoint_expected or actual != report_expected:
            raise ValueError(
                f"{split_name} data hash mismatch: actual={actual}, "
                f"checkpoint={checkpoint_expected}, report={report_expected}"
            )
    return checkpoint, data_report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, data_report = load_and_verify_inputs(args)
    dt_seconds = float(checkpoint["event_definition"]["dt_seconds"])
    if not math.isclose(
        dt_seconds,
        float(data_report["fixed_dt_seconds"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("checkpoint and data report disagree on fixed time step")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model = build_model(checkpoint).to(device)
    thresholds = checkpoint["event_definition"]["thresholds"]
    cooldown_steps = int(checkpoint["model_config"]["cooldown_steps"])

    split_results: dict[str, Any] = {}
    deduplicated_by_split: dict[str, dict[str, Any]] = {}
    consistency: dict[str, Any] = {}
    for split_name in ("val", "test"):
        split = load_split(args.data_dir, split_name)
        fragments = reconstruct_trajectories(split)
        labelled = label_trajectories(
            fragments,
            thresholds,
            dt_seconds=dt_seconds,
            cooldown_steps=cooldown_steps,
        )
        targets = map_labels_to_windows(split, labelled)
        consistency[split_name] = assert_onset_parameter_consistency(targets)
        predictions = infer_windows(
            model,
            split,
            batch_size=args.batch_size,
            device=device,
        )
        deduplicated = deduplicate_predictions(split, targets, predictions)
        deduplicated_by_split[split_name] = deduplicated
        split_results[split_name] = {
            "benchmark_status": (
                "validation" if split_name == "val" else "previously inspected benchmark"
            ),
            "trajectory_fragments": int(len(fragments)),
            "deduplication": deduplicated["deduplication"],
            "hazard": {},
            "marks": {},
        }
        for branch_name in BRANCHES:
            branch = deduplicated["branches"][branch_name]
            eligible = branch["eligible"]
            split_results[split_name]["hazard"][branch_name] = binary_metrics(
                branch["onset"][eligible], branch["probability"][eligible]
            )

    threshold_selection: dict[str, Any] = {}
    for branch_name in BRANCHES:
        validation_branch = deduplicated_by_split["val"]["branches"][branch_name]
        eligible = validation_branch["eligible"]
        selection = select_f1_threshold(
            validation_branch["onset"][eligible],
            validation_branch["probability"][eligible],
        )
        threshold_selection[branch_name] = selection
        threshold = float(selection["threshold"])
        for split_name in ("val", "test"):
            branch = deduplicated_by_split[split_name]["branches"][branch_name]
            split_eligible = branch["eligible"]
            split_results[split_name]["hazard"][branch_name][
                "classification_at_frozen_threshold"
            ] = classification_counts(
                branch["onset"][split_eligible],
                branch["probability"][split_eligible],
                threshold,
            )
            split_results[split_name]["hazard"][branch_name][
                "frozen_threshold"
            ] = threshold
            split_results[split_name]["hazard"][branch_name][
                "event_one_to_one_matching"
            ] = event_metrics(
                deduplicated_by_split[split_name], branch_name, threshold
            )

    parameter_layout = {
        branch: list(checkpoint["model_config"][f"{branch}_parameter_layout"])
        for branch in BRANCHES
    }
    for split_name in ("val", "test"):
        for branch_name in BRANCHES:
            split_results[split_name]["marks"][branch_name] = mark_metrics(
                deduplicated_by_split[split_name],
                model,
                branch_name,
                parameter_layout[branch_name],
                dt_seconds,
            )

    model_hash = sha256_file(args.model)
    result = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "status": "success",
        "evaluation": "Normal HMD v4 teacher-forced predictive hazard and mark evaluation",
        "test_benchmark_status": "previously inspected benchmark",
        "test_benchmark_warning": "The test split was viewed in earlier development/reporting; these results cannot support an untouched-holdout claim.",
        "causality": {
            "predictor": "causal teacher forcing: prediction at t uses observed state through t-1 only",
            "target": "offline non-causal pseudo-label: centered Savitzky-Golay smoothing uses future context",
        },
        "threshold_selection": {
            "scope": "one branch-specific threshold selected on validation and frozen unchanged for test",
            "purpose": "deterministic classification diagnostic only; runtime remains seeded Bernoulli hazard sampling",
            "branches": threshold_selection,
        },
        "deduplication_policy": "Before every metric, retain one row per (trajectory_index, absolute_index), choosing the largest target step / longest causal context.",
        "provenance": {
            "evaluation_script": str(Path(__file__).resolve()),
            "evaluation_script_sha256": sha256_file(Path(__file__)),
            "checkpoint": str(args.model.resolve()),
            "checkpoint_sha256": model_hash,
            "checkpoint_schema_version": checkpoint["schema_version"],
            "data_directory": str(args.data_dir.resolve()),
            "data_report": str(args.data_report.resolve()),
            "data_report_sha256": sha256_file(args.data_report),
            "dataset_fingerprint_sha256": checkpoint["dataset_fingerprint_sha256"],
            "prepared_split_sha256": {
                name: checkpoint["prepared_split_sha256"][name]
                for name in ("val", "test")
            },
            "label_threshold_source": "checkpoint event_definition.thresholds (train-derived and frozen)",
            "hazard_probability": "sigmoid(raw teacher-forced logit + checkpoint validation calibration bias)",
            "fixed_dt_seconds": dt_seconds,
        },
        "label_consistency": consistency,
        "splits": split_results,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "threads": args.threads,
        },
    }
    return result


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Normal HMD v4 predictive evaluation",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "> **Benchmark status:** test is a **previously inspected benchmark**. It was viewed during earlier development/reporting and cannot support an untouched-holdout claim.",
        "",
        "## Protocol",
        "",
        "Predictions are teacher-forced and causal: the prediction at frame `t` sees observed state only through `t-1`. The SG/hysteresis pseudo-labels are offline and non-causal because their centered smoother uses future context. All metrics are computed after deduplicating overlapping windows by `(trajectory_index, absolute_index)`; the retained view has the largest target step and therefore the longest causal context.",
        "",
        "Hazard probabilities include the checkpoint's validation-fitted calibration bias. One threshold per branch maximizes unweighted eligible-frame F1 on validation (ties choose the higher threshold), and that exact threshold is frozen for test. Thresholded results are classification diagnostics only; the runtime policy remains seeded Bernoulli sampling.",
        "",
        "PR-AUC is average precision (the step integral of the precision-recall curve). ECE uses 10 deterministic equal-frequency bins. Event matches are chronological, one-to-one, and trajectory-local.",
        "",
        "## Provenance and deduplication",
        "",
        f"- Checkpoint SHA-256: `{result['provenance']['checkpoint_sha256']}`",
        f"- Dataset fingerprint: `{result['provenance']['dataset_fingerprint_sha256']}`",
        f"- Fixed timestep: `{result['provenance']['fixed_dt_seconds']}` seconds",
        "",
        "| Split | Raw window-target rows | Unique physical frames | Removed duplicates | Retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in ("val", "test"):
        dedup = result["splits"][split_name]["deduplication"]
        lines.append(
            f"| {split_name} | {dedup['raw_window_target_rows']} | "
            f"{dedup['unique_physical_frames']} | {dedup['duplicate_rows_removed']} | "
            f"{dedup['retained_fraction']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Eligible risk-set hazard metrics",
            "",
            "| Split | Branch | Eligible | Onsets | Log loss | Brier | ECE-10 | PR-AUC | AUROC | sum(p)/onsets | Frozen threshold | Frame F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ("val", "test"):
        for branch_name in BRANCHES:
            hazard = result["splits"][split_name]["hazard"][branch_name]
            classification = hazard["classification_at_frozen_threshold"]
            lines.append(
                f"| {split_name} | {branch_name} | {hazard['eligible_frames']} | "
                f"{hazard['true_onsets']} | {hazard['unweighted_log_loss']:.4f} | "
                f"{hazard['brier_score']:.4f} | {hazard['ece_10_equal_frequency']:.4f} | "
                f"{format_value(hazard['pr_auc'])} | {format_value(hazard['auroc'])} | "
                f"{format_value(hazard['sum_probability_over_true_onsets'])} | "
                f"{hazard['frozen_threshold']:.6f} | {classification['f1']:.4f} |"
            )

    lines.extend(["", "## One-to-one event matching", ""])
    for split_name in ("val", "test"):
        lines.extend(
            [
                f"### {split_name}",
                "",
                "| Branch | Tolerance | Predicted | True | Matched | Precision | Recall | F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for branch_name in BRANCHES:
            matching = result["splits"][split_name]["hazard"][branch_name][
                "event_one_to_one_matching"
            ]
            for label in ("exact", "plus_minus_1", "plus_minus_3", "plus_minus_5"):
                metric = matching[label]
                tolerance = "exact" if label == "exact" else f"±{metric['tolerance_frames']}"
                lines.append(
                    f"| {branch_name} | {tolerance} | {metric['predicted_events']} | "
                    f"{metric['true_events']} | {metric['matched']} | "
                    f"{metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} |"
                )
        lines.append("")

    lines.extend(["## Marks at true complete onsets", ""])
    for split_name in ("val", "test"):
        lines.extend([f"### {split_name}", ""])
        for branch_name in BRANCHES:
            marks = result["splits"][split_name]["marks"][branch_name]
            lines.extend(
                [
                    f"**{branch_name}** — {marks['complete_true_onsets']} complete true onsets; joint correlated Gaussian NLL = {marks['joint_correlated_gaussian_nll_mean']:.4f}.",
                    "",
                    "| Endpoint axis | MAE | Pearson r | 50% coverage | 90% coverage | Marginal NLL |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for name, metric in marks["endpoint_axes"].items():
                lines.append(
                    f"| {name} | {metric['mae']:.4f} | {format_value(metric['pearson_r'])} | "
                    f"{metric['gaussian_50_interval_coverage']:.4f} | "
                    f"{metric['gaussian_90_interval_coverage']:.4f} | "
                    f"{metric['marginal_gaussian_nll_mean']:.4f} |"
                )
            duration = marks["duration"]
            lines.extend(
                [
                    "",
                    f"Duration MAE: **{duration['mae_steps']:.3f} steps** ({duration['mae_seconds']:.3f} s). Gaussian log-duration interval coverage: 50% = {duration['gaussian_50_interval_coverage']:.4f}, 90% = {duration['gaussian_90_interval_coverage']:.4f}.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- Mark metrics condition on true complete onsets and do not depend on the deterministic hazard threshold.",
            "- Gaussian duration is modeled in log-steps; the point duration is `exp(mean_log_duration)`, the lognormal median.",
            "- Interval coverage is marginal per parameter. Joint NLL uses the checkpoint's fixed parameter correlation and includes the full Gaussian normalization term.",
            "- Because test was previously inspected, test figures are descriptive benchmark results rather than an untouched generalization estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data-report", type=Path, default=DEFAULT_DATA_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch size and thread count must be positive")
    try:
        result = evaluate(args)
        atomic_json(args.json_output, result)
        atomic_text(args.markdown_output, markdown_report(result))
    except Exception as exc:
        print(f"Normal HMD v4 predictive evaluation failed: {exc}", file=sys.stderr)
        return 1
    print("Normal HMD v4 predictive evaluation complete")
    print(f"  test status: {result['test_benchmark_status']}")
    print(f"  JSON: {args.json_output.resolve()}")
    print(f"  Markdown: {args.markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
