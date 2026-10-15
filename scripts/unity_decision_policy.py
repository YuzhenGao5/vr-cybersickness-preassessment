#!/usr/bin/env python3
"""Shared model, feature, metric, and export utilities for Decision Brain v0.

The model is deliberately small and set-oriented.  It scores each candidate
with shared weights, so neither candidate order nor a target identifier is a
feature.  Runtime safety remains outside this model: reachability, cooldown,
interaction authorization, and legal-skill masks are supplied by Unity and are
applied before an action can execute.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


# PyTorch's deterministic CUDA mode requires a CuBLAS workspace policy on
# CUDA 10.2+.  Set the documented deterministic option before the first model
# operation so the normal ``--device cuda`` command is reproducible without a
# caller-specific environment-variable workaround.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "unity_decision_brain_v0.json"
FROZEN_SEED = 42
FROZEN_ALLOWED_SOURCES = (
    "rule_bootstrap",
    "scripted_bootstrap",
    "public_human",
    "unity_human",
)
FROZEN_HUMAN_SOURCES = ("public_human", "unity_human")
FROZEN_SKILLS = ("Navigate", "LookAt", "Observe", "Interact", "Replan", "Stop")
UNITY_FLAT_SCHEMA_VERSION = "unity-decision-brain-unity-flat-1.0"

# The names are a deployment contract, not an implementation detail.  Each
# PyTorch Linear layer is exported as a row-major float[] plus explicit input
# and output sizes so Unity's JsonUtility never has to deserialize a dictionary
# or a jagged array.
UNITY_LINEAR_LAYERS = (
    ("globalEncoderLinear1", "global_encoder.0"),
    ("globalEncoderLinear2", "global_encoder.2"),
    ("candidateEncoderLinear1", "candidate_encoder.0"),
    ("candidateEncoderLinear2", "candidate_encoder.2"),
    ("targetHeadLinear1", "target_head.0"),
    ("targetHeadLinear2", "target_head.3"),
    ("skillHeadLinear1", "skill_head.0"),
    ("skillHeadLinear2", "skill_head.3"),
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "context_schema_version",
        "outcome_schema_version",
        "allowed_source_kinds",
        "human_source_kinds",
        "skills",
        "brain_states",
        "planner_modes",
        "semantic_types",
        "global_numeric_features",
        "candidate_numeric_features",
        "candidate_boolean_features",
        "split",
        "model",
        "training",
        "claims",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Decision Brain config is missing keys: {sorted(missing)}")
    if payload["claims"].get("human_like_claim_allowed") is not False:
        raise ValueError("Decision Brain v0 must keep human_like_claim_allowed=false")
    if tuple(payload["allowed_source_kinds"]) != FROZEN_ALLOWED_SOURCES:
        raise ValueError(
            "Decision Brain v0 freezes allowed_source_kinds to rule_bootstrap, "
            "scripted_bootstrap, public_human, unity_human"
        )
    if tuple(payload["human_source_kinds"]) != FROZEN_HUMAN_SOURCES:
        raise ValueError(
            "Decision Brain v0 freezes human_source_kinds to public_human and unity_human"
        )
    if tuple(payload["skills"]) != FROZEN_SKILLS:
        raise ValueError("Decision Brain v0 freezes the six macro-skill labels and order")
    if payload["claims"].get("rule_bootstrap_is_human_data") is not False:
        raise ValueError("rule_bootstrap_is_human_data must remain false")
    if payload["claims"].get("risk_score_is_a_decision_label") is not False:
        raise ValueError("risk_score_is_a_decision_label must remain false")
    if payload["context_schema_version"] != "unity-decision-context-1.0":
        raise ValueError("Decision Brain v0 requires unity-decision-context-1.0")
    if payload["outcome_schema_version"] != "unity-decision-outcome-1.0":
        raise ValueError("Decision Brain v0 requires unity-decision-outcome-1.0")
    if int(payload["split"].get("seed", -1)) != FROZEN_SEED or int(
        payload["training"].get("seed", -1)
    ) != FROZEN_SEED:
        raise ValueError("Decision Brain v0 freezes split.seed and training.seed at 42")
    return payload


def global_feature_names(config: dict[str, Any]) -> list[str]:
    names = list(config["global_numeric_features"])
    names.extend(f"brain_state::{name}" for name in config["brain_states"])
    names.extend(f"previous_skill::{name}" for name in [*config["skills"], "Unknown"])
    names.extend(f"planner_mode::{name}" for name in config["planner_modes"])
    return names


def candidate_feature_names(config: dict[str, Any]) -> list[str]:
    names = list(config["candidate_numeric_features"])
    names.extend(config["candidate_boolean_features"])
    names.extend(f"semantic_type::{name}" for name in config["semantic_types"])
    forbidden = ("target_id", "scene_id", "scene_name", "project_id", "absolute_")
    if any(any(token in name.lower() for token in forbidden) for name in names):
        raise ValueError("Identifier or absolute-coordinate leakage in candidate features")
    return names


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some older CUDA installations cannot make every kernel deterministic.
        # CPU remains the default for this intentionally small model.
        pass


def masked_fill_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {logits.shape} vs {mask.shape}")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if torch.any(~mask.any(dim=1)):
        raise ValueError("Every decision sample must expose at least one legal choice")
    return logits.masked_fill(~mask, -1.0e9)


class UnityDecisionRanker(nn.Module):
    """Permutation-equivariant candidate ranker plus a macro-skill head."""

    def __init__(
        self,
        global_dim: int,
        candidate_dim: int,
        skill_count: int,
        *,
        global_hidden_dim: int = 32,
        candidate_hidden_dim: int = 48,
        joint_hidden_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if min(global_dim, candidate_dim, skill_count) < 1:
            raise ValueError("Model dimensions must be positive")
        if min(global_hidden_dim, candidate_hidden_dim, joint_hidden_dim) < 1:
            raise ValueError("Hidden dimensions must be positive")
        self.global_dim = int(global_dim)
        self.candidate_dim = int(candidate_dim)
        self.skill_count = int(skill_count)
        self.global_hidden_dim = int(global_hidden_dim)
        self.candidate_hidden_dim = int(candidate_hidden_dim)
        self.joint_hidden_dim = int(joint_hidden_dim)
        self.dropout_probability = float(dropout)

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden_dim),
            nn.ReLU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.ReLU(),
        )
        joint_input = global_hidden_dim + candidate_hidden_dim * 3
        self.target_head = nn.Sequential(
            nn.Linear(joint_input, joint_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_dim, 1),
        )
        self.skill_head = nn.Sequential(
            nn.Linear(joint_input, joint_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_dim, skill_count),
        )

    def architecture(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "global_dim": self.global_dim,
            "candidate_dim": self.candidate_dim,
            "skill_count": self.skill_count,
            "global_hidden_dim": self.global_hidden_dim,
            "candidate_hidden_dim": self.candidate_hidden_dim,
            "joint_hidden_dim": self.joint_hidden_dim,
            "dropout": self.dropout_probability,
            "pooling": ["masked_mean", "masked_max"],
            "activation": "ReLU",
        }

    def forward(
        self,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        selected_target_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_features.ndim != 2:
            raise ValueError("global_features must be [batch, feature]")
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must be [batch, candidate, feature]")
        if candidate_mask.ndim != 2:
            raise ValueError("candidate_mask must be [batch, candidate]")
        if candidate_features.shape[:2] != candidate_mask.shape:
            raise ValueError("candidate tensor and mask dimensions do not agree")
        if global_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("global and candidate batch dimensions do not agree")

        mask = candidate_mask.bool()
        legal_target_mask = mask if target_mask is None else target_mask.bool()
        if legal_target_mask.shape != mask.shape:
            raise ValueError("target_mask and candidate_mask dimensions do not agree")
        if torch.any(legal_target_mask & ~mask):
            raise ValueError("target_mask cannot authorize a padded candidate")
        if torch.any(~mask.any(dim=1)):
            raise ValueError("Every sample requires at least one present candidate")
        if torch.any(~legal_target_mask.any(dim=1)):
            raise ValueError("Every sample requires at least one legal target")
        encoded_global = self.global_encoder(global_features)
        encoded_candidates = self.candidate_encoder(candidate_features)
        float_mask = mask.unsqueeze(-1).to(encoded_candidates.dtype)
        candidate_count = float_mask.sum(dim=1).clamp_min(1.0)
        pooled_mean = (encoded_candidates * float_mask).sum(dim=1) / candidate_count
        max_input = encoded_candidates.masked_fill(~mask.unsqueeze(-1), -1.0e9)
        pooled_max = max_input.max(dim=1).values

        batch, candidate_count_int, _ = encoded_candidates.shape
        repeated_global = encoded_global.unsqueeze(1).expand(-1, candidate_count_int, -1)
        repeated_mean = pooled_mean.unsqueeze(1).expand(-1, candidate_count_int, -1)
        repeated_max = pooled_max.unsqueeze(1).expand(-1, candidate_count_int, -1)
        joint_candidates = torch.cat(
            [repeated_global, encoded_candidates, repeated_mean, repeated_max], dim=-1
        )
        raw_target_logits = self.target_head(joint_candidates).squeeze(-1)
        masked_target_logits = masked_fill_logits(raw_target_logits, legal_target_mask)

        if selected_target_indices is None:
            selected_target_indices = masked_target_logits.argmax(dim=1)
        if selected_target_indices.shape != (batch,):
            raise ValueError("selected_target_indices must be [batch]")
        if torch.any(selected_target_indices < 0) or torch.any(
            selected_target_indices >= candidate_count_int
        ):
            raise ValueError("selected target index is outside the padded candidate dimension")
        selected_valid = mask.gather(1, selected_target_indices[:, None]).squeeze(1)
        if torch.any(~selected_valid):
            raise ValueError("selected target index points at an invalid candidate")
        selected = encoded_candidates[
            torch.arange(batch, device=encoded_candidates.device), selected_target_indices
        ]
        joint_selected = torch.cat(
            [encoded_global, selected, pooled_mean, pooled_max], dim=-1
        )
        raw_skill_logits = self.skill_head(joint_selected)
        return raw_target_logits, masked_target_logits, raw_skill_logits


def normalization_from_training(
    global_features: np.ndarray,
    candidate_features: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, list[float]]:
    if len(global_features) == 0:
        raise ValueError("Cannot fit normalization on an empty training split")
    valid_candidates = candidate_features[candidate_mask.astype(bool)]
    if len(valid_candidates) == 0:
        raise ValueError("Training split contains no valid candidates")
    global_mean = global_features.astype(np.float64).mean(axis=0)
    global_std = global_features.astype(np.float64).std(axis=0)
    candidate_mean = valid_candidates.astype(np.float64).mean(axis=0)
    candidate_std = valid_candidates.astype(np.float64).std(axis=0)
    global_std = np.maximum(global_std, 1.0e-6)
    candidate_std = np.maximum(candidate_std, 1.0e-6)
    return {
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
        "candidate_mean": candidate_mean.tolist(),
        "candidate_std": candidate_std.tolist(),
        "fitted_on": "train split only",
    }


def apply_normalization(
    global_features: np.ndarray,
    candidate_features: np.ndarray,
    normalization: dict[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    global_mean = np.asarray(normalization["global_mean"], dtype=np.float32)
    global_std = np.asarray(normalization["global_std"], dtype=np.float32)
    candidate_mean = np.asarray(normalization["candidate_mean"], dtype=np.float32)
    candidate_std = np.asarray(normalization["candidate_std"], dtype=np.float32)
    if global_features.shape[-1] != len(global_mean):
        raise ValueError("Global normalization dimension mismatch")
    if candidate_features.shape[-1] != len(candidate_mean):
        raise ValueError("Candidate normalization dimension mismatch")
    normalized_global = (global_features.astype(np.float32) - global_mean) / global_std
    normalized_candidate = (
        candidate_features.astype(np.float32) - candidate_mean
    ) / candidate_std
    return normalized_global.astype(np.float32), normalized_candidate.astype(np.float32)


def confusion_and_macro_f1(
    truth: np.ndarray, prediction: np.ndarray, class_count: int
) -> tuple[list[list[int]], float, list[float]]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(truth.astype(int), prediction.astype(int), strict=True):
        if 0 <= actual < class_count and 0 <= predicted < class_count:
            matrix[actual, predicted] += 1
    f1_values: list[float] = []
    for index in range(class_count):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - matrix[index, index])
        false_negative = float(matrix[index, :].sum() - matrix[index, index])
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return matrix.tolist(), float(np.mean(f1_values)), f1_values


def evaluate_predictions(
    *,
    raw_target_logits: np.ndarray,
    candidate_mask: np.ndarray,
    target_index: np.ndarray,
    raw_skill_logits: np.ndarray,
    skill_mask: np.ndarray,
    skill_index: np.ndarray,
    teacher_utilities: np.ndarray,
    skill_names: Sequence[str],
    target_label_available: np.ndarray | None = None,
    skill_conditioning: str = (
        "model-selected target for target_selection; logged active target for "
        "skill_transition"
    ),
) -> dict[str, Any]:
    raw_target_logits = np.asarray(raw_target_logits, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    target_index = np.asarray(target_index, dtype=np.int64)
    raw_skill_logits = np.asarray(raw_skill_logits, dtype=np.float64)
    skill_mask = np.asarray(skill_mask, dtype=bool)
    skill_index = np.asarray(skill_index, dtype=np.int64)
    teacher_utilities = np.asarray(teacher_utilities, dtype=np.float64)
    sample_count = len(target_index)
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty split")
    if raw_target_logits.shape != candidate_mask.shape:
        raise ValueError("Target logits and candidate mask shape mismatch")
    if teacher_utilities.shape != candidate_mask.shape:
        raise ValueError("Teacher utility and candidate mask shape mismatch")
    if raw_skill_logits.shape != skill_mask.shape:
        raise ValueError("Skill logits and skill mask shape mismatch")
    if np.any(~candidate_mask.any(axis=1)) or np.any(~skill_mask.any(axis=1)):
        raise ValueError("Every evaluated sample needs a legal target and skill")
    target_available = (
        np.ones(sample_count, dtype=bool)
        if target_label_available is None
        else np.asarray(target_label_available, dtype=bool)
    )
    if target_available.shape != (sample_count,):
        raise ValueError("target_label_available must be [sample]")

    raw_target_prediction = raw_target_logits.argmax(axis=1)
    target_illegal_before = ~candidate_mask[
        np.arange(sample_count), raw_target_prediction
    ]
    masked_target_logits = np.where(candidate_mask, raw_target_logits, -1.0e30)
    target_prediction = masked_target_logits.argmax(axis=1)
    target_illegal_after = ~candidate_mask[np.arange(sample_count), target_prediction]
    first_legal_target_prediction = candidate_mask.argmax(axis=1)

    ranks: list[int] = []
    for row in np.flatnonzero(target_available):
        valid = np.flatnonzero(candidate_mask[row])
        ordered = valid[np.argsort(-masked_target_logits[row, valid], kind="stable")]
        location = np.flatnonzero(ordered == target_index[row])
        if len(location) != 1:
            raise ValueError(f"Target label is not legal for sample {row}")
        ranks.append(int(location[0]) + 1)
    ranks_array = np.asarray(ranks, dtype=np.float64)

    finite_regret: list[float] = []
    for row in np.flatnonzero(target_available):
        predicted = target_prediction[row]
        actual_value = teacher_utilities[row, target_index[row]]
        predicted_value = teacher_utilities[row, predicted]
        if math.isfinite(actual_value) and math.isfinite(predicted_value):
            finite_regret.append(max(0.0, float(actual_value - predicted_value)))

    raw_skill_prediction = raw_skill_logits.argmax(axis=1)
    skill_illegal_before = ~skill_mask[np.arange(sample_count), raw_skill_prediction]
    masked_skill_logits = np.where(skill_mask, raw_skill_logits, -1.0e30)
    skill_prediction = masked_skill_logits.argmax(axis=1)
    skill_illegal_after = ~skill_mask[np.arange(sample_count), skill_prediction]
    confusion, macro_f1, per_class_f1 = confusion_and_macro_f1(
        skill_index, skill_prediction, len(skill_names)
    )

    if len(ranks_array):
        labelled_target_illegal_before = target_illegal_before[target_available]
        labelled_target_illegal_after = target_illegal_after[target_available]
        target_metrics: dict[str, Any] = {
            "status": "reported",
            "labelled_samples": int(len(ranks_array)),
            "top1_accuracy": float(np.mean(ranks_array <= 1)),
            "top3_accuracy": float(np.mean(ranks_array <= 3)),
            "mrr": float(np.mean(1.0 / ranks_array)),
            "ndcg": float(np.mean(1.0 / np.log2(ranks_array + 1.0))),
            "mean_rank": float(np.mean(ranks_array)),
            "illegal_rate_before_mask": float(np.mean(labelled_target_illegal_before)),
            "illegal_rate_after_mask": float(np.mean(labelled_target_illegal_after)),
            "baselines": {
                "first_legal_top1_accuracy": float(
                    np.mean(
                        first_legal_target_prediction[target_available]
                        == target_index[target_available]
                    )
                ),
                "uniform_legal_expected_top1_accuracy": float(
                    np.mean(1.0 / candidate_mask[target_available].sum(axis=1))
                ),
            },
            "teacher_utility_regret": {
                "available_samples": len(finite_regret),
                "mean": None if not finite_regret else float(np.mean(finite_regret)),
                "median": None if not finite_regret else float(np.median(finite_regret)),
            },
        }
    else:
        target_metrics = {
            "status": "unavailable_no_target_selection_labels",
            "labelled_samples": 0,
            "illegal_rate_before_mask": None,
            "illegal_rate_after_mask": None,
        }

    return {
        "samples": sample_count,
        "target": target_metrics,
        "skill": {
            "macro_f1": macro_f1,
            "per_class_f1": {
                name: value for name, value in zip(skill_names, per_class_f1, strict=True)
            },
            "confusion_matrix": {
                "labels": list(skill_names),
                "rows_actual_columns_predicted": confusion,
            },
            "illegal_rate_before_mask": float(np.mean(skill_illegal_before)),
            "illegal_rate_after_mask": float(np.mean(skill_illegal_after)),
            "baselines": {
                "first_legal_accuracy": float(
                    np.mean(skill_mask.argmax(axis=1) == skill_index)
                ),
            },
            "conditioning": skill_conditioning,
        },
    }


def state_dict_to_json(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "values": tensor.detach().cpu().numpy().tolist(),
        }
        for name, tensor in state_dict.items()
    }


def _flat_float_values(value: torch.Tensor | np.ndarray | Sequence[float]) -> list[float]:
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    ).astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Unity export cannot contain non-finite floats")
    return [float(item) for item in array.reshape(-1)]


def _flat_bool_values(value: np.ndarray | Sequence[bool]) -> list[bool]:
    return [bool(item) for item in np.asarray(value, dtype=bool).reshape(-1)]


def _add_flat_linear(
    payload: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    export_prefix: str,
    state_prefix: str,
) -> None:
    weight_key = f"{state_prefix}.weight"
    bias_key = f"{state_prefix}.bias"
    if weight_key not in state_dict or bias_key not in state_dict:
        raise ValueError(f"Checkpoint is missing deployment layer {state_prefix}")
    weight = state_dict[weight_key].detach().cpu().numpy().astype(np.float32)
    bias = state_dict[bias_key].detach().cpu().numpy().astype(np.float32)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError(f"Unexpected Linear tensor shape for {state_prefix}")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError(f"Non-finite deployment weights in {state_prefix}")
    payload[f"{export_prefix}InputSize"] = int(weight.shape[1])
    payload[f"{export_prefix}OutputSize"] = int(weight.shape[0])
    payload[f"{export_prefix}Weights"] = _flat_float_values(weight)
    payload[f"{export_prefix}Bias"] = _flat_float_values(bias)


def _linear_from_flat_export(
    payload: dict[str, Any], prefix: str, values: np.ndarray
) -> np.ndarray:
    input_size = int(payload[f"{prefix}InputSize"])
    output_size = int(payload[f"{prefix}OutputSize"])
    weight = np.asarray(payload[f"{prefix}Weights"], dtype=np.float32).reshape(
        output_size, input_size
    )
    bias = np.asarray(payload[f"{prefix}Bias"], dtype=np.float32)
    if bias.shape != (output_size,) or values.shape[-1] != input_size:
        raise ValueError(f"Unity flat layer dimension mismatch for {prefix}")
    return values.astype(np.float32) @ weight.T + bias


def predict_numpy_from_unity_flat_export(
    payload: dict[str, Any],
    global_features: np.ndarray | Sequence[float],
    candidate_features: np.ndarray | Sequence[Sequence[float]],
    candidate_mask: np.ndarray | Sequence[bool],
    target_mask: np.ndarray | Sequence[bool],
    *,
    selected_target_index: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Execute one Decision Brain sample using only the Unity-flat JSON fields.

    This intentionally mirrors the small C# inference path and is used as the
    independent golden-vector oracle. Inputs are encoded but unnormalised.
    """

    if payload.get("schemaVersion") != UNITY_FLAT_SCHEMA_VERSION:
        raise ValueError("Unsupported Unity Decision Brain flat schema")
    global_values = np.asarray(global_features, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidate_features, dtype=np.float32)
    present = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    legal = np.asarray(target_mask, dtype=bool).reshape(-1)
    if candidates.ndim != 2:
        raise ValueError("candidate_features must be [candidate, feature]")
    if global_values.shape != (int(payload["globalDim"]),):
        raise ValueError("Unity flat global feature dimension mismatch")
    if candidates.shape[1] != int(payload["candidateDim"]):
        raise ValueError("Unity flat candidate feature dimension mismatch")
    if present.shape != (len(candidates),) or legal.shape != present.shape:
        raise ValueError("Unity flat candidate mask dimension mismatch")
    if np.any(legal & ~present) or not present.any() or not legal.any():
        raise ValueError("Unity flat masks contain no legal target or authorize padding")

    global_mean = np.asarray(payload["globalMean"], dtype=np.float32)
    global_std = np.asarray(payload["globalStd"], dtype=np.float32)
    candidate_mean = np.asarray(payload["candidateMean"], dtype=np.float32)
    candidate_std = np.asarray(payload["candidateStd"], dtype=np.float32)
    normalized_global = (global_values - global_mean) / global_std
    normalized_candidates = (candidates - candidate_mean) / candidate_std

    encoded_global = np.maximum(
        _linear_from_flat_export(
            payload, "globalEncoderLinear1", normalized_global
        ),
        0.0,
    )
    encoded_global = np.maximum(
        _linear_from_flat_export(payload, "globalEncoderLinear2", encoded_global),
        0.0,
    )
    encoded_candidates = np.maximum(
        _linear_from_flat_export(
            payload, "candidateEncoderLinear1", normalized_candidates
        ),
        0.0,
    )
    encoded_candidates = np.maximum(
        _linear_from_flat_export(
            payload, "candidateEncoderLinear2", encoded_candidates
        ),
        0.0,
    )
    pooled_mean = encoded_candidates[present].mean(axis=0)
    pooled_max = encoded_candidates[present].max(axis=0)
    repeated_global = np.repeat(encoded_global[None, :], len(candidates), axis=0)
    repeated_mean = np.repeat(pooled_mean[None, :], len(candidates), axis=0)
    repeated_max = np.repeat(pooled_max[None, :], len(candidates), axis=0)
    joint_candidates = np.concatenate(
        [repeated_global, encoded_candidates, repeated_mean, repeated_max], axis=1
    )
    target_hidden = np.maximum(
        _linear_from_flat_export(payload, "targetHeadLinear1", joint_candidates),
        0.0,
    )
    raw_target_logits = _linear_from_flat_export(
        payload, "targetHeadLinear2", target_hidden
    ).reshape(-1)
    masked_target_logits = np.where(legal, raw_target_logits, -1.0e9).astype(
        np.float32
    )
    if selected_target_index is None:
        selected_target_index = int(masked_target_logits.argmax())
    selected_target_index = int(selected_target_index)
    if not 0 <= selected_target_index < len(candidates) or not present[
        selected_target_index
    ]:
        raise ValueError("Unity flat selected target is absent")
    joint_selected = np.concatenate(
        [
            encoded_global,
            encoded_candidates[selected_target_index],
            pooled_mean,
            pooled_max,
        ]
    )
    skill_hidden = np.maximum(
        _linear_from_flat_export(payload, "skillHeadLinear1", joint_selected),
        0.0,
    )
    raw_skill_logits = _linear_from_flat_export(
        payload, "skillHeadLinear2", skill_hidden
    ).reshape(-1)
    return {
        "normalized_global_features": normalized_global.astype(np.float32),
        "normalized_candidate_features": normalized_candidates.astype(np.float32),
        "raw_target_logits": raw_target_logits.astype(np.float32),
        "masked_target_logits": masked_target_logits,
        "selected_target_index": selected_target_index,
        "raw_skill_logits": raw_skill_logits.astype(np.float32),
    }


def assert_unity_flat_payload(payload: dict[str, Any]) -> None:
    """Reject structures that Unity JsonUtility cannot bind to the flat DTO."""

    if payload.get("schemaVersion") != UNITY_FLAT_SCHEMA_VERSION:
        raise ValueError("Unity flat payload schemaVersion is missing or unsupported")
    primitive_types = (str, int, float, bool, type(None))
    for name, value in payload.items():
        if isinstance(value, dict):
            raise ValueError(f"Unity flat field {name} must not be a dictionary")
        if isinstance(value, (list, tuple)):
            if any(not isinstance(item, primitive_types) for item in value):
                raise ValueError(
                    f"Unity flat array {name} must contain primitive values only"
                )
        elif not isinstance(value, primitive_types):
            raise ValueError(f"Unity flat field {name} has unsupported type {type(value)}")


def build_unity_flat_export(
    *,
    state_dict: dict[str, torch.Tensor],
    architecture: dict[str, Any],
    normalization: dict[str, Sequence[float]],
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    generated_at_utc: str,
    data_status: str,
    synthetic_fixture: bool,
    checkpoint_sha256: str,
    python_export_sha256: str,
    data_fingerprint_sha256: str,
    config_sha256: str,
    source_counts: dict[str, int],
    decision_kind_counts: dict[str, int],
    human_decision_samples: int,
    seed: int,
    best_epoch: int,
    golden_global_features: np.ndarray,
    golden_candidate_features: np.ndarray,
    golden_candidate_mask: np.ndarray,
    golden_target_mask: np.ndarray,
    golden_skill_mask: np.ndarray,
    golden_target_index: int,
    golden_decision_kind: str,
    golden_decision_id: str,
    golden_episode_id: str,
    golden_split_index: int = 0,
) -> dict[str, Any]:
    """Build a self-contained, dictionary-free Unity JsonUtility asset."""

    if golden_decision_kind not in {"target_selection", "skill_transition"}:
        raise ValueError("Golden decision kind is unsupported")
    source_names = sorted(source_counts)
    decision_kind_names = sorted(decision_kind_counts)
    stop_contract = preprocessing["virtual_stop_candidate"]
    candidate_values = np.asarray(golden_candidate_features, dtype=np.float32)
    if candidate_values.ndim != 2:
        raise ValueError("Golden candidate features must be [candidate, feature]")
    selected_target_index: int | None = (
        None
        if golden_decision_kind == "target_selection"
        else int(golden_target_index)
    )
    payload: dict[str, Any] = {
        "schemaVersion": UNITY_FLAT_SCHEMA_VERSION,
        "generatedAtUtc": str(generated_at_utc),
        "modelName": str(config["claims"]["model_name"]),
        "scope": str(config["claims"]["scope"]),
        "dataStatus": str(data_status),
        "syntheticFixture": bool(synthetic_fixture),
        "modelSha256": str(checkpoint_sha256),
        "checkpointSha256": str(checkpoint_sha256),
        "pythonExportSha256": str(python_export_sha256),
        "dataFingerprintSha256": str(data_fingerprint_sha256),
        "configSha256": str(config_sha256),
        "contextSchemaVersion": str(config["context_schema_version"]),
        "outcomeSchemaVersion": str(config["outcome_schema_version"]),
        "preprocessingSchemaVersion": str(preprocessing["schema_version"]),
        "tensorLayout": "row-major; candidate tensors are candidate-major",
        "activation": "ReLU",
        "pooling": "masked_mean+masked_max over present candidates",
        "dropoutAppliedAtInference": False,
        "globalDim": int(architecture["global_dim"]),
        "candidateDim": int(architecture["candidate_dim"]),
        "skillCount": int(architecture["skill_count"]),
        "globalHiddenDim": int(architecture["global_hidden_dim"]),
        "candidateHiddenDim": int(architecture["candidate_hidden_dim"]),
        "jointHiddenDim": int(architecture["joint_hidden_dim"]),
        "trainingDropout": float(architecture["dropout"]),
        "globalFeatureNames": list(global_feature_names(config)),
        "candidateFeatureNames": list(candidate_feature_names(config)),
        "skills": list(config["skills"]),
        "brainStates": list(config["brain_states"]),
        "previousSkillVocabulary": [*config["skills"], "Unknown"],
        "plannerModes": list(config["planner_modes"]),
        "semanticTypes": list(config["semantic_types"]),
        "globalMean": _flat_float_values(normalization["global_mean"]),
        "globalStd": _flat_float_values(normalization["global_std"]),
        "candidateMean": _flat_float_values(normalization["candidate_mean"]),
        "candidateStd": _flat_float_values(normalization["candidate_std"]),
        "normalizationFittedOn": str(normalization["fitted_on"]),
        "appendStopCandidate": bool(preprocessing["append_stop_candidate"]),
        "stopCandidateTargetId": str(
            stop_contract["raw_template"]["targetId"]
        ),
        "stopCandidateEncodedFeatures": _flat_float_values(
            stop_contract["encoded_feature_vector"]
        ),
        "stopCandidateLegalWhenSkill": "Stop",
        "globalNumericFields": [
            str(item["field"]) for item in preprocessing["global_numeric"]
        ],
        "globalOneHotFields": [
            str(item["field"]) for item in preprocessing["global_one_hot"]
        ],
        "candidateNumericFields": [
            str(item["field"]) for item in preprocessing["candidate_numeric"]
        ],
        "candidateBooleanFields": [
            str(item["field"]) for item in preprocessing["candidate_boolean"]
        ],
        "candidateSemanticField": str(preprocessing["candidate_one_hot"]["field"]),
        "unknownBrainState": "Unknown",
        "unknownPreviousSkill": "Unknown",
        "unknownPlannerMode": "unknown",
        "unknownSemanticType": "unknown",
        "targetMaskPreferredField": "selectable",
        "targetMaskFallbackRule": str(preprocessing["target_mask"]["fallback"]),
        "skillMaskSourceField": "validSkills",
        "targetSelectionSkillConditioning": "model-ranked legal target",
        "skillTransitionSkillConditioning": "runtime active target index",
        "sourceCountNames": source_names,
        "sourceCountValues": [int(source_counts[name]) for name in source_names],
        "decisionKindCountNames": decision_kind_names,
        "decisionKindCountValues": [
            int(decision_kind_counts[name]) for name in decision_kind_names
        ],
        "humanDecisionSamples": int(human_decision_samples),
        "humanLikeClaimAllowed": False,
        "trainingSeed": int(seed),
        "bestEpoch": int(best_epoch),
        "modelMayRankTargets": True,
        "modelMayRecommendMacroSkill": True,
        "modelMayBypassNavmeshOrCollisionChecks": False,
        "modelMayAuthorizeInteractionMethods": False,
        "modelMayBypassSkillMask": False,
        "goldenSplit": "test",
        "goldenSplitIndex": int(golden_split_index),
        "goldenEpisodeId": str(golden_episode_id),
        "goldenDecisionId": str(golden_decision_id),
        "goldenDecisionKind": str(golden_decision_kind),
        "goldenCandidateCount": int(candidate_values.shape[0]),
        "goldenGlobalFeatures": _flat_float_values(golden_global_features),
        "goldenCandidateFeatures": _flat_float_values(candidate_values),
        "goldenCandidateMask": _flat_bool_values(golden_candidate_mask),
        "goldenTargetMask": _flat_bool_values(golden_target_mask),
        "goldenSkillMask": _flat_bool_values(golden_skill_mask),
        "goldenLoggedTargetIndex": int(golden_target_index),
        "goldenTolerance": 0.00005,
    }
    for export_prefix, state_prefix in UNITY_LINEAR_LAYERS:
        _add_flat_linear(payload, state_dict, export_prefix, state_prefix)
    prediction = predict_numpy_from_unity_flat_export(
        payload,
        golden_global_features,
        candidate_values,
        golden_candidate_mask,
        golden_target_mask,
        selected_target_index=selected_target_index,
    )
    skill_mask = np.asarray(golden_skill_mask, dtype=bool).reshape(-1)
    if skill_mask.shape != (int(architecture["skill_count"]),) or not skill_mask.any():
        raise ValueError("Golden skill mask has the wrong size or no legal skill")
    raw_skill_logits = np.asarray(prediction["raw_skill_logits"], dtype=np.float32)
    payload.update(
        {
            "goldenNormalizedGlobalFeatures": _flat_float_values(
                prediction["normalized_global_features"]
            ),
            "goldenNormalizedCandidateFeatures": _flat_float_values(
                prediction["normalized_candidate_features"]
            ),
            "goldenSkillConditioningTargetIndex": int(
                prediction["selected_target_index"]
            ),
            "goldenExpectedRawTargetLogits": _flat_float_values(
                prediction["raw_target_logits"]
            ),
            "goldenExpectedMaskedTargetLogits": _flat_float_values(
                prediction["masked_target_logits"]
            ),
            "goldenExpectedRawSkillLogits": _flat_float_values(raw_skill_logits),
            "goldenExpectedMaskedSkillLogits": _flat_float_values(
                np.where(skill_mask, raw_skill_logits, -1.0e9)
            ),
        }
    )
    assert_unity_flat_payload(payload)
    return payload


def count_human_decisions(source_kinds: Iterable[str], config: dict[str, Any]) -> int:
    human = set(config["human_source_kinds"])
    return sum(str(value) in human for value in source_kinds)


def human_metrics_or_refusal(
    source_kinds: np.ndarray,
    config: dict[str, Any],
    evaluator: Any,
) -> dict[str, Any]:
    human = set(config["human_source_kinds"])
    indices = np.asarray(
        [index for index, source in enumerate(source_kinds.tolist()) if source in human],
        dtype=np.int64,
    )
    if len(indices) == 0:
        return {
            "status": "refused_no_human_decisions",
            "human_decision_samples": 0,
            "reason": (
                "No public_human or unity_human decision labels are present; "
                "rule/scripted bootstrap performance is not a human-behaviour metric."
            ),
        }
    return {
        "status": "reported_descriptive_only",
        "human_decision_samples": int(len(indices)),
        "human_like_claim_allowed": False,
        "metrics": evaluator(indices),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
