#!/usr/bin/env python3
"""Small, auditable six-action policy used by the Habitat-Web decision prior.

The network intentionally consumes only numeric features that can be recreated
at runtime from the current pose, a supplied target waypoint, and past actions.
It has no visual encoder, map, scene identifier, target category, or interaction
head.  Keeping the model here separate from preparation and training makes the
runtime contract and JSON export straightforward to test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ACTIONS: tuple[str, ...] = (
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LOOK_UP",
    "LOOK_DOWN",
    "STOP",
)
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTIONS)}

FEATURE_NAMES: tuple[str, ...] = (
    "goal_local_right_m",
    "goal_local_up_m",
    "goal_local_forward_m",
    "goal_horizontal_distance_m",
    "goal_distance_m",
    "goal_direction_right",
    "goal_direction_up",
    "goal_direction_forward",
    "current_forward_vertical",
    "current_up_vertical",
    "current_height_minus_start_m",
    "travelled_distance_m",
    "log1p_prior_steps",
    *(f"previous_action_{name}" for name in ACTIONS),
    *(f"prior_action_fraction_{name}" for name in ACTIONS),
    "same_action_run_fraction",
    "last_step_local_right_m",
    "last_step_local_up_m",
    "last_step_local_forward_m",
    "last_rotation_angle_rad",
)


class HabitatDecisionMLP(nn.Module):
    """Two-hidden-layer MLP for a single six-action decision."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dims: Sequence[int] = (64, 64),
        dropout: float = 0.1,
        output_dim: int = len(ACTIONS),
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim != len(ACTIONS):
            raise ValueError("invalid input/output dimension")
        if not hidden_dims or any(int(width) < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        layers.append(nn.Linear(previous, int(output_dim)))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)
        self.output_dim = int(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        return self.network(features)


def build_model(config: Mapping[str, Any]) -> HabitatDecisionMLP:
    return HabitatDecisionMLP(
        input_dim=int(config.get("input_dim", len(FEATURE_NAMES))),
        hidden_dims=tuple(config.get("hidden_dims", (64, 64))),
        dropout=float(config.get("dropout", 0.1)),
        output_dim=int(config.get("output_dim", len(ACTIONS))),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_features(
    features: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if center.shape != (len(FEATURE_NAMES),) or scale.shape != center.shape:
        raise ValueError("normalization vectors do not match FEATURE_NAMES")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("normalization std must be finite and positive")
    return (array - center) / scale


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    required = {
        "schema_version",
        "state_dict",
        "model_config",
        "feature_names",
        "action_classes",
        "normalization",
        "provenance",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    if list(checkpoint["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("checkpoint feature contract differs from runtime")
    if list(checkpoint["action_classes"]) != list(ACTIONS):
        raise ValueError("checkpoint action contract differs from runtime")
    normalization = checkpoint["normalization"]
    normalize_features(
        np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        normalization["mean"],
        normalization["std"],
    )
    return checkpoint


def _linear_layers(model: HabitatDecisionMLP) -> list[nn.Linear]:
    return [module for module in model.network if isinstance(module, nn.Linear)]


def predict_numpy_from_export(
    exported: Mapping[str, Any], raw_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference NumPy inference for validating the self-contained export."""

    normalization = exported["normalization"]
    value = normalize_features(
        np.asarray(raw_features, dtype=np.float32),
        normalization["mean"],
        normalization["std"],
    ).astype(np.float64)
    layers = exported["layers"]
    for index, layer in enumerate(layers):
        weight = np.asarray(layer["weight"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        value = value @ weight.T + bias
        if index + 1 < len(layers):
            value = np.maximum(value, 0.0)
    logits = value
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return logits, probabilities


def export_runtime_json(
    *,
    model: HabitatDecisionMLP,
    normalization: Mapping[str, Sequence[float]],
    output_path: str | Path,
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
    golden_raw_features: Sequence[float],
) -> dict[str, Any]:
    model = model.cpu().eval()
    raw = np.asarray(golden_raw_features, dtype=np.float32).reshape(1, -1)
    normalized = normalize_features(raw, normalization["mean"], normalization["std"])
    with torch.no_grad():
        logits = model(torch.from_numpy(normalized)).numpy()
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    payload: dict[str, Any] = {
        "schema_version": "habitat-decision-prior-runtime-1.0",
        "model_type": "feedforward_relu_mlp",
        "feature_names": list(FEATURE_NAMES),
        "action_classes": list(ACTIONS),
        "normalization": {
            "mean": [float(value) for value in normalization["mean"]],
            "std": [float(value) for value in normalization["std"]],
        },
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dims": list(model.hidden_dims),
            "dropout_training_only": model.dropout,
            "output_dim": model.output_dim,
        },
        "layers": [
            {
                "weight": layer.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": layer.bias.detach().cpu().numpy().astype(float).tolist(),
            }
            for layer in _linear_layers(model)
        ],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metadata": dict(metadata),
        "golden_vector": {
            "raw_features": raw[0].astype(float).tolist(),
            "normalized_features": normalized[0].astype(float).tolist(),
            "logits": logits[0].astype(float).tolist(),
            "probabilities": probabilities[0].astype(float).tolist(),
            "predicted_action": ACTIONS[int(np.argmax(probabilities[0]))],
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    exported_logits, exported_probabilities = predict_numpy_from_export(payload, raw)
    if not np.allclose(exported_logits, logits, atol=2e-5, rtol=2e-5):
        raise RuntimeError("JSON export logits failed the golden-vector check")
    if not np.allclose(exported_probabilities, probabilities, atol=2e-5, rtol=2e-5):
        raise RuntimeError("JSON export probabilities failed the golden-vector check")
    return payload

