#!/usr/bin/env python3
"""Export Normal HMD Agent v3 to the dependency-free Unity runtime.

The export freezes four related contracts in one JSON asset:

* the shared two-layer LSTM and seven-dimensional main-motion decoder;
* the six raw vertical-event heads and their physical normalisation;
* the closed vertical event controller used for HeadLocal Y and pitch; and
* a root-only joint AR(1) innovation process.  HeadLocal innovation is
  structurally zero.

Golden vectors are regenerated on CPU from the frozen checkpoint instead of
being copied from a training run.  They therefore test the same FP32 execution
order that the dependency-free C# implementation is expected to mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from normal_hmd_common import STATE_FEATURE_NAMES, sha256_file
from normal_hmd_v3_model import (
    MAIN_STATE_INDICES,
    PITCH_INDEX,
    VERTICAL_INDEX,
    NormalHMDV3,
    V3Step,
    VerticalControllerState,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "normal_hmd_v3.pth"
DEFAULT_METADATA = ROOT / "models" / "normal_hmd_v3_metadata.json"
DEFAULT_METRICS = ROOT / "results" / "normal_hmd_v3_training_metrics.json"
DEFAULT_DATA_REPORT = ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_ASSET = (
    ROOT
    / "demo"
    / "UnityNormalHmdAgentDemo"
    / "Assets"
    / "Resources"
    / "NormalHmdAgentModel.json"
)
DEFAULT_INTEGRITY = (
    ROOT
    / "demo"
    / "UnityNormalHmdAgentDemo"
    / "Assets"
    / "Scripts"
    / "GeneratedNormalHmdIntegrity.cs"
)
DEFAULT_REPORT = ROOT / "results" / "normal_hmd_v3_unity_export_report.json"

SCHEMA_VERSION = "normal-hmd-agent-unity-3.0"
DEPLOYMENT_ROLL_LIMIT_RAD = np.float32(0.12)
MAIN_HEAD_SMOOTHING_INDICES = (3, 5, 7, 8)
DEFAULT_MAIN_HEAD_SMOOTHING_TAU_SECONDS = 0.05
RAW_HEAD_NAMES = (
    "event_gate_logit",
    "vertical_velocity_normalized",
    "vertical_amplitude_normalized",
    "pitch_velocity_normalized",
    "pitch_amplitude_normalized",
    "vertical_log_duration_normalized",
)
CONTROLLER_STATE_NAMES = (
    "active",
    "previous_gate",
    "elapsed_steps",
    "duration_steps",
    "baseline_y_physical",
    "amplitude_y_physical",
    "baseline_pitch_physical",
    "amplitude_pitch_physical",
)


def floats(value: Any) -> list[float]:
    """Flatten as IEEE-754 single values before JSON encoding."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def require_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def deployment_state_clamps(
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deployment comfort envelope without changing training data.

    State index 8 is the HeadLocal rotation-vector Z component.  The learned
    distribution is retained in the checkpoint, while the visible demo uses a
    tighter roll envelope after inference and before recurrent feedback.
    """

    low = np.asarray(checkpoint["state_clip_low"], dtype=np.float32).copy()
    high = np.asarray(checkpoint["state_clip_high"], dtype=np.float32).copy()
    low[8] = max(float(low[8]), -float(DEPLOYMENT_ROLL_LIMIT_RAD))
    high[8] = min(float(high[8]), float(DEPLOYMENT_ROLL_LIMIT_RAD))
    if np.any(high <= low):
        raise ValueError("deployment comfort envelope produced an invalid clamp")
    return low, high


def choose_seed_windows(
    test: dict[str, np.ndarray], per_trajectory: int
) -> list[dict[str, Any]]:
    required = {"X", "trajectory_index", "trajectory_ids", "start_index"}
    missing = required - set(test)
    if missing:
        raise ValueError(f"held-out NPZ is missing seed fields: {sorted(missing)}")
    trajectory_index = test["trajectory_index"].astype(np.int64)
    seeds: list[dict[str, Any]] = []
    for index, trajectory_id in enumerate(test["trajectory_ids"]):
        candidates = np.flatnonzero(trajectory_index == index)
        if not len(candidates):
            continue
        count = min(per_trajectory, len(candidates))
        selected = candidates[np.linspace(0, len(candidates) - 1, count, dtype=np.int64)]
        for row in selected:
            history = np.asarray(test["X"][row], dtype=np.float32)
            require_finite(f"held-out seed row {row}", history)
            seeds.append(
                {
                    "trajectoryId": str(trajectory_id),
                    "split": "test",
                    "startFrame": int(test["start_index"][row]),
                    "historyRaw": floats(history),
                }
            )
    if not seeds:
        raise ValueError("no held-out seed histories found")
    return seeds


class PortableGaussian:
    """xorshift32 plus cached Box-Muller, mirrored by the Unity runtime."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & 0xFFFFFFFF
        if self.state == 0:
            self.state = 0x6D2B79F5
        self.spare: float | None = None

    def next_uint(self) -> int:
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state

    def uniform_open(self) -> float:
        return (self.next_uint() + 0.5) / 4294967296.0

    def gaussian(self) -> float:
        if self.spare is not None:
            result = self.spare
            self.spare = None
            return result
        u1 = self.uniform_open()
        u2 = self.uniform_open()
        magnitude = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self.spare = magnitude * math.sin(angle)
        return magnitude * math.cos(angle)


class RootInnovation(NamedTuple):
    rho: np.ndarray
    stationary_cholesky_physical: np.ndarray
    default_noise_scale: float
    normalized_std: np.ndarray
    source: str


def resolve_root_innovation(
    checkpoint: dict[str, Any], state_std: np.ndarray
) -> RootInnovation:
    """Resolve the deployed root-only joint AR model.

    v3 training deliberately augments only normalized BodyRoot feedback.  Its
    default augmentation is isotropic, represented here as a diagonal joint
    stationary covariance.  A future checkpoint may freeze a non-diagonal
    Cholesky factor without changing the Unity schema.
    """

    configured = checkpoint.get("root_innovation_model")
    if configured is not None:
        rho_value = configured.get("ar1_rho", configured.get("rho"))
        cholesky_value = configured.get(
            "stationary_cholesky_physical",
            configured.get("cholesky_physical"),
        )
        if rho_value is None:
            raise ValueError("root_innovation_model requires ar1_rho")
        rho = np.broadcast_to(np.asarray(rho_value, dtype=np.float32), (3,)).copy()
        if cholesky_value is not None:
            cholesky = np.asarray(cholesky_value, dtype=np.float32).reshape(3, 3)
        elif configured.get("physical_covariance") is not None:
            covariance = np.asarray(
                configured["physical_covariance"], dtype=np.float64
            ).reshape(3, 3)
            cholesky = np.linalg.cholesky(covariance).astype(np.float32)
        elif configured.get("stationary_std_physical") is not None:
            physical_std = np.asarray(
                configured["stationary_std_physical"], dtype=np.float32
            ).reshape(3)
            cholesky = np.diag(physical_std).astype(np.float32)
        else:
            raise ValueError(
                "root_innovation_model requires a stationary physical "
                "Cholesky factor, covariance, or standard deviation"
            )
        default_scale = float(configured.get("default_noise_scale", 1.0))
        normalized_value = configured.get(
            "stationary_std_normalized", configured.get("normalized_std", 0.0)
        )
        normalized_std = np.broadcast_to(
            np.asarray(normalized_value, dtype=np.float32), (3,)
        ).copy()
        source = "checkpoint.root_innovation_model"
    else:
        policy = checkpoint.get("stochastic_policy", {})
        normalized_scalar = float(
            policy.get(
                "deployment_root_noise_std_normalized",
                policy.get("training_root_feedback_noise_std_normalized", 0.04),
            )
        )
        normalized_std = np.full(3, normalized_scalar, dtype=np.float32)
        rho = np.full(3, np.float32(0.90), dtype=np.float32)
        cholesky = np.diag(
            np.asarray(state_std[:3], dtype=np.float32) * np.float32(normalized_scalar)
        ).astype(np.float32)
        default_scale = 1.0
        source = "training root-feedback augmentation (diagonal stationary covariance)"

    require_finite("root innovation rho", rho)
    require_finite("root innovation Cholesky", cholesky)
    require_finite("root innovation normalized standard deviation", normalized_std)
    if np.any(np.abs(rho) >= 1.0):
        raise ValueError("root innovation AR coefficients must have abs(rho) < 1")
    if not np.allclose(cholesky, np.tril(cholesky), atol=0.0, rtol=0.0):
        raise ValueError("root innovation Cholesky must be lower triangular")
    if np.any(np.diag(cholesky) < 0.0):
        raise ValueError("root innovation Cholesky diagonal cannot be negative")
    if not math.isfinite(default_scale) or default_scale < 0.0:
        raise ValueError("default root noise scale must be finite and non-negative")
    return RootInnovation(rho, cholesky, default_scale, normalized_std, source)


def portable_joint_root_innovation(
    innovation: RootInnovation, steps: int, seed: int
) -> np.ndarray:
    """Generate the exact root AR residual expected from the C# implementation.

    For each frame three Gaussian values are consumed first.  Each triangular
    product and accumulation is rounded to float32.  The Cholesky factor is for
    the *stationary* covariance, hence sqrt(1-rho^2) is applied afterward.
    """

    rng = PortableGaussian(seed)
    residual = np.zeros(3, dtype=np.float32)
    output = np.empty((steps, 3), dtype=np.float32)
    rho = np.asarray(innovation.rho, dtype=np.float32)
    cholesky = np.asarray(
        innovation.stationary_cholesky_physical, dtype=np.float32
    )
    for step_index in range(steps):
        gaussian = np.asarray([rng.gaussian() for _ in range(3)], dtype=np.float32)
        epsilon = np.zeros(3, dtype=np.float32)
        for row in range(3):
            accumulated = np.float32(0.0)
            for column in range(row + 1):
                product = np.float32(cholesky[row, column] * gaussian[column])
                accumulated = np.float32(accumulated + product)
            epsilon[row] = accumulated
        for feature in range(3):
            rho_square = np.float32(rho[feature] * rho[feature])
            gain = math.sqrt(max(0.0, 1.0 - float(rho_square)))
            autoregressive = np.float32(rho[feature] * residual[feature])
            residual[feature] = np.float32(
                float(autoregressive) + gain * float(epsilon[feature])
            )
        output[step_index] = residual
    return output


def build_model(checkpoint: dict[str, Any]) -> NormalHMDV3:
    config = checkpoint["model_config"]
    normalization = checkpoint["normalization"]
    model = NormalHMDV3(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        fixed_dt_seconds=float(checkpoint["dt_seconds"]),
        vertical_state_mean=float(normalization["state_mean"][VERTICAL_INDEX]),
        vertical_state_std=float(normalization["state_std"][VERTICAL_INDEX]),
        vertical_velocity_mean=float(normalization["vertical_velocity_mean_mps"]),
        vertical_velocity_std=float(normalization["vertical_velocity_std_mps"]),
        vertical_amplitude_mean=float(normalization["vertical_amplitude_mean_m"]),
        vertical_amplitude_std=float(normalization["vertical_amplitude_std_m"]),
        vertical_log_duration_mean=float(
            normalization["vertical_log_duration_mean"]
        ),
        vertical_log_duration_std=float(normalization["vertical_log_duration_std"]),
        pitch_state_mean=float(normalization["state_mean"][PITCH_INDEX]),
        pitch_state_std=float(normalization["state_std"][PITCH_INDEX]),
        pitch_velocity_mean=float(normalization["pitch_velocity_mean_radps"]),
        pitch_velocity_std=float(normalization["pitch_velocity_std_radps"]),
        pitch_amplitude_mean=float(normalization["pitch_amplitude_mean_rad"]),
        pitch_amplitude_std=float(normalization["pitch_amplitude_std_rad"]),
        event_gate_threshold=float(config.get("event_gate_threshold", 0.5)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def choose_golden_history(
    model: NormalHMDV3,
    test: dict[str, np.ndarray],
    checkpoint: dict[str, Any],
    candidate_count: int = 512,
) -> tuple[np.ndarray, int, float]:
    """Choose a held-out history that actually exercises the event controller.

    The first candidate row is not guaranteed to cross the event threshold.
    Selecting the largest first-step gate probability makes the golden vector
    cover amplitude/duration latching and the closed bump whenever the trained
    model emits an event, without looking at held-out labels.
    """

    histories = np.asarray(test["X"], dtype=np.float32)
    if not len(histories):
        raise ValueError("held-out split has no histories")
    count = min(candidate_count, len(histories))
    candidates = np.linspace(0, len(histories) - 1, count, dtype=np.int64)
    mean = np.asarray(checkpoint["normalization"]["state_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["state_std"], dtype=np.float32)
    best_row = int(candidates[0])
    best_probability = -math.inf
    for start in range(0, len(candidates), 64):
        rows = candidates[start : start + 64]
        normalized = ((histories[rows] - mean) / std).astype(np.float32)
        prediction, _ = model.warm_history(torch.from_numpy(normalized))
        probability = torch.sigmoid(prediction.vertical_event_logit)
        local = int(torch.argmax(probability).item())
        value = float(probability[local].item())
        if value > best_probability:
            best_probability = value
            best_row = int(rows[local])
    return histories[best_row], best_row, best_probability


def step_arrays(step: V3Step) -> tuple[np.ndarray, np.ndarray]:
    main = step.main_normalized[0].detach().cpu().numpy().astype(np.float32)
    heads = np.asarray(
        [
            step.vertical_event_logit[0].item(),
            step.vertical_velocity_normalized[0].item(),
            step.vertical_amplitude_normalized[0].item(),
            step.pitch_velocity_normalized[0].item(),
            step.pitch_amplitude_normalized[0].item(),
            step.vertical_log_duration_normalized[0].item(),
        ],
        dtype=np.float32,
    )
    return main, heads


def controller_array(controller: VerticalControllerState) -> np.ndarray:
    return np.asarray(
        [
            float(bool(controller.active[0].item())),
            float(bool(controller.previous_gate[0].item())),
            controller.elapsed_steps[0].item(),
            controller.duration_steps[0].item(),
            controller.baseline_y_physical[0].item(),
            controller.amplitude_physical[0].item(),
            controller.baseline_pitch_physical[0].item(),
            controller.pitch_amplitude_physical[0].item(),
        ],
        dtype=np.float32,
    )


class GoldenRollout(NamedTuple):
    main_normalized: np.ndarray
    raw_heads: np.ndarray
    raw_composed_before_smoothing: np.ndarray
    composed_before_innovation: np.ndarray
    applied: np.ndarray
    controller: np.ndarray
    root_innovation: np.ndarray


@torch.no_grad()
def cpu_closed_loop_rollout(
    model: NormalHMDV3,
    history_physical: np.ndarray,
    checkpoint: dict[str, Any],
    steps: int,
    root_innovation: np.ndarray | None = None,
    root_noise_scale: float = 1.0,
    main_head_smoothing_tau_seconds: float = (
        DEFAULT_MAIN_HEAD_SMOOTHING_TAU_SECONDS
    ),
) -> GoldenRollout:
    """Run one CPU FP32 closed loop with the deployment feedback contract."""

    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    std = np.asarray(normalization["state_std"], dtype=np.float32)
    clip_low, clip_high = deployment_state_clamps(checkpoint)
    history = np.asarray(history_physical, dtype=np.float32)
    normalized_history = ((history - mean) / std).astype(np.float32)
    prediction, hidden = model.warm_history(torch.from_numpy(normalized_history)[None])
    previous = torch.from_numpy(normalized_history[-1])[None]
    controller = model.initial_vertical_controller(previous)
    smoothing_indices = np.asarray(MAIN_HEAD_SMOOTHING_INDICES, dtype=np.int64)
    smoothing_state = np.clip(history[-1], clip_low, clip_high)[
        smoothing_indices
    ].astype(np.float32)
    smoothing_alpha = np.float32(
        1.0
        if main_head_smoothing_tau_seconds <= 0.0
        else 1.0
        - math.exp(
            -float(checkpoint["dt_seconds"])
            / main_head_smoothing_tau_seconds
        )
    )
    main_rows: list[np.ndarray] = []
    head_rows: list[np.ndarray] = []
    raw_composed_rows: list[np.ndarray] = []
    composed_rows: list[np.ndarray] = []
    applied_rows: list[np.ndarray] = []
    controller_rows: list[np.ndarray] = []
    innovation_rows: list[np.ndarray] = []
    for step_index in range(steps):
        main, heads = step_arrays(prediction)
        composed_normalized, controller = model.compose_state(
            previous, prediction, controller
        )
        composed_physical = (
            composed_normalized[0].detach().cpu().numpy().astype(np.float32) * std
            + mean
        ).astype(np.float32)
        raw_composed_physical = composed_physical.copy()
        smoothing_state = (
            smoothing_state
            + smoothing_alpha
            * (composed_physical[smoothing_indices] - smoothing_state)
        ).astype(np.float32)
        composed_physical[smoothing_indices] = smoothing_state
        composed_clipped = np.clip(composed_physical, clip_low, clip_high).astype(
            np.float32
        )
        applied = composed_clipped.copy()
        current_innovation = np.zeros(3, dtype=np.float32)
        if root_innovation is not None:
            current_innovation = np.asarray(
                root_innovation[step_index], dtype=np.float32
            )
            applied[:3] = (
                applied[:3]
                + current_innovation * np.float32(root_noise_scale)
            ).astype(np.float32)
            applied = np.clip(applied, clip_low, clip_high).astype(np.float32)

        main_rows.append(main)
        head_rows.append(heads)
        raw_composed_rows.append(raw_composed_physical)
        composed_rows.append(composed_clipped)
        applied_rows.append(applied)
        controller_rows.append(controller_array(controller))
        innovation_rows.append(current_innovation)
        # The smoother is part of the recurrent deployment state.  Reset it to
        # the final visible/applied pose after safety clamps, so it cannot hide
        # an out-of-envelope latent value and release it on a later frame.
        smoothing_state = applied[smoothing_indices].copy()
        if step_index + 1 == steps:
            break
        feedback = ((applied - mean) / std).astype(np.float32)
        previous = torch.from_numpy(feedback)[None]
        prediction, hidden = model.advance(previous, hidden)
    result = GoldenRollout(
        np.asarray(main_rows, dtype=np.float32),
        np.asarray(head_rows, dtype=np.float32),
        np.asarray(raw_composed_rows, dtype=np.float32),
        np.asarray(composed_rows, dtype=np.float32),
        np.asarray(applied_rows, dtype=np.float32),
        np.asarray(controller_rows, dtype=np.float32),
        np.asarray(innovation_rows, dtype=np.float32),
    )
    for field_name, value in zip(result._fields, result):
        require_finite(f"golden {field_name}", value)
    return result


def load_and_verify_sources(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, str],
]:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "state_dict",
        "model_config",
        "normalization",
        "state_clip_low",
        "state_clip_high",
        "dt_seconds",
        "feature_names",
        "vertical_event_definition",
        "stochastic_policy",
        "runtime_feedback_contract",
        "data_source",
        "synthetic",
        "dataset_fingerprint_sha256",
        "data_report_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"v3 checkpoint missing fields: {sorted(missing)}")
    if checkpoint.get("schema_version") != "3.0.0":
        raise ValueError("expected Normal HMD checkpoint schema 3.0.0")
    if checkpoint["data_source"] != "EyeNavGS" or bool(checkpoint["synthetic"]):
        raise ValueError("Unity v3 export requires a real, non-synthetic EyeNavGS model")

    source_hash = sha256_file(checkpoint_path)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if str(metadata.get("model_sha256", "")) != source_hash:
        raise ValueError("metadata/checkpoint SHA-256 mismatch")
    if str(metrics.get("artifacts", {}).get("model_sha256", "")) != source_hash:
        raise ValueError("training metrics/checkpoint SHA-256 mismatch")
    if metrics.get("status") != "success":
        raise ValueError("training metrics do not report success")
    if metrics.get("data_source") != "EyeNavGS" or bool(metrics.get("synthetic")):
        raise ValueError("training metrics do not identify real EyeNavGS data")
    if bool(metrics.get("smoke_subset")) and not args.allow_smoke:
        raise ValueError("refusing to export a smoke-subset model as the demo asset")

    data_report_hash = sha256_file(args.data_report)
    if str(checkpoint["data_report_sha256"]) != data_report_hash:
        raise ValueError("checkpoint/data report SHA-256 mismatch")
    if data_report.get("status") != "success" or data_report.get("synthetic") is not False:
        raise ValueError("frozen data report is not a successful real-data report")
    if data_report.get("source_type") != "EyeNavGS":
        raise ValueError("frozen data report does not identify EyeNavGS")
    if (
        str(checkpoint["dataset_fingerprint_sha256"])
        != str(data_report.get("dataset_fingerprint_sha256", ""))
    ):
        raise ValueError("checkpoint/data report dataset fingerprint mismatch")

    split_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = args.data_dir / f"normal_hmd_{split}.npz"
        expected = str(data_report.get("splits", {}).get(split, {}).get("sha256", ""))
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"{split} NPZ hash differs from the frozen real-data report: "
                f"expected={expected!r}, actual={actual!r}"
            )
        split_hashes[split] = actual
    with np.load(args.data_dir / "normal_hmd_test.npz", allow_pickle=False) as archive:
        test = {name: archive[name] for name in archive.files}
    return checkpoint, metadata, metrics, test, split_hashes


def validate_contract(checkpoint: dict[str, Any], model: NormalHMDV3) -> None:
    config = checkpoint["model_config"]
    expected_architecture = "shared_lstm_7d_main_plus_y_pitch_event_heads"
    if str(config.get("architecture")) != expected_architecture:
        raise ValueError("checkpoint is not the frozen Normal HMD v3 architecture")
    if int(config.get("input_dim", -1)) != 9:
        raise ValueError("Unity v3 contract requires a 9-D recurrent state")
    if int(config.get("main_output_dim", -1)) != 7:
        raise ValueError("Unity v3 contract requires a 7-D main decoder")
    if list(config.get("main_state_indices", [])) != list(MAIN_STATE_INDICES):
        raise ValueError("checkpoint main-state index order differs from Unity")
    if int(config.get("vertical_state_index", -1)) != VERTICAL_INDEX:
        raise ValueError("checkpoint vertical state index differs from Unity")
    if int(config.get("pitch_state_index", -1)) != PITCH_INDEX:
        raise ValueError("checkpoint pitch state index differs from Unity")
    if int(config.get("num_layers", -1)) != 2:
        raise ValueError("dependency-free Unity runtime requires exactly two LSTM layers")
    if list(checkpoint["feature_names"]) != list(STATE_FEATURE_NAMES):
        raise ValueError("checkpoint feature order differs from the frozen 9-D contract")

    normalization = checkpoint["normalization"]
    mean = require_finite("state mean", normalization["state_mean"]).reshape(-1)
    std = require_finite("state std", normalization["state_std"]).reshape(-1)
    low = require_finite("state clip low", checkpoint["state_clip_low"]).reshape(-1)
    high = require_finite("state clip high", checkpoint["state_clip_high"]).reshape(-1)
    if any(len(value) != 9 for value in (mean, std, low, high)):
        raise ValueError("state normalization and clamp vectors must all have length 9")
    if np.any(std <= 0.0) or np.any(high <= low):
        raise ValueError("invalid state standard deviation or clamp interval")
    scalar_scales = (
        "vertical_velocity_std_mps",
        "vertical_amplitude_std_m",
        "vertical_log_duration_std",
        "pitch_velocity_std_radps",
        "pitch_amplitude_std_rad",
    )
    if any(float(normalization[name]) <= 0.0 for name in scalar_scales):
        raise ValueError("vertical-head normalisation scales must be positive")

    event = checkpoint["vertical_event_definition"]
    duration_bounds = list(event.get("duration_bounds_steps", []))
    if len(duration_bounds) != 2 or not (
        1 <= int(duration_bounds[0]) <= int(duration_bounds[1])
    ):
        raise ValueError("invalid vertical event duration bounds")
    if "16*u^2*(1-u)^2" not in str(event.get("event_template", "")):
        raise ValueError("Unity export requires the zero-endpoint event bump")
    if "none" not in str(checkpoint["stochastic_policy"].get("headlocal", "")).lower():
        raise ValueError("v3 Unity export forbids HeadLocal stochastic innovation")

    state = checkpoint["state_dict"]
    hidden = int(config["hidden_dim"])
    expected_shapes = {
        "lstm.weight_ih_l0": (4 * hidden, 9),
        "lstm.weight_hh_l0": (4 * hidden, hidden),
        "lstm.bias_ih_l0": (4 * hidden,),
        "lstm.bias_hh_l0": (4 * hidden,),
        "lstm.weight_ih_l1": (4 * hidden, hidden),
        "lstm.weight_hh_l1": (4 * hidden, hidden),
        "lstm.bias_ih_l1": (4 * hidden,),
        "lstm.bias_hh_l1": (4 * hidden,),
        "main_output.weight": (7, hidden),
        "main_output.bias": (7,),
        "vertical_event.weight": (1, hidden),
        "vertical_event.bias": (1,),
        "vertical_velocity.weight": (1, hidden),
        "vertical_velocity.bias": (1,),
        "vertical_amplitude.weight": (1, hidden),
        "vertical_amplitude.bias": (1,),
        "pitch_velocity.weight": (1, hidden),
        "pitch_velocity.bias": (1,),
        "pitch_amplitude.weight": (1, hidden),
        "pitch_amplitude.bias": (1,),
        "vertical_log_duration.weight": (1, hidden),
        "vertical_log_duration.bias": (1,),
    }
    if set(state) != set(expected_shapes):
        unexpected = sorted(set(state) - set(expected_shapes))
        missing = sorted(set(expected_shapes) - set(state))
        raise ValueError(
            f"unexpected v3 state_dict keys; missing={missing}, unexpected={unexpected}"
        )
    for name, shape in expected_shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(state[name].shape)} != {shape}")
        require_finite(name, state[name].detach().cpu().numpy())
    if sum(value.numel() for value in state.values()) != sum(
        value.numel() for value in model.state_dict().values()
    ):
        raise ValueError("reconstructed model parameter count differs from checkpoint")


def validate_production_metrics(
    metrics: dict[str, Any], checkpoint: dict[str, Any], allow_smoke: bool
) -> None:
    if allow_smoke:
        return
    probe = metrics.get("closed_loop_probe", {})
    dt = float(checkpoint["dt_seconds"])
    if int(probe.get("steps", 0)) * dt < 60.0:
        raise ValueError("production export requires a >=60-second closed-loop probe")
    if not bool(probe.get("finite")) or not bool(probe.get("passed")):
        raise ValueError("v3 60-second closed-loop probe did not pass")
    clip_fraction = float(probe.get("clip_step_fraction", math.inf))
    if not math.isfinite(clip_fraction) or clip_fraction > 0.01:
        raise ValueError("v3 probe clips more than one percent of generated steps")
    evaluations = metrics.get("evaluations", {})
    for split in ("val", "test"):
        loss = float(evaluations.get(split, {}).get("closed_loop_loss", math.nan))
        if not math.isfinite(loss):
            raise ValueError(f"{split} closed-loop loss is not finite")


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, metadata, metrics, test, split_hashes = load_and_verify_sources(args)
    model = build_model(checkpoint)
    validate_contract(checkpoint, model)
    validate_production_metrics(metrics, checkpoint, args.allow_smoke)
    seeds = choose_seed_windows(test, args.seeds_per_trajectory)
    config = checkpoint["model_config"]
    history_steps = int(config["history_length"])
    golden_history, golden_source_row, golden_initial_event_probability = (
        choose_golden_history(model, test, checkpoint)
    )
    if golden_history.shape != (history_steps, 9):
        raise ValueError(
            f"held-out golden history shape {golden_history.shape} != {(history_steps, 9)}"
        )

    deterministic = cpu_closed_loop_rollout(
        model,
        golden_history,
        checkpoint,
        args.golden_steps,
        main_head_smoothing_tau_seconds=args.main_head_smoothing_tau_seconds,
    )
    normalization = checkpoint["normalization"]
    state_std = np.asarray(normalization["state_std"], dtype=np.float32)
    innovation = resolve_root_innovation(checkpoint, state_std)
    if args.root_noise_scale is not None:
        innovation = innovation._replace(default_noise_scale=args.root_noise_scale)
    innovation_golden = portable_joint_root_innovation(
        innovation, args.innovation_golden_steps, args.innovation_golden_seed
    )
    combined_innovation = portable_joint_root_innovation(
        innovation, args.combined_golden_steps, args.combined_golden_seed
    )
    combined = cpu_closed_loop_rollout(
        model,
        golden_history,
        checkpoint,
        args.combined_golden_steps,
        root_innovation=combined_innovation,
        root_noise_scale=innovation.default_noise_scale,
        main_head_smoothing_tau_seconds=args.main_head_smoothing_tau_seconds,
    )

    state = checkpoint["state_dict"]
    event = checkpoint["vertical_event_definition"]
    duration_bounds = event["duration_bounds_steps"]
    source_hash = sha256_file(args.checkpoint)
    data_report_hash = sha256_file(args.data_report)
    coordinate_system = json.loads(args.data_report.read_text(encoding="utf-8")).get(
        "coordinate_system", {}
    )
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    deployment_clip_low, deployment_clip_high = deployment_state_clamps(checkpoint)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "exportedAtUtc": datetime.now(timezone.utc).isoformat(),
        "dataSource": "EyeNavGS",
        "syntheticTraining": False,
        "sourceModelSha256": source_hash,
        "sourceDataReportSha256": data_report_hash,
        "datasetFingerprintSha256": str(checkpoint["dataset_fingerprint_sha256"]),
        "preparedTrainSha256": split_hashes["train"],
        "preparedValSha256": split_hashes["val"],
        "preparedTestSha256": split_hashes["test"],
        "modelTask": "event_conditioned_normal_hmd_generation",
        "architecture": str(config["architecture"]),
        "parameterCount": int(sum(value.numel() for value in state.values())),
        "lstmGateOrder": "ifgo",
        "tensorLayout": "row-major",
        "historyLength": history_steps,
        "trainedRolloutLength": int(config["rollout_length"]),
        "inputDim": 9,
        "hiddenDim": int(config["hidden_dim"]),
        "mainOutputDim": 7,
        "verticalHeadCount": 6,
        "numLayers": 2,
        "fixedDtSeconds": float(checkpoint["dt_seconds"]),
        "trainedRolloutSeconds": float(
            int(config["rollout_length"]) * float(checkpoint["dt_seconds"])
        ),
        "nominalEyeHeightM": float(
            data_report.get("body_proxy", {}).get("nominal_eye_height_m", 1.65)
        ),
        "coordinateSystemJson": json.dumps(
            coordinate_system, ensure_ascii=False, separators=(",", ":")
        ),
        "stateFeatureNames": list(STATE_FEATURE_NAMES),
        "mainStateIndices": list(MAIN_STATE_INDICES),
        "verticalStateIndex": VERTICAL_INDEX,
        "pitchStateIndex": PITCH_INDEX,
        "rawHeadNames": list(RAW_HEAD_NAMES),
        "xMean": floats(normalization["state_mean"]),
        "xStd": floats(normalization["state_std"]),
        "stateClipLow": floats(deployment_clip_low),
        "stateClipHigh": floats(deployment_clip_high),
        "trainingStateClipLow": floats(checkpoint["state_clip_low"]),
        "trainingStateClipHigh": floats(checkpoint["state_clip_high"]),
        "deploymentRollClampRad": float(DEPLOYMENT_ROLL_LIMIT_RAD),
        "deploymentComfortEnvelope": (
            "HeadLocal rotvec-Z index 8 is clamped to +/-0.12 rad after inference "
            "and before recurrent feedback; this is a display/deployment comfort "
            "envelope, not the learned EyeNavGS training distribution"
        ),
        "mainHeadSmoothingIndices": list(MAIN_HEAD_SMOOTHING_INDICES),
        "mainHeadSmoothingTauSeconds": float(
            args.main_head_smoothing_tau_seconds
        ),
        "mainHeadSmoothingAlpha": float(
            1.0
            if args.main_head_smoothing_tau_seconds <= 0.0
            else 1.0
            - math.exp(
                -float(checkpoint["dt_seconds"])
                / args.main_head_smoothing_tau_seconds
            )
        ),
        "mainHeadSmoothingPolicy": (
            "deterministic first-order response on HeadLocal X/Z/yaw/roll; "
            "initialized from deployment-clipped final history state and reset "
            "to final applied state every frame before recurrent feedback"
        ),
        "verticalVelocityMean": float(normalization["vertical_velocity_mean_mps"]),
        "verticalVelocityStd": float(normalization["vertical_velocity_std_mps"]),
        "verticalAmplitudeMean": float(normalization["vertical_amplitude_mean_m"]),
        "verticalAmplitudeStd": float(normalization["vertical_amplitude_std_m"]),
        "pitchVelocityMean": float(normalization["pitch_velocity_mean_radps"]),
        "pitchVelocityStd": float(normalization["pitch_velocity_std_radps"]),
        "pitchAmplitudeMean": float(normalization["pitch_amplitude_mean_rad"]),
        "pitchAmplitudeStd": float(normalization["pitch_amplitude_std_rad"]),
        "verticalLogDurationMean": float(
            normalization["vertical_log_duration_mean"]
        ),
        "verticalLogDurationStd": float(normalization["vertical_log_duration_std"]),
        "verticalEventThreshold": float(config.get("event_gate_threshold", 0.5)),
        "verticalDurationMinSteps": int(duration_bounds[0]),
        "verticalDurationMaxSteps": int(duration_bounds[1]),
        "verticalEventTemplate": "16*u^2*(1-u)^2",
        "verticalEventDefinitionJson": json.dumps(
            event, ensure_ascii=False, separators=(",", ":")
        ),
        "runtimeFeedbackContract": str(checkpoint["runtime_feedback_contract"]),
        "headInnovationPolicy": str(checkpoint["stochastic_policy"]["headlocal"]),
        "rootInnovationPolicy": str(checkpoint["stochastic_policy"]["bodyroot"]),
        "rootInnovationDim": 3,
        "rootInnovationRho": floats(innovation.rho),
        "rootInnovationCholesky": floats(
            innovation.stationary_cholesky_physical
        ),
        "defaultRootNoiseScale": float(innovation.default_noise_scale),
        "headInnovationStd": [0.0] * 6,
        "rootInnovationFeedsBack": True,
        "rootInnovationRng": "xorshift32_plus_cached_box_muller",
        "rootInnovationFormula": (
            "epsilon=L*z; residual_i=rho_i*previous_i+"
            "sqrt(1-rho_i^2)*epsilon_i; L is stationary physical covariance Cholesky"
        ),
        "weightIhL0": floats(state["lstm.weight_ih_l0"]),
        "weightHhL0": floats(state["lstm.weight_hh_l0"]),
        "biasIhL0": floats(state["lstm.bias_ih_l0"]),
        "biasHhL0": floats(state["lstm.bias_hh_l0"]),
        "weightIhL1": floats(state["lstm.weight_ih_l1"]),
        "weightHhL1": floats(state["lstm.weight_hh_l1"]),
        "biasIhL1": floats(state["lstm.bias_ih_l1"]),
        "biasHhL1": floats(state["lstm.bias_hh_l1"]),
        "mainFcWeight": floats(state["main_output.weight"]),
        "mainFcBias": floats(state["main_output.bias"]),
        "verticalEventWeight": floats(state["vertical_event.weight"]),
        "verticalEventBias": float(state["vertical_event.bias"].item()),
        "verticalVelocityWeight": floats(state["vertical_velocity.weight"]),
        "verticalVelocityBias": float(state["vertical_velocity.bias"].item()),
        "verticalAmplitudeWeight": floats(state["vertical_amplitude.weight"]),
        "verticalAmplitudeBias": float(state["vertical_amplitude.bias"].item()),
        "pitchVelocityWeight": floats(state["pitch_velocity.weight"]),
        "pitchVelocityBias": float(state["pitch_velocity.bias"].item()),
        "pitchAmplitudeWeight": floats(state["pitch_amplitude.weight"]),
        "pitchAmplitudeBias": float(state["pitch_amplitude.bias"].item()),
        "verticalLogDurationWeight": floats(
            state["vertical_log_duration.weight"]
        ),
        "verticalLogDurationBias": float(
            state["vertical_log_duration.bias"].item()
        ),
        "seedWindows": seeds,
        "goldenSourceTestRow": golden_source_row,
        "goldenInitialEventProbability": golden_initial_event_probability,
        "goldenHistoryRaw": floats(golden_history),
        "goldenStepCount": args.golden_steps,
        "goldenExpectedMainNormalized": floats(deterministic.main_normalized),
        "goldenExpectedRawHeads": floats(deterministic.raw_heads),
        "goldenExpectedRawComposedStatePhysical": floats(
            deterministic.raw_composed_before_smoothing
        ),
        "goldenExpectedComposedStateRaw": floats(deterministic.applied),
        "goldenControllerStateNames": list(CONTROLLER_STATE_NAMES),
        "goldenExpectedControllerState": floats(deterministic.controller),
        "goldenTolerance": 0.0005,
        "rootInnovationGoldenSeed": int(args.innovation_golden_seed),
        "rootInnovationGoldenStepCount": int(args.innovation_golden_steps),
        "rootInnovationGoldenExpected": floats(innovation_golden),
        "combinedStochasticGoldenSeed": int(args.combined_golden_seed),
        "combinedStochasticGoldenStepCount": int(args.combined_golden_steps),
        "combinedStochasticGoldenNoiseScale": float(
            innovation.default_noise_scale
        ),
        "combinedStochasticGoldenExpectedMainNormalized": floats(
            combined.main_normalized
        ),
        "combinedStochasticGoldenExpectedRawHeads": floats(combined.raw_heads),
        "combinedStochasticGoldenExpectedRawComposedStatePhysical": floats(
            combined.raw_composed_before_smoothing
        ),
        "combinedStochasticGoldenExpectedComposedBeforeInnovation": floats(
            combined.composed_before_innovation
        ),
        "combinedStochasticGoldenExpectedControllerState": floats(
            combined.controller
        ),
        "combinedStochasticGoldenExpectedApplied": floats(combined.applied),
    }

    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    atomic_write_bytes(args.asset, encoded)
    asset_hash = sha256_file(args.asset)
    integrity_source = (
        "// Generated by scripts/normal_hmd_v3_export.py. Do not edit.\n"
        "namespace NormalHmdAgentDemo\n"
        "{\n"
        "    public static class GeneratedNormalHmdIntegrity\n"
        "    {\n"
        f'        public const string SchemaVersion = "{SCHEMA_VERSION}";\n'
        f'        public const string ExportSha256 = "{asset_hash}";\n'
        f'        public const string SourceModelSha256 = "{source_hash}";\n'
        f'        public const string SourceDataReportSha256 = "{data_report_hash}";\n'
        f'        public const string DatasetFingerprintSha256 = "{checkpoint["dataset_fingerprint_sha256"]}";\n'
        f'        public const string PreparedTrainSha256 = "{split_hashes["train"]}";\n'
        f'        public const string PreparedValSha256 = "{split_hashes["val"]}";\n'
        f'        public const string PreparedTestSha256 = "{split_hashes["test"]}";\n'
        f"        public const int SeedWindowCount = {len(seeds)};\n"
        f"        public const int GoldenStepCount = {args.golden_steps};\n"
        f"        public const int RootInnovationGoldenStepCount = {args.innovation_golden_steps};\n"
        f"        public const int CombinedStochasticGoldenStepCount = {args.combined_golden_steps};\n"
        "    }\n"
        "}\n"
    )
    atomic_write_text(args.integrity, integrity_source)

    report = {
        "schema_version": "3.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "unity_schema_version": SCHEMA_VERSION,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": source_hash,
        "source_metadata": str(args.metadata.resolve()),
        "source_metrics": str(args.metrics.resolve()),
        "source_training_is_synthetic": False,
        "source_data_report": str(args.data_report.resolve()),
        "source_data_report_sha256": data_report_hash,
        "dataset_fingerprint_sha256": str(checkpoint["dataset_fingerprint_sha256"]),
        "verified_prepared_split_sha256": split_hashes,
        "asset": str(args.asset.resolve()),
        "asset_sha256": asset_hash,
        "integrity_source": str(args.integrity.resolve()),
        "parameter_count": payload["parameterCount"],
        "seed_windows": len(seeds),
        "held_out_participants": len(
            {":".join(seed["trajectoryId"].split(":")[:2]) for seed in seeds}
        ),
        "held_out_scenes": sorted(
            {seed["trajectoryId"].split(":")[2] for seed in seeds}
        ),
        "main_state_indices": list(MAIN_STATE_INDICES),
        "raw_head_names": list(RAW_HEAD_NAMES),
        "event_controller_state_names": list(CONTROLLER_STATE_NAMES),
        "event_template": payload["verticalEventTemplate"],
        "head_innovation_std": payload["headInnovationStd"],
        "root_innovation": {
            "source": innovation.source,
            "rho": floats(innovation.rho),
            "stationary_cholesky_physical": floats(
                innovation.stationary_cholesky_physical
            ),
            "normalized_std": floats(innovation.normalized_std),
            "default_noise_scale": innovation.default_noise_scale,
            "feeds_back": True,
            "export_scale_override_used": args.root_noise_scale is not None,
        },
        "deployment_comfort_envelope": {
            "state_index": 8,
            "semantic": "HeadLocal rotation-vector Z / roll component",
            "lower_rad": float(deployment_clip_low[8]),
            "upper_rad": float(deployment_clip_high[8]),
            "limit_degrees": math.degrees(float(DEPLOYMENT_ROLL_LIMIT_RAD)),
            "application": "post-inference, pre-feedback and display",
            "scope": (
                "display/deployment comfort safety envelope; not a claim about "
                "or modification of the EyeNavGS training distribution"
            ),
        },
        "main_head_smoothing": {
            "indices": list(MAIN_HEAD_SMOOTHING_INDICES),
            "tau_seconds": args.main_head_smoothing_tau_seconds,
            "alpha": payload["mainHeadSmoothingAlpha"],
            "feedback": "final applied state",
            "validation_warning": (
                "Val long-run sweep supports this as a comfort de-jitter filter, "
                "but no tested EMA setting repaired the model's collapsed "
                "horizontal motion amplitude or human-reference occupancy."
            ),
        },
        "deterministic_golden_steps": args.golden_steps,
        "golden_source_test_row": golden_source_row,
        "golden_initial_event_probability": golden_initial_event_probability,
        "golden_exercises_event_controller": (
            golden_initial_event_probability >= model.event_gate_threshold
        ),
        "root_innovation_golden_seed": args.innovation_golden_seed,
        "root_innovation_golden_steps": args.innovation_golden_steps,
        "combined_stochastic_golden_seed": args.combined_golden_seed,
        "combined_stochastic_golden_steps": args.combined_golden_steps,
        "unity_golden_reference": (
            "CPU FP32 closed loop reconstructed from the frozen v3 checkpoint; "
            "safe/clipped final 9-D state is recurrent feedback"
        ),
        "training_metrics_model_sha256_verified": str(
            metrics["artifacts"]["model_sha256"]
        )
        == source_hash,
        "metadata_model_sha256_verified": str(metadata["model_sha256"])
        == source_hash,
    }
    atomic_write_text(
        args.report,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--data-report", type=Path, default=DEFAULT_DATA_REPORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds-per-trajectory", type=int, default=4)
    parser.add_argument("--golden-steps", type=int, default=64)
    parser.add_argument("--innovation-golden-seed", type=int, default=246813579)
    parser.add_argument("--innovation-golden-steps", type=int, default=64)
    parser.add_argument("--combined-golden-seed", type=int, default=97531)
    parser.add_argument("--combined-golden-steps", type=int, default=64)
    parser.add_argument(
        "--main-head-smoothing-tau-seconds",
        type=float,
        default=DEFAULT_MAIN_HEAD_SMOOTHING_TAU_SECONDS,
        help=(
            "Deterministic EMA time constant for HeadLocal X/Z/yaw/roll. "
            "Use zero only for an explicit no-smoothing diagnostic export."
        ),
    )
    parser.add_argument(
        "--root-noise-scale",
        type=float,
        default=None,
        help=(
            "Optional independent multiplier for root-only innovation. The default "
            "uses the checkpoint value; HeadLocal innovation remains zero."
        ),
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow a smoke-subset checkpoint; never use for the shipped demo.",
    )
    args = parser.parse_args(argv)
    counts = (
        args.seeds_per_trajectory,
        args.golden_steps,
        args.innovation_golden_steps,
        args.combined_golden_steps,
    )
    if any(value < 1 for value in counts):
        parser.error("seed-window and golden-vector counts must be positive")
    if args.root_noise_scale is not None and (
        not math.isfinite(args.root_noise_scale)
        or not 0.0 <= args.root_noise_scale <= 2.0
    ):
        parser.error("--root-noise-scale must be finite and between 0 and 2")
    if (
        not math.isfinite(args.main_head_smoothing_tau_seconds)
        or not 0.0 <= args.main_head_smoothing_tau_seconds <= 0.25
    ):
        parser.error(
            "--main-head-smoothing-tau-seconds must be finite and between 0 and 0.25"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    report = export(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
