#!/usr/bin/env python3
"""Export the independent Normal HMD v4 branches and portable Unity contract.

All tensors and golden arrays are flattened row-major/step-major. The 25 heads
retain the model's order, including its auxiliary horizontal pose proposal.
Production uses independent eligible-only Bernoulli streams and samples event
parameters once per onset. Deterministic goldens use an explicit test policy:
p >= .5 and mean parameters. Neither policy adds per-frame HeadLocal noise.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normal_hmd_common import STATE_FEATURE_NAMES, sha256_file
from normal_hmd_v4_data import HORIZONTAL_INDICES, ROOT_INDICES, VERTICAL_INDICES, load_split
from normal_hmd_v4_model import NormalHMDV4, V4Step
from normal_hmd_v4_train import run_self_checks as training_self_checks
from normal_hmd_v3_export import PortableGaussian, atomic_write_bytes, atomic_write_text, floats, require_finite


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "normal_hmd_v4.pth"
DEFAULT_METADATA = ROOT / "models" / "normal_hmd_v4_metadata.json"
DEFAULT_METRICS = ROOT / "results" / "normal_hmd_v4_training_metrics.json"
DEFAULT_DATA_REPORT = ROOT / "results" / "normal_hmd_v2_data_report.json"
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "normal_hmd_v2"
DEFAULT_V3_CHECKPOINT = ROOT / "models" / "normal_hmd_v3.pth"
DEFAULT_ASSET = ROOT / "demo" / "UnityNormalHmdAgentDemo" / "Assets" / "Resources" / "NormalHmdAgentModel.json"
DEFAULT_INTEGRITY = ROOT / "demo" / "UnityNormalHmdAgentDemo" / "Assets" / "Scripts" / "GeneratedNormalHmdIntegrity.cs"
DEFAULT_REPORT = ROOT / "results" / "normal_hmd_v4_unity_export_report.json"
SCHEMA_VERSION = "normal-hmd-agent-unity-4.0"
ROOT_SEED_SALT = 0xA341316C
HORIZONTAL_SEED_SALT = 0xC8013EA4
VERTICAL_SEED_SALT = 0xAD90777D
SEED_WINDOW_SALT = 0x7E95761E
F32 = np.float32
HEAD_NAMES = (
    "root_right_velocity_normalized", "root_forward_velocity_normalized", "root_yaw_rate_normalized",
    "horizontal_x_pose_normalized", "horizontal_z_pose_normalized",
    "horizontal_yaw_pose_normalized", "horizontal_roll_pose_normalized",
    "horizontal_hazard_logit",
    "horizontal_end_x_mean_normalized", "horizontal_end_z_mean_normalized",
    "horizontal_end_yaw_mean_normalized", "horizontal_end_roll_mean_normalized",
    "horizontal_log_duration_mean_normalized",
    "horizontal_end_x_logstd", "horizontal_end_z_logstd", "horizontal_end_yaw_logstd",
    "horizontal_end_roll_logstd", "horizontal_log_duration_logstd",
    "vertical_hazard_logit", "vertical_end_y_mean_normalized",
    "vertical_end_pitch_mean_normalized", "vertical_log_duration_mean_normalized",
    "vertical_end_y_logstd", "vertical_end_pitch_logstd", "vertical_log_duration_logstd",
)
DIAGNOSTIC_NAMES = (
    "horizontal_trigger", "vertical_trigger", "horizontal_eligible", "vertical_eligible",
    "horizontal_probability", "vertical_probability",
)
HEAD_MODULES = {
    "root_output": 3, "horizontal_pose": 4, "horizontal_hazard": 1,
    "horizontal_parameter_mean": 5, "horizontal_parameter_logstd": 5,
    "vertical_hazard": 1, "vertical_parameter_mean": 3, "vertical_parameter_logstd": 3,
}
BUFFER_SHAPES = {
    "state_mean": (9,), "state_std": (9,),
    "horizontal_parameter_center": (5,), "horizontal_parameter_scale": (5,),
    "vertical_parameter_center": (3,), "vertical_parameter_scale": (3,),
    "horizontal_parameter_low": (5,), "horizontal_parameter_high": (5,),
    "vertical_parameter_low": (3,), "vertical_parameter_high": (3,),
    "horizontal_pose_low": (4,), "horizontal_pose_high": (4,),
    "vertical_pose_low": (2,), "vertical_pose_high": (2,),
    "horizontal_parameter_cholesky": (5, 5), "vertical_parameter_cholesky": (3, 3),
    "horizontal_hazard_bias": (), "vertical_hazard_bias": (),
}


def camel(name: str) -> str:
    words = name.split("_")
    return words[0] + "".join("LogStd" if word == "logstd" else word[:1].upper() + word[1:] for word in words[1:])


def splitmix32(value: int) -> int:
    """Unsigned 32-bit SplitMix finalizer; C# must use unchecked uint products."""
    z = (int(value) + 0x9E3779B9) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x21F0AAAD) & 0xFFFFFFFF
    z = ((z ^ (z >> 15)) * 0x735A2D97) & 0xFFFFFFFF
    return (z ^ (z >> 15)) & 0xFFFFFFFF


def derived_seeds(run_seed: int) -> tuple[int, int, int, int]:
    return tuple(splitmix32((int(run_seed) & 0xFFFFFFFF) ^ salt) for salt in
                 (ROOT_SEED_SALT, HORIZONTAL_SEED_SALT, VERTICAL_SEED_SALT, SEED_WINDOW_SALT))


def triangular_product(matrix: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    result = np.zeros(len(gaussian), dtype=np.float32)
    for row in range(len(gaussian)):
        for column in range(row + 1):
            result[row] = F32(result[row] + F32(matrix[row, column] * gaussian[column]))
    return result


class PortableRoot:
    """Stationary initialization precedes the first AR update, even at scale 0."""

    def __init__(
        self,
        state_std: np.ndarray,
        run_seed: int,
        *,
        normalized_std: float = .15,
        rho: float = .95,
    ) -> None:
        if normalized_std < 0.0:
            raise ValueError("Root innovation standard deviation cannot be negative")
        if not 0.0 <= rho < 1.0:
            raise ValueError("Root innovation rho must be in [0, 1)")
        self.rho = np.full(3, F32(rho), dtype=np.float32)
        self.cholesky = np.diag(
            np.asarray(state_std[:3], dtype=np.float32) * F32(normalized_std)
        )
        self.rng = PortableGaussian(derived_seeds(run_seed)[0])
        self.residual = self._epsilon()
        self.initial_residual = self.residual.copy()

    def _epsilon(self) -> np.ndarray:
        z = np.asarray([self.rng.gaussian() for _ in range(3)], dtype=np.float32)
        return triangular_product(self.cholesky, z)

    def next(self) -> np.ndarray:
        epsilon = self._epsilon()
        for index in range(3):
            gain = F32(math.sqrt(max(0., 1. - float(F32(self.rho[index] * self.rho[index])))))
            self.residual[index] = F32(F32(self.rho[index] * self.residual[index]) + F32(gain * epsilon[index]))
        return self.residual.copy()


@dataclass
class Controller:
    active: bool
    cooldown: int
    elapsed: int
    duration: int
    start: np.ndarray
    target: np.ndarray
    amplitude: np.ndarray
    comfort_activated_this_step: bool = False

    @classmethod
    def initial(cls, pose: np.ndarray, minimum_duration: int) -> "Controller":
        return cls(False, 0, 0, minimum_duration, pose.copy(), pose.copy(), np.zeros_like(pose))

    def array(self) -> np.ndarray:
        return np.concatenate((np.asarray([self.active, self.cooldown, self.elapsed, self.duration], dtype=np.float32), self.start, self.target, self.amplitude))


def controller_names() -> list[str]:
    names = []
    for branch, axes in (("horizontal", ("x", "z", "yaw", "roll")), ("vertical", ("y", "pitch"))):
        names.extend(f"{branch}_{field}" for field in ("active", "cooldown_remaining", "elapsed_steps", "duration_steps"))
        for field in ("start", "target", "amplitude"):
            names.extend(f"{branch}_{field}_{axis}_physical" for axis in axes)
    return names


def portable_branch(
    payload: dict[str, Any], branch: str, controller: Controller, previous: np.ndarray,
    raw: np.ndarray, rng: PortableGaussian, *, deterministic: bool = False,
) -> tuple[np.ndarray, Controller, bool, bool, float]:
    """Scalar FP32 arithmetic and double transcendentals mirror dependency-free C#."""
    horizontal = branch == "horizontal"
    indices = HORIZONTAL_INDICES if horizontal else VERTICAL_INDICES
    hazard_index, mean_start, logstd_start, dimension = (7, 8, 13, 5) if horizontal else (18, 19, 22, 3)
    logit = F32(raw[hazard_index] + F32(payload[f"{branch}HazardCalibrationBias"]))
    if logit >= 0:
        exponential = F32(math.exp(-float(logit)))
        probability = F32(F32(1.) / F32(F32(1.) + exponential))
    else:
        exponential = F32(math.exp(float(logit)))
        probability = F32(exponential / F32(F32(1.) + exponential))
    eligible = not controller.active and controller.cooldown <= 0
    # No uniform is consumed during an event or cooldown. Cached Gaussians are
    # also branch-local; a Bernoulli draw never invalidates a Box-Muller spare.
    trigger = eligible and (float(probability) >= .5 if deterministic else rng.uniform_open() < float(probability))
    was_active = controller.active
    controller.comfort_activated_this_step = False
    if trigger:
        mean = raw[mean_start:mean_start + dimension].copy()
        if not deterministic:
            z = np.asarray([rng.gaussian() for _ in range(dimension)], dtype=np.float32)
            cholesky = np.asarray(payload[f"{branch}ParameterCholesky"], dtype=np.float32).reshape(dimension, dimension)
            noise = triangular_product(cholesky, z)
            for index in range(dimension):
                sigma = F32(math.exp(float(np.clip(raw[logstd_start + index], -5., 2.))))
                mean[index] = F32(mean[index] + F32(sigma * noise[index]))
        center = np.asarray(payload[f"{branch}ParameterCenter"], dtype=np.float32)
        scale = np.asarray(payload[f"{branch}ParameterScale"], dtype=np.float32)
        low = np.asarray(payload[f"{branch}ParameterLow"], dtype=np.float32)
        high = np.asarray(payload[f"{branch}ParameterHigh"], dtype=np.float32)
        parameters = np.empty(dimension, dtype=np.float32)
        for index in range(dimension):
            physical = F32(F32(mean[index] * scale[index]) + center[index])
            midpoint = F32(F32(low[index] + high[index]) * F32(.5))
            half_range = F32(max(float(F32(F32(high[index] - low[index]) * F32(.5))), 1e-6))
            normalized = F32(F32(physical - midpoint) / half_range)
            parameters[index] = F32(midpoint + F32(half_range * F32(math.tanh(float(normalized)))))
        target = np.clip(parameters[:-1], payload[f"{branch}PoseLow"], payload[f"{branch}PoseHigh"]).astype(np.float32)
        deployment_target = np.clip(target, np.asarray(payload["stateClipLow"])[list(indices)], np.asarray(payload["stateClipHigh"])[list(indices)]).astype(np.float32)
        controller.comfort_activated_this_step = bool(np.any(target != deployment_target))
        target = deployment_target
        duration_value = math.exp(float(parameters[-1])) * float(payload[f"{branch}DurationScale"])
        duration = min(payload["durationMaxSteps"], max(payload["durationMinSteps"], int(round(float(duration_value)))))
        controller.start = previous[list(indices)].copy()
        controller.target = target
        controller.amplitude = (target - controller.start).astype(np.float32)
        controller.duration = duration
        controller.elapsed = 0
    elif was_active:
        controller.elapsed += 1
    active_now = was_active or trigger
    if active_now:
        u = F32(min(1., max(0., float(F32(controller.elapsed / max(controller.duration - 1, 1))))))
        # Each multiplication/addition is float32; no fused multiply-add.
        u3 = F32(F32(u * u) * u)
        weight = F32(u3 * F32(F32(10.) + F32(u * F32(F32(-15.) + F32(F32(6.) * u)))))
        pose = (controller.start + ((controller.target - controller.start).astype(np.float32) * weight).astype(np.float32)).astype(np.float32)
    else:
        pose = controller.target.copy()
    finished = active_now and controller.elapsed >= controller.duration - 1
    if finished:
        pose = controller.target.copy()
        controller.active = False
        controller.cooldown = payload["cooldownSteps"]
        controller.elapsed = 0
    else:
        controller.active = active_now
        if not active_now:
            controller.cooldown = max(0, controller.cooldown - 1)
    return pose, controller, trigger, eligible, float(probability)


def initial_controllers(payload: dict[str, Any], initial: np.ndarray) -> tuple[Controller, Controller]:
    initial = np.clip(initial, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
    return tuple(Controller.initial(initial[list(indices)], payload["durationMinSteps"]) for indices in (HORIZONTAL_INDICES, VERTICAL_INDICES))


def compose(
    payload: dict[str, Any], raw: np.ndarray, previous: np.ndarray,
    controllers: tuple[Controller, Controller], streams: tuple[PortableGaussian, PortableGaussian],
    *, deterministic: bool = False,
) -> tuple[np.ndarray, tuple[Controller, Controller], np.ndarray]:
    h, hc, ht, he, hp = portable_branch(payload, "horizontal", controllers[0], previous, raw, streams[0], deterministic=deterministic)
    v, vc, vt, ve, vp = portable_branch(payload, "vertical", controllers[1], previous, raw, streams[1], deterministic=deterministic)
    state = np.empty(9, dtype=np.float32)
    state[:3] = (raw[:3] * np.asarray(payload["stateStd"][:3], dtype=np.float32) + np.asarray(payload["stateMean"][:3], dtype=np.float32)).astype(np.float32)
    state[list(HORIZONTAL_INDICES)], state[list(VERTICAL_INDICES)] = h, v
    return state, (hc, vc), np.asarray([ht, vt, he, ve, hp, vp], dtype=np.float32)


def build_model(checkpoint: dict[str, Any]) -> NormalHMDV4:
    state, config = checkpoint["state_dict"], checkpoint["model_config"]
    values = {name: state[name].tolist() for name in BUFFER_SHAPES if name not in
              ("horizontal_parameter_cholesky", "vertical_parameter_cholesky", "horizontal_hazard_bias", "vertical_hazard_bias")}
    model = NormalHMDV4(
        input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], num_layers=config["num_layers"], dropout=config["dropout"],
        horizontal_parameter_correlation=checkpoint["parameter_correlation"]["horizontal"],
        vertical_parameter_correlation=checkpoint["parameter_correlation"]["vertical"],
        cooldown_steps=config["cooldown_steps"], minimum_duration_steps=config["duration_bounds_steps"][0], maximum_duration_steps=config["duration_bounds_steps"][1], **values,
    )
    model.load_state_dict(state, strict=True)
    return model.cpu().eval()


def validate_contract(checkpoint: dict[str, Any], allow_smoke: bool) -> None:
    if checkpoint.get("schema_version") != "4.0.0":
        raise ValueError("expected checkpoint schema 4.0.0")
    config, state = checkpoint["model_config"], checkpoint["state_dict"]
    if config.get("architecture") != "independent_root_horizontal_vertical_lstm_holding_events":
        raise ValueError("unexpected v4 architecture")
    if config.get("input_dim") != 9 or config.get("num_layers") != 1 or (config.get("hidden_dim") != 96 and not allow_smoke):
        raise ValueError("production contract requires independent full-9D, single-layer, hidden96 branches")
    if config.get("dropout") != 0. or config.get("cooldown_steps") != 4:
        raise ValueError("v4 deployment requires dropout0 and cooldown4")
    for name, indices in (("root_indices", ROOT_INDICES), ("horizontal_indices", HORIZONTAL_INDICES), ("vertical_indices", VERTICAL_INDICES)):
        if config.get(name) != list(indices):
            raise ValueError(f"{name} differs from frozen state contract")
    if config.get("duration_bounds_steps") != [5, 148]:
        raise ValueError("v4 deployment duration bounds must be 5..148")
    hidden = config["hidden_dim"]
    expected = dict(BUFFER_SHAPES)
    for branch in ("root", "horizontal", "vertical"):
        expected.update({f"{branch}_lstm.weight_ih_l0": (4 * hidden, 9), f"{branch}_lstm.weight_hh_l0": (4 * hidden, hidden), f"{branch}_lstm.bias_ih_l0": (4 * hidden,), f"{branch}_lstm.bias_hh_l0": (4 * hidden,)})
    for name, dimension in HEAD_MODULES.items():
        expected[f"{name}.weight"], expected[f"{name}.bias"] = (dimension, hidden), (dimension,)
    if set(state) != set(expected):
        raise ValueError(f"state_dict keys mismatch: missing={sorted(set(expected)-set(state))}, extra={sorted(set(state)-set(expected))}")
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape or state[name].dtype != torch.float32:
            raise ValueError(f"{name} must be float32 shape {shape}")
        require_finite(name, state[name].numpy())
    for name in ("state_std", "horizontal_parameter_scale", "vertical_parameter_scale"):
        if torch.any(state[name] <= 0):
            raise ValueError(f"{name} must be positive")
    for branch in ("horizontal", "vertical"):
        for field in ("parameter", "pose"):
            if torch.any(state[f"{branch}_{field}_low"] >= state[f"{branch}_{field}_high"]):
                raise ValueError(f"unordered {branch} {field} bounds")
        chol = state[f"{branch}_parameter_cholesky"]
        if not torch.equal(chol, torch.tril(chol)) or torch.any(torch.diag(chol) <= 0):
            raise ValueError(f"invalid {branch} parameter Cholesky")
        correlation = torch.tensor(checkpoint["parameter_correlation"][branch], dtype=torch.float32)
        if not torch.allclose(chol @ chol.T, correlation, atol=2e-6, rtol=0.):
            raise ValueError(f"{branch} correlation/buffer mismatch")
        if not math.isclose(float(state[f"{branch}_hazard_bias"]), float(checkpoint["hazard_calibration"][branch]["bias"]), abs_tol=1e-6):
            raise ValueError(f"{branch} calibrated hazard/buffer mismatch")
    for name, value in checkpoint["normalization"].items():
        if name in state and not np.allclose(state[name].numpy(), np.asarray(value, dtype=np.float32), atol=0, rtol=0):
            raise ValueError(f"normalization/buffer mismatch for {name}")


def deployment_state_clamps(
    checkpoint: dict[str, Any],
    v3: dict[str, Any],
    roll_limit_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if v3.get("schema_version") != "3.0.0" or v3.get("synthetic") is not False:
        raise ValueError("root safety bounds must come from the real frozen v3 checkpoint")
    low = np.empty(9, dtype=np.float32)
    high = np.empty(9, dtype=np.float32)
    low[:3], high[:3] = np.asarray(v3["state_clip_low"], dtype=np.float32)[:3], np.asarray(v3["state_clip_high"], dtype=np.float32)[:3]
    for branch, indices in (("horizontal", HORIZONTAL_INDICES), ("vertical", VERTICAL_INDICES)):
        low[list(indices)] = checkpoint["state_dict"][f"{branch}_pose_low"].numpy()
        high[list(indices)] = checkpoint["state_dict"][f"{branch}_pose_high"].numpy()
    low[8], high[8] = (
        max(low[8], F32(-roll_limit_rad)),
        min(high[8], F32(roll_limit_rad)),
    )
    require_finite("stateClipLow", low)
    require_finite("stateClipHigh", high)
    if np.any(low >= high):
        raise ValueError("invalid deployment comfort envelope")
    return low, high


def choose_seed_windows(validation: dict[str, np.ndarray], per_trajectory: int) -> list[dict[str, Any]]:
    seeds = []
    for trajectory, identity in enumerate(validation["trajectory_ids"]):
        rows = np.flatnonzero(validation["trajectory_index"] == trajectory)
        if not len(rows):
            continue
        selected = rows[np.linspace(0, len(rows) - 1, min(len(rows), per_trajectory), dtype=np.int64)]
        for row in selected:
            history = np.asarray(validation["X"][row], dtype=np.float32)
            require_finite(f"held-out val history {row}", history)
            seeds.append({"trajectoryId": str(identity), "split": "val", "startFrame": int(validation["start_index"][row]), "historyRaw": floats(history)})
    if not seeds:
        raise ValueError("no held-out validation histories")
    return seeds


def raw_heads(step: V4Step) -> np.ndarray:
    return np.concatenate([value[0].detach().cpu().numpy().reshape(-1) for value in step]).astype(np.float32)


@torch.no_grad()
def closed_loop_rollout(model: NormalHMDV4, payload: dict[str, Any], history: np.ndarray, steps: int, run_seed: int, *, deterministic: bool = False, root_scale: float = 1.) -> dict[str, np.ndarray]:
    mean, std = np.asarray(payload["stateMean"], dtype=np.float32), np.asarray(payload["stateStd"], dtype=np.float32)
    normalized_history = ((history.astype(np.float32) - mean) / std).astype(np.float32)
    prediction, hidden = model.warm_history(torch.from_numpy(normalized_history)[None])
    previous = np.clip(history[-1], payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
    controllers = initial_controllers(payload, previous)
    seeds = derived_seeds(run_seed)
    streams = PortableGaussian(seeds[1]), PortableGaussian(seeds[2])
    root = PortableRoot(
        std,
        run_seed,
        normalized_std=float(payload["rootInnovationNormalizedStd"]),
        rho=float(payload["rootInnovationRho"][0]),
    )
    rows = {name: [] for name in ("rawHeads", "composed", "applied", "controller", "diagnostics", "rootInnovation", "comfortEnvelopeActivation")}
    for index in range(steps):
        raw = raw_heads(prediction)
        composed, controllers, diagnostics = compose(payload, raw, previous, controllers, streams, deterministic=deterministic)
        residual = np.zeros(3, dtype=np.float32) if deterministic else root.next()
        candidate = composed.copy()
        candidate[:3] = (candidate[:3] + residual * F32(root_scale)).astype(np.float32)
        applied = np.clip(candidate, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
        for name, value in (("rawHeads", raw), ("composed", composed), ("applied", applied), ("controller", np.concatenate([controller.array() for controller in controllers])), ("diagnostics", diagnostics), ("rootInnovation", residual), ("comfortEnvelopeActivation", np.asarray([controllers[0].comfort_activated_this_step or controllers[1].comfort_activated_this_step], dtype=np.float32))):
            rows[name].append(value)
        previous = applied
        if index + 1 < steps:
            feedback = torch.from_numpy(((applied - mean) / std).astype(np.float32))[None]
            prediction, hidden = model.advance(feedback, hidden)
    result = {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}
    for name, value in result.items():
        require_finite(f"closed loop {name}", value)
    return result


def rollout_safety_metrics(payload: dict[str, Any], rollout: dict[str, np.ndarray], scale: float) -> dict[str, Any]:
    candidate = rollout["composed"].copy()
    candidate[:, :3] = (candidate[:, :3] + rollout["rootInnovation"] * F32(scale)).astype(np.float32)
    difference = np.abs(candidate - rollout["applied"])
    root_clipped = difference[:, :3] > 0.
    # Head arithmetic may differ by a few ulps at a latched safety endpoint.
    # A substantive excursion remains unexpected; intentional endpoint caps
    # are separately reported on onset rather than counted as runtime errors.
    unexpected = difference > 0.
    unexpected[:, 8] = difference[:, 8] > F32(
        4. * float(np.spacing(F32(payload["deploymentRollClampRad"])))
    )
    root_distance = float(np.sum(np.hypot(rollout["applied"][:, 0], rollout["applied"][:, 1]), dtype=np.float64) * payload["fixedDtSeconds"])
    return {"finite": bool(np.isfinite(rollout["applied"]).all()),
            "root_clip_step_rate": float(np.mean(np.any(root_clipped, axis=1))),
            "root_clip_value_rate": float(np.mean(root_clipped)),
            "unexpected_clip_step_rate": float(np.mean(np.any(unexpected, axis=1))),
            "unexpected_clip_value_rate": float(np.mean(unexpected)),
            "comfort_envelope_activation_rate": float(np.mean(rollout["comfortEnvelopeActivation"])),
            "comfort_envelope_activation_count": int(rollout["comfortEnvelopeActivation"].sum()),
            "clip_feature_counts": np.sum(unexpected, axis=0).astype(int).tolist(),
            "maximum_clip_excess_by_feature": np.max(difference, axis=0).astype(float).tolist(),
            "root_distance_m": root_distance,
            "distance_in_human_val_band": 4.82 <= root_distance <= 33.34,
            "horizontal_event_count": int(rollout["diagnostics"][:, 0].sum()),
            "vertical_event_count": int(rollout["diagnostics"][:, 1].sum())}


def calibrate_root_scale(model: NormalHMDV4, payload: dict[str, Any], validation: dict[str, np.ndarray], source_hash: str, output: Path) -> dict[str, Any]:
    """Validation-only fixed 16-history, 5-scale sweep; retain every candidate."""
    trajectories = np.unique(validation["trajectory_index"])
    if len(trajectories) < 16:
        raise ValueError("root calibration requires at least16 distinct held-out val trajectories")
    selected_trajectories = trajectories[np.linspace(0, len(trajectories) - 1, 16, dtype=np.int64)]
    rows = []
    for index, trajectory in enumerate(selected_trajectories):
        candidates = np.flatnonzero(validation["trajectory_index"] == trajectory)
        row = int(candidates[0] if index == 0 else candidates[len(candidates) // 2])
        rows.append(row)
    steps = int(math.ceil(60. / payload["fixedDtSeconds"]))
    candidates_report = []
    for scale in (1., .9, .8, .7, .6):
        rollouts = []
        for index, row in enumerate(rows):
            seed = 7045 + 104729 * index
            generated = closed_loop_rollout(model, payload, np.asarray(validation["X"][row], dtype=np.float32), steps, seed, root_scale=scale)
            metrics = rollout_safety_metrics(payload, generated, scale)
            metrics.update({"val_row": row, "trajectory_index": int(validation["trajectory_index"][row]), "trajectory_id": str(validation["trajectory_ids"][int(validation["trajectory_index"][row])]), "run_seed": seed})
            rollouts.append(metrics)
        maximum_root_rate = max(item["root_clip_step_rate"] for item in rollouts)
        aggregate_value_rate = float(np.mean([item["unexpected_clip_value_rate"] for item in rollouts]))
        median_distance = float(np.median([item["root_distance_m"] for item in rollouts]))
        band_fraction = float(np.mean([item["distance_in_human_val_band"] for item in rollouts]))
        passed = (all(item["finite"] for item in rollouts) and maximum_root_rate < .01 and
                  max(item["unexpected_clip_step_rate"] for item in rollouts) < .01 and
                  aggregate_value_rate < .001 and 4.82 <= median_distance <= 33.34 and band_fraction >= .6)
        candidate = {"scale": scale, "rollouts": rollouts, "maximum_root_clip_step_rate": maximum_root_rate,
                     "aggregate_unexpected_clip_value_rate": aggregate_value_rate,
                     "aggregate_root_clip_value_rate": float(np.mean([item["root_clip_value_rate"] for item in rollouts])),
                     "median_root_distance_m": median_distance, "fraction_distance_in_human_val_band": band_fraction, "passed": passed}
        candidates_report.append(candidate)
        print(f"root validation scale={scale:g}: maxclip={maximum_root_rate:.5%}, values={aggregate_value_rate:.5%}, distanceMedian={median_distance:.3f}m, band={band_fraction:.2%}, passed={passed}", flush=True)
    passed_candidates = [candidate["scale"] for candidate in candidates_report if candidate["passed"]]
    selected = max(passed_candidates) if passed_candidates else None
    result = {"schema_version": "4.0.0", "status": "passed" if selected is not None else "failed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_model_sha256": source_hash, "prepared_val_sha256": payload["preparedValSha256"],
              "selection_split": "val", "test_evaluated": False, "histories": len(rows), "history_selection": "16 spread distinct val trajectories; first window on first trajectory, median window on others",
              "steps_per_rollout": steps, "seconds_per_rollout": steps * payload["fixedDtSeconds"],
              "root_innovation_normalized_std": payload["rootInnovationNormalizedStd"],
              "root_innovation_rho": payload["rootInnovationRho"],
              "deployment_roll_clamp_rad": payload["deploymentRollClampRad"],
              "candidate_scales": [1., .9, .8, .7, .6], "selected_default_root_noise_scale": selected,
              "criteria": {"per_rollout_root_clip_step_rate_strict_max": .01, "per_rollout_unexpected_clip_step_rate_strict_max": .01,
                           "aggregate_unexpected_clip_value_rate_strict_max": .001, "human_val_root_distance_band_m": [4.82, 33.34], "minimum_fraction_distance_in_band": .6,
                           "median_distance_must_be_in_band": True},
              "distance_formula": "sum(hypot(root_right_velocity,root_forward_velocity))*fixed_dt_seconds",
              "comfort_activation_definition": "onset parameter endpoint is clipped by tighter deployment comfort envelope; excluded from unexpected state clipping",
              "horizontal_duration_scale": 1., "vertical_duration_scale": 1., "headlocal_noise": "none", "candidates": candidates_report}
    atomic_write_text(output, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def verify_saved_calibration(
    calibration: dict[str, Any],
    source_hash: str,
    validation_hash: str,
    payload: dict[str, Any],
) -> None:
    """Recompute frozen gates from each saved rollout, never trust pass labels."""
    if (calibration.get("status") != "passed" or calibration.get("source_model_sha256") != source_hash or
            calibration.get("prepared_val_sha256") != validation_hash or calibration.get("test_evaluated") is not False or
            calibration.get("selection_split") != "val" or calibration.get("histories", 0) < 16 or
            calibration.get("steps_per_rollout", 0) < 2223 or calibration.get("candidate_scales") != [1., .9, .8, .7, .6] or
            not math.isclose(float(calibration.get("root_innovation_normalized_std", -1.0)), float(payload["rootInnovationNormalizedStd"]), rel_tol=0.0, abs_tol=1e-8) or
            not math.isclose(float(calibration.get("deployment_roll_clamp_rad", -1.0)), float(payload["deploymentRollClampRad"]), rel_tol=0.0, abs_tol=1e-8) or
            not np.allclose(np.asarray(calibration.get("root_innovation_rho", []), dtype=np.float64), np.asarray(payload["rootInnovationRho"], dtype=np.float64), rtol=0.0, atol=1e-8)):
        raise ValueError("saved root calibration does not match frozen protocol and sources")
    candidates = calibration.get("candidates", [])
    if len(candidates) != 5 or [item.get("scale") for item in candidates] != [1., .9, .8, .7, .6]:
        raise ValueError("saved root calibration must retain all5 candidate results")
    passed = []
    for candidate in candidates:
        rollouts = candidate.get("rollouts", [])
        if len(rollouts) != calibration["histories"] or len({item["trajectory_index"] for item in rollouts}) < 16 or len({item["run_seed"] for item in rollouts}) != len(rollouts):
            raise ValueError("saved root calibration has insufficient independent val histories/seeds")
        distances = [float(item["root_distance_m"]) for item in rollouts]
        median = float(np.median(distances))
        band_fraction = float(np.mean([4.82 <= distance <= 33.34 for distance in distances]))
        clip_step = max(float(item["root_clip_step_rate"]) for item in rollouts)
        unexpected_step = max(float(item["unexpected_clip_step_rate"]) for item in rollouts)
        clip_value = float(np.mean([item["unexpected_clip_value_rate"] for item in rollouts]))
        if (all(item["finite"] for item in rollouts) and clip_step < .01 and unexpected_step < .01 and
                clip_value < .001 and 4.82 <= median <= 33.34 and band_fraction >= .6):
            passed.append(candidate["scale"])
    if not passed or calibration.get("selected_default_root_noise_scale") != max(passed):
        raise ValueError("saved root noise scale is not the largest candidate satisfying all frozen gates")


def event_fixture(payload: dict[str, Any], steps: int, run_seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Real portable RNG with forced high/low probabilities, no trigger override."""
    initial = np.asarray(payload["stateMean"], dtype=np.float32)
    initial[list(HORIZONTAL_INDICES)] = [0.03, 0.02, 0.12, -0.06]
    initial[list(VERTICAL_INDICES)] = [0.01, -0.15]
    initial = np.clip(initial, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
    raw = np.zeros((steps, 25), dtype=np.float32)
    for index in range(steps):
        raw[index, :3] = (initial[:3] - np.asarray(payload["stateMean"][:3], dtype=np.float32)) / np.asarray(payload["stateStd"][:3], dtype=np.float32)
        raw[index, 3:7] = F32(99.)  # Auxiliary pose must not affect controller output.
        high = index == 0 or index >= 12
        raw[index, 7] = F32((80. if high else -80.) - payload["horizontalHazardCalibrationBias"])
        raw[index, 18] = F32((80. if high else -80.) - payload["verticalHazardCalibrationBias"])
        raw[index, 8:12] = F32(.4 if index % 2 else -.4)
        raw[index, 19:21] = F32(-.5 if index % 2 else .5)
        raw[index, 12] = raw[index, 21] = F32(-80.)  # Both event durations reach minimum5.
        raw[index, 13:18], raw[index, 22:25] = F32(-.8), F32(-.9)
    controllers = initial_controllers(payload, initial)
    seeds = derived_seeds(run_seed)
    streams = PortableGaussian(seeds[1]), PortableGaussian(seeds[2])
    previous = initial.copy()
    rows = {name: [] for name in ("applied", "controller", "diagnostics")}
    for frame in raw:
        composed, controllers, diagnostic = compose(payload, frame, previous, controllers, streams)
        previous = np.clip(composed, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
        rows["applied"].append(previous.copy())
        rows["controller"].append(np.concatenate([controller.array() for controller in controllers]))
        rows["diagnostics"].append(diagnostic)
    return initial, raw, {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}


def load_and_verify_sources(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = json.loads(args.data_report.read_text(encoding="utf-8"))
    source_hash = sha256_file(args.checkpoint)
    for label, document in (("metadata", metadata), ("metrics", metrics)):
        if document.get("status") != "success" or document.get("schema_version") != "4.0.0":
            raise ValueError(f"{label} does not report successful v4 training")
        if document.get("artifacts", {}).get("model_sha256") != source_hash:
            raise ValueError(f"{label}/checkpoint SHA-256 mismatch")
        if document.get("smoke") and not args.allow_smoke:
            raise ValueError("smoke checkpoint cannot be deployed as production")
        if document.get("self_checks", {}).get("status") != "passed":
            raise ValueError(f"{label} training self checks did not pass")
    if report.get("status") != "success" or report.get("synthetic") is not False or report.get("source_type") != "EyeNavGS":
        raise ValueError("expected successful non-synthetic EyeNavGS frozen data report")
    if checkpoint.get("data_report_sha256") != sha256_file(args.data_report):
        raise ValueError("checkpoint/data report hash mismatch")
    if checkpoint.get("dataset_fingerprint_sha256") != report.get("dataset_fingerprint_sha256"):
        raise ValueError("dataset fingerprint mismatch")
    if metadata.get("dataset_fingerprint_sha256") != checkpoint["dataset_fingerprint_sha256"]:
        raise ValueError("metadata dataset fingerprint mismatch")
    hashes = {}
    for name in ("train", "val", "test"):
        hashes[name] = sha256_file(args.data_dir / f"normal_hmd_{name}.npz")
        expected = report.get("splits", {}).get(name, {}).get("sha256")
        if hashes[name] != expected or hashes[name] != checkpoint.get("prepared_split_sha256", {}).get(name):
            raise ValueError(f"{name} frozen split/checkpoint hash mismatch")
        for label, document in (("metadata", metadata), ("metrics", metrics)):
            if hashes[name] != document.get("prepared_split_sha256", {}).get(name):
                raise ValueError(f"{name} {label} prepared split hash mismatch")
    validation = load_split(args.data_dir, "val")
    if not args.allow_smoke:
        for name in ("train", "val", "test"):
            with np.load(args.data_dir / f"normal_hmd_{name}.npz", allow_pickle=False) as archive:
                if metrics.get("segments_used", {}).get(name) != len(archive["X"]):
                    raise ValueError("production export requires all frozen segments")
        probe = metrics.get("closed_loop_probe", {})
        if float(probe.get("seconds_generated", 0)) < 60.:
            raise ValueError("production export requires a training 60-second probe")
        for name in ("best_val_loss", "test_loss"):
            if not math.isfinite(float(metrics.get(name, math.nan))):
                raise ValueError(f"production {name} is not finite")
    v3 = torch.load(args.root_bounds_checkpoint, map_location="cpu", weights_only=True)
    if v3.get("dataset_fingerprint_sha256") != checkpoint["dataset_fingerprint_sha256"]:
        raise ValueError("v3 root safety bounds refer to a different dataset")
    return checkpoint, metadata, metrics, validation, hashes, report


def portable_lstm_heads(payload: dict[str, Any], history: np.ndarray) -> np.ndarray:
    """Independent scalar-accumulation reference, verifying tensor orientation."""
    hidden_dim = payload["hiddenDim"]
    normalized = ((history.astype(np.float32) - np.asarray(payload["stateMean"], dtype=np.float32)) / np.asarray(payload["stateStd"], dtype=np.float32)).astype(np.float32)
    outputs = {}
    for branch in ("root", "horizontal", "vertical"):
        ih = np.asarray(payload[f"{branch}WeightIhL0"], dtype=np.float32).reshape(4 * hidden_dim, 9)
        hh = np.asarray(payload[f"{branch}WeightHhL0"], dtype=np.float32).reshape(4 * hidden_dim, hidden_dim)
        bi = np.asarray(payload[f"{branch}BiasIhL0"], dtype=np.float32)
        bh = np.asarray(payload[f"{branch}BiasHhL0"], dtype=np.float32)
        h, c = np.zeros(hidden_dim, dtype=np.float32), np.zeros(hidden_dim, dtype=np.float32)
        for x in normalized:
            gates = (bi + bh).astype(np.float32)
            for column in range(9):
                gates = (gates + (ih[:, column] * x[column]).astype(np.float32)).astype(np.float32)
            for column in range(hidden_dim):
                gates = (gates + (hh[:, column] * h[column]).astype(np.float32)).astype(np.float32)
            i, f, g, o = np.split(gates, 4)
            i = np.asarray([1. / (1. + math.exp(-float(v))) for v in i], dtype=np.float32)
            f = np.asarray([1. / (1. + math.exp(-float(v))) for v in f], dtype=np.float32)
            o = np.asarray([1. / (1. + math.exp(-float(v))) for v in o], dtype=np.float32)
            g = np.asarray([math.tanh(float(v)) for v in g], dtype=np.float32)
            c = ((f * c).astype(np.float32) + (i * g).astype(np.float32)).astype(np.float32)
            h = (o * np.asarray([math.tanh(float(v)) for v in c], dtype=np.float32)).astype(np.float32)
        for name, dimension in HEAD_MODULES.items():
            if name.split("_")[0] != branch:
                continue
            weight = np.asarray(payload[camel(name) + "Weight"], dtype=np.float32).reshape(dimension, hidden_dim)
            output = np.asarray(payload[camel(name) + "Bias"], dtype=np.float32).copy()
            for column in range(hidden_dim):
                output = (output + (weight[:, column] * h[column]).astype(np.float32)).astype(np.float32)
            if name.endswith("logstd"):
                output = np.clip(output, -5., 2.)
            outputs[name] = output
    return np.concatenate([outputs[name] for name in HEAD_MODULES]).astype(np.float32)


def run_export_self_checks(model: NormalHMDV4, payload: dict[str, Any], history: np.ndarray) -> dict[str, Any]:
    training = training_self_checks()
    normalized = ((history.astype(np.float32) - np.asarray(payload["stateMean"], dtype=np.float32)) / np.asarray(payload["stateStd"], dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        step, _ = model.warm_history(torch.from_numpy(normalized)[None])
    error = float(np.max(np.abs(raw_heads(step) - portable_lstm_heads(payload, history))))
    if error > 5e-5:
        raise AssertionError(f"independent portable LSTM/heads differs from torch: {error}")
    initial, fixture, event = event_fixture(payload, 32, 314159265)
    if not np.all(event["diagnostics"][0, :2] == 1):
        raise AssertionError("event fixture did not trigger H/V simultaneously")
    if np.any(event["diagnostics"][1:9, :2]):
        raise AssertionError("event fixture retriggered during event/cooldown")
    for left, right in ((0, 4), (4, 9)):
        if left == 0:
            for offset, dimension in ((0, 4), (16, 2)):
                target = event["controller"][0, offset + 4 + dimension:offset + 4 + 2 * dimension]
                if not np.all(event["controller"][left:right+1, offset + 4 + dimension:offset + 4 + 2 * dimension] == target):
                    raise AssertionError("latched target changed during event")
        else:
            if not np.all(event["applied"][left:right+1, 3:] == event["applied"][left, 3:]):
                raise AssertionError("HeadLocal pose not held during cooldown")
    # Force portable parameters into torch's independent controller API and
    # compare its minimum-jerk/hold timing from a nonzero physical start.
    portable_controllers = initial_controllers(payload, initial)
    seeds = derived_seeds(314159265)
    streams = PortableGaussian(seeds[1]), PortableGaussian(seeds[2])
    previous = initial.copy()
    runtime = model.initial_runtime_state(torch.from_numpy(((initial - np.asarray(payload["stateMean"], dtype=np.float32)) / np.asarray(payload["stateStd"], dtype=np.float32)).astype(np.float32))[None])
    controller_error = 0.
    for frame in fixture:
        composed, portable_controllers, diagnostic = compose(payload, frame, previous, portable_controllers, streams)
        # Recreate tensor heads without invoking the networks.
        slices = (frame[:3], frame[3:7], frame[7:8], frame[8:13], frame[13:18], frame[18:19], frame[19:22], frame[22:25])
        v4step = V4Step(*(torch.from_numpy(value.copy())[None] if index not in (2, 5) else torch.tensor([float(value[0])]) for index, value in enumerate(slices)))
        kwargs = {}
        for branch, controller, triggered in (("horizontal", portable_controllers[0], diagnostic[0]), ("vertical", portable_controllers[1], diagnostic[1])):
            kwargs[f"{branch}_trigger_override"] = torch.tensor([bool(triggered)])
            if triggered:
                parameters = np.concatenate((controller.target, [math.log(controller.duration)])).astype(np.float32)
                kwargs[f"{branch}_parameter_override_physical"] = torch.from_numpy(parameters)[None]
        with torch.no_grad():
            physical_previous = torch.from_numpy(((previous - np.asarray(payload["stateMean"], dtype=np.float32)) / np.asarray(payload["stateStd"], dtype=np.float32)).astype(np.float32))[None]
            state, runtime, _ = model.runtime_step(physical_previous, v4step, runtime, model.seeded_generator(torch.device("cpu"), 1), **kwargs)
        physical = (state[0].numpy() * np.asarray(payload["stateStd"], dtype=np.float32) + np.asarray(payload["stateMean"], dtype=np.float32)).astype(np.float32)
        controller_error = max(controller_error, float(np.max(np.abs(composed - physical))))
        previous = np.clip(composed, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
    if controller_error > 2e-6:
        raise AssertionError(f"portable controller differs from training controller: {controller_error}")
    # Verify exact consumption as well as outcomes. This detects an accidental
    # per-frame Bernoulli/Gaussian draw while a controller is ineligible.
    class CountingGaussian(PortableGaussian):
        def __init__(self, seed: int) -> None:
            super().__init__(seed)
            self.uint_count = 0
            self.gaussian_count = 0

        def next_uint(self) -> int:
            self.uint_count += 1
            return super().next_uint()

        def gaussian(self) -> float:
            self.gaussian_count += 1
            return super().gaussian()

    counted_seeds = derived_seeds(314159265)
    counted = CountingGaussian(counted_seeds[1]), CountingGaussian(counted_seeds[2])
    counted_controllers = initial_controllers(payload, initial)
    counted_previous = initial.copy()
    counted_diagnostics = []
    for frame in fixture:
        composed, counted_controllers, diagnostic = compose(payload, frame, counted_previous, counted_controllers, counted)
        counted_previous = np.clip(composed, payload["stateClipLow"], payload["stateClipHigh"]).astype(np.float32)
        counted_diagnostics.append(diagnostic)
    counted_diagnostics = np.asarray(counted_diagnostics)
    consumption = {}
    for branch, index, dimension in (("horizontal", 0, 5), ("vertical", 1, 3)):
        gaussian_count = int(counted_diagnostics[:, index].sum()) * dimension
        eligible_count = int(counted_diagnostics[:, index + 2].sum())
        expected_uint = eligible_count + 2 * int(math.ceil(gaussian_count / 2.))
        if counted[index].gaussian_count != gaussian_count or counted[index].uint_count != expected_uint:
            raise AssertionError(f"{branch} RNG consumption violates eligible-only/onset-only contract")
        consumption[branch] = {"eligible_uniforms": eligible_count, "onset_gaussians": gaussian_count, "xorshift_uints": expected_uint}
    # Stream independence: extra horizontal draws cannot alter V or root.
    seed_values = derived_seeds(0)
    baseline = [PortableGaussian(seed_values[2]).next_uint(), PortableGaussian(seed_values[0]).next_uint()]
    horizontal_rng = PortableGaussian(seed_values[1])
    for _ in range(31):
        horizontal_rng.gaussian()
    if baseline != [PortableGaussian(seed_values[2]).next_uint(), PortableGaussian(seed_values[0]).next_uint()]:
        raise AssertionError("portable streams are coupled")
    # Stationary initial residual must be nonzero and affect the first update.
    root = PortableRoot(
        np.asarray(payload["stateStd"], dtype=np.float32),
        246813579,
        normalized_std=float(payload["rootInnovationNormalizedStd"]),
        rho=float(payload["rootInnovationRho"][0]),
    )
    initial_root = root.initial_residual.copy()
    first_root = root.next()
    if np.all(initial_root == 0) or np.array_equal(initial_root, first_root):
        raise AssertionError("root process was not initialized from stationarity")
    return {"status": "passed", "training_self_checks": training, "portable_lstm_raw_heads_max_abs_error": error,
            "portable_controller_torch_max_abs_error": controller_error, "simultaneous_onset": True,
            "latched_parameters_and_hold": True, "eligible_only_uniform": True,
            "event_rng_consumption": consumption,
            "independent_random_streams": True, "stationary_root_initialization": True,
            "headlocal_per_frame_noise": False}


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint, metadata, metrics, validation, hashes, data_report = load_and_verify_sources(args)
    validate_contract(checkpoint, args.allow_smoke)
    model = build_model(checkpoint)
    v3 = torch.load(args.root_bounds_checkpoint, map_location="cpu", weights_only=True)
    low, high = deployment_state_clamps(checkpoint, v3, args.roll_limit_rad)
    config, state = checkpoint["model_config"], checkpoint["state_dict"]
    source_hash, report_hash = sha256_file(args.checkpoint), sha256_file(args.data_report)
    histories = np.asarray(validation["X"], dtype=np.float32)
    history = histories[0].copy()
    payload: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION, "exportedAtUtc": datetime.now(timezone.utc).isoformat(),
        "dataSource": "EyeNavGS", "syntheticTraining": False, "smokeTraining": bool(metrics.get("smoke")),
        "sourceModelSha256": source_hash, "sourceDataReportSha256": report_hash,
        "sourceRootBoundsModelSha256": sha256_file(args.root_bounds_checkpoint),
        "datasetFingerprintSha256": checkpoint["dataset_fingerprint_sha256"],
        "preparedTrainSha256": hashes["train"], "preparedValSha256": hashes["val"], "preparedTestSha256": hashes["test"],
        "modelTask": "independent_hazard_event_normal_hmd_generation", "architecture": config["architecture"],
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "inputDim": 9, "hiddenDim": config["hidden_dim"], "numLayers": 1,
        "rootOutputDim": 3, "horizontalPoseOutputDim": 4, "horizontalParameterDim": 5, "verticalParameterDim": 3,
        "rawHeadCount": 25, "lstmGateOrder": "ifgo", "tensorLayout": "row-major", "goldenArrayLayout": "step-major; index=step*width+channel",
        "historyLength": histories.shape[1], "trainedRolloutLength": validation["Y"].shape[1],
        "fixedDtSeconds": checkpoint["event_definition"]["dt_seconds"],
        "trainedRolloutSeconds": validation["Y"].shape[1] * checkpoint["event_definition"]["dt_seconds"],
        "nominalEyeHeightM": data_report.get("body_proxy", {}).get("nominal_eye_height_m", 1.65),
        "coordinateSystemJson": json.dumps(data_report.get("coordinate_system", {}), ensure_ascii=False, separators=(",", ":")),
        "stateFeatureNames": list(STATE_FEATURE_NAMES), "rootStateIndices": list(ROOT_INDICES),
        "horizontalStateIndices": list(HORIZONTAL_INDICES), "verticalStateIndices": list(VERTICAL_INDICES),
        "horizontalParameterNames": config["horizontal_parameter_layout"], "verticalParameterNames": config["vertical_parameter_layout"],
        "rawHeadNames": list(HEAD_NAMES), "stateClipLow": floats(low), "stateClipHigh": floats(high),
        "deploymentRollClampRad": float(F32(args.roll_limit_rad)),
        "durationMinSteps": config["duration_bounds_steps"][0], "durationMaxSteps": config["duration_bounds_steps"][1],
        "cooldownSteps": 4, "horizontalDurationScale": 1., "verticalDurationScale": 1., "eventInterpolation": "10u^3-15u^4+6u^5",
        "eventDefinitionJson": json.dumps(checkpoint["event_definition"], ensure_ascii=False, separators=(",", ":")),
        "eventParameterSampling": "mean+exp(clamp(logstd,-5,2))*(L*z), physical=center+scale*sample; midpoint+halfRange*tanh((physical-midpoint)/halfRange); absolute target clipped then latched; duration round-to-even(exp(logDuration))",
        "eventTiming": "onset frame elapsed0; active later frames elapsed+1; finish at duration-1; hold target; next four frames ineligible cooldown",
        "runtimeFeedbackContract": "final applied 9D physical state after root innovation and safety clip is normalized and fed to all three LSTMs",
        "horizontalPosePolicy": "auxiliary continuous pose head is exported for parity; applied horizontal pose comes from holding event controller",
        "headInnovationPolicy": "none; only eligible Bernoulli and 5/3 Gaussian event parameters once per onset",
        "headInnovationStd": [0.] * 6, "rootInnovationDim": 3,
        "rootInnovationNormalizedStd": float(F32(args.root_innovation_std)),
        "rootInnovationRho": [float(F32(args.root_innovation_rho))] * 3,
        "rootInnovationCholesky": floats(
            np.diag(
                state["state_std"].numpy()[:3]
                * F32(args.root_innovation_std)
            )
        ),
        "defaultRootNoiseScale": 1., "rootInnovationFeedsBack": True,
        "rootInnovationInitialization": "consume3Gaussian; residual=L*z from stationary N(0,LLT) before first Next",
        "rootInnovationFormula": "each Next consume3Gaussian; epsilon=L*z; residual_i=rho_i*residual_i+sqrt(1-rho_i^2)*epsilon_i; appliedRoot=prediction+scale*residual; std/rho selected by Root smoothness validation",
        "portableRng": "xorshift32(13,17,5), zero seed fallback0x6D2B79F5, openUniform=(uint+.5)/4294967296; cached BoxMuller double math, Gaussian castfloat",
        "seedMixAlgorithm": "z=x+0x9E3779B9; z=(z^(z>>16))*0x21F0AAAD; z=(z^(z>>15))*0x735A2D97; return z^(z>>15); unchecked uint32",
        "rootSeedSalt": ROOT_SEED_SALT, "horizontalSeedSalt": HORIZONTAL_SEED_SALT, "verticalSeedSalt": VERTICAL_SEED_SALT, "seedWindowSalt": SEED_WINDOW_SALT,
        "rootSeedXorConstant": ROOT_SEED_SALT, "horizontalSeedXorConstant": HORIZONTAL_SEED_SALT, "verticalSeedXorConstant": VERTICAL_SEED_SALT, "seedWindowXorConstant": SEED_WINDOW_SALT,
        "randomAlgorithm": "xorshift32_plus_cached_box_muller",
        "seedDerivationAlgorithm": "splitmix32_xor_stream_constant",
        "seedDerivation": "streamSeed=SplitMix32(uint32(runSeed)^streamSalt); seed window index=xorshift32(derivedWindowSeed).NextUInt()%seedWindows.Length",
        "seedWindows": choose_seed_windows(validation, args.seeds_per_trajectory),
        "goldenSourceValRow": 0, "goldenHistoryRaw": floats(history),
        "goldenControllerStateNames": controller_names(), "goldenControllerStateWidth": 26,
        "goldenDiagnosticNames": list(DIAGNOSTIC_NAMES), "goldenDiagnosticWidth": 6,
        "goldenTolerance": .0005, "goldenDeterministicPolicy": "test only: eligible && p>=.5, mean parameters, root scale0; no event RNG draws",
    }
    for branch in ("root", "horizontal", "vertical"):
        for parameter in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            payload[branch + camel(parameter)[:1].upper() + camel(parameter)[1:]] = floats(state[f"{branch}_lstm.{parameter}"])
    for name in HEAD_MODULES:
        payload[camel(name) + "Weight"] = floats(state[name + ".weight"])
        payload[camel(name) + "Bias"] = floats(state[name + ".bias"])
    for name in BUFFER_SHAPES:
        field = camel(name).replace("HazardBias", "HazardCalibrationBias")
        payload[field] = float(state[name]) if state[name].ndim == 0 else floats(state[name])
    self_checks = run_export_self_checks(model, payload, history)
    if args.allow_smoke:
        calibration = {"status": "not_run_diagnostic_only", "selected_default_root_noise_scale": 1., "test_evaluated": False}
    elif args.reuse_root_calibration:
        calibration = json.loads(args.root_calibration_report.read_text(encoding="utf-8"))
        verify_saved_calibration(calibration, source_hash, hashes["val"], payload)
    else:
        calibration = calibrate_root_scale(model, payload, validation, source_hash, args.root_calibration_report)
    if calibration.get("selected_default_root_noise_scale") is None:
        failure = {"schema_version": "4.0.0", "status": "blocked_quality_gate", "reason": "no validation-only root noise candidate meets frozen safety/distance criteria",
                   "source_checkpoint_sha256": source_hash, "self_checks": self_checks, "root_calibration_report": str(args.root_calibration_report.resolve()), "asset_written": False}
        atomic_write_text(args.report, json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
        raise ValueError(failure["reason"])
    selected_scale = float(calibration["selected_default_root_noise_scale"])
    payload["defaultRootNoiseScale"] = selected_scale
    payload["rootNoiseCalibrationSplit"] = "val"
    payload["rootNoiseCalibrationReportSha256"] = sha256_file(args.root_calibration_report) if not args.allow_smoke else ""
    deterministic = closed_loop_rollout(model, payload, history, args.golden_steps, 0, deterministic=True)
    payload["goldenStepCount"] = args.golden_steps
    for field, name in (("RawHeads", "rawHeads"), ("ComposedBeforeInnovation", "composed"), ("Applied", "applied"), ("ControllerState", "controller"), ("Diagnostics", "diagnostics")):
        payload["goldenExpected" + field] = floats(deterministic[name])
    initial, fixture, event = event_fixture(payload, args.event_golden_steps, args.event_golden_seed)
    payload.update({"eventGoldenRunSeed": args.event_golden_seed, "eventGoldenStepCount": args.event_golden_steps,
                    "eventGoldenInitialPhysical": floats(initial), "eventGoldenRawHeads": floats(fixture),
                    "eventGoldenDerivedHorizontalSeed": derived_seeds(args.event_golden_seed)[1], "eventGoldenDerivedVerticalSeed": derived_seeds(args.event_golden_seed)[2]})
    for field, name in (("Applied", "applied"), ("ControllerState", "controller"), ("Diagnostics", "diagnostics")):
        payload["eventGoldenExpected" + field] = floats(event[name])
    root = PortableRoot(
        state["state_std"].numpy(),
        args.innovation_golden_seed,
        normalized_std=args.root_innovation_std,
        rho=args.root_innovation_rho,
    )
    payload.update({"rootInnovationGoldenSeed": args.innovation_golden_seed, "rootInnovationGoldenDerivedSeed": derived_seeds(args.innovation_golden_seed)[0],
                    "rootInnovationGoldenStepCount": args.innovation_golden_steps, "rootInnovationGoldenInitialResidual": floats(root.initial_residual),
                    "rootInnovationGoldenExpected": floats(np.asarray([root.next() for _ in range(args.innovation_golden_steps)], dtype=np.float32))})
    combined = closed_loop_rollout(model, payload, history, args.combined_golden_steps, args.combined_golden_seed, root_scale=selected_scale)
    payload.update({"combinedStochasticGoldenSeed": args.combined_golden_seed, "combinedStochasticGoldenStepCount": args.combined_golden_steps, "combinedStochasticGoldenNoiseScale": selected_scale})
    for field, index in (("Root", 0), ("Horizontal", 1), ("Vertical", 2)):
        payload["combinedStochasticGoldenDerived" + field + "Seed"] = derived_seeds(args.combined_golden_seed)[index]
    for field, name in (("RawHeads", "rawHeads"), ("ComposedBeforeInnovation", "composed"), ("Applied", "applied"), ("ControllerState", "controller"), ("Diagnostics", "diagnostics"), ("RootInnovation", "rootInnovation")):
        payload["combinedStochasticGoldenExpected" + field] = floats(combined[name])
    # Validate the exact deployment policy for 60 seconds. Passing numerical
    # checks does not imply statistical equivalence or qualify the model as a
    # predictor of cybersickness; the separate validation gate remains required.
    probe_steps = int(math.ceil(60. / payload["fixedDtSeconds"]))
    probe = closed_loop_rollout(model, payload, history, probe_steps, 7045, root_scale=selected_scale)
    safety = rollout_safety_metrics(payload, probe, selected_scale)
    clip_fraction = safety["unexpected_clip_step_rate"]
    if clip_fraction >= .01 and not args.allow_smoke:
        raise ValueError(f"production deployment probe unexpectedly clips {clip_fraction:.3%} frames (>=1%)")
    probe_report = {"seconds_generated": probe_steps * payload["fixedDtSeconds"], "steps": probe_steps, "finite": True,
                    "clip_step_fraction": clip_fraction, "passed": clip_fraction <= .01,
                    "horizontal_event_count": int(probe["diagnostics"][:, 0].sum()), "vertical_event_count": int(probe["diagnostics"][:, 1].sum()),
                    "unexpectedClipStepRate": safety["unexpected_clip_step_rate"], "comfortEnvelopeActivationRate": safety["comfort_envelope_activation_rate"], "metrics": safety}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # Smoke exports are restricted to a diagnostic destination.
    if args.allow_smoke and (args.asset.resolve() == DEFAULT_ASSET.resolve() or args.integrity.resolve() == DEFAULT_INTEGRITY.resolve()):
        raise ValueError("--allow-smoke requires diagnostic asset and integrity paths")
    atomic_write_bytes(args.asset, encoded)
    asset_hash = sha256_file(args.asset)
    integrity = "// Generated by scripts/normal_hmd_v4_export.py. Do not edit.\nnamespace NormalHmdAgentDemo\n{\n    public static class GeneratedNormalHmdIntegrity\n    {\n"
    for field, value in (("SchemaVersion", SCHEMA_VERSION), ("ExportSha256", asset_hash), ("SourceModelSha256", source_hash), ("SourceDataReportSha256", report_hash),
                         ("DatasetFingerprintSha256", checkpoint["dataset_fingerprint_sha256"]), ("PreparedTrainSha256", hashes["train"]), ("PreparedValSha256", hashes["val"]), ("PreparedTestSha256", hashes["test"])):
        integrity += f'        public const string {field} = "{value}";\n'
    for field, value in (("SeedWindowCount", len(payload["seedWindows"])), ("GoldenStepCount", args.golden_steps), ("EventGoldenStepCount", args.event_golden_steps), ("RootInnovationGoldenStepCount", args.innovation_golden_steps), ("CombinedStochasticGoldenStepCount", args.combined_golden_steps)):
        integrity += f"        public const int {field} = {value};\n"
    integrity += "    }\n}\n"
    atomic_write_text(args.integrity, integrity)
    result = {"schema_version": "4.0.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success", "unity_schema_version": SCHEMA_VERSION,
              "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": source_hash, "source_metadata": str(args.metadata.resolve()), "source_metrics": str(args.metrics.resolve()),
              "source_training_is_synthetic": False, "smoke": bool(metrics["smoke"]), "source_data_report_sha256": report_hash,
              "dataset_fingerprint_sha256": checkpoint["dataset_fingerprint_sha256"], "verified_prepared_split_sha256": hashes,
              "asset": str(args.asset.resolve()), "asset_sha256": asset_hash, "integrity_source": str(args.integrity.resolve()),
              "parameter_count": payload["parameterCount"], "seed_windows": len(payload["seedWindows"]), "seed_split": "val", "golden_source_val_row": 0,
              "raw_head_names": list(HEAD_NAMES), "controller_state_names": controller_names(), "diagnostic_names": list(DIAGNOSTIC_NAMES),
              "array_layout": payload["goldenArrayLayout"], "self_checks": self_checks, "deployment_probe": probe_report,
              "root_innovation": {"rho": payload["rootInnovationRho"], "stationary_cholesky_physical": payload["rootInnovationCholesky"], "normalized_std": [float(F32(args.root_innovation_std))] * 3, "default_noise_scale": selected_scale, "initialization": "stationary N(0,LLT)", "feeds_back": True},
              "root_noise_calibration": calibration, "root_noise_calibration_report": str(args.root_calibration_report.resolve()),
              "golden_step_counts": {"deterministic": args.golden_steps, "event": args.event_golden_steps, "root": args.innovation_golden_steps, "combined": args.combined_golden_steps},
              "event_fixture_simultaneous_onsets": int(np.sum((event["diagnostics"][:, 0] != 0) & (event["diagnostics"][:, 1] != 0))),
              "deployment_state_clip_low": floats(low), "deployment_state_clip_high": floats(high),
              "duration_scale": {"horizontal": 1., "vertical": 1.},
              "validation_scope": "export parity and 60-second numerical safety; statistical fidelity and sickness-preassessment claims require separate validation",
              "statistical_validation_warning": "event occupancy is below the provisional human band; no artificial HeadLocal jitter or duration inflation is used to conceal it",
              "training_metrics_model_sha256_verified": True, "metadata_model_sha256_verified": True}
    atomic_write_text(args.report, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("checkpoint", DEFAULT_CHECKPOINT), ("metadata", DEFAULT_METADATA), ("metrics", DEFAULT_METRICS), ("data-report", DEFAULT_DATA_REPORT), ("data-dir", DEFAULT_DATA_DIR), ("root-bounds-checkpoint", DEFAULT_V3_CHECKPOINT), ("asset", DEFAULT_ASSET), ("integrity", DEFAULT_INTEGRITY), ("report", DEFAULT_REPORT)):
        parser.add_argument("--" + flag, type=Path, default=default)
    parser.add_argument("--seeds-per-trajectory", type=int, default=4)
    parser.add_argument("--golden-steps", type=int, default=64)
    parser.add_argument("--event-golden-seed", type=int, default=314159265)
    parser.add_argument("--event-golden-steps", type=int, default=32)
    parser.add_argument("--innovation-golden-seed", type=int, default=246813579)
    parser.add_argument("--innovation-golden-steps", type=int, default=64)
    parser.add_argument("--combined-golden-seed", type=int, default=97531)
    parser.add_argument("--combined-golden-steps", type=int, default=128)
    parser.add_argument(
        "--root-innovation-std",
        type=float,
        default=.15,
        help="Stationary Root innovation standard deviation in normalized units.",
    )
    parser.add_argument(
        "--root-innovation-rho",
        type=float,
        default=.95,
        help="AR(1) persistence for the Root innovation process.",
    )
    parser.add_argument(
        "--roll-limit-rad",
        type=float,
        default=.12,
        help="Deployment HeadLocal roll comfort clamp in radians (at most .12).",
    )
    parser.add_argument("--allow-smoke", action="store_true", help="diagnostic paths only; does not alter frozen production dimensions")
    parser.add_argument("--root-calibration-report", type=Path, default=ROOT / "results" / "normal_hmd_v4_root_noise_calibration.json")
    parser.add_argument("--reuse-root-calibration", action="store_true", help="reuse matching frozen-source validation-only 16-history scale report")
    args = parser.parse_args(argv)
    if min(args.seeds_per_trajectory, args.golden_steps, args.event_golden_steps, args.innovation_golden_steps, args.combined_golden_steps) < 1:
        parser.error("all window/golden counts must be positive")
    if args.event_golden_steps < 16:
        parser.error("event golden requires >=16 steps to cover event/hold/cooldown/retrigger")
    if args.root_innovation_std < 0.0:
        parser.error("--root-innovation-std cannot be negative")
    if not 0.0 <= args.root_innovation_rho < 1.0:
        parser.error("--root-innovation-rho must be in [0, 1)")
    if not 0.0 < args.roll_limit_rad <= .12:
        parser.error("--roll-limit-rad must be in (0, .12]")
    return args


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    result = export(parse_args(argv))
    summary = {key: result[key] for key in ("status", "unity_schema_version", "asset", "asset_sha256", "parameter_count", "seed_windows", "deployment_probe", "root_innovation", "golden_step_counts")}
    summary["report"] = str(parse_args(argv).report.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
