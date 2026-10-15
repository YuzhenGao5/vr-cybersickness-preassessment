#!/usr/bin/env python3
"""Export Normal HMD Agent v2 weights, seeds and parity vectors to Unity JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normal_hmd_common import STATE_FEATURE_NAMES, sha256_file
from normal_hmd_model import NormalHMDLSTM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "normal_hmd_v2.pth"
DEFAULT_METADATA = ROOT / "models" / "normal_hmd_v2_metadata.json"
DEFAULT_TEST_DATA = ROOT / "data" / "processed" / "normal_hmd_v2" / "normal_hmd_test.npz"
DEFAULT_METRICS = ROOT / "results" / "normal_hmd_v2_training_metrics.json"
DEFAULT_ASSET = (
    ROOT / "demo" / "UnityNormalHmdAgentDemo" / "Assets" / "Resources" / "NormalHmdAgentModel.json"
)
DEFAULT_INTEGRITY = (
    ROOT
    / "demo"
    / "UnityNormalHmdAgentDemo"
    / "Assets"
    / "Scripts"
    / "GeneratedNormalHmdIntegrity.cs"
)
DEFAULT_REPORT = ROOT / "results" / "normal_hmd_v2_unity_export_report.json"


def floats(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()


def choose_seed_windows(test: dict[str, np.ndarray], per_trajectory: int) -> list[dict[str, Any]]:
    trajectory_index = test["trajectory_index"].astype(np.int64)
    seeds: list[dict[str, Any]] = []
    for index, trajectory_id in enumerate(test["trajectory_ids"]):
        candidates = np.flatnonzero(trajectory_index == index)
        if not len(candidates):
            continue
        count = min(per_trajectory, len(candidates))
        selected = candidates[np.linspace(0, len(candidates) - 1, count, dtype=np.int64)]
        for row in selected:
            history = test["X"][row]
            if not np.isfinite(history).all():
                raise ValueError(f"non-finite held-out seed at row {row}")
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
    """xorshift32 + Box-Muller RNG mirrored exactly by the Unity runtime."""

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


def build_model(checkpoint: dict[str, Any]) -> NormalHMDLSTM:
    config = checkpoint["model_config"]
    model = NormalHMDLSTM(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def cpu_deterministic_rollout(
    model: NormalHMDLSTM,
    history_physical: np.ndarray,
    checkpoint: dict[str, Any],
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the deployment golden vector with the CPU FP32 reference path.

    Training may run through CUDA/cuDNN, whose reduction order need not exactly
    match the dependency-free C# loop.  Unity parity should therefore compare
    against a CPU FP32 rollout reconstructed from the frozen weights, while the
    export report records any difference from the trainer's CUDA probe.
    """

    normalization = checkpoint["normalization"]
    mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    std = np.asarray(normalization["state_std"], dtype=np.float32)
    clip_low = np.asarray(checkpoint["state_clip_low"], dtype=np.float32)
    clip_high = np.asarray(checkpoint["state_clip_high"], dtype=np.float32)
    normalized_history = (
        (np.asarray(history_physical, dtype=np.float32) - mean) / std
    )
    prediction, hidden = model.warm_history(torch.from_numpy(normalized_history)[None])
    raw_rows: list[np.ndarray] = []
    clipped_rows: list[np.ndarray] = []
    for step in range(steps):
        raw = prediction[0].numpy().astype(np.float32) * std + mean
        clipped = np.clip(raw, clip_low, clip_high).astype(np.float32)
        raw_rows.append(raw)
        clipped_rows.append(clipped)
        if step + 1 < steps:
            feedback = ((clipped - mean) / std).astype(np.float32)
            prediction, hidden = model.advance(torch.from_numpy(feedback)[None], hidden)
    return np.asarray(raw_rows), np.asarray(clipped_rows)


@torch.no_grad()
def portable_stochastic_rollout(
    model: NormalHMDLSTM,
    history_physical: np.ndarray,
    checkpoint: dict[str, Any],
    steps: int,
    seed: int,
    root_yaw_dc_tau_seconds: float,
    head_pose_dc_tau_seconds: float,
    head_innovation_smoothing_tau_seconds: float,
    head_innovation_smoothing_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalization = checkpoint["normalization"]
    # Match the dependency-free C# runtime's float state and double
    # Box-Muller calculations closely enough for a combined parity vector.
    mean = np.asarray(normalization["state_mean"], dtype=np.float32)
    std = np.asarray(normalization["state_std"], dtype=np.float32)
    clip_low = np.asarray(checkpoint["state_clip_low"], dtype=np.float32)
    clip_high = np.asarray(checkpoint["state_clip_high"], dtype=np.float32)
    innovation = checkpoint["innovation_model"]
    innovation_std = np.asarray(innovation["innovation_std_physical"], dtype=np.float32)
    rho = np.asarray(innovation["innovation_ar1_rho"], dtype=np.float32)
    noise_scale = np.float32(innovation["default_noise_scale"])
    normalized_history = (history_physical.astype(np.float32) - mean) / std
    prediction, hidden = model.warm_history(
        torch.from_numpy(normalized_history.astype(np.float32))[None]
    )
    generator = PortableGaussian(seed)
    residual = np.zeros(len(mean), dtype=np.float32)
    root_yaw_dc = np.float32(0.0)
    root_yaw_alpha = np.float32(
        1.0 - math.exp(-float(checkpoint["dt_seconds"]) / root_yaw_dc_tau_seconds)
    )
    head_alpha = np.float32(
        1.0 - math.exp(-float(checkpoint["dt_seconds"]) / head_pose_dc_tau_seconds)
    )
    head_innovation_alpha = np.float32(
        1.0
        - math.exp(
            -float(checkpoint["dt_seconds"])
            / head_innovation_smoothing_tau_seconds
        )
    )
    anchor_margin_sigma = np.float32(3.0)
    anchor_margin = (
        anchor_margin_sigma * noise_scale * innovation_std[3:9]
    ).astype(np.float32)
    head_anchor = np.clip(
        history_physical[-1, 3:9].astype(np.float32),
        clip_low[3:9] + anchor_margin,
        clip_high[3:9] - anchor_margin,
    ).astype(np.float32)
    # A momentary last-frame head tilt must not become the 60-second horizon.
    # Keep the learned deviations/noise, but center pitch/roll on upright.
    head_anchor[3] = np.float32(0.0)  # state feature 6: rotvec X / pitch
    head_anchor[5] = np.float32(0.0)  # state feature 8: rotvec Z / roll
    head_dc = np.zeros(6, dtype=np.float32)
    smoothed_head_innovation = np.zeros(6, dtype=np.float32)
    head_dc_initialized = False
    raw_rows: list[np.ndarray] = []
    clipped_rows: list[np.ndarray] = []
    for step in range(steps):
        base = (prediction[0].numpy().astype(np.float32) * std + mean).astype(np.float32)
        stabilized_base = base.copy()
        root_yaw_dc = np.float32(
            root_yaw_dc + root_yaw_alpha * np.float32(base[2] - root_yaw_dc)
        )
        stabilized_base[2] = np.float32(base[2] - root_yaw_dc)
        if not head_dc_initialized:
            head_dc[:] = base[3:9]
            head_dc_initialized = True
        else:
            head_dc += head_alpha * (base[3:9] - head_dc)
        stabilized_base[3:9] = head_anchor + (base[3:9] - head_dc)
        for feature in range(len(mean)):
            gaussian = generator.gaussian()
            rho_square_float = np.float32(rho[feature] * rho[feature])
            innovation_gain = math.sqrt(max(0.0, 1.0 - float(rho_square_float)))
            autoregressive_float = np.float32(rho[feature] * residual[feature])
            residual[feature] = np.float32(
                float(autoregressive_float)
                + innovation_gain * float(innovation_std[feature]) * gaussian
            )
        applied_innovation = (residual * noise_scale).astype(np.float32)
        # BodyRoot innovation remains unchanged.  HeadLocal innovation is an
        # absolute-pose residual, so applying it directly makes its first
        # difference nearly white.  Smooth only this output-layer term; it is
        # still excluded from recurrent feedback.
        smoothed_head_innovation += head_innovation_alpha * (
            applied_innovation[3:9] - smoothed_head_innovation
        )
        applied_innovation[3:9] = (
            smoothed_head_innovation * np.float32(head_innovation_smoothing_gain)
        ).astype(np.float32)
        raw = (stabilized_base + applied_innovation).astype(np.float32)
        clipped = np.clip(raw, clip_low, clip_high).astype(np.float32)
        raw_rows.append(raw)
        clipped_rows.append(clipped)
        if step + 1 < steps:
            # The innovation is an output-layer observation residual.  It is
            # displayed and logged, but must not perturb the LSTM's latent
            # intention trajectory on the following step.
            base_feedback = np.clip(base, clip_low, clip_high)
            feedback = ((base_feedback - mean) / std).astype(np.float32)
            prediction, hidden = model.advance(torch.from_numpy(feedback)[None], hidden)
    return np.asarray(raw_rows), np.asarray(clipped_rows)


def portable_innovation_golden(
    innovation_std_values: Any,
    rho_values: Any,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Mirror SeededHmdInnovation.Next(1f), including float residual rounding."""

    innovation_std = np.asarray(innovation_std_values, dtype=np.float32)
    rho = np.asarray(rho_values, dtype=np.float32)
    residual = np.zeros_like(innovation_std, dtype=np.float32)
    generator = PortableGaussian(seed)
    output = np.empty((steps, len(residual)), dtype=np.float32)
    for step in range(steps):
        for feature in range(len(residual)):
            gaussian = generator.gaussian()
            rho_square_float = np.float32(rho[feature] * rho[feature])
            innovation_gain = math.sqrt(max(0.0, 1.0 - float(rho_square_float)))
            autoregressive_float = np.float32(rho[feature] * residual[feature])
            residual[feature] = np.float32(
                float(autoregressive_float)
                + innovation_gain * float(innovation_std[feature]) * gaussian
            )
            output[step, feature] = residual[feature]
    return output


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "state_dict", "model_config", "normalization", "state_clip_low",
        "state_clip_high", "innovation_model", "dt_seconds", "feature_names",
        "coordinate_system", "golden_stateful_rollout", "data_source", "synthetic",
        "prepared_split_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    if checkpoint["data_source"] != "EyeNavGS" or bool(checkpoint["synthetic"]):
        raise ValueError("Unity export requires the real EyeNavGS checkpoint")
    config = checkpoint["model_config"]
    if int(config["input_dim"]) != 9 or int(config["output_dim"]) != 9:
        raise ValueError("Unity contract requires 9-D state input/output")
    if int(config["num_layers"]) != 2:
        raise ValueError("Unity contract currently requires exactly two LSTM layers")
    if list(checkpoint["feature_names"]) != list(STATE_FEATURE_NAMES):
        raise ValueError("checkpoint feature order does not match the Unity 9-D state contract")

    source_hash = sha256_file(checkpoint_path)
    if args.metadata.is_file():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        declared_hash = str(metadata.get("model_sha256", ""))
        if declared_hash and declared_hash != source_hash:
            raise ValueError(
                f"checkpoint hash mismatch: metadata={declared_hash}, actual={source_hash}"
            )
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("status") != "success":
        raise ValueError("training metrics do not report success")
    declared_model_hash = str(metrics.get("artifacts", {}).get("model_sha256", ""))
    if declared_model_hash != source_hash:
        raise ValueError(
            "training metrics/checkpoint identity mismatch: "
            f"metrics={declared_model_hash!r}, checkpoint={source_hash!r}"
        )
    if bool(metrics.get("smoke_subset")) and not args.allow_smoke:
        raise ValueError("refusing to export a smoke-subset model as a production Unity asset")
    expected_test_hash = str(checkpoint.get("prepared_split_sha256", {}).get("test", ""))
    actual_test_hash = sha256_file(args.test_data)
    if not expected_test_hash or expected_test_hash != actual_test_hash:
        raise ValueError(
            "held-out test NPZ does not match the checkpoint provenance: "
            f"expected={expected_test_hash!r}, actual={actual_test_hash!r}"
        )
    if metrics.get("prepared_split_sha256") != checkpoint.get("prepared_split_sha256"):
        raise ValueError("training metrics/checkpoint prepared split hashes disagree")
    if not args.allow_smoke:
        probes = metrics.get("closed_loop_probe", {})
        deterministic = probes.get("deterministic", {})
        stochastic = probes.get("seeded_stochastic", [])
        if float(deterministic.get("requested_seconds", 0.0)) < 60.0:
            raise ValueError("production export requires a >=60-second closed-loop probe")
        deterministic_checks = deterministic.get("engineering_checks", {})
        required_deterministic_checks = (
            "finite",
            "ran_requested_steps",
            "clip_step_fraction_at_most_1_percent",
            "body_pitch_roll_structurally_zero",
            "head_angular_speed_p99_at_most_720_degps",
            "head_height_start_end_drift_at_most_0_25m",
        )
        if not all(bool(deterministic_checks.get(key)) for key in required_deterministic_checks):
            raise ValueError("deterministic 60-second stability checks did not pass")
        deterministic_distance = float(deterministic.get("integrated_planar_distance_m", math.nan))
        if not math.isfinite(deterministic_distance) or not 0.0 <= deterministic_distance <= 30.0:
            raise ValueError("deterministic 60-second trajectory distance is unstable")
        if len(stochastic) < 2 or not all(
            float(item.get("requested_seconds", 0.0)) >= 60.0
            and bool(item.get("engineering_probe_passed"))
            for item in stochastic
        ):
            raise ValueError("at least two stochastic 60-second engineering probes must pass")
        if not bool(probes.get("same_seed_bitwise_reproducible")):
            raise ValueError("seeded stochastic rollout is not reproducible")
        diversity = probes.get("different_seed_state_rms_difference")
        if diversity is None or not math.isfinite(float(diversity)) or float(diversity) <= 1e-6:
            raise ValueError("different stochastic seeds did not produce measurably different states")

    with np.load(args.test_data, allow_pickle=False) as data:
        test = {key: data[key] for key in data.files}
    seeds = choose_seed_windows(test, args.seeds_per_trajectory)
    golden = checkpoint["golden_stateful_rollout"]
    golden_history = np.asarray(golden["history_state_physical"], dtype=np.float32)
    if golden_history.shape != (int(config["history_length"]), 9):
        raise ValueError("checkpoint golden history has wrong shape")
    trainer_deterministic_expected = np.asarray(
        golden["raw_output_physical"], dtype=np.float32
    )
    # Constructing and strictly loading the module here catches a mismatched
    # checkpoint before emitting an asset, even though golden vectors are
    # already embedded by the trainer.
    verified_model = build_model(checkpoint)
    deterministic_expected, deterministic_clipped = cpu_deterministic_rollout(
        verified_model,
        golden_history,
        checkpoint,
        len(trainer_deterministic_expected),
    )
    trainer_vs_cpu_golden_max_abs_error = float(
        np.max(np.abs(trainer_deterministic_expected - deterministic_expected))
    )
    innovation_seed = int(args.innovation_golden_seed)
    innovation_steps = min(args.innovation_golden_steps, len(deterministic_expected))
    innovation_expected = portable_innovation_golden(
        checkpoint["innovation_model"]["innovation_std_physical"],
        checkpoint["innovation_model"]["innovation_ar1_rho"],
        innovation_steps,
        innovation_seed,
    )
    combined_seed = int(args.combined_golden_seed)
    combined_steps = min(args.combined_golden_steps, len(deterministic_expected))
    _, combined_expected_applied = portable_stochastic_rollout(
        verified_model,
        golden_history,
        checkpoint,
        combined_steps,
        combined_seed,
        args.root_yaw_dc_tau_seconds,
        args.head_pose_dc_tau_seconds,
        args.head_innovation_smoothing_tau_seconds,
        args.head_innovation_smoothing_gain,
    )

    state = checkpoint["state_dict"]
    normalization = checkpoint["normalization"]
    innovation = checkpoint["innovation_model"]
    payload = {
        "schemaVersion": "normal-hmd-agent-unity-1.2",
        "exportedAtUtc": datetime.now(timezone.utc).isoformat(),
        "dataSource": "EyeNavGS",
        "syntheticTraining": False,
        "sourceModelSha256": source_hash,
        "datasetFingerprintSha256": str(checkpoint["dataset_fingerprint_sha256"]),
        "modelTask": "normal_hmd_9d_autoregressive",
        "architecture": "stateful_lstm",
        "parameterCount": int(sum(value.numel() for value in state.values())),
        "lstmGateOrder": "ifgo",
        "tensorLayout": "row-major",
        "historyLength": int(config["history_length"]),
        "trainedRolloutLength": int(config["rollout_length"]),
        "inputDim": 9,
        "hiddenDim": int(config["hidden_dim"]),
        "outputDim": 9,
        "numLayers": 2,
        "fixedDtSeconds": float(checkpoint["dt_seconds"]),
        "trainedRolloutSeconds": float(
            int(config["rollout_length"]) * float(checkpoint["dt_seconds"])
        ),
        "nominalEyeHeightM": float(checkpoint["nominal_eye_height_m"]),
        "stateFeatureNames": list(STATE_FEATURE_NAMES),
        # Autoregressive state input and output use the same physical definition.
        "xMean": floats(normalization["state_mean"]),
        "xStd": floats(normalization["state_std"]),
        "yMean": floats(normalization["state_mean"]),
        "yStd": floats(normalization["state_std"]),
        "stateClipLow": floats(checkpoint["state_clip_low"]),
        "stateClipHigh": floats(checkpoint["state_clip_high"]),
        "innovationStd": floats(innovation["innovation_std_physical"]),
        "innovationRho": floats(innovation["innovation_ar1_rho"]),
        "defaultNoiseScale": float(innovation["default_noise_scale"]),
        "headInnovationPolicy": "BodyRoot AR unchanged; HeadLocal AR output EMA-smoothed with calibrated gain before application and excluded from LSTM feedback",
        "headInnovationSmoothingTauSeconds": float(
            args.head_innovation_smoothing_tau_seconds
        ),
        "headInnovationSmoothingGain": float(args.head_innovation_smoothing_gain),
        "deploymentStabilizer": "output-only EMA DC removal; never enters LSTM feedback",
        "rootYawDcRemovalTauSeconds": float(args.root_yaw_dc_tau_seconds),
        "headPoseDcRemovalTauSeconds": float(args.head_pose_dc_tau_seconds),
        "headPoseAnchor": "3-sigma-safe last history pose with pitch and roll neutralized",
        "headPoseAnchorMarginSigma": 3.0,
        "weightIhL0": floats(state["lstm.weight_ih_l0"]),
        "weightHhL0": floats(state["lstm.weight_hh_l0"]),
        "biasIhL0": floats(state["lstm.bias_ih_l0"]),
        "biasHhL0": floats(state["lstm.bias_hh_l0"]),
        "weightIhL1": floats(state["lstm.weight_ih_l1"]),
        "weightHhL1": floats(state["lstm.weight_hh_l1"]),
        "biasIhL1": floats(state["lstm.bias_ih_l1"]),
        "biasHhL1": floats(state["lstm.bias_hh_l1"]),
        "fcWeight": floats(state["output.weight"]),
        "fcBias": floats(state["output.bias"]),
        "seedWindows": seeds,
        "goldenHistoryRaw": floats(golden_history),
        "goldenStepCount": int(len(deterministic_expected)),
        "goldenExpectedClosedLoopRaw": floats(deterministic_expected),
        "goldenTolerance": 0.0005,
        "innovationGoldenSeed": innovation_seed,
        "innovationGoldenStepCount": innovation_steps,
        "innovationGoldenExpected": floats(innovation_expected),
        "combinedStochasticGoldenSeed": combined_seed,
        "combinedStochasticGoldenStepCount": combined_steps,
        "combinedStochasticGoldenNoiseScale": float(innovation["default_noise_scale"]),
        "combinedStochasticGoldenExpectedApplied": floats(combined_expected_applied),
    }
    args.asset.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.asset.write_bytes(encoded)
    asset_hash = sha256_file(args.asset)

    args.integrity.parent.mkdir(parents=True, exist_ok=True)
    args.integrity.write_text(
        "// Generated by scripts/normal_hmd_export.py. Do not edit.\n"
        "namespace NormalHmdAgentDemo\n"
        "{\n"
        "    public static class GeneratedNormalHmdIntegrity\n"
        "    {\n"
        f'        public const string ExportSha256 = "{asset_hash}";\n'
        f'        public const string SourceModelSha256 = "{source_hash}";\n'
        f"        public const int SeedWindowCount = {len(seeds)};\n"
        f"        public const int GoldenStepCount = {len(deterministic_expected)};\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_hash,
        "source_training_is_synthetic": False,
        "asset": str(args.asset.resolve()),
        "asset_sha256": asset_hash,
        "integrity_source": str(args.integrity.resolve()),
        "seed_windows": len(seeds),
        "held_out_participants": len(
            {":".join(seed["trajectoryId"].split(":")[:2]) for seed in seeds}
        ),
        "held_out_scenes": sorted(
            {seed["trajectoryId"].split(":")[2] for seed in seeds}
        ),
        "deterministic_golden_steps": int(len(deterministic_expected)),
        "innovation_golden_seed": innovation_seed,
        "innovation_golden_steps": innovation_steps,
        "combined_stochastic_golden_seed": combined_seed,
        "combined_stochastic_golden_steps": combined_steps,
        "combined_stochastic_golden_noise_scale": float(innovation["default_noise_scale"]),
        "head_innovation_policy": payload["headInnovationPolicy"],
        "head_innovation_smoothing_tau_seconds": float(
            args.head_innovation_smoothing_tau_seconds
        ),
        "head_innovation_smoothing_gain": float(
            args.head_innovation_smoothing_gain
        ),
        "deployment_stabilizer": payload["deploymentStabilizer"],
        "root_yaw_dc_removal_tau_seconds": float(args.root_yaw_dc_tau_seconds),
        "head_pose_dc_removal_tau_seconds": float(args.head_pose_dc_tau_seconds),
        "head_pose_anchor": payload["headPoseAnchor"],
        "head_pose_anchor_margin_sigma": payload["headPoseAnchorMarginSigma"],
        "portable_rng": "xorshift32 plus Box-Muller with cached sine sample",
        "trainer_vs_cpu_golden_max_abs_error": trainer_vs_cpu_golden_max_abs_error,
        "unity_golden_reference": "CPU FP32 rollout reconstructed from frozen checkpoint",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds-per-trajectory", type=int, default=4)
    parser.add_argument("--innovation-golden-seed", type=int, default=246813579)
    parser.add_argument("--innovation-golden-steps", type=int, default=64)
    parser.add_argument("--combined-golden-seed", type=int, default=97531)
    parser.add_argument("--combined-golden-steps", type=int, default=64)
    parser.add_argument("--root-yaw-dc-tau-seconds", type=float, default=2.5)
    parser.add_argument("--head-pose-dc-tau-seconds", type=float, default=2.5)
    parser.add_argument(
        "--head-innovation-smoothing-tau-seconds",
        type=float,
        default=0.15,
        help="Output-only EMA time constant for HeadLocal stochastic innovation.",
    )
    parser.add_argument(
        "--head-innovation-smoothing-gain",
        type=float,
        default=1.13,
        help="Calibrated amplitude compensation applied after HeadLocal EMA.",
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow a short smoke-subset artifact; never use this for the shipped demo.",
    )
    args = parser.parse_args(argv)
    if (
        args.seeds_per_trajectory < 1
        or args.innovation_golden_steps < 1
        or args.combined_golden_steps < 1
        or not math.isfinite(args.root_yaw_dc_tau_seconds)
        or args.root_yaw_dc_tau_seconds <= 0.0
        or not math.isfinite(args.head_pose_dc_tau_seconds)
        or args.head_pose_dc_tau_seconds <= 0.0
        or not (0.05 <= args.head_innovation_smoothing_tau_seconds <= 1.0)
        or not (0.5 <= args.head_innovation_smoothing_gain <= 2.0)
    ):
        parser.error(
            "counts/time constants must be positive; HeadLocal smoothing tau must "
            "be 0.05..1.0 s and gain must be 0.5..2.0"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    report = export(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
