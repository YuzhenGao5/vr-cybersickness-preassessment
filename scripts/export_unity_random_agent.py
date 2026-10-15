#!/usr/bin/env python3
"""Export the trained EyeNavGS LSTM and held-out seed histories for Unity.

The Unity demo uses the exact trained PyTorch weights.  It does not fall back to
random weights or a random walk.  Each run randomly chooses one held-out
participant/scene history and then executes the exported LSTM autoregressively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exploration_policy import build_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "exploration_model.pth"
DEFAULT_TEST_DATA = ROOT / "data" / "processed" / "exploration_test.npz"
DEFAULT_METRICS = ROOT / "results" / "exploration_training_metrics.json"
DEFAULT_ASSET = (
    ROOT
    / "demo"
    / "UnityRandomAgentDemo"
    / "Assets"
    / "Resources"
    / "EyeNavAgentModel.json"
)
DEFAULT_INTEGRITY = (
    ROOT
    / "demo"
    / "UnityRandomAgentDemo"
    / "Assets"
    / "Scripts"
    / "GeneratedModelIntegrity.cs"
)
DEFAULT_REPORT = ROOT / "results" / "unity_random_agent_export_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floats(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()


def choose_seed_windows(test: Any, per_trajectory: int) -> list[dict[str, Any]]:
    windows = np.asarray(test["X"], dtype=np.float32)
    trajectory_index = np.asarray(test["trajectory_index"], dtype=np.int64)
    trajectory_ids = [str(item) for item in test["trajectory_ids"]]
    seeds: list[dict[str, Any]] = []

    for index, trajectory_id in enumerate(trajectory_ids):
        candidates = np.flatnonzero(trajectory_index == index)
        if candidates.size == 0:
            continue
        count = min(per_trajectory, int(candidates.size))
        positions = np.linspace(0, candidates.size - 1, num=count, dtype=np.int64)
        for rank, position in enumerate(positions):
            window_index = int(candidates[int(position)])
            window = windows[window_index]
            if window.shape != (8, 6) or not np.isfinite(window).all():
                raise ValueError(f"Invalid held-out seed window at row {window_index}")
            seeds.append(
                {
                    "trajectoryId": trajectory_id,
                    "windowIndex": window_index,
                    "seedRank": rank,
                    "values": floats(window),
                }
            )

    if not seeds:
        raise ValueError("No valid held-out seed windows were found")
    return seeds


def predict_reference(checkpoint: dict[str, Any], window: np.ndarray) -> np.ndarray:
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    x_mean = np.asarray(checkpoint["normalization"]["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["normalization"]["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["normalization"]["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["normalization"]["y_std"], dtype=np.float32)
    normalized = (np.asarray(window, dtype=np.float32) - x_mean) / x_std
    with torch.no_grad():
        result = model(torch.from_numpy(normalized[None])).cpu().numpy()[0]
    result = result * y_std + y_mean
    return np.clip(
        result,
        np.asarray(checkpoint["action_clip_low"], dtype=np.float32),
        np.asarray(checkpoint["action_clip_high"], dtype=np.float32),
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "state_dict",
        "model_config",
        "normalization",
        "action_clip_low",
        "action_clip_high",
        "median_dt_seconds",
        "data_source",
        "synthetic",
        "dataset_fingerprint_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing required fields: {sorted(missing)}")
    if checkpoint["data_source"] != "EyeNavGS" or bool(checkpoint["synthetic"]):
        raise ValueError("Unity demo requires the real-data EyeNavGS checkpoint")

    source_hash = sha256_file(checkpoint_path)
    metadata_path = ROOT / "models" / "exploration_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_hash = str(metadata.get("model_sha256", ""))
    if declared_hash != source_hash:
        raise ValueError(
            f"Checkpoint hash mismatch: metadata={declared_hash!r}, actual={source_hash!r}"
        )

    with np.load(args.test_data, allow_pickle=False) as test:
        seeds = choose_seed_windows(test, args.seeds_per_trajectory)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    test_model = metrics["evaluations"]["test"]["model"]
    persistence = metrics["evaluations"]["test"]["persistence_baseline"]
    state = checkpoint["state_dict"]
    first_window = np.asarray(seeds[0]["values"], dtype=np.float32).reshape(8, 6)
    self_test_expected = predict_reference(checkpoint, first_window)

    payload = {
        "schemaVersion": "1.0.0",
        "dataSource": str(checkpoint["data_source"]),
        "syntheticTraining": bool(checkpoint["synthetic"]),
        "sourceModelSha256": source_hash,
        "datasetFingerprintSha256": str(checkpoint["dataset_fingerprint_sha256"]),
        "modelTask": "one-step 6-DoF head-motion behaviour cloning",
        "parameterCount": int(sum(value.numel() for value in state.values())),
        "featureNames": list(checkpoint.get("feature_names", [])),
        "lstmGateOrder": "ifgo",
        "tensorLayout": "row-major",
        "sourceCoordinateSystem": "OpenXR right-handed, -Z forward",
        "unityCoordinateAdapter": "linear=(x,y,-z); angular=(-wx,-wy,wz)",
        "testDataSha256": sha256_file(args.test_data),
        "historyLength": int(checkpoint["model_config"]["history_length"]),
        "inputDim": int(checkpoint["model_config"]["input_dim"]),
        "hiddenDim": int(checkpoint["model_config"]["hidden_dim"]),
        "outputDim": int(checkpoint["model_config"]["output_dim"]),
        "numLayers": int(checkpoint["model_config"]["num_layers"]),
        "medianDtSeconds": float(checkpoint["median_dt_seconds"]),
        "xMean": floats(checkpoint["normalization"]["x_mean"]),
        "xStd": floats(checkpoint["normalization"]["x_std"]),
        "yMean": floats(checkpoint["normalization"]["y_mean"]),
        "yStd": floats(checkpoint["normalization"]["y_std"]),
        "actionClipLow": floats(checkpoint["action_clip_low"]),
        "actionClipHigh": floats(checkpoint["action_clip_high"]),
        "weightIhL0": floats(state["lstm.weight_ih_l0"]),
        "weightHhL0": floats(state["lstm.weight_hh_l0"]),
        "biasIhL0": floats(state["lstm.bias_ih_l0"]),
        "biasHhL0": floats(state["lstm.bias_hh_l0"]),
        "weightIhL1": floats(state["lstm.weight_ih_l1"]),
        "weightHhL1": floats(state["lstm.weight_hh_l1"]),
        "biasIhL1": floats(state["lstm.bias_ih_l1"]),
        "biasHhL1": floats(state["lstm.bias_hh_l1"]),
        "fcWeight": floats(state["fc.weight"]),
        "fcBias": floats(state["fc.bias"]),
        "seedWindows": seeds,
        "selfTestInput": floats(first_window),
        "selfTestExpectedOutput": floats(self_test_expected),
        "testSampleCount": int(test_model["samples"]),
        "testTrajectoryCount": int(test_model["trajectory_count"]),
        "testLinearVelocityVectorMaeMps": float(
            test_model["linear_velocity_vector_mae_mps"]
        ),
        "testAngularVelocityVectorMaeDegps": float(
            test_model["angular_velocity_vector_mae_degps"]
        ),
        "persistenceLinearVelocityVectorMaeMps": float(
            persistence["linear_velocity_vector_mae_mps"]
        ),
        "persistenceAngularVelocityVectorMaeDegps": float(
            persistence["angular_velocity_vector_mae_degps"]
        ),
    }

    args.asset.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.asset.write_bytes(encoded)
    export_hash = sha256_file(args.asset)

    args.integrity.parent.mkdir(parents=True, exist_ok=True)
    args.integrity.write_text(
        "// Generated by scripts/export_unity_random_agent.py. Do not edit.\n"
        "namespace RandomAgentDemo\n"
        "{\n"
        "    public static class GeneratedModelIntegrity\n"
        "    {\n"
        f'        public const string ExportSha256 = "{export_hash}";\n'
        f'        public const string SourceModelSha256 = "{source_hash}";\n'
        f"        public const int SeedWindowCount = {len(seeds)};\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )

    trajectory_ids = {item["trajectoryId"] for item in seeds}
    participants = {":".join(item.split(":")[:2]) for item in trajectory_ids}
    scenes = {item.split(":")[-1] for item in trajectory_ids}
    report = {
        "status": "success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "source_model_sha256": source_hash,
        "source_training_is_synthetic": False,
        "unity_asset": str(args.asset.resolve()),
        "unity_asset_sha256": export_hash,
        "seed_windows": len(seeds),
        "held_out_trajectories": len(trajectory_ids),
        "held_out_participants": len(participants),
        "held_out_scenes": sorted(scenes),
        "runtime_self_test_expected_output": floats(self_test_expected),
        "test_metrics": {
            "samples": int(test_model["samples"]),
            "linear_velocity_vector_mae_mps": float(
                test_model["linear_velocity_vector_mae_mps"]
            ),
            "angular_velocity_vector_mae_degps": float(
                test_model["angular_velocity_vector_mae_degps"]
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds-per-trajectory", type=int, default=6)
    args = parser.parse_args()
    if args.seeds_per_trajectory < 1:
        parser.error("--seeds-per-trajectory must be at least 1")
    return args


def main() -> int:
    report = export(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
