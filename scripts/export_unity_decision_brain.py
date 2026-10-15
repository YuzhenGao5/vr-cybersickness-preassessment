#!/usr/bin/env python3
"""Re-export a trained Decision Brain checkpoint for Unity without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_unity_decision_brain import load_split, sha256_file
from unity_decision_policy import (
    DEFAULT_CONFIG,
    build_unity_flat_export,
    candidate_feature_names,
    global_feature_names,
    load_config,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "features" / "unity_decision_brain_v0"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "unity_decision_brain_v0.pth"
DEFAULT_PYTHON_EXPORT = REPO_ROOT / "models" / "unity_decision_brain_v0.json"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "unity_decision_brain_v0_unity.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_python_export_weights(
    checkpoint: dict[str, Any], python_export: dict[str, Any]
) -> None:
    if python_export.get("schema_version") != "unity-decision-brain-json-1.0":
        raise ValueError("Unsupported Python Decision Brain JSON schema")
    exported_weights = python_export.get("weights")
    if not isinstance(exported_weights, dict):
        raise ValueError("Python Decision Brain export has no weights dictionary")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(exported_weights):
        raise ValueError("Checkpoint/Python-export parameter names differ")
    for name, tensor in state.items():
        exported = exported_weights[name]
        if not isinstance(exported, dict) or "values" not in exported:
            raise ValueError(f"Python export parameter {name} is malformed")
        expected = tensor.detach().cpu().numpy().astype(np.float32)
        actual = np.asarray(exported["values"], dtype=np.float32)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"Checkpoint/Python-export parameter mismatch: {name}")


def export(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    manifest = load_json(arguments.data_dir / "dataset_manifest.json", "dataset manifest")
    if manifest.get("schema_version") != "unity-decision-prepared-data-1.0":
        raise ValueError("Unsupported prepared data manifest")
    if manifest.get("status") == "smoke_only" and not arguments.allow_smoke:
        raise ValueError("Prepared data is smoke_only; pass --allow-smoke explicitly")
    if manifest.get("synthetic_fixture") and not arguments.allow_synthetic_fixture:
        raise ValueError(
            "Synthetic fixture data is test-only; pass --allow-synthetic-fixture explicitly"
        )
    config_sha256 = sha256_file(arguments.config)
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("Prepared data was produced with a different config")
    if manifest.get("global_feature_names") != global_feature_names(config):
        raise ValueError("Prepared global feature order differs from config")
    if manifest.get("candidate_feature_names") != candidate_feature_names(config):
        raise ValueError("Prepared candidate feature order differs from config")

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "unity-decision-brain-checkpoint-1.0":
        raise ValueError("Unsupported Decision Brain checkpoint")
    python_export = load_json(arguments.python_export, "Python Decision Brain export")
    verify_python_export_weights(checkpoint, python_export)
    for name in (
        "data_fingerprint_sha256",
        "source_counts",
        "decision_kind_counts",
        "human_decision_samples",
    ):
        if checkpoint.get(name) != (
            manifest.get(name)
            if name != "human_decision_samples"
            else int(manifest.get(name, -1))
        ):
            raise ValueError(f"Checkpoint/prepared-data mismatch: {name}")
    provenance = python_export.get("training_provenance", {})
    if provenance.get("data_fingerprint_sha256") != manifest.get(
        "data_fingerprint_sha256"
    ):
        raise ValueError("Python export/prepared-data fingerprint mismatch")

    artifact_hashes = manifest.get("artifact_sha256", {})
    artifact_sizes = manifest.get("artifact_size_bytes", {})
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        path = arguments.data_dir / f"unity_decision_{split_name}.npz"
        if artifact_hashes.get(split_name) != sha256_file(path):
            raise ValueError(f"Prepared {split_name} split hash mismatch")
        if int(artifact_sizes.get(split_name, -1)) != path.stat().st_size:
            raise ValueError(f"Prepared {split_name} split size mismatch")
        splits[split_name] = load_split(path, config)

    test = splits["test"]
    if len(test["target_index"]) < 1:
        raise ValueError("Cannot make a golden vector from an empty test split")
    golden_index = 0
    payload = build_unity_flat_export(
        state_dict=checkpoint["state_dict"],
        architecture=checkpoint["architecture"],
        normalization=checkpoint["normalization"],
        config=config,
        preprocessing=checkpoint["preprocessing"],
        generated_at_utc=utc_now(),
        data_status=str(manifest["status"]),
        synthetic_fixture=bool(manifest.get("synthetic_fixture")),
        checkpoint_sha256=sha256_file(arguments.checkpoint),
        python_export_sha256=sha256_file(arguments.python_export),
        data_fingerprint_sha256=str(manifest["data_fingerprint_sha256"]),
        config_sha256=config_sha256,
        source_counts=checkpoint["source_counts"],
        decision_kind_counts=checkpoint["decision_kind_counts"],
        human_decision_samples=int(checkpoint["human_decision_samples"]),
        seed=int(checkpoint["seed"]),
        best_epoch=int(provenance["best_epoch"]),
        golden_global_features=test["global_features"][golden_index],
        golden_candidate_features=test["candidate_features"][golden_index],
        golden_candidate_mask=test["candidate_mask"][golden_index],
        golden_target_mask=test["target_mask"][golden_index],
        golden_skill_mask=test["skill_mask"][golden_index],
        golden_target_index=int(test["target_index"][golden_index]),
        golden_decision_kind=str(test["decision_kind"][golden_index]),
        golden_decision_id=str(test["decision_id"][golden_index]),
        golden_episode_id=str(test["episode_id"][golden_index]),
        golden_split_index=golden_index,
    )
    write_json_atomic(arguments.output, payload)
    return {
        "status": "success",
        "data_status": payload["dataStatus"],
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256_file(arguments.output),
        "checkpoint_sha256": payload["checkpointSha256"],
        "python_export_sha256": payload["pythonExportSha256"],
        "data_fingerprint_sha256": payload["dataFingerprintSha256"],
        "golden_decision_id": payload["goldenDecisionId"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python-export", type=Path, default=DEFAULT_PYTHON_EXPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        result = export(build_arg_parser().parse_args(argv))
    except Exception as exc:
        print(f"Unity Decision Brain export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
