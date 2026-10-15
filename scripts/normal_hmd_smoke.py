#!/usr/bin/env python3
"""Run a tiny real-data prepare -> train -> Unity-export Normal HMD v2 smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / "normal_hmd_v2_smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    data = output / "data"
    data_report = output / "data_report.json"
    model = output / "normal_hmd_smoke.pth"
    metadata = output / "metadata.json"
    metrics = output / "metrics.json"
    history = output / "history.csv"
    asset = output / "NormalHmdAgentModel.json"
    integrity = output / "GeneratedNormalHmdIntegrity.cs"
    export_report = output / "export_report.json"
    python = sys.executable

    run(
        [
            python,
            "scripts/normal_hmd_prepare.py",
            "--output-dir", str(data),
            "--report", str(data_report),
            "--history", "8",
            "--rollout", "8",
            "--stride", "4",
            "--max-segments-per-trajectory", "8",
            "--max-trajectories-per-split", "1",
        ]
    )
    run(
        [
            python,
            "scripts/normal_hmd_train.py",
            "--data-dir", str(data),
            "--data-report", str(data_report),
            "--model", str(model),
            "--metadata", str(metadata),
            "--metrics", str(metrics),
            "--history-output", str(history),
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "8",
            "--hidden-dim", "16",
            "--num-layers", "2",
            "--dropout", "0",
            "--probe-seconds", "2",
            "--innovation-calibration-segments", "8",
            "--device", args.device,
            "--threads", "2",
        ]
    )
    run(
        [
            python,
            "scripts/normal_hmd_export.py",
            "--checkpoint", str(model),
            "--metadata", str(metadata),
            "--test-data", str(data / "normal_hmd_test.npz"),
            "--metrics", str(metrics),
            "--asset", str(asset),
            "--integrity", str(integrity),
            "--report", str(export_report),
            "--seeds-per-trajectory", "2",
            "--innovation-golden-steps", "8",
            "--allow-smoke",
        ]
    )

    prepared = json.loads(data_report.read_text(encoding="utf-8"))
    trained = json.loads(metrics.read_text(encoding="utf-8"))
    exported = json.loads(export_report.read_text(encoding="utf-8"))
    checkpoint = torch.load(model, map_location="cpu", weights_only=True)
    payload = json.loads(asset.read_text(encoding="utf-8"))
    assert prepared["status"] == "success" and prepared["synthetic"] is False
    assert trained["status"] == "success" and trained["synthetic"] is False
    assert "innovation_model" in checkpoint
    assert trained["closed_loop_probe"]["same_seed_bitwise_reproducible"] is True
    assert payload["schemaVersion"] == "normal-hmd-agent-unity-1.0"
    assert payload["modelTask"] == "normal_hmd_9d_autoregressive"
    assert payload["architecture"] == "stateful_lstm"
    assert payload["fixedDtSeconds"] > 0
    assert len(payload["stateFeatureNames"]) == 9
    assert len(payload["innovationStd"]) == 9
    assert len(payload["innovationRho"]) == 9
    assert len(payload["fcWeight"]) == 9 * payload["hiddenDim"]
    assert len(payload["fcBias"]) == 9
    assert len(payload["seedWindows"]) >= 2
    assert len(payload["seedWindows"][0]["historyRaw"]) == payload["historyLength"] * 9
    assert len(payload["innovationGoldenExpected"]) == payload["innovationGoldenStepCount"] * 9
    assert exported["status"] == "success"
    print("Normal HMD v2 smoke test PASSED")
    print(f"  artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
