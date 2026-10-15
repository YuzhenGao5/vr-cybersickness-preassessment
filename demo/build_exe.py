#!/usr/bin/env python3
"""Build and smoke-test the standalone Windows motion-imitation demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "demo" / "agent_demo.py"
MODEL = REPO_ROOT / "models" / "exploration_model.pth"
MODEL_METADATA = REPO_ROOT / "models" / "exploration_model_metadata.json"
SCENE_REPORT = REPO_ROOT / "results" / "unity_parse_result.json"
DIST_DIR = REPO_ROOT / "demo"
BUILD_DIR = REPO_ROOT / "demo" / "build"
SPEC_DIR = REPO_ROOT / "demo"
EXE_PATH = DIST_DIR / "AgentExplorationDemo.exe"
STATUS_PATH = REPO_ROOT / "results" / "exe_build_status.json"
LOG_PATH = REPO_ROOT / "results" / "exe_build.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pyinstaller_version() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_command() -> list[str]:
    # PyInstaller accepts os.pathsep here; on Windows this is a semicolon.
    separator = ";" if sys.platform == "win32" else ":"
    add_data = [
        f"{MODEL}{separator}models",
        f"{MODEL_METADATA}{separator}models",
        f"{SCENE_REPORT}{separator}results",
    ]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "AgentExplorationDemo",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(SPEC_DIR),
        "--paths",
        str(REPO_ROOT / "scripts"),
        "--hidden-import",
        "matplotlib.backends.backend_tkagg",
        "--hidden-import",
        "PIL._tkinter_finder",
    ]
    # The upstream PyInstaller torch hook enumerates every torch submodule. In
    # a research environment this can otherwise pull in unrelated optional
    # stacks merely because they are installed (Transformers, OpenCV, pandas,
    # scikit-learn, TensorFlow, etc.).  None is imported by this LSTM inference
    # app; excluding them reduces build time and avoids shipping unrelated code.
    optional_stacks = (
        "cv2",
        "fsspec",
        "jax",
        "pandas",
        "scipy",
        "sklearn",
        "tensorflow",
        "tensorboard",
        "transformers",
        "torch.onnx",
        "torch.utils.tensorboard",
        "torchaudio",
        "torchtext",
        "torchvision",
    )
    for module in optional_stacks:
        command.extend(["--exclude-module", module])
    for resource in add_data:
        command.extend(["--add-data", resource])
    command.append(str(DEMO_SCRIPT))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Build only. By default the EXE is launched from an unrelated temporary directory.",
    )
    parser.add_argument("--smoke-timeout", type=int, default=240)
    args = parser.parse_args(argv)

    resources = [DEMO_SCRIPT, MODEL, MODEL_METADATA, SCENE_REPORT]
    missing = [str(path) for path in resources if not path.is_file()]
    started = time.perf_counter()
    base_status: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "target": str(EXE_PATH),
        "resources": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in resources
            if path.is_file()
        ],
        "resource_policy": "Only the model, its metadata, and the validated Unity parse JSON are bundled as data.",
    }
    if missing:
        base_status.update(
            {
                "status": "failed",
                "reason": f"Required build inputs are missing: {missing}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(STATUS_PATH, base_status)
        print(base_status["reason"], file=sys.stderr)
        return 1

    try:
        version = pyinstaller_version()
    except (OSError, subprocess.CalledProcessError) as exc:
        base_status.update(
            {
                "status": "failed",
                "reason": f"PyInstaller is not available in {sys.executable}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(STATUS_PATH, base_status)
        print(base_status["reason"], file=sys.stderr)
        return 1

    command = build_command()
    base_status["pyinstaller_version"] = version
    base_status["command"] = command
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_PATH.open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0 or not EXE_PATH.is_file():
            raise RuntimeError(
                f"PyInstaller returned {completed.returncode}; inspect {LOG_PATH}"
            )

        smoke_test: dict[str, Any]
        if args.skip_smoke_test:
            smoke_test = {"status": "skipped", "reason": "--skip-smoke-test"}
        else:
            with tempfile.TemporaryDirectory(prefix="agent-demo-smoke-") as temporary:
                smoke_started = time.perf_counter()
                diagnostic_path = Path(temporary) / "self_test_error.log"
                smoke_environment = os.environ.copy()
                smoke_environment["AGENT_DEMO_ERROR_LOG"] = str(diagnostic_path)
                smoke = subprocess.run(
                    [str(EXE_PATH), "--self-test"],
                    cwd=temporary,
                    check=False,
                    timeout=args.smoke_timeout,
                    env=smoke_environment,
                )
                smoke_test = {
                    "status": "passed" if smoke.returncode == 0 else "failed",
                    "return_code": smoke.returncode,
                    "cwd_was_outside_repository": True,
                    "elapsed_seconds": time.perf_counter() - smoke_started,
                }
                if diagnostic_path.is_file():
                    smoke_test["diagnostic"] = diagnostic_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:8000]
                base_status["smoke_test"] = smoke_test
                if smoke.returncode != 0:
                    raise RuntimeError(
                        f"Packaged EXE self-test returned {smoke.returncode}: "
                        f"{smoke_test.get('diagnostic', 'no diagnostic was written')}"
                    )

        base_status.update(
            {
                "status": "success",
                "exe_bytes": EXE_PATH.stat().st_size,
                "exe_sha256": sha256_file(EXE_PATH),
                "smoke_test": smoke_test,
                "log": str(LOG_PATH.relative_to(REPO_ROOT)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(STATUS_PATH, base_status)
        print(f"Built: {EXE_PATH}")
        print(f"Size : {base_status['exe_bytes']} bytes")
        print(f"SHA256: {base_status['exe_sha256']}")
        print(f"Smoke test: {smoke_test['status']}")
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        base_status.update(
            {
                "status": "failed",
                "reason": str(exc),
                "log": str(LOG_PATH.relative_to(REPO_ROOT)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if EXE_PATH.is_file():
            base_status.update(
                {
                    "exe_bytes": EXE_PATH.stat().st_size,
                    "exe_sha256": sha256_file(EXE_PATH),
                }
            )
        atomic_json(STATUS_PATH, base_status)
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
