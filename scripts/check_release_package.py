"""Fail closed when the public Git candidate set contains private or bulky artifacts.

The check examines tracked files plus untracked files that are not ignored.  It is
therefore useful both before staging a release and in CI after clone.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024
REQUIRED = {
    "README.md",
    "docs/PROJECT_REPORT_2026-10-15.md",
    "docs/AI_HANDOFF.md",
    "docs/DATASETS.md",
    "docs/REPRODUCIBILITY.md",
    "release/research_snapshot_2026-10-15.json",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE_STATUS.md",
    "LICENSES/Apache-2.0.txt",
    "demo/scene_agent_terminal.py",
    "demo/generic_scene_agent/GenericSceneAgentBrain.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Resources/NormalHmdAgentModel.json",
    "models/normal_hmd_v4_metadata.json",
    "models/unity_decision_brain_v0_unity.json",
    "results/unity_semantic_fixture/README.md",
    "results/unity_semantic_fixture/parse_result.json",
}

BANNED_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".dmp",
    ".exe",
    ".joblib",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pth",
    ".tar",
    ".zip",
}

BANNED_DIRECTORY_PARTS = {
    ".venv",
    ".venv-normal-hmd",
    "Library",
    "Temp",
    "Obj",
    "Logs",
    "UserSettings",
    "MemoryCaptures",
    "UnityGenericSceneAgentProjects",
    "UnityExternalSceneAgentProjects",
    "GenericSceneAgentBuilds",
    "ExternalSceneAgentBuilds",
    "SceneAgentTerminalBuild",
}

TOKEN_PATTERNS = {
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "GitHub token": re.compile(rb"gh[pousr]_[0-9A-Za-z]{36,}"),
    "OpenAI-style secret": re.compile(rb"sk-[0-9A-Za-z_-]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def git_candidate_paths() -> list[str]:
    command = [
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return sorted({p for p in completed.stdout.decode("utf-8").split("\0") if p})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    candidates = git_candidate_paths()
    candidate_set = set(candidates)
    total_bytes = 0

    missing = sorted(REQUIRED - candidate_set)
    errors.extend(f"required public file is missing from candidate set: {p}" for p in missing)

    for relative in candidates:
        path = ROOT / relative
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        posix = Path(relative).as_posix()
        parts = set(Path(relative).parts)

        if parts & BANNED_DIRECTORY_PARTS:
            errors.append(f"generated/private directory is publishable: {posix}")
        if path.suffix.lower() in BANNED_SUFFIXES:
            errors.append(f"banned binary/archive is publishable: {posix}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(
                f"file exceeds conservative 50 MiB release limit: {posix} "
                f"({path.stat().st_size} bytes)"
            )

        if posix.startswith("data/") and not (
            posix in {"data/.gitkeep", "data/cyre_status.md"}
            or posix.startswith("data/test_scenes/")
        ):
            errors.append(f"non-fixture data file is publishable: {posix}")
        if posix.startswith(("features/", "audit/", "tmp/", "figures/")) and not posix.endswith(
            "/.gitkeep"
        ):
            errors.append(f"research data/output directory is publishable: {posix}")
        if posix.startswith("results/") and not (
            posix == "results/.gitkeep"
            or posix.startswith("results/unity_semantic_fixture/")
        ):
            errors.append(f"local result is publishable instead of release summary: {posix}")
        if posix.startswith("models/") and posix not in {
            "models/.gitkeep",
            "models/normal_hmd_v4_metadata.json",
            "models/unity_decision_brain_v0_unity.json",
        }:
            errors.append(f"unreviewed model artifact is publishable: {posix}")

        if path.stat().st_size <= 2 * 1024 * 1024:
            content = path.read_bytes()
            for label, pattern in TOKEN_PATTERNS.items():
                if pattern.search(content):
                    errors.append(f"possible {label} in: {posix}")

    manifest_path = ROOT / "release/research_snapshot_2026-10-15.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            manifest["unity"]["canonicalBrain"]["path"]:
                manifest["unity"]["canonicalBrain"]["sha256"],
            manifest["models"]["normalHmdV4"]["runtimeExport"]["path"]:
                manifest["models"]["normalHmdV4"]["runtimeExport"]["sha256"],
            manifest["models"]["legacyEyeNavRandomAgent"]["runtimeExport"]["path"]:
                manifest["models"]["legacyEyeNavRandomAgent"]["runtimeExport"]["sha256"],
            manifest["models"]["unityDecisionBrainV0"]["runtimeExport"]["path"]:
                manifest["models"]["unityDecisionBrainV0"]["runtimeExport"]["sha256"],
        }
        for relative, expected_hash in expected.items():
            actual_hash = sha256(ROOT / relative)
            if actual_hash != expected_hash:
                errors.append(
                    f"snapshot hash mismatch for {relative}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot validate release manifest: {exc}")

    if errors:
        print("Release package check FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        "Release package check passed: "
        f"{len(candidates)} candidate files, {total_bytes} bytes, "
        "required artifacts and snapshot hashes verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
