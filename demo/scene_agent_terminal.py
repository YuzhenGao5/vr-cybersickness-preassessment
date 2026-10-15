#!/usr/bin/env python3
"""Windows control terminal for scene-aware Normal HMD Agent demonstrations.

The terminal deliberately distinguishes three states:
  * agent_ready: a scene has a reviewed runtime adapter and can be executed;
  * analysis_only: static Unity evidence can be parsed, but interactions are not guessed;
  * blocked: automating the scene would create unsafe or misleading side effects.

It never edits a selected Unity project.  Adapted players are built separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


APP_TITLE = "Unity Scene Agent Terminal"
SKIP_DIRECTORIES = {
    ".git", ".vs", "Library", "Logs", "Temp", "obj", "Build", "build",
    "Builds", "SceneAgentTerminalBuild", "UnitySceneAwareAgentDemoBuild",
    "UnityNormalHmdAgentDemoBuild", "UnityRandomAgentDemoBuild",
    "UnityExternalSceneAgentProjects", "ExternalSceneAgentBuilds",
    "UnityGenericSceneAgentProjects", "GenericSceneAgentBuilds", "__pycache__",
}

GENERIC_UNITY_VERSION = "6000.2.9f1"
GENERIC_RUNTIME_SOURCE_FILES = (
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/GeneratedNormalHmdIntegrity.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdAgentController.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdDemoBootstrap.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdLstmRuntime.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdModelData.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdVerticalEventController.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/SceneAwareContracts.cs",
    "demo/UnityNormalHmdAgentDemo/Assets/Scripts/SeededRootInnovation.cs",
    "demo/generic_scene_agent/GenericSceneContract.cs",
    "demo/generic_scene_agent/UnityDecisionBrainRuntime.cs",
    "demo/generic_scene_agent/GenericSceneAgentBrain.cs",
    "demo/generic_scene_agent/GenericSceneAgentBootstrap.cs",
    "demo/generic_scene_agent/GenericRenderProbe.cs",
    "demo/generic_scene_agent/GenericSceneAgentBuilder.cs",
)
GENERIC_BUILD_PIPELINE_FILES = (
    "demo/build_generic_scene_agent.ps1",
    "scripts/materialize_generic_runtime_contract.py",
    "scripts/validate_render_capture.py",
)
DECISION_BRAIN_MODEL_RELATIVE = Path("models/unity_decision_brain_v0_unity.json")
DECISION_BRAIN_MODEL_SCHEMA = "unity-decision-brain-unity-flat-1.0"
DECISION_BRAIN_MODES = ("rule", "shadow", "model")
DECISION_BRAIN_MODE_LABELS = {
    "rule": "Rule（稳定规则，总可用）",
    "shadow": "Shadow（模型旁路，不控制）",
    "model": "Model research（模型控制，研究模式）",
}
WINDOWS_NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PARSER_TIMEOUT_SECONDS = 300
BACKGROUND_HEARTBEAT_SECONDS = 3.0
BACKGROUND_CONFLICTING_ACTIONS = ("analyze", "build", "launch", "test", "export")
BACKGROUND_NON_CONFLICTING_CONTROLS = ("project", "scene", "browse", "open")


@dataclass
class SceneChoice:
    id: str
    label: str
    scene_path: str
    support: str = "analysis_only"
    support_reason: str = "尚未提供经过验证的运行时交互适配器。"
    launcher: str = ""
    launcher_args: list[str] = field(default_factory=list)
    scenario_index: int | None = None
    runtime_kind: str = ""
    build_status: str = ""


@dataclass
class UnityProject:
    name: str
    path: str
    unity_version: str
    scenes: list[SceneChoice]


@dataclass(frozen=True)
class DecisionBrainModelInfo:
    path: Path
    status: str
    detail: str
    sha256: str = ""
    schema_version: str = ""
    data_status: str = ""


def inspect_decision_brain_model(
    repository: Path, explicit_path: Path | None = None
) -> DecisionBrainModelInfo:
    """Inspect only the frozen Unity-flat export; never accept the Python JSON by accident."""
    path = (explicit_path or repository / DECISION_BRAIN_MODEL_RELATIVE).resolve()
    if not path.is_file():
        return DecisionBrainModelInfo(path, "missing", f"Unity 扁平模型不存在：{path}")
    payload = load_json(path)
    if payload is None:
        return DecisionBrainModelInfo(path, "invalid", "模型不是可读取的 JSON 对象。")
    schema = str(payload.get("schemaVersion", ""))
    data_status = str(payload.get("dataStatus", ""))
    if schema != DECISION_BRAIN_MODEL_SCHEMA:
        return DecisionBrainModelInfo(
            path, "invalid",
            f"模型 schema={schema or 'missing'}；需要 {DECISION_BRAIN_MODEL_SCHEMA}。",
            schema_version=schema, data_status=data_status,
        )
    if data_status not in {"smoke_only", "trained", "production_candidate"}:
        return DecisionBrainModelInfo(
            path, "invalid",
            f"模型 dataStatus={data_status or 'missing'} 不受支持。",
            schema_version=schema, data_status=data_status,
        )
    try:
        digest = sha256_file(path)
    except OSError as exception:
        return DecisionBrainModelInfo(path, "invalid", f"模型哈希读取失败：{exception}")
    return DecisionBrainModelInfo(
        path, "ready", f"模型可用；dataStatus={data_status or 'unknown'}。",
        sha256=digest, schema_version=schema, data_status=data_status,
    )


def decision_brain_mode_availability(
    model: DecisionBrainModelInfo, mode: str, research_override: bool
) -> tuple[bool, str]:
    if mode not in DECISION_BRAIN_MODES:
        return False, f"未知 Decision Brain 模式：{mode}"
    if mode == "rule":
        return True, "Rule 使用确定性高层 Brain；模型缺失或无效也不影响使用。"
    if model.status != "ready":
        return False, f"{DECISION_BRAIN_MODE_LABELS[mode]}不可用：{model.detail}"
    if mode == "shadow":
        return True, "Shadow 会运行模型并记录分歧，但执行权仍属于 Rule。"
    if model.data_status != "production_candidate" and not research_override:
        return False, (
            f"当前模型 dataStatus={model.data_status}，不是 production_candidate；"
            "Model research 必须显式勾选研究覆盖。"
        )
    return True, "Model research 将模型置于闭环决策权；仅供研究验证。"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes} 分 {remaining_seconds:02d} 秒"
    return f"{remaining_seconds} 秒"


def parse_utc_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def same_path(left: str | Path, right: str | Path) -> bool:
    if not str(left).strip() or not str(right).strip():
        return False
    try:
        return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))
    except (OSError, RuntimeError):
        return False


def is_within(path: Path, root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def slug(value: str, max_length: int = 32) -> str:
    safe = "".join(character.lower() if character.isalnum() else "-" for character in value)
    normalized = "-".join(part for part in safe.split("-") if part) or "scene"
    if len(normalized) <= max_length:
        return normalized
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:max_length - len(suffix) - 1].rstrip('-')}-{suffix}"


def locate_repository() -> Path:
    starts = [Path(sys.executable).resolve().parent] if getattr(sys, "frozen", False) else []
    starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "scripts" / "unity_parser.py").is_file() and (candidate / "demo").is_dir():
                return candidate
    configured = os.environ.get("SCENE_AGENT_REPOSITORY")
    if configured and (Path(configured) / "scripts" / "unity_parser.py").is_file():
        return Path(configured).resolve()
    raise RuntimeError("找不到 vr-cybersickness-preassessment 仓库。请从交付目录运行终端。")


def read_unity_version(project: Path) -> str:
    version_file = project / "ProjectSettings" / "ProjectVersion.txt"
    try:
        for line in version_file.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("m_EditorVersion:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def discover_unity_projects(search_roots: list[Path]) -> list[UnityProject]:
    found: dict[str, UnityProject] = {}
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for root_text, directories, files in os.walk(search_root):
            skipped = {name.casefold() for name in SKIP_DIRECTORIES}
            directories[:] = [name for name in directories if name.casefold() not in skipped]
            root = Path(root_text)
            if root.name == "ProjectSettings" and "ProjectVersion.txt" in files:
                project = root.parent.resolve()
                assets = project / "Assets"
                scenes = []
                if assets.is_dir():
                    for scene_path in sorted(assets.rglob("*.unity")):
                        relative = scene_path.relative_to(project).as_posix()
                        scenes.append(SceneChoice(
                            id=relative,
                            label=scene_path.stem,
                            scene_path=str(scene_path),
                        ))
                if not scenes:
                    scenes.append(SceneChoice(
                        id="no-scene",
                        label="（没有找到 .unity Scene）",
                        scene_path="",
                        support="blocked",
                        support_reason="项目中没有可选择的 .unity Scene。",
                    ))
                found[str(project).casefold()] = UnityProject(
                    name=project.name,
                    path=str(project),
                    unity_version=read_unity_version(project),
                    scenes=scenes,
                )
                directories[:] = []
    return sorted(found.values(), key=lambda item: (item.name.casefold(), item.path.casefold()))


def load_json(path: Path) -> dict[str, object] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_file_hashes(repository: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    """Hash the exact local sources that authorize a reusable Player build."""
    return {
        relative_path: sha256_file(repository / Path(relative_path))
        for relative_path in relative_paths
    }


def reviewed_binding_matches_contract(
    binding: dict[str, object], contract: dict[str, object]
) -> bool:
    """Require static evidence for a reviewed method or the narrow UI Button adapter.

    ``UnityEngine.UI.Button.Press`` is implemented by the runtime adapter rather
    than by a source-declared method. It is accepted only for the exact reviewed
    target when deterministic evidence contains both a Button component and an
    affordance. The caller separately verifies the central manifest and its
    pinned scene/script hashes.
    """
    target_id = binding.get("target_node_id")
    script_relative = str(binding.get("script_path", "")).replace("\\", "/")
    method_candidates = contract.get("method_candidates", [])
    if isinstance(method_candidates, list) and any(
        isinstance(item, dict)
        and item.get("target_node_id") == target_id
        and str(item.get("script_path", "")).replace("\\", "/") == script_relative
        and item.get("declaring_type") == binding.get("component_type")
        and item.get("method") == binding.get("method")
        and item.get("access") == "public"
        and item.get("parameter_count") == 0
        and not item.get("is_static")
        for item in method_candidates
    ):
        return True

    component_type = str(binding.get("component_type", ""))
    if component_type not in {"Button", "UnityEngine.UI.Button"} or binding.get("method") != "Press":
        return False
    nodes = contract.get("semantic_nodes", [])
    if not isinstance(nodes, list):
        return False
    node = next((
        item for item in nodes
        if isinstance(item, dict) and item.get("id") == target_id
    ), None)
    if node is None:
        return False
    evidence_text = " ".join(
        str(item.get("observation") or "")
        for item in node.get("evidence", [])
        if isinstance(item, dict)
    ).casefold()
    affordances = contract.get("affordances", [])
    has_target_affordance = isinstance(affordances, list) and any(
        isinstance(item, dict) and item.get("target_node_id") == target_id
        for item in affordances
    )
    return "button" in evidence_text and has_target_affordance


def validated_build_time(build_status_path: Path, executable: Path) -> float | None:
    report = load_json(build_status_path)
    if report is None or report.get("status") != "success" or not executable.is_file():
        return None
    reported_executable = report.get("executable")
    if not isinstance(reported_executable, str) or not same_path(reported_executable, executable):
        return None
    finished = parse_utc_timestamp(report.get("finishedAtUtc"))
    try:
        file_time = build_status_path.stat().st_mtime
        artifact_times = [executable.stat().st_mtime]
        data_root = executable.parent / f"{executable.stem}_Data"
        for artifact in (
            executable.parent / "UnityPlayer.dll",
            data_root / "globalgamemanagers",
            data_root / "Managed" / "Assembly-CSharp.dll",
        ):
            if artifact.is_file():
                artifact_times.append(artifact.stat().st_mtime)
    except OSError:
        return None
    return max(file_time, finished or 0.0, *artifact_times)


def acceptance_is_fresh(acceptance_path: Path, acceptance: dict[str, object], build_time: float) -> bool:
    generated = parse_utc_timestamp(acceptance.get("generatedAtUtc"))
    try:
        file_time = acceptance_path.stat().st_mtime
    except OSError:
        return False
    # Some external reports do not serialize generatedAtUtc, so their file mtime is
    # the audit clock. When a content timestamp exists, require both clocks so merely
    # touching an old report cannot make it authorize a newer build. Five seconds
    # tolerates filesystem timestamp granularity.
    if file_time + 5.0 < build_time:
        return False
    return generated is None or generated + 5.0 >= build_time


def completed_acceptance_run(
    report: object,
    *,
    expected_interactions: int | None = None,
    expected_objectives: int | None = None,
) -> bool:
    if not isinstance(report, dict) or report.get("status") != "passed":
        return False
    try:
        if float(report.get("generatedSeconds", 0)) < 59.9:
            return False
        if int(report.get("telemetryRowCount", 0)) < 2000:
            return False
        interactions_expected = int(report.get("interactionsExpected", 0))
        interactions_succeeded = int(report.get("interactionsSucceeded", 0))
        objectives_expected = int(report.get("objectivesExpected", 0))
        objectives_completed = int(report.get("objectivesCompleted", 0))
    except (TypeError, ValueError):
        return False
    if interactions_succeeded < interactions_expected or objectives_completed < objectives_expected:
        return False
    if expected_interactions is not None and interactions_expected != expected_interactions:
        return False
    if expected_objectives is not None and objectives_expected != expected_objectives:
        return False
    failures = report.get("failures", [])
    if failures not in (None, []) and bool(failures):
        return False
    return report.get("completionReason") == "60_seconds_complete"


def completed_generic_acceptance_run(
    report: object, requested_decision_mode: str | None = None
) -> bool:
    """Validate a complete, collision-safe generic run with interaction-lock evidence."""
    if not isinstance(report, dict) or report.get("status") != "passed":
        return False
    try:
        simulated_seconds = float(report.get("simulatedSeconds", 0))
        telemetry_rows = int(report.get("telemetryRows", 0))
        targets_declared = int(report.get("targetsDeclared", 0))
        targets_resolved = int(report.get("targetsResolved", 0))
        targets_visited = int(report.get("targetsVisited", 0))
        unfinished_targets = int(report.get("unfinishedTargetCount", -1))
        interactions_declared = int(report.get("interactionsDeclared", 0))
        interactions_attempted = int(report.get("interactionsAttempted", 0))
        interactions_required = int(report.get("interactionsRequired", interactions_declared))
        interactions_required_verified = int(
            report.get("interactionsRequiredVerified", report.get("interactionsVerified", 0))
        )
        minimum_evidence_seconds = float(report.get("minimumEvidenceSeconds", 0))
        lock_acquisitions = int(report.get("interactionMotionLockAcquisitions", -1))
        lock_releases = int(report.get("interactionMotionLockReleases", -1))
        locked_rows = int(report.get("interactionMotionLockedTelemetryRows", -1))
        lock_violations = int(report.get("interactionMotionLockViolations", -1))
        head_motion_rows = int(report.get("interactionHeadMotionTelemetryRows", -1))
        collision_overlaps = int(report.get("collisionOverlapDetections", -1))
    except (TypeError, ValueError):
        return False
    completion_reason = report.get("completionReason")
    if completion_reason != "content_complete":
        return False
    if (
        report.get("contentCompletionSatisfied") is not True
        or minimum_evidence_seconds < 1.0
        or simulated_seconds + 0.1 < minimum_evidence_seconds
        or telemetry_rows < 1
        or targets_visited != targets_declared
        or unfinished_targets != 0
        # A runtime walkable-sample/positionHint target has no backing
        # GameObject by design, so it may be visited without contributing to
        # targetsResolved.  Resolved is diagnostic; coverage is authoritative.
        or targets_resolved < 0
        or targets_resolved > targets_declared
    ):
        return False
    if targets_declared < 1:
        return False
    if interactions_required_verified < interactions_required:
        return False
    if (
        report.get("interactionMotionLockPassed") is not True
        or lock_violations != 0
        or lock_acquisitions != lock_releases
        or interactions_attempted > lock_acquisitions
        or report.get("collisionSafetyMode")
        != "full-height-capsule-sweep+stable-side-steering+inertia-hard-stop"
        or report.get("collisionSafetyPassed") is not True
        or collision_overlaps != 0
    ):
        return False
    if interactions_attempted > 0 and (locked_rows < 1 or head_motion_rows < 1):
        return False
    failures = report.get("failures", [])
    if failures not in (None, []) and bool(failures):
        return False
    if requested_decision_mode is not None:
        expected = {
            "rule": ("not_requested", "rule"),
            "shadow": ("ready", "rule_shadow"),
            "model": ("ready", "model"),
        }.get(requested_decision_mode)
        if expected is None:
            return False
        if (
            report.get("decisionBrainModeRequested") != requested_decision_mode
            or report.get("decisionBrainModelStatus") != expected[0]
            or report.get("policyAuthority") != expected[1]
        ):
            return False
        try:
            inference_count = int(report.get("decisionBrainInferences", 0))
            fallback_count = int(report.get("decisionBrainFallbacks", 0))
        except (TypeError, ValueError):
            return False
        if requested_decision_mode == "rule" and inference_count != 0:
            return False
        if requested_decision_mode in {"shadow", "model"}:
            if inference_count < 1 or fallback_count != 0:
                return False
            if (
                report.get("decisionBrainGoldenSelfTestPassed") is not True
                or report.get("decisionBrainAllInferenceLogitsFinite") is not True
                or report.get("decisionBrainModelSchemaVersion") != DECISION_BRAIN_MODEL_SCHEMA
                or report.get("decisionBrainDataStatus")
                not in {"smoke_only", "trained", "production_candidate"}
            ):
                return False
    return True


def apply_known_adapters(repository: Path, projects: list[UnityProject]) -> None:
    scene_aware_exe = repository / "demo" / "UnitySceneAwareAgentDemoBuild" / "SceneAwareAgentDemo.exe"
    heartisland_exe = repository / "demo" / "ExternalSceneAgentBuilds" / "HeartislandAgent" / "HeartislandAgent.exe"
    scene_build_status = repository / "results" / "scene_aware_agent_build_status.json"
    scene_acceptance_path = repository / "results" / "scene_aware_agent_acceptance.json"
    heart_build_status = repository / "results" / "heartisland_agent_build_status.json"
    heart_acceptance_path = repository / "results" / "heartisland_agent_acceptance.json"
    accepted_scenarios: set[int] = set()
    acceptance = load_json(scene_acceptance_path)
    scene_build_time = validated_build_time(scene_build_status, scene_aware_exe)
    if acceptance is not None and scene_build_time is not None:
        reported_executable = acceptance.get("executable")
        reported_build_status = acceptance.get("buildStatusPath")
        if (
            acceptance.get("status") == "passed"
            and acceptance.get("allScenariosPassed") is True
            and isinstance(reported_executable, str)
            and same_path(reported_executable, scene_aware_exe)
            and isinstance(reported_build_status, str)
            and same_path(reported_build_status, scene_build_status)
            and acceptance_is_fresh(scene_acceptance_path, acceptance, scene_build_time)
        ):
            scenario_reports = acceptance.get("scenarios", [])
            expected_counts = {0: (1, 2), 1: (1, 2), 2: (1, 2), 3: (2, 3)}
            if isinstance(scenario_reports, list):
                for item in scenario_reports:
                    if not isinstance(item, dict):
                        continue
                    try:
                        scenario_index = int(item["scenarioIndex"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    counts = expected_counts.get(scenario_index)
                    if counts is not None and completed_acceptance_run(
                        item, expected_interactions=counts[0], expected_objectives=counts[1]
                    ):
                        accepted_scenarios.add(scenario_index)
    heartisland_accepted = False
    external_acceptance = load_json(heart_acceptance_path)
    heart_build_time = validated_build_time(heart_build_status, heartisland_exe)
    if external_acceptance is not None and heart_build_time is not None:
        heartisland_accepted = (
            external_acceptance.get("project") == "HeartislandUnity"
            and external_acceptance.get("sceneName") == "ResonanceValley"
            and acceptance_is_fresh(heart_acceptance_path, external_acceptance, heart_build_time)
            and completed_acceptance_run(
                external_acceptance, expected_interactions=3, expected_objectives=4
            )
        )
    built_in = [
        (0, "Calm Door Baseline", "平静环境：寻路、按钮、开门、观察。"),
        (1, "Multi-turn Corridor", "多次 90° 转向、出口开关和状态验证。"),
        (2, "Dynamic Visual Motion Lab", "启动动态视觉装置并观察场景变化。"),
        (3, "Roller Coaster Experience", "上车、启动、随轨道运动并记录载具与头显运动。"),
    ]
    for project in projects:
        project_path = Path(project.path)
        if project_path == (repository / "demo" / "UnityNormalHmdAgentDemo").resolve():
            # Keep the real .unity files visible.  Older versions replaced the
            # raw scene list with four logical reference scenarios, which made
            # it look as though other scenes in the project did not exist.
            # The references remain available as separately labelled, already
            # reviewed demonstrations.
            project.scenes.extend([
                SceneChoice(
                    id=f"scene-aware-{index}",
                    label=f"【已验收参考】{label}",
                    scene_path=str(project_path / "Assets" / "Scenes" / "SceneAwareAgentDemo.unity"),
                    support="agent_ready" if scene_aware_exe.is_file() and index in accepted_scenarios else "blocked",
                    support_reason=(
                        description
                        if scene_aware_exe.is_file() and index in accepted_scenarios
                        else "该场景尚未通过独立 60 秒 Player 验收。"
                    ),
                    launcher=str(scene_aware_exe),
                    launcher_args=[f"--scenario={index}"],
                    scenario_index=index,
                    runtime_kind="legacy_scene_aware",
                    build_status=str(scene_build_status),
                )
                for index, label, description in built_in
            ])
        elif same_path(project_path, repository.parent.parent / "Game" / "HeartislandUnity"):
            for scene in project.scenes:
                is_adapted_scene = Path(scene.scene_path).name.casefold() == "resonancevalley.unity"
                scene.support = (
                    "agent_ready"
                    if is_adapted_scene and heartisland_exe.is_file() and heartisland_accepted
                    else "analysis_only" if not is_adapted_scene else "blocked"
                )
                scene.support_reason = (
                    "已适配运行时代码生成的山谷：移动、修复花圃、接近精灵并执行一次共鸣互动。"
                    if is_adapted_scene and heartisland_exe.is_file() and heartisland_accepted
                    else "该 Heartisland Scene 没有经过运行时交互适配；只提供静态解析。"
                    if not is_adapted_scene
                    else "Heartisland Agent 适配 Player 尚未构建或未通过 60 秒验收。"
                )
                if is_adapted_scene:
                    scene.launcher = str(heartisland_exe)
                    scene.launcher_args = ["--agent-experience"]
                    scene.runtime_kind = "legacy_external"
                    scene.build_status = str(heart_build_status)
        elif project.name.casefold() == "teach2selfprototype":
            for scene in project.scenes:
                scene.support = "analysis_only"
                scene.support_reason = (
                    "这是参与者实验/记录流程。终端可只读解析，但不会伪造参与者身份、回答或研究数据；"
                    "需要单独定义测试身份和无污染 sandbox adapter。"
                )


def default_search_roots(repository: Path) -> list[Path]:
    roots = [repository.parent, Path.home() / "Documents" / "ChatGPT"]
    saved_path = repository / "results" / "scene_agent_terminal" / "search_roots.json"
    saved = load_json(saved_path)
    if saved is not None and isinstance(saved.get("roots"), list):
        for value in saved["roots"]:
            if isinstance(value, str) and value.strip():
                roots.append(Path(value))
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


class TerminalBackend:
    def __init__(self, repository: Path):
        self.repository = repository
        self.results_root = repository / "results" / "scene_agent_terminal"
        self.runs_root = self.results_root / "runs"
        self.semantic_root = self.results_root / "semantic"
        self.exports_root = self.results_root / "exports"
        for directory in (self.runs_root, self.semantic_root, self.exports_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.projects: list[UnityProject] = []
        self.last_run_by_scene: dict[str, Path] = {}
        self.last_analysis_by_scene: dict[str, Path] = {}
        self.active_processes: dict[Path, subprocess.Popen[bytes]] = {}

    @staticmethod
    def scene_key(project: UnityProject, scene: SceneChoice) -> str:
        return f"{os.path.normcase(str(Path(project.path).resolve()))}|{scene.id}"

    def refresh(self, extra_root: Path | None = None) -> list[UnityProject]:
        roots = default_search_roots(self.repository)
        if extra_root is not None:
            resolved_extra = extra_root.resolve()
            roots.insert(0, resolved_extra)
            self.remember_search_root(resolved_extra)
        self.projects = discover_unity_projects(roots)
        apply_known_adapters(self.repository, self.projects)
        self.apply_generic_builds(self.projects)
        return self.projects

    def remember_search_root(self, root: Path) -> None:
        if not root.is_dir():
            return
        path = self.results_root / "search_roots.json"
        current = load_json(path) or {}
        values = current.get("roots", [])
        roots = [Path(value).resolve() for value in values if isinstance(value, str) and value.strip()]
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
        path.write_text(json.dumps({
            "schemaVersion": "scene-agent-search-roots-1.0",
            "updatedAtUtc": utc_now(),
            "roots": [str(item) for item in roots],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def latest_analysis_record(
        self, project: UnityProject, scene: SceneChoice
    ) -> tuple[Path, dict[str, object]] | None:
        key = self.scene_key(project, scene)
        remembered = self.last_analysis_by_scene.get(key)
        candidates = [remembered] if remembered is not None else []
        candidates.extend(sorted(
            self.semantic_root.glob("*_index.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ))
        seen: set[Path] = set()
        for index_path in candidates:
            if index_path is None or index_path in seen or not index_path.is_file():
                continue
            seen.add(index_path)
            index = load_json(index_path)
            if index is None:
                continue
            indexed_project = index.get("project")
            indexed_scene = index.get("selectedScene")
            if not isinstance(indexed_project, dict) or not isinstance(indexed_scene, dict):
                continue
            if (
                same_path(str(indexed_project.get("path", "")), project.path)
                and indexed_scene.get("id") == scene.id
            ):
                return index_path, index
        return None

    def trusted_reviewed_bindings(
        self, project: UnityProject, scene: SceneChoice, contract_path: Path
    ) -> dict[str, object] | None:
        """Resolve an explicitly approved central binding manifest.

        Project files and LLM output are intentionally not trust roots. A
        manifest must live in the repository's central review directory and
        pin the exact source scene content.
        """
        review_root = self.repository / "configs" / "scene_agent_reviewed_bindings"
        if not review_root.is_dir():
            return None
        matches: list[dict[str, object]] = []
        scene_hash = sha256_file(Path(scene.scene_path))
        contract = load_json(contract_path)
        if contract is None:
            raise RuntimeError("无法验证 reviewed bindings：Agent 契约不可读。")
        node_ids = {
            str(item.get("id")) for item in contract.get("semantic_nodes", [])
            if isinstance(item, dict) and item.get("id")
        }
        for candidate in sorted(review_root.glob("*.json")):
            if candidate.is_symlink() or not is_within(candidate, review_root):
                continue
            payload = load_json(candidate)
            if payload is None:
                continue
            if not same_path(str(payload.get("projectPath", "")), project.path):
                continue
            if str(payload.get("scenePath", "")).replace("\\", "/") != scene.id.replace("\\", "/"):
                continue
            if payload.get("schemaVersion") != "scene-agent-reviewed-bindings-1.0":
                raise RuntimeError(f"reviewed bindings schema 不受支持：{candidate}")
            if payload.get("approvalStatus") != "approved":
                continue
            if str(payload.get("sourceSceneSha256", "")).casefold() != scene_hash.casefold():
                raise RuntimeError(f"reviewed bindings 对应旧 Scene 内容，请重新审核：{candidate}")
            bindings = payload.get("bindings")
            if not isinstance(bindings, list):
                raise RuntimeError(f"reviewed bindings 缺少 bindings 数组：{candidate}")
            for binding in bindings:
                required = {
                    "target_node_id", "component_type", "method", "success_member",
                    "expected_value", "reviewed_by", "reviewed_at_utc",
                    "script_path", "script_sha256",
                }
                if not isinstance(binding, dict) or not required.issubset(binding):
                    raise RuntimeError(f"reviewed binding 缺少审计字段：{candidate}")
                if str(binding.get("target_node_id")) not in node_ids:
                    raise RuntimeError(f"reviewed binding 目标不在当前契约：{candidate}")
                if parse_utc_timestamp(binding.get("reviewed_at_utc")) is None:
                    raise RuntimeError(f"reviewed binding 的 reviewed_at_utc 无效：{candidate}")
                script_relative = str(binding.get("script_path", "")).replace("\\", "/")
                script_path = (Path(project.path) / script_relative).resolve()
                if (
                    not script_relative.startswith("Assets/")
                    or not script_relative.casefold().endswith(".cs")
                    or not script_path.is_file()
                    or not is_within(script_path, Path(project.path))
                ):
                    raise RuntimeError(f"reviewed binding 的 script_path 不安全或不存在：{candidate}")
                if sha256_file(script_path).casefold() != str(binding.get("script_sha256", "")).casefold():
                    raise RuntimeError(f"reviewed binding 对应脚本已经变化，请重新审核：{script_path}")
                if not reviewed_binding_matches_contract(binding, contract):
                    raise RuntimeError(f"reviewed binding 方法不在当前静态证据候选中：{candidate}")
            matches.append({
                "path": candidate.resolve(),
                "sha256": sha256_file(candidate),
                "count": len(bindings),
            })
        if len(matches) > 1:
            raise RuntimeError("同一项目/Scene 存在多个已批准 reviewed bindings；请撤销到只保留一个。")
        return matches[0] if matches else None

    @staticmethod
    def indexed_reviewed_bindings(index: dict[str, object]) -> dict[str, object] | None:
        value = index.get("reviewedBindings")
        return value if isinstance(value, dict) and isinstance(value.get("path"), str) else None

    def ensure_explicit_observe_only_index(
        self, project: UnityProject, scene: SceneChoice
    ) -> tuple[Path, dict[str, object]]:
        """Upgrade a pre-provenance index without rerunning or altering its LLM report."""
        analysis = self.latest_analysis_record(project, scene)
        if analysis is None:
            raise RuntimeError("找不到当前项目/Scene 的解析索引。")
        index_path, index = analysis
        if "reviewedBindings" in index:
            return analysis
        contract_value = index.get("agentContract")
        if not isinstance(contract_value, str) or not Path(contract_value).is_file():
            raise RuntimeError("旧解析索引没有可核验的 Agent 契约。")
        current = self.trusted_reviewed_bindings(project, scene, Path(contract_value))
        if current is not None:
            raise RuntimeError("发现新的人工审核交互清单；必须重新执行第 1 步，不能把它静默加入旧解析。")
        upgraded = dict(index)
        upgraded["schemaVersion"] = "scene-agent-terminal-analysis-2.1"
        upgraded["bindingProvenanceAddedAtUtc"] = utc_now()
        upgraded["derivedFromAnalysisIndex"] = str(index_path)
        upgraded["reviewedBindings"] = {
            "path": None,
            "sha256": None,
            "count": 0,
            "trustRoot": "configs/scene_agent_reviewed_bindings",
            "interactionMode": "observe_only",
        }
        upgraded_path = index_path.with_name(index_path.stem[:-len("_index")] + "_binding_index.json")
        upgraded_path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_analysis_by_scene[self.scene_key(project, scene)] = upgraded_path
        return upgraded_path, upgraded

    def generic_build_record(
        self,
        project: UnityProject,
        scene: SceneChoice,
        *,
        required_contract: Path | None = None,
        decision_brain_mode: str = "rule",
        decision_brain_research_override: bool = False,
    ) -> dict[str, object] | None:
        """Return only a build whose source, contract and acceptance still match."""
        model = inspect_decision_brain_model(self.repository)
        mode_ready, _ = decision_brain_mode_availability(
            model, decision_brain_mode, decision_brain_research_override
        )
        if not mode_ready:
            return None
        scene_path = Path(scene.scene_path)
        if not scene_path.is_file() or not scene.id.startswith("Assets/"):
            return None
        analysis = self.latest_analysis_record(project, scene)
        if analysis is None:
            return None
        analysis_index = analysis[1]
        current_contract = required_contract
        if current_contract is None:
            raw_contract = analysis_index.get("agentContract")
            if not isinstance(raw_contract, str):
                return None
            current_contract = Path(raw_contract)
        if not current_contract.is_file() or not is_within(current_contract, self.semantic_root):
            return None

        indexed_bindings = self.indexed_reviewed_bindings(analysis_index)
        current_bindings = self.trusted_reviewed_bindings(project, scene, current_contract)
        if indexed_bindings is None:
            if current_bindings is not None:
                return None  # A new approval requires a fresh analysis index.
        elif (
            current_bindings is None
            or not same_path(str(indexed_bindings.get("path", "")), current_bindings["path"])
            or str(indexed_bindings.get("sha256", "")).casefold() != str(current_bindings["sha256"]).casefold()
            or int(indexed_bindings.get("count", -1)) != int(current_bindings["count"])
        ):
            return None

        expected_contract_hash = sha256_file(current_contract)
        expected_scene_hash = sha256_file(scene_path)
        try:
            expected_runtime_sources = repository_file_hashes(
                self.repository, GENERIC_RUNTIME_SOURCE_FILES
            )
            expected_build_pipeline = repository_file_hashes(
                self.repository, GENERIC_BUILD_PIPELINE_FILES
            )
            expected_normal_hmd_model_hash = sha256_file(
                self.repository
                / "demo" / "UnityNormalHmdAgentDemo" / "Assets"
                / "Resources" / "NormalHmdAgentModel.json"
            )
        except OSError:
            return None
        results_root = self.repository / "results" / "generic_scene_agent"
        candidates = sorted(
            results_root.glob("*_build_status.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ) if results_root.is_dir() else []
        for status_path in candidates:
            status = load_json(status_path)
            if status is None or status.get("status") != "success":
                continue
            if status.get("sourceProjectWasModified") is not False:
                continue
            if not same_path(str(status.get("sourceProject", "")), project.path):
                continue
            if str(status.get("sourceScene", "")).replace("\\", "/") != scene.id.replace("\\", "/"):
                continue
            if not same_path(str(status.get("compiledContractPath", "")), current_contract):
                continue
            if str(status.get("compiledContractSha256", "")).casefold() != expected_contract_hash.casefold():
                continue
            if str(status.get("sourceSceneSha256", "")).casefold() != expected_scene_hash.casefold():
                continue
            runtime_sources = status.get("runtimeSourceSha256")
            build_pipeline = status.get("buildPipelineSha256")
            if not isinstance(runtime_sources, dict) or not isinstance(build_pipeline, dict):
                continue
            if {
                str(key): str(value).casefold() for key, value in runtime_sources.items()
            } != expected_runtime_sources:
                continue
            if {
                str(key): str(value).casefold() for key, value in build_pipeline.items()
            } != expected_build_pipeline:
                continue
            if str(status.get("normalHmdModelArtifactSha256", "")).casefold() != expected_normal_hmd_model_hash:
                continue
            if status.get("decisionBrainModeValidated") != decision_brain_mode:
                continue
            if decision_brain_mode == "rule":
                if status.get("decisionBrainModelStatus") not in {"not_required", "bundled"}:
                    continue
            elif (
                status.get("decisionBrainModelStatus") != "bundled"
                or str(status.get("decisionBrainModelSha256", "")).casefold() != model.sha256.casefold()
                or status.get("decisionBrainModelSchemaVersion") != DECISION_BRAIN_MODEL_SCHEMA
            ):
                continue
            if decision_brain_mode == "model" and model.data_status != "production_candidate":
                if status.get("decisionBrainResearchOverride") is not True:
                    continue
            try:
                status_binding_count = int(status.get("reviewedBindingCount", -1))
            except (TypeError, ValueError):
                continue
            if indexed_bindings is None:
                if (
                    status.get("reviewedBindingsPath") not in (None, "")
                    or status.get("reviewedBindingsSha256") not in (None, "")
                    or status_binding_count != 0
                ):
                    continue
            elif (
                not same_path(str(status.get("reviewedBindingsPath", "")), str(indexed_bindings["path"]))
                or str(status.get("reviewedBindingsSha256", "")).casefold()
                != str(indexed_bindings["sha256"]).casefold()
                or status_binding_count != int(indexed_bindings["count"])
            ):
                continue
            raw_executable = status.get("executable")
            if not isinstance(raw_executable, str):
                continue
            executable = Path(raw_executable)
            build_time = validated_build_time(status_path, executable)
            if build_time is None:
                continue

            prefix = status_path.name[:-len("_build_status.json")]
            runtime_contract = results_root / f"{prefix}_runtime_contract.json"
            acceptance_path = results_root / f"{prefix}_acceptance.json"
            render_validation = results_root / f"{prefix}_render_validation.json"
            camera_audit = results_root / f"{prefix}_camera_audit.json"
            screenshot = results_root / f"{prefix}_first_person.png"
            if not runtime_contract.is_file():
                continue
            if str(status.get("runtimeContractSha256", "")).casefold() != sha256_file(runtime_contract).casefold():
                continue
            acceptance = load_json(acceptance_path)
            render = load_json(render_validation)
            camera = load_json(camera_audit)
            try:
                rendered_artifacts_are_fresh = all(
                    artifact.stat().st_mtime + 5.0 >= build_time
                    for artifact in (render_validation, camera_audit, screenshot)
                )
            except OSError:
                rendered_artifacts_are_fresh = False
            if (
                acceptance is None
                or not acceptance_is_fresh(acceptance_path, acceptance, build_time)
                or not completed_generic_acceptance_run(acceptance, decision_brain_mode)
                or acceptance.get("projectId") != project.name
                or acceptance.get("sceneName") != Path(scene.scene_path).stem
                or (
                    decision_brain_mode != "rule"
                    and (
                        str(acceptance.get("decisionBrainModelSha256", "")).casefold()
                        != model.sha256.casefold()
                        or acceptance.get("decisionBrainModelSchemaVersion")
                        != model.schema_version
                        or acceptance.get("decisionBrainDataStatus") != model.data_status
                    )
                )
                or render is None or render.get("status") != "passed"
                or not same_path(str(render.get("image", "")), screenshot)
                or camera is None or camera.get("status") != "captured"
                or "ParticipantCamera" not in str(camera.get("activeCameraName", ""))
                or not same_path(str(camera.get("screenshotPath", "")), screenshot)
                or not screenshot.is_file() or screenshot.stat().st_size <= 0
                or not rendered_artifacts_are_fresh
            ):
                continue
            return {
                "statusPath": status_path,
                "status": status,
                "executable": executable,
                "runtimeContract": runtime_contract,
                "acceptance": acceptance_path,
                "renderValidation": render_validation,
                "cameraAudit": camera_audit,
                "screenshot": screenshot,
            }
        return None

    def apply_generic_builds(self, projects: list[UnityProject]) -> None:
        for project in projects:
            if project.name.casefold() == "teach2selfprototype":
                continue
            for scene in project.scenes:
                if scene.support == "blocked" or not scene.id.startswith("Assets/"):
                    continue
                try:
                    record = self.generic_build_record(project, scene)
                    if record is None:
                        record = self.generic_build_record(
                            project, scene, decision_brain_mode="shadow"
                        )
                    if record is None:
                        record = self.generic_build_record(
                            project,
                            scene,
                            decision_brain_mode="model",
                            decision_brain_research_override=True,
                        )
                except RuntimeError as exception:
                    scene.support = "blocked"
                    scene.support_reason = f"人工审核交互清单无效：{exception}"
                    continue
                if record is None:
                    continue
                scene.support = "agent_ready"
                scene.support_reason = (
                    "AI 解析契约已注入隔离副本；Generic Brain、Normal HMD v4、第一人称渲染和内容完成闭环均已验收。"
                )
                scene.launcher = str(record["executable"])
                scene.launcher_args = []
                scene.runtime_kind = "generic"
                scene.build_status = str(record["statusPath"])

    def generic_build_eligibility(
        self,
        project: UnityProject,
        scene: SceneChoice,
        decision_brain_mode: str = "rule",
        decision_brain_research_override: bool = False,
    ) -> tuple[bool, str]:
        model = inspect_decision_brain_model(self.repository)
        mode_ready, mode_reason = decision_brain_mode_availability(
            model, decision_brain_mode, decision_brain_research_override
        )
        if not mode_ready:
            return False, mode_reason
        if project.name.casefold() == "teach2selfprototype":
            return False, "参与者实验项目默认只读；需先建立不会污染研究数据的测试身份与 sandbox。"
        if scene.support == "blocked":
            return False, scene.support_reason
        if not scene.id.startswith("Assets/") or not scene.id.casefold().endswith(".unity"):
            return False, "逻辑参考场景已有专用 Player；请选择真实的 Assets/*.unity Scene 进行通用构建。"
        if project.unity_version != GENERIC_UNITY_VERSION:
            return False, (
                f"当前通用构建器已验证 Unity {GENERIC_UNITY_VERSION}；该项目为 {project.unity_version}，"
                "可解析，但不能冒充已验证构建。"
            )
        analysis = self.latest_analysis_record(project, scene)
        if analysis is None:
            return False, "请先完成第 1 步 AI 解析与契约生成。"
        contract = analysis[1].get("agentContract")
        if not isinstance(contract, str) or not Path(contract).is_file():
            return False, "最新解析没有有效 Agent 契约，请重新执行第 1 步。"
        indexed_bindings = self.indexed_reviewed_bindings(analysis[1])
        try:
            current_bindings = self.trusted_reviewed_bindings(project, scene, Path(contract))
        except RuntimeError as exception:
            return False, f"人工审核交互清单无效：{exception}"
        if (indexed_bindings is None) != (current_bindings is None):
            return False, "人工审核交互清单在解析后发生变化；请重新执行第 1 步建立新的证据索引。"
        if indexed_bindings is None:
            return True, (
                f"{mode_reason} 可以隔离构建 Brain + Normal HMD v4；当前没有人工审核交互清单，"
                "本次 Agent 只导航和观察，不会调用项目方法。"
            )
        return True, (
            f"{mode_reason} 可以隔离构建；将启用 {indexed_bindings['count']} 个带审核人/时间证据的白名单交互。"
        )

    def generic_conflict_suppression(self, project: UnityProject, scene: SceneChoice) -> list[str]:
        """Known source-player controllers that would fight the injected Agent.

        This is deliberately an exact compatibility profile, not a heuristic
        that disables arbitrary gameplay managers in an unknown project.
        """
        random_demo = (self.repository / "demo" / "UnityRandomAgentDemo").resolve()
        if same_path(project.path, random_demo) and scene.id == "Assets/Scenes/AuroraMotionCourtyard.unity":
            return ["RandomAgentDemo.RandomAgentController"]
        return []

    def python_executable(self) -> Path:
        candidates = [
            self.repository / ".venv" / "Scripts" / "python.exe",
            self.repository / ".venv-normal-hmd" / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise RuntimeError("找不到项目 Python 环境，无法运行 Unity 解析器。")

    def analyze(self, project: UnityProject, scene: SceneChoice, log: Callable[[str], None]) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = f"{slug(project.name)}_{slug(scene.label)}_{stamp}"
        project_output = self.semantic_root / f"{prefix}_project.json"
        status_output = self.semantic_root / f"{prefix}_status.json"
        scene_output = self.semantic_root / f"{prefix}_scene.csv"
        parser_log = self.semantic_root / f"{prefix}_scene.log"
        contract_output = self.semantic_root / f"{prefix}_agent_contract.json"
        contract_status = self.semantic_root / f"{prefix}_contract_status.json"
        human_report = self.semantic_root / f"{prefix}_ai_report.md"
        python = self.python_executable()
        project_command = [
            str(python), str(self.repository / "scripts" / "unity_parser.py"),
            "--project", project.path, "--output", str(project_output), "--status", str(status_output),
        ]
        log("运行项目语义解析器……")
        parser_environment = os.environ.copy()
        parser_environment["PYTHONUTF8"] = "1"
        try:
            completed = subprocess.run(
                project_command, cwd=self.repository, capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=parser_environment,
                timeout=PARSER_TIMEOUT_SECONDS, creationflags=WINDOWS_NO_CONSOLE,
            )
        except subprocess.TimeoutExpired as exception:
            raise RuntimeError(
                f"项目语义解析超过 {PARSER_TIMEOUT_SECONDS} 秒，已停止等待；请检查项目规模或解析日志。"
            ) from exception
        if completed.stdout.strip():
            log(completed.stdout.strip())
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"项目解析失败，退出码 {completed.returncode}")
        if scene.scene_path:
            scene_command = [
                str(python), str(self.repository / "scripts" / "parse_unity_scene.py"),
                "--scene", scene.scene_path, "--output", str(scene_output), "--log", str(parser_log),
            ]
            try:
                completed = subprocess.run(
                    scene_command, cwd=self.repository, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", env=parser_environment,
                    timeout=PARSER_TIMEOUT_SECONDS, creationflags=WINDOWS_NO_CONSOLE,
                )
            except subprocess.TimeoutExpired as exception:
                raise RuntimeError(
                    f"Scene 静态解析超过 {PARSER_TIMEOUT_SECONDS} 秒，已停止等待：{scene.label}"
                ) from exception
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"Scene 解析失败，退出码 {completed.returncode}")
            log(f"Scene 静态特征解析完成：{scene_output}")
        else:
            raise RuntimeError("所选项目没有可编译的 .unity Scene。")

        project_root = Path(project.path).resolve()
        selected_scene_path = Path(scene.scene_path).resolve()
        try:
            selected_scene_relative = selected_scene_path.relative_to(project_root).as_posix()
        except ValueError as exception:
            raise RuntimeError("所选 Scene 不在该 Unity 项目内，拒绝生成执行契约。") from exception

        contract_command = [
            str(python), str(self.repository / "scripts" / "compile_scene_agent_contract.py"),
            "--project", project.path,
            "--semantic", str(project_output),
            "--scene", selected_scene_relative,
            "--contract", str(contract_output),
            "--report", str(human_report),
            "--status", str(contract_status),
            "--narrative-backend", "auto",
        ]
        log("启动只读 LLM 场景理解，并编译 Agent 可执行契约……")
        try:
            completed = subprocess.run(
                contract_command, cwd=self.repository, capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=parser_environment, timeout=480,
                creationflags=WINDOWS_NO_CONSOLE,
            )
        except subprocess.TimeoutExpired as exception:
            raise RuntimeError("AI 场景理解超过 480 秒，已停止等待；请稍后重试。") from exception
        if completed.stdout.strip():
            log(completed.stdout.strip())
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"场景契约编译失败，退出码 {completed.returncode}")
        compiled_status = load_json(contract_status)
        if compiled_status is None or compiled_status.get("status") != "success":
            raise RuntimeError(f"场景契约没有通过结构校验：{contract_status}")
        narrative_backend = str(compiled_status.get("narrative_backend_used", "unknown"))
        if narrative_backend == "codex_cli":
            log("[OK] LLM 已以只读方式检查项目证据；运行时只使用经校验的契约字段。")
        else:
            reason = str(compiled_status.get("narrative_fallback_reason", "未说明原因"))
            log(f"[WARN] 本次未使用真实 LLM，已明确回退到确定性报告：{reason}")
        if human_report.is_file():
            report_text = human_report.read_text(encoding="utf-8-sig", errors="replace")
            log("── 场景理解报告 ──\n" + report_text)
        reviewed = self.trusted_reviewed_bindings(project, scene, contract_output)
        reviewed_index = {
            "path": str(reviewed["path"]) if reviewed is not None else None,
            "sha256": str(reviewed["sha256"]) if reviewed is not None else None,
            "count": int(reviewed["count"]) if reviewed is not None else 0,
            "trustRoot": "configs/scene_agent_reviewed_bindings",
            "interactionMode": "reviewed_allowlist" if reviewed is not None else "observe_only",
        }
        if reviewed is None:
            log("[INFO] 当前没有匹配的人工审核交互清单；Generic Brain 将只导航/观察，不会调用项目方法。")
        else:
            log(f"[OK] 加载 {reviewed['count']} 个可追责 reviewed bindings：{reviewed['path']}")
        index_path = self.semantic_root / f"{prefix}_index.json"
        index_path.write_text(json.dumps({
            "schemaVersion": "scene-agent-terminal-analysis-2.0",
            "generatedAtUtc": utc_now(),
            "project": asdict(project),
            "selectedScene": asdict(scene),
            "projectReport": str(project_output),
            "projectStatus": str(status_output),
            "sceneReport": str(scene_output) if scene_output.exists() else None,
            "sceneLog": str(parser_log) if parser_log.exists() else None,
            "agentContract": str(contract_output),
            "contractStatus": str(contract_status),
            "humanReport": str(human_report),
            "narrativeBackend": narrative_backend,
            "reviewedBindings": reviewed_index,
            "claimBoundary": (
                "LLM findings are advisory. The deterministic contract and runtime verification gate all actions; "
                "static evidence is not proof that the Agent executed an interaction."
            ),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_analysis_by_scene[self.scene_key(project, scene)] = index_path
        return index_path

    def build_generic(
        self,
        project: UnityProject,
        scene: SceneChoice,
        log: Callable[[str], None],
        decision_brain_mode: str = "rule",
        decision_brain_research_override: bool = False,
    ) -> Path:
        eligible, reason = self.generic_build_eligibility(
            project, scene, decision_brain_mode, decision_brain_research_override
        )
        if not eligible:
            raise RuntimeError(reason)
        analysis = self.ensure_explicit_observe_only_index(project, scene)
        contract_value = analysis[1].get("agentContract")
        if not isinstance(contract_value, str):
            raise RuntimeError("解析索引没有 Agent 契约路径。")
        contract_path = Path(contract_value).resolve()
        contract = load_json(contract_path)
        if contract is None or contract.get("schema_version") != "scene-agent-contract-1.0":
            raise RuntimeError("Agent 契约缺失或 schema 不受支持。")
        runtime_scene = contract.get("runtime_scene")
        contract_project = contract.get("project")
        if not isinstance(runtime_scene, dict) or str(runtime_scene.get("path", "")).replace("\\", "/") != scene.id:
            raise RuntimeError("最新 Agent 契约不属于当前选择的 Scene；请重新执行第 1 步。")
        if not isinstance(contract_project, dict) or not same_path(str(contract_project.get("root", "")), project.path):
            raise RuntimeError("最新 Agent 契约不属于当前选择的 Unity 项目；请重新执行第 1 步。")
        indexed_bindings = self.indexed_reviewed_bindings(analysis[1])
        current_bindings = self.trusted_reviewed_bindings(project, scene, contract_path)
        if (indexed_bindings is None) != (current_bindings is None):
            raise RuntimeError("reviewed bindings 在解析后发生变化，请重新执行第 1 步。")
        if indexed_bindings is not None and (
            current_bindings is None
            or not same_path(str(indexed_bindings.get("path", "")), current_bindings["path"])
            or str(indexed_bindings.get("sha256", "")).casefold() != str(current_bindings["sha256"]).casefold()
        ):
            raise RuntimeError("reviewed bindings 路径或哈希与解析索引不一致，请重新审核/解析。")

        build_script = self.repository / "demo" / "build_generic_scene_agent.ps1"
        if not build_script.is_file():
            raise RuntimeError(f"通用 Agent 构建脚本不存在：{build_script}")
        seed = random.SystemRandom().randrange(1, 2_000_000_000)
        # Prefer PowerShell 7 when installed, but Windows PowerShell 5.1 is a
        # supported fallback.  The build script performs every JSON read with
        # explicit UTF-8 decoding, so the legacy host no longer corrupts
        # Chinese scene contracts through the active ANSI code page.
        powershell = (
            shutil.which("pwsh.exe")
            or shutil.which("pwsh")
            or shutil.which("powershell.exe")
            or shutil.which("powershell")
        )
        if not powershell:
            raise RuntimeError("没有找到可用的 PowerShell（pwsh 或 powershell.exe）。")
        log(f"构建 Shell：{Path(powershell).name}；UTF-8 契约由构建脚本显式解码。")
        command = [
            powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(build_script),
            "-SourceProject", project.path,
            "-CompiledContract", str(contract_path),
            "-ScenePath", scene.id,
            "-Seed", str(seed),
            "-DecisionBrainMode", decision_brain_mode,
        ]
        model = inspect_decision_brain_model(self.repository)
        if model.status == "ready":
            command.extend(["-DecisionBrainModel", str(model.path)])
        if decision_brain_research_override:
            command.append("-DecisionBrainResearchOverride")
        if indexed_bindings is not None:
            command.extend(["-ReviewedBindings", str(indexed_bindings["path"])])
            log(
                f"交互授权：{indexed_bindings['count']} 个 reviewed bindings；"
                f"SHA-256={indexed_bindings['sha256']}。"
            )
        else:
            log("交互授权：无人工审核清单，本次严格为 observe-only。")
        suppressed_behaviours = self.generic_conflict_suppression(project, scene)
        if suppressed_behaviours:
            # PowerShell's string[] binder accepts the single exact type used by
            # this reviewed compatibility profile. Unknown projects never get
            # automatic gameplay-manager suppression.
            command.extend(["-DisableBehaviourType", suppressed_behaviours[0]])
            log(
                "兼容性门控：隔离 Player 将停用源项目自己的 AI 控制器 "
                f"{suppressed_behaviours[0]}，防止双控制器/双 HUD；这不是预设路线。"
            )
        log(
            "开始隔离构建：源项目保持只读；构建器会复制项目、注入 Brain/运动技能并运行内容完成验收。"
        )
        log(f"Decision Brain：mode={decision_brain_mode}；{model.detail}")
        process = subprocess.Popen(
            command,
            cwd=self.repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=WINDOWS_NO_CONSOLE,
        )
        if process.stdout is not None:
            for line in process.stdout:
                if line.strip():
                    log(line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"通用 Agent 隔离构建或验收失败，退出码 {return_code}。请查看上方构建日志。")
        record = self.generic_build_record(
            project,
            scene,
            required_contract=contract_path,
            decision_brain_mode=decision_brain_mode,
            decision_brain_research_override=decision_brain_research_override,
        )
        if record is None:
            raise RuntimeError("构建脚本已结束，但场景/契约哈希或内容完成渲染验收没有形成一致的证据链。")
        status = record.get("status")
        if not isinstance(status, dict) or status.get("sourceProjectWasModified") is not False:
            raise RuntimeError("构建状态没有证明源项目保持未修改，拒绝授权启动。")
        scene.support = "agent_ready"
        scene.support_reason = (
            "AI 契约已注入隔离副本；Generic Brain、EyeNavGS Normal HMD v4、第一人称渲染和内容完成闭环均已验收。"
        )
        scene.launcher = str(record["executable"])
        scene.launcher_args = []
        scene.runtime_kind = "generic"
        scene.build_status = str(record["statusPath"])
        log(f"[OK] Agent 已就绪：{scene.launcher}")
        log(f"[OK] 构建证据：{scene.build_status}")
        log(f"[OK] 内容完成验收：{record['acceptance']}")
        log(f"[OK] 第一人称渲染：{record['screenshot']}")
        return Path(scene.launcher)

    def launch(
        self,
        project: UnityProject,
        scene: SceneChoice,
        automated: bool,
        log: Callable[[str], None],
        decision_brain_mode: str = "rule",
        decision_brain_research_override: bool = False,
    ) -> Path:
        analysis_files = self.latest_analysis_files(project, scene)
        if not analysis_files:
            raise RuntimeError("请先执行“1  AI 解析 + 生成 Agent 契约”；本终端不允许跳过场景理解直接投放 Agent。")
        if scene.support != "agent_ready":
            raise RuntimeError(scene.support_reason)
        model = inspect_decision_brain_model(self.repository)
        mode_ready, mode_reason = decision_brain_mode_availability(
            model, decision_brain_mode, decision_brain_research_override
        )
        if not mode_ready:
            raise RuntimeError(mode_reason)
        if scene.runtime_kind != "generic" and decision_brain_mode != "rule":
            raise RuntimeError("该专用参考 Player 尚未集成训练 Decision Brain；请选择 Rule。")
        if scene.runtime_kind == "generic":
            generic_record = self.generic_build_record(
                project,
                scene,
                decision_brain_mode=decision_brain_mode,
                decision_brain_research_override=decision_brain_research_override,
            )
            if generic_record is None:
                raise RuntimeError("源 Scene 或 AI 契约已经变化，旧的通用构建不再授权启动；请重新执行第 2 步。")
            scene.launcher = str(generic_record["executable"])
            scene.build_status = str(generic_record["statusPath"])
        executable = Path(scene.launcher)
        if not executable.is_file():
            raise RuntimeError(f"适配 Player 不存在：{executable}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed = random.SystemRandom().randrange(1, 2_000_000_000)
        scenario_tag = (
            f"s{scene.scenario_index}" if scene.scenario_index is not None
            else "generic" if scene.runtime_kind == "generic" else "external"
        )
        # Unity appends another descriptive episode directory below this root.
        # Keep the terminal-owned segment deliberately short for legacy Win32/
        # Unity APIs that still fail near MAX_PATH even when Python supports it.
        run_root = self.runs_root / f"run_{stamp}_{scenario_tag}_seed_{seed}"
        run_root.mkdir(parents=True, exist_ok=False)
        analysis_index = load_json(analysis_files[0]) or {}
        reviewed_bindings = self.indexed_reviewed_bindings(analysis_index)
        manifest = {
            "schemaVersion": "scene-agent-terminal-run-1.0",
            "createdAtUtc": utc_now(),
            "project": {"name": project.name, "path": project.path, "unityVersion": project.unity_version},
            "scene": asdict(scene),
            "seed": seed,
            "automatedAcceptance": automated,
            "launchMode": "automated_acceptance" if automated else "visual_experience",
            "completionPolicy": (
                "content_complete" if scene.runtime_kind == "generic" else "fixed_60_seconds"
            ),
            "minimumEvidenceSeconds": 15 if scene.runtime_kind == "generic" else 60,
            "safetyTimeoutSeconds": 900 if scene.runtime_kind == "generic" else 60,
            "decisionBrainModeRequested": decision_brain_mode,
            "decisionBrainResearchOverride": decision_brain_research_override,
            "decisionBrainModel": ({
                "path": str(model.path),
                "sha256": model.sha256,
                "schemaVersion": model.schema_version,
                "dataStatus": model.data_status,
                "status": model.status,
            } if decision_brain_mode != "rule" else {
                "path": None, "sha256": None, "schemaVersion": None,
                "dataStatus": None, "status": "not_requested",
            }),
            "outputRoot": str(run_root),
            "analysisIndex": str(analysis_files[0]),
            "buildStatus": scene.build_status or None,
            "reviewedBindings": ({
                "path": str(reviewed_bindings["path"]),
                "sha256": str(reviewed_bindings["sha256"]),
                "count": int(reviewed_bindings["count"]),
            } if reviewed_bindings is not None else {
                "path": None, "sha256": None, "count": 0, "interactionMode": "observe_only",
            }),
            "agentBoundary": (
                "EyeNavGS trains Normal HMD HeadLocal motion and the root-speed envelope. The high-level brain, "
                "target choice and route steering use the explicitly recorded Decision Brain mode. The learned v0 "
                "model is rule-bootstrap smoke research, not a learned visual policy or a human-equivalent policy. "
                "Interactions require an explicit reviewed allowlist and runtime state verification."
            ),
        }
        (run_root / "terminal_run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_run_by_scene[self.scene_key(project, scene)] = run_root
        arguments = [str(executable), *scene.launcher_args, f"--seed={seed}"]
        if scene.runtime_kind == "generic":
            normal_hmd_root = run_root / "normal_hmd"
            normal_hmd_root.mkdir(parents=True, exist_ok=True)
            arguments.extend([
                f"--generic-agent-log-root={run_root}",
                f"--log-root={normal_hmd_root}",
                f"--decision-brain-mode={decision_brain_mode}",
            ])
            if decision_brain_research_override:
                arguments.append("--decision-brain-research-override")
        else:
            arguments.append(f"--scene-log-root={run_root}")
        report = run_root / "terminal_acceptance.json"
        player_log = run_root / "player.log"
        if automated:
            if scene.runtime_kind == "generic":
                arguments.extend([
                    "-batchmode", "-nographics", "--generic-agent-test",
                    f"--generic-agent-test-output={report}", "-logFile", str(player_log),
                ])
            elif scene.scenario_index is not None:
                arguments.extend([
                    "-batchmode", "-nographics", "--scene-aware-test",
                    f"--scene-aware-test-output={report}", "-logFile", str(player_log),
                ])
            else:
                arguments.extend([
                    "-batchmode", "-nographics", "--external-agent-test",
                    f"--external-agent-test-output={report}", "-logFile", str(player_log),
                ])
            log(f"自动验收 seed={seed}，运行到场景内容完成（含无进展/安全上限）……")
            try:
                completed = subprocess.run(arguments, cwd=executable.parent, timeout=180)
            except subprocess.TimeoutExpired as exception:
                raise RuntimeError(f"Player 自动验收超过 180 秒，已终止；日志：{player_log}") from exception
            if completed.returncode != 0:
                raise RuntimeError(f"Player 自动验收失败，退出码 {completed.returncode}；日志：{player_log}")
            if not report.is_file():
                raise RuntimeError(f"Player 未写出验收报告：{report}")
            parsed = json.loads(report.read_text(encoding="utf-8-sig"))
            valid_report = (
                completed_generic_acceptance_run(parsed, decision_brain_mode)
                if scene.runtime_kind == "generic"
                else completed_acceptance_run(parsed)
            )
            if not valid_report:
                raise RuntimeError(f"验收报告未通过：{report}")
            log(f"验收通过：{report}")
            if scene.runtime_kind == "generic":
                summary = self.write_target_exposure_summary(run_root)
                if summary is not None:
                    log(f"目标级工程暴露排序：{summary[1]}")
        else:
            log(f"启动可视体验 seed={seed}……")
            arguments.extend(["-logFile", str(player_log)])
            process = subprocess.Popen(arguments, cwd=executable.parent)
            self.active_processes[run_root] = process
            log("Player 已启动。每次运行使用新 seed；场景内容体验完成后日志会完整落盘。")
        return run_root

    def latest_run(self, project: UnityProject, scene: SceneChoice) -> Path | None:
        key = self.scene_key(project, scene)
        remembered = self.last_run_by_scene.get(key)
        if remembered is not None and remembered.is_dir():
            return remembered
        matches: list[Path] = []
        for manifest_path in self.runs_root.glob("*/terminal_run_manifest.json"):
            report = load_json(manifest_path)
            if report is None:
                continue
            report_project = report.get("project")
            report_scene = report.get("scene")
            if not isinstance(report_project, dict) or not isinstance(report_scene, dict):
                continue
            if (
                same_path(str(report_project.get("path", "")), project.path)
                and report_scene.get("id") == scene.id
            ):
                matches.append(manifest_path.parent)
        if not matches:
            return None
        return max(matches, key=lambda item: item.stat().st_mtime)

    def latest_analysis_files(self, project: UnityProject, scene: SceneChoice) -> list[Path]:
        analysis = self.latest_analysis_record(project, scene)
        if analysis is None:
            return []
        index_path, index = analysis
        files = [index_path]
        for field_name in (
            "projectReport", "projectStatus", "sceneReport", "sceneLog",
            "agentContract", "contractStatus", "humanReport", "derivedFromAnalysisIndex",
        ):
            raw_path = index.get(field_name)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            path = Path(raw_path)
            if path.is_file() and is_within(path, self.semantic_root) and not path.is_symlink():
                files.append(path)
        return files

    def latest_analysis_report(self, project: UnityProject, scene: SceneChoice) -> Path | None:
        analysis = self.latest_analysis_record(project, scene)
        if analysis is None:
            return None
        value = analysis[1].get("humanReport")
        if not isinstance(value, str):
            return None
        report = Path(value)
        return report if report.is_file() and is_within(report, self.semantic_root) else None

    def write_target_exposure_summary(self, run_root: Path) -> tuple[Path, Path] | None:
        """Aggregate the generic motion proxy by the brain's active target."""
        telemetry_files = sorted(
            run_root.rglob("generic_agent_telemetry.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not telemetry_files:
            return None
        telemetry = telemetry_files[0]
        contract_files = sorted(
            telemetry.parent.glob("agent_scene_contract.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        labels: dict[str, str] = {}
        if contract_files:
            contract = load_json(contract_files[0]) or {}
            for target in contract.get("targets", []):
                if isinstance(target, dict) and target.get("id"):
                    labels[str(target["id"])] = str(target.get("label") or target["id"])
        inventory = load_json(telemetry.parent / "runtime_scene_inventory.json") or {}
        for target in inventory.get("targetResolutions", []):
            if isinstance(target, dict) and target.get("targetId"):
                labels[str(target["targetId"])] = str(
                    target.get("requestedNameContains") or labels.get(str(target["targetId"]), target["targetId"])
                )

        groups: dict[str, dict[str, object]] = {}
        previous_time: float | None = None
        with telemetry.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {
                "sim_time", "target_id", "brain_state", "camera_speed", "camera_angular_speed",
                "camera_acceleration", "engineering_exposure_proxy",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                return None
            for row in reader:
                target_id = (row.get("target_id") or "").strip()
                try:
                    sample_time = float(row["sim_time"])
                    speed = float(row["camera_speed"])
                    angular = float(row["camera_angular_speed"])
                    acceleration = float(row["camera_acceleration"])
                    exposure = float(row["engineering_exposure_proxy"])
                except (KeyError, TypeError, ValueError):
                    previous_time = None
                    continue
                delta = 0.0 if previous_time is None else max(0.0, min(0.25, sample_time - previous_time))
                previous_time = sample_time
                if not target_id:
                    continue
                group = groups.setdefault(target_id, {
                    "targetId": target_id,
                    "label": labels.get(target_id, target_id),
                    "samples": 0,
                    "durationSeconds": 0.0,
                    "states": set(),
                    "speedSum": 0.0,
                    "maxCameraSpeed": 0.0,
                    "angularSum": 0.0,
                    "maxCameraAngularSpeed": 0.0,
                    "accelerationSum": 0.0,
                    "maxCameraAcceleration": 0.0,
                    "exposureSum": 0.0,
                    "maxEngineeringExposureProxy": 0.0,
                })
                group["samples"] = int(group["samples"]) + 1
                group["durationSeconds"] = float(group["durationSeconds"]) + delta
                states = group["states"]
                if isinstance(states, set) and row.get("brain_state"):
                    states.add(str(row["brain_state"]))
                for sum_key, value in (
                    ("speedSum", speed), ("angularSum", angular),
                    ("accelerationSum", acceleration), ("exposureSum", exposure),
                ):
                    group[sum_key] = float(group[sum_key]) + value
                group["maxCameraSpeed"] = max(float(group["maxCameraSpeed"]), speed)
                group["maxCameraAngularSpeed"] = max(float(group["maxCameraAngularSpeed"]), angular)
                group["maxCameraAcceleration"] = max(float(group["maxCameraAcceleration"]), acceleration)
                group["maxEngineeringExposureProxy"] = max(
                    float(group["maxEngineeringExposureProxy"]), exposure
                )

        targets: list[dict[str, object]] = []
        for group in groups.values():
            samples = max(1, int(group.pop("samples")))
            states = group.pop("states")
            group["samples"] = samples
            group["durationSeconds"] = round(float(group["durationSeconds"]), 6)
            group["brainStates"] = sorted(states) if isinstance(states, set) else []
            group["meanCameraSpeed"] = float(group.pop("speedSum")) / samples
            group["meanCameraAngularSpeed"] = float(group.pop("angularSum")) / samples
            group["meanCameraAcceleration"] = float(group.pop("accelerationSum")) / samples
            group["meanEngineeringExposureProxy"] = float(group.pop("exposureSum")) / samples
            targets.append(group)
        targets.sort(
            key=lambda item: (
                -float(item["meanEngineeringExposureProxy"]),
                -float(item["maxEngineeringExposureProxy"]),
                str(item["targetId"]),
            )
        )
        for rank, target in enumerate(targets, start=1):
            target["engineeringReviewRank"] = rank

        payload = {
            "schemaVersion": "scene-agent-target-exposure-summary-1.0",
            "generatedAtUtc": utc_now(),
            "sourceTelemetry": str(telemetry),
            "targetCount": len(targets),
            "rankingMeaning": (
                "Uncalibrated engineering prioritization within this simulated rollout; not a VR.net probability, "
                "individual sickness probability, causal attribution or medical diagnosis."
            ),
            "targets": targets,
        }
        json_path = run_root / "target_exposure_summary.json"
        markdown_path = run_root / "target_exposure_summary.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = [
            "# Target-level engineering exposure summary",
            "",
            "> This is an uncalibrated within-run engineering ranking. It is not a VR.net probability, "
            "individual sickness probability, causal attribution or medical diagnosis.",
            "",
            "| Rank | Target | Samples | Duration (s) | Mean / max speed | Mean / max angular speed | "
            "Mean / max acceleration | Mean / max proxy |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for target in targets:
            markdown.append(
                "| {engineeringReviewRank} | {label} | {samples} | {durationSeconds:.2f} | "
                "{meanCameraSpeed:.3f} / {maxCameraSpeed:.3f} | "
                "{meanCameraAngularSpeed:.2f} / {maxCameraAngularSpeed:.2f} | "
                "{meanCameraAcceleration:.3f} / {maxCameraAcceleration:.3f} | "
                "{meanEngineeringExposureProxy:.2f} / {maxEngineeringExposureProxy:.2f} |".format(**target)
            )
        markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
        return json_path, markdown_path

    def export(self, project: UnityProject, scene: SceneChoice, log: Callable[[str], None]) -> Path:
        run_root = self.latest_run(project, scene)
        if run_root is None:
            raise RuntimeError("该场景还没有运行结果。请先启动体验或运行自动验收。")
        process = self.active_processes.get(run_root)
        runtime_reports = []
        for path in run_root.rglob("*.json"):
            if path.name not in {"scene_acceptance.json", "generic_agent_acceptance.json", "terminal_acceptance.json"}:
                continue
            if not path.is_file() or path.is_symlink() or not is_within(path, run_root):
                continue
            report_data = load_json(path)
            if report_data is not None and (
                completed_acceptance_run(report_data) or completed_generic_acceptance_run(report_data)
            ):
                runtime_reports.append(path)
        if not runtime_reports:
            if process is not None and process.poll() is None:
                raise RuntimeError("Agent 仍在记录本次体验。请等待场景内容完成后再导出。")
            raise RuntimeError("本次运行没有内容完成报告，不能声称是完整证据包；请重新运行验收。")
        if process is not None and process.poll() is not None:
            self.active_processes.pop(run_root, None)
        if scene.runtime_kind == "generic":
            summary = self.write_target_exposure_summary(run_root)
            if summary is None:
                raise RuntimeError("通用运行缺少带目标级运动指标的 telemetry，无法生成区域排序。")
            log(f"目标级工程暴露排序已生成：{summary[1]}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive = self.exports_root / f"{run_root.name}_export_{stamp}.zip"
        partial_archive = archive.with_name(archive.name + ".partial")
        inventory: list[dict[str, object]] = []
        skipped: list[str] = []

        def add_file(
            bundle: zipfile.ZipFile, file_path: Path, relative: Path, allowed_root: Path
        ) -> None:
            if file_path.is_symlink() or not is_within(file_path, allowed_root):
                skipped.append(str(file_path))
                return
            digest = hashlib.sha256()
            byte_count = 0
            with file_path.open("rb") as source, bundle.open(relative.as_posix(), "w") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            inventory.append({
                "path": relative.as_posix(),
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
            })

        try:
            with zipfile.ZipFile(partial_archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for file_path in sorted(run_root.rglob("*")):
                    if not file_path.is_file():
                        continue
                    relative = Path("run") / file_path.relative_to(run_root)
                    add_file(bundle, file_path, relative, run_root)
                for semantic in self.latest_analysis_files(project, scene):
                    add_file(bundle, semantic, Path("semantic") / semantic.name, self.semantic_root)
                build_evidence: list[str] = []
                review_evidence: list[str] = []
                if scene.runtime_kind == "generic":
                    run_manifest = load_json(run_root / "terminal_run_manifest.json") or {}
                    run_mode = str(run_manifest.get("decisionBrainModeRequested", "rule"))
                    run_override = run_manifest.get("decisionBrainResearchOverride") is True
                    record = self.generic_build_record(
                        project,
                        scene,
                        decision_brain_mode=run_mode,
                        decision_brain_research_override=run_override,
                    )
                    generic_root = self.repository / "results" / "generic_scene_agent"
                    if record is None:
                        raise RuntimeError("通用构建证据与当前契约/Scene 不再匹配，拒绝导出混合证据包。")
                    for field_name in (
                        "statusPath", "runtimeContract", "acceptance", "renderValidation", "cameraAudit", "screenshot",
                    ):
                        artifact = record.get(field_name)
                        if not isinstance(artifact, Path) or not artifact.is_file():
                            continue
                        add_file(bundle, artifact, Path("build") / artifact.name, generic_root)
                        build_evidence.append(artifact.name)
                    analysis = self.latest_analysis_record(project, scene)
                    indexed_bindings = self.indexed_reviewed_bindings(analysis[1]) if analysis else None
                    if indexed_bindings is not None:
                        manifest = Path(str(indexed_bindings["path"]))
                        review_root = self.repository / "configs" / "scene_agent_reviewed_bindings"
                        add_file(bundle, manifest, Path("review") / manifest.name, review_root)
                        review_evidence.append(manifest.name)
                export_manifest = json.dumps({
                    "schemaVersion": "scene-agent-terminal-export-1.0",
                    "exportedAtUtc": utc_now(),
                    "sourceRun": str(run_root),
                    "project": project.path,
                    "scene": scene.label,
                    "files": inventory,
                    "skippedUnsafePaths": skipped,
                    "runtimeReports": [str(path.relative_to(run_root)) for path in runtime_reports],
                    "buildEvidence": build_evidence,
                    "reviewedBindingEvidence": review_evidence,
                    "riskMeaning": (
                        "Motion/exposure telemetry is research evidence. Any uncalibrated exposure proxy is not a "
                        "VR.net probability, individual sickness probability, or medical diagnosis."
                    ),
                }, ensure_ascii=False, indent=2).encode("utf-8")
                bundle.writestr("export_manifest.json", export_manifest)
            os.replace(partial_archive, archive)
        except Exception:
            try:
                partial_archive.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        log(f"结果包已导出：{archive}")
        return archive


def self_test(
    output: Path | None,
    decision_brain_mode: str = "rule",
    decision_brain_model: Path | None = None,
    decision_brain_research_override: bool = False,
) -> int:
    try:
        repository = locate_repository()
        backend = TerminalBackend(repository)
        projects = backend.refresh()
        supported = [
            (project.name, scene.label)
            for project in projects for scene in project.scenes if scene.support == "agent_ready"
        ]
        export_writable = os.access(backend.exports_root, os.W_OK)
        model = inspect_decision_brain_model(repository, decision_brain_model)
        mode_ready, mode_detail = decision_brain_mode_availability(
            model, decision_brain_mode, decision_brain_research_override
        )
        button_binding = {
            "target_node_id": "node:reviewed-button",
            "component_type": "UnityEngine.UI.Button",
            "method": "Press",
            "script_path": "Assets/Scripts/ReviewedButtonHandler.cs",
        }
        button_contract = {
            "semantic_nodes": [{
                "id": "node:reviewed-button",
                "evidence": [{"observation": "Component evidence: UnityEngine.UI.Button"}],
            }],
            "affordances": [{"target_node_id": "node:reviewed-button", "verb": "press"}],
            "method_candidates": [],
        }
        button_policy_test = (
            reviewed_binding_matches_contract(button_binding, button_contract)
            and not reviewed_binding_matches_contract(
                button_binding, {**button_contract, "affordances": []}
            )
            and not reviewed_binding_matches_contract(
                {**button_binding, "method": "Invoke"}, button_contract
            )
        )
        ui_feedback_policy_test = (
            1.0 <= BACKGROUND_HEARTBEAT_SECONDS <= 5.0
            and format_elapsed(-1) == "0 秒"
            and format_elapsed(65) == "1 分 05 秒"
            and not set(BACKGROUND_CONFLICTING_ACTIONS).intersection(BACKGROUND_NON_CONFLICTING_CONTROLS)
            and {"browse", "open"}.issubset(BACKGROUND_NON_CONFLICTING_CONTROLS)
        )
        result = {
            "status": (
                "passed"
                if projects and export_writable and button_policy_test and ui_feedback_policy_test and mode_ready
                else "failed"
            ),
            "generatedAtUtc": utc_now(),
            "repository": str(repository),
            "projectCount": len(projects),
            "projects": [asdict(project) for project in projects],
            "agentReadyScenes": supported,
            "agentReadySceneCount": len(supported),
            "decisionBrainModeRequested": decision_brain_mode,
            "decisionBrainModeAvailable": mode_ready,
            "decisionBrainModeDetail": mode_detail,
            "decisionBrainModel": {
                "path": str(model.path),
                "status": model.status,
                "sha256": model.sha256 or None,
                "schemaVersion": model.schema_version or None,
                "dataStatus": model.data_status or None,
            },
            "exportWritable": export_writable,
            "reviewedButtonPolicySelfTest": button_policy_test,
            "uiBackgroundFeedbackSelfTest": ui_feedback_policy_test,
            "uiBackgroundHeartbeatSeconds": BACKGROUND_HEARTBEAT_SECONDS,
            "uiControlsAvailableDuringBackgroundWork": list(BACKGROUND_NON_CONFLICTING_CONTROLS),
            "readinessMeaning": (
                "Zero ready scenes does not fail terminal health; runtime launch remains disabled until a fresh "
                "post-build content-completion acceptance report exists."
            ),
        }
    except Exception as exception:  # pragma: no cover - diagnostic entrypoint
        result = {"status": "failed", "generatedAtUtc": utc_now(), "error": str(exception)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    # PyInstaller --windowed sets stdout to None on Windows. When a report path
    # is supplied the JSON file is authoritative, so avoid a second encoded copy
    # passing through PowerShell's native-output decoder.
    if output is None and sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
        print(text)
    return 0 if result["status"] == "passed" else 1


def run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    repository = locate_repository()
    backend = TerminalBackend(repository)
    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    projects: list[UnityProject] = []

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1240x820")
    root.minsize(1040, 700)
    root.configure(bg="#0b1520")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TFrame", background="#0b1520")
    style.configure("Panel.TFrame", background="#132333")
    style.configure("TLabel", background="#0b1520", foreground="#dceaf1", font=("Microsoft YaHei UI", 10))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"), foreground="#48dfd2")
    style.configure("Section.TLabel", font=("Microsoft YaHei UI", 12, "bold"), foreground="#6fb8ff")
    style.configure("Status.TLabel", background="#132333", foreground="#dceaf1", font=("Microsoft YaHei UI", 10))
    style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(9, 7))
    style.configure("TCombobox", padding=6)

    outer = ttk.Frame(root, padding=18)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="Unity Scene Agent Terminal", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        outer,
        text="选择项目/Scene → AI 解析并编译契约 → 隔离注入 Brain + 运动技能 → 第一人称体验 → 导出证据",
    ).pack(anchor="w", pady=(2, 14))

    selector = ttk.Frame(outer, style="Panel.TFrame", padding=14)
    selector.pack(fill="x")
    selector.columnconfigure(1, weight=1)
    ttk.Label(selector, text="Unity 项目", style="Status.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
    project_var = tk.StringVar()
    project_box = ttk.Combobox(selector, textvariable=project_var, state="readonly")
    project_box.grid(row=0, column=1, sticky="ew")
    browse_button = ttk.Button(selector, text="添加项目目录…")
    browse_button.grid(row=0, column=2, padx=(10, 0))
    ttk.Label(selector, text="Scene / 场景", style="Status.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
    scene_var = tk.StringVar()
    scene_box = ttk.Combobox(selector, textvariable=scene_var, state="readonly")
    scene_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0))

    ttk.Label(selector, text="Decision Brain", style="Status.TLabel").grid(
        row=2, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
    decision_mode_var = tk.StringVar(value=DECISION_BRAIN_MODE_LABELS["rule"])
    decision_mode_box = ttk.Combobox(
        selector,
        textvariable=decision_mode_var,
        values=[DECISION_BRAIN_MODE_LABELS[item] for item in DECISION_BRAIN_MODES],
        state="readonly",
    )
    decision_mode_box.grid(row=2, column=1, sticky="ew", pady=(10, 0))
    research_override_var = tk.BooleanVar(value=False)
    research_override_check = ttk.Checkbutton(
        selector,
        text="显式允许 smoke-only 模型闭环（研究）",
        variable=research_override_var,
    )
    research_override_check.grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(10, 0))

    status_var = tk.StringVar(value="正在扫描 Unity 项目……")
    detail_var = tk.StringVar(value="")
    ttk.Label(selector, textvariable=status_var, style="Section.TLabel").grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(14, 2))
    ttk.Label(selector, textvariable=detail_var, style="Status.TLabel", wraplength=1020).grid(
        row=4, column=0, columnspan=3, sticky="w")
    progress_bar = ttk.Progressbar(selector, mode="indeterminate")
    progress_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    progress_bar.grid_remove()

    buttons = ttk.Frame(outer, padding=(0, 14))
    buttons.pack(fill="x")
    analyze_button = ttk.Button(buttons, text="1  AI 解析 + 生成契约")
    analyze_button.pack(side="left", padx=(0, 8))
    build_button = ttk.Button(buttons, text="2  隔离构建 Agent")
    build_button.pack(side="left", padx=8)
    launch_button = ttk.Button(buttons, text="3  启动 Agent 体验")
    launch_button.pack(side="left", padx=8)
    test_button = ttk.Button(buttons, text="内容完成复验")
    test_button.pack(side="left", padx=8)
    export_button = ttk.Button(buttons, text="4  导出结果 ZIP")
    export_button.pack(side="left", padx=8)
    open_button = ttk.Button(buttons, text="打开结果目录")
    open_button.pack(side="left", padx=8)

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True, pady=(4, 0))
    report_tab = ttk.Frame(notebook, padding=4)
    log_tab = ttk.Frame(notebook, padding=4)
    notebook.add(report_tab, text="AI 场景理解报告")
    notebook.add(log_tab, text="运行 / 构建日志")
    report_text = tk.Text(
        report_tab, height=16, bg="#071019", fg="#dceaf1", insertbackground="white",
        relief="flat", font=("Microsoft YaHei UI", 10), padx=12, pady=10, wrap="word",
    )
    report_text.pack(fill="both", expand=True)
    report_text.insert("1.0", "请选择场景并执行第 1 步。这里会显示 LLM/确定性回退生成的完整场景理解报告。")
    report_text.configure(state="disabled")
    log_text = tk.Text(
        log_tab, height=16, bg="#071019", fg="#cbe4ee", insertbackground="white",
        relief="flat", font=("Cascadia Mono", 9), padx=12, pady=10, wrap="word",
    )
    log_text.pack(fill="both", expand=True)
    footer_var = tk.StringVar(value=(
        "运动技能：EyeNavGS Normal HMD v4（HeadLocal + root-speed prior）  |  "
        "大脑：Rule / Shadow / Model research（显式门控）  |  交互：白名单 + Unity 状态验证"
    ))
    ttk.Label(outer, textvariable=footer_var).pack(anchor="w", pady=(9, 0))

    active_task: dict[str, object] | None = None

    def write_log_line(message: str) -> None:
        log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        log_text.see("end")

    def append_log(message: str) -> None:
        event_queue.put(("log", f"[{datetime.now().strftime('%H:%M:%S')}] {message}"))

    def show_active_task(*, heartbeat: bool = False) -> None:
        if active_task is None:
            return
        started_at = float(active_task["started_at"])
        elapsed = format_elapsed(time.monotonic() - started_at)
        task_name = str(active_task["name"])
        subject = str(active_task["subject"])
        status_var.set(f"⏳ {task_name}正在后台运行 · 已用 {elapsed}")
        detail_var.set(
            f"正在处理：{subject}\n"
            "界面仍可操作；可切换项目/Scene 或打开结果目录。任务完成或失败后按钮会自动恢复。"
        )
        if heartbeat:
            write_log_line(f"[进度] {task_name}仍在运行，已用 {elapsed}。")

    def apply_active_task_locks() -> None:
        if active_task is None:
            return
        for button in active_task["locked_buttons"]:  # type: ignore[union-attr]
            button.configure(state="disabled")

    def selected() -> tuple[UnityProject, SceneChoice] | None:
        project_index = project_box.current()
        scene_index = scene_box.current()
        if project_index < 0 or scene_index < 0 or project_index >= len(projects):
            return None
        project = projects[project_index]
        if scene_index >= len(project.scenes):
            return None
        return project, project.scenes[scene_index]

    def selected_decision_mode() -> tuple[str, bool]:
        label = decision_mode_var.get()
        mode = next(
            (key for key, value in DECISION_BRAIN_MODE_LABELS.items() if value == label),
            "rule",
        )
        return mode, bool(research_override_var.get()) if mode == "model" else False

    def update_report_view(project: UnityProject, scene: SceneChoice) -> None:
        report_path = backend.latest_analysis_report(project, scene)
        if report_path is None:
            contents = "尚未找到这个项目/Scene 的 AI 场景理解报告。\n\n请执行第 1 步。"
        else:
            try:
                contents = report_path.read_text(encoding="utf-8-sig", errors="replace")
                contents = f"报告文件：{report_path}\n\n{contents}"
            except OSError as exception:
                contents = f"无法读取报告：{exception}"
        report_text.configure(state="normal")
        report_text.delete("1.0", "end")
        report_text.insert("1.0", contents)
        report_text.configure(state="disabled")

    def update_scene_details(*_args: object) -> None:
        item = selected()
        if item is None:
            return
        project, scene = item
        decision_mode, research_override = selected_decision_mode()
        if decision_mode != "model" and research_override_var.get():
            research_override_var.set(False)
            research_override = False
        analysis_ready = backend.latest_analysis_record(project, scene) is not None
        labels = {"agent_ready": "✓ Agent 可体验", "analysis_only": "△ 待 AI 解析", "blocked": "× 暂不可运行"}
        state_label = labels.get(scene.support, scene.support)
        if scene.support == "analysis_only" and analysis_ready:
            state_label = "◇ AI 契约已生成 · 待隔离构建"
        status_var.set(f"{state_label}  ·  Unity {project.unity_version}")
        buildable, build_reason = backend.generic_build_eligibility(
            project, scene, decision_mode, research_override
        )
        mode_record = None
        if (
            analysis_ready
            and scene.id.startswith("Assets/")
            and scene.runtime_kind in {"", "generic"}
        ):
            try:
                mode_record = backend.generic_build_record(
                    project,
                    scene,
                    decision_brain_mode=decision_mode,
                    decision_brain_research_override=research_override,
                )
            except RuntimeError:
                mode_record = None
        detail = scene.support_reason
        if analysis_ready:
            detail = build_reason
        if mode_record is not None:
            state_label = f"✓ {DECISION_BRAIN_MODE_LABELS[decision_mode]} 已验收"
            detail = (
                f"当前 Player 已按 {decision_mode} 模式完成场景内容验收。"
                if decision_mode != "model"
                else "当前 Player 已按 Model research 完成场景内容研究验收；不是生产策略。"
            )
            scene.support = "agent_ready"
            scene.runtime_kind = "generic"
            scene.launcher = str(mode_record["executable"])
            scene.build_status = str(mode_record["statusPath"])
        detail_var.set(f"{project.path}\n{detail}")
        if scene.runtime_kind == "generic" or (
            scene.id.startswith("Assets/") and scene.runtime_kind == ""
        ):
            runnable = mode_record is not None
        else:
            runnable = scene.support == "agent_ready" and analysis_ready and decision_mode == "rule"
        build_button.configure(state="normal" if buildable else "disabled")
        launch_button.configure(state="normal" if runnable else "disabled")
        test_button.configure(state="normal" if runnable else "disabled")
        update_report_view(project, scene)
        research_override_check.configure(state="normal" if decision_mode == "model" else "disabled")
        if active_task is not None:
            apply_active_task_locks()
            show_active_task()

    def update_project(*_args: object) -> None:
        index = project_box.current()
        if index < 0 or index >= len(projects):
            return
        scene_box["values"] = [scene.label for scene in projects[index].scenes]
        scene_box.current(0)
        update_scene_details()

    def set_projects(new_projects: list[UnityProject]) -> None:
        nonlocal projects
        projects = new_projects
        project_box["values"] = [f"{project.name}  —  {project.path}" for project in projects]
        if projects:
            generic_choice = next((
                (project_index, scene_index)
                for project_index, project in enumerate(projects)
                for scene_index, scene in enumerate(project.scenes)
                if scene.runtime_kind == "generic" and scene.id.startswith("Assets/")
            ), None)
            preferred = (
                generic_choice[0] if generic_choice is not None
                else next((i for i, p in enumerate(projects) if p.name == "UnityNormalHmdAgentDemo"), 0)
            )
            project_box.current(preferred)
            update_project()
            if generic_choice is not None:
                scene_box.current(generic_choice[1])
                update_scene_details()
        append_log(f"发现 {len(projects)} 个 Unity 项目。终端不会修改这些源项目。")

    action_buttons = {
        "analyze": analyze_button,
        "build": build_button,
        "launch": launch_button,
        "test": test_button,
        "export": export_button,
    }

    def run_background(
        task_name: str,
        project: UnityProject,
        scene: SceneChoice,
        label: str,
        action: Callable[[], object],
    ) -> None:
        nonlocal active_task
        if active_task is not None:
            show_active_task()
            write_log_line(f"[提示] {active_task['name']}仍在运行；请等待当前任务完成。")
            return

        task_id = time.monotonic_ns()
        locked_buttons = tuple(action_buttons[name] for name in BACKGROUND_CONFLICTING_ACTIONS)
        saved_states = tuple((button, str(button.cget("state"))) for button in locked_buttons)
        active_task = {
            "id": task_id,
            "name": task_name,
            "subject": f"{project.name} / {scene.label}",
            "project": project,
            "scene": scene,
            "started_at": time.monotonic(),
            "last_heartbeat_at": time.monotonic(),
            "locked_buttons": locked_buttons,
            "saved_states": saved_states,
        }
        apply_active_task_locks()
        progress_bar.grid()
        progress_bar.start(50)
        notebook.select(log_tab)
        write_log_line(label)
        write_log_line("[进度] 后台任务已启动；等待外部解析器输出，界面不会被阻塞。")
        show_active_task()
        root.update_idletasks()

        def worker() -> None:
            try:
                result = action()
                event_queue.put(("done", (task_id, result)))
            except Exception as exception:
                message = str(exception).strip() or repr(exception)
                event_queue.put(("error", (task_id, message)))

        threading.Thread(target=worker, daemon=True).start()

    def restore_buttons(task_id: int) -> dict[str, object] | None:
        nonlocal active_task
        if active_task is None or active_task.get("id") != task_id:
            return None
        finished_task = active_task
        active_task = None
        progress_bar.stop()
        progress_bar.grid_remove()
        for button, state in finished_task["saved_states"]:  # type: ignore[union-attr]
            button.configure(state=state)
        update_scene_details()
        return finished_task

    def on_analyze() -> None:
        item = selected()
        if item:
            project, scene = item
            run_background(
                "AI 解析与契约编译",
                project,
                scene,
                "开始只读 AI 解析与契约编译……",
                lambda: backend.analyze(project, scene, append_log),
            )

    def on_build() -> None:
        item = selected()
        if item:
            project, scene = item
            decision_mode, research_override = selected_decision_mode()
            run_background(
                "隔离构建与内容完成验收",
                project,
                scene,
                "准备隔离构建并执行内容完成闭环验收……",
                lambda: backend.build_generic(
                    project, scene, append_log, decision_mode, research_override
                ),
            )

    def on_launch(automated: bool) -> None:
        item = selected()
        if item:
            project, scene = item
            decision_mode, research_override = selected_decision_mode()
            # Close any native ttk combobox popdown before Unity creates its
            # window.  On Windows an unposted popup is a separate top-level
            # surface and can otherwise remain above the Player, looking like
            # a Unity rendering defect.  Minimise the terminal for the visible
            # experience so the first-person view receives focus.
            for combo in (project_box, scene_box, decision_mode_box):
                try:
                    combo.event_generate("<Escape>")
                except tk.TclError:
                    pass
            root.focus_set()
            root.update_idletasks()
            if not automated:
                root.iconify()
            run_background(
                "内容完成复验" if automated else "Agent 体验",
                project,
                scene,
                "准备 Agent 运行……",
                lambda: backend.launch(
                    project,
                    scene,
                    automated,
                    append_log,
                    decision_mode,
                    research_override,
                ),
            )

    def on_export() -> None:
        item = selected()
        if item:
            project, scene = item
            run_background(
                "结果导出",
                project,
                scene,
                "整理运行证据……",
                lambda: backend.export(project, scene, append_log),
            )

    def on_browse() -> None:
        selected_path = filedialog.askdirectory(title="选择 Unity 项目或其上级目录")
        if not selected_path:
            return
        set_projects(backend.refresh(Path(selected_path)))

    def on_open() -> None:
        os.startfile(backend.results_root)  # type: ignore[attr-defined]

    def poll_events() -> None:
        try:
            while True:
                kind, payload = event_queue.get_nowait()
                if kind == "log":
                    log_text.insert("end", str(payload) + "\n")
                    log_text.see("end")
                elif kind == "done":
                    task_id, result = payload  # type: ignore[misc]
                    finished_task = restore_buttons(int(task_id))
                    if finished_task is None:
                        continue
                    elapsed = format_elapsed(time.monotonic() - float(finished_task["started_at"]))
                    write_log_line(f"[完成] {finished_task['name']}，总耗时 {elapsed}：{result}")
                    item = selected()
                    completed_project = finished_task["project"]
                    completed_scene = finished_task["scene"]
                    same_selection = (
                        item is not None
                        and isinstance(completed_project, UnityProject)
                        and isinstance(completed_scene, SceneChoice)
                        and backend.scene_key(*item) == backend.scene_key(completed_project, completed_scene)
                    )
                    if same_selection:
                        update_report_view(completed_project, completed_scene)
                    elif isinstance(result, Path) and result.name.endswith("_index.json"):
                        write_log_line(
                            f"[提示] {finished_task['subject']} 的报告已生成；切回该 Scene 即可查看。"
                        )
                    if (
                        same_selection
                        and isinstance(result, Path)
                        and result.name.endswith("_index.json")
                    ):
                        notebook.select(report_tab)
                elif kind == "error":
                    task_id, error = payload  # type: ignore[misc]
                    root.deiconify()
                    root.lift()
                    finished_task = restore_buttons(int(task_id))
                    if finished_task is None:
                        continue
                    elapsed = format_elapsed(time.monotonic() - float(finished_task["started_at"]))
                    write_log_line(f"[失败] {finished_task['name']}（{elapsed}）：{error}")
                    status_var.set(f"× {finished_task['name']}失败 · 已恢复按钮")
                    detail_var.set(str(error))
                    messagebox.showerror(APP_TITLE, str(error))
        except queue.Empty:
            pass
        except Exception as exception:  # keep the UI event pump alive after an unexpected handler error
            try:
                write_log_line(f"[界面错误] {exception}")
                if active_task is not None:
                    status_var.set("⚠ 界面反馈异常 · 后台任务仍受保护")
                    detail_var.set(
                        f"{exception}\n后台任务可能仍在运行，因此没有提前解锁冲突按钮。"
                    )
            except Exception:
                pass
        finally:
            try:
                if active_task is not None:
                    now = time.monotonic()
                    last_heartbeat = float(active_task["last_heartbeat_at"])
                    if now - last_heartbeat >= BACKGROUND_HEARTBEAT_SECONDS:
                        active_task["last_heartbeat_at"] = now
                        show_active_task(heartbeat=True)
            except (tk.TclError, ValueError, TypeError):
                pass
            try:
                root.after(120, poll_events)
            except tk.TclError:
                pass

    project_box.bind("<<ComboboxSelected>>", update_project)
    scene_box.bind("<<ComboboxSelected>>", update_scene_details)
    decision_mode_box.bind("<<ComboboxSelected>>", update_scene_details)
    research_override_check.configure(command=update_scene_details)
    analyze_button.configure(command=on_analyze)
    build_button.configure(command=on_build)
    launch_button.configure(command=lambda: on_launch(False))
    test_button.configure(command=lambda: on_launch(True))
    export_button.configure(command=on_export)
    open_button.configure(command=on_open)
    browse_button.configure(command=on_browse)

    set_projects(backend.refresh())
    poll_events()
    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--decision-brain-mode", choices=DECISION_BRAIN_MODES, default="rule")
    parser.add_argument("--decision-brain-model", type=Path)
    parser.add_argument("--decision-brain-research-override", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test(
            args.self_test_output,
            args.decision_brain_mode,
            args.decision_brain_model,
            args.decision_brain_research_override,
        )
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
