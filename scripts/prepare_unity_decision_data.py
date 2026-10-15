#!/usr/bin/env python3
"""Validate and prepare Unity Decision Brain decision contexts.

Production inputs must be Unity-emitted JSONL using
``unity-decision-context-1.0`` and ``unity-decision-outcome-1.0``.  This script
never invents records.  Explicit synthetic fixtures are accepted only with
``--allow-synthetic-fixture`` so unit tests cannot be mistaken for research
data.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from unity_decision_policy import (
    DEFAULT_CONFIG,
    candidate_feature_names,
    count_human_decisions,
    global_feature_names,
    load_config,
    write_json_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "features" / "unity_decision_brain_v0"


class PreparationError(ValueError):
    pass


FLAT_CONTEXT_REQUIRED = {
    "schemaVersion",
    "utc",
    "episodeId",
    "decisionId",
    "projectId",
    "sceneName",
    "seed",
    "sourceKind",
    "humanDemonstration",
    "simulatedTime",
    "decisionKind",
    "state",
    "previousSkill",
    "bodyPosition",
    "bodyYawDegrees",
    "coverage",
    "replans",
    "plannerMode",
    "candidates",
    "validSkills",
    "chosenTargetIndex",
    "chosenTargetId",
    "chosenSkill",
    "labelSource",
    "humanAccepted",
    "humanCorrectionTargetId",
    "humanCorrectionSkill",
}
FLAT_CANDIDATE_REQUIRED = {
    "targetId",
    "semanticType",
    "priority",
    "relativeX",
    "relativeY",
    "relativeZ",
    "distance",
    "bearingSin",
    "bearingCos",
    "resolved",
    "reachabilityKnown",
    "reachable",
    "visibilityKnown",
    "visible",
    "navMeshProjected",
    "visited",
    "cooldownSeconds",
    "interactionAvailable",
    "interactionRequired",
    "riskCandidate",
    "baseUtility",
    "randomJitter",
    "totalUtility",
}
FLAT_OUTCOME_REQUIRED = {
    "schemaVersion",
    "utc",
    "episodeId",
    "decisionId",
    "projectId",
    "sceneName",
    "seed",
    "simulatedTime",
    "elapsedSeconds",
    "chosenTargetId",
    "chosenSkill",
    "reached",
    "visited",
    "interactionAttempted",
    "interactionVerified",
    "timedOut",
    "collisionRecoveriesDelta",
    "coverageBefore",
    "coverageAfter",
    "coverageGain",
    "terminalState",
    "evidence",
}


@dataclass(frozen=True)
class PreparedExample:
    episode_id: str
    decision_id: str
    group_id: str
    source_kind: str
    human_demonstration: bool
    decision_kind: str
    global_features: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    target_mask: np.ndarray
    target_label_available: bool
    skill_mask: np.ndarray
    target_index: int
    skill_index: int
    teacher_utilities: np.ndarray
    sample_weight: float
    outcome_reached: bool
    outcome_interaction_verified: bool
    outcome_coverage_gain: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick(mapping: dict[str, Any] | None, *names: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _required_text(value: Any, label: str) -> str:
    if value is None or not str(value).strip():
        raise PreparationError(f"{label} must be a non-empty string")
    return str(value).strip()


def _finite_float(value: Any, label: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PreparationError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return int(default)
    if isinstance(value, bool):
        raise PreparationError(f"{label} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{label} must be an integer") from exc
    return result


def _boolean(value: Any, label: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return bool(default)
    if type(value) is not bool:
        raise PreparationError(f"{label} must be a JSON boolean")
    return value


def _optional_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _normal_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _require_keys(mapping: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise PreparationError(f"{location}: canonical flat record is missing {missing}")


def validate_flat_context_shape(raw: dict[str, Any], location: str) -> None:
    """Require every field emitted by the canonical flat C# logging contract."""

    if "schemaVersion" not in raw:
        return
    _require_keys(raw, FLAT_CONTEXT_REQUIRED, location)
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PreparationError(f"{location}.candidates must be a non-empty array")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise PreparationError(f"{location}.candidates[{index}] must be an object")
        _require_keys(candidate, FLAT_CANDIDATE_REQUIRED, f"{location}.candidates[{index}]")


def validate_flat_outcome_shape(raw: dict[str, Any], location: str) -> None:
    if "schemaVersion" in raw:
        _require_keys(raw, FLAT_OUTCOME_REQUIRED, location)


def _enum_value(value: Any, vocabulary: list[str], unknown: str) -> str:
    wanted = _normal_key(value)
    for candidate in vocabulary:
        if _normal_key(candidate) == wanted:
            return candidate
    return unknown


def _read_jsonl(paths: Iterable[Path], label: str) -> list[tuple[dict[str, Any], str]]:
    records: list[tuple[dict[str, Any], str]] = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} JSONL is missing: {path}")
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreparationError(
                        f"Invalid JSON in {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PreparationError(f"{path}:{line_number} is not a JSON object")
                records.append((value, f"{path}:{line_number}"))
    if not records:
        raise PreparationError(f"No {label} records were found")
    return records


def normalize_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the canonical flat C# schema and the earlier nested draft."""

    episode = _pick(raw, "episode", default={})
    provenance = _pick(raw, "provenance", default={})
    observation = _pick(raw, "observation", default={})
    global_state = _pick(observation, "global", default={})
    allowed = _pick(raw, "allowed", default={})
    label = _pick(raw, "label", default={})

    schema_version = _pick(raw, "schemaVersion", "schema_version")
    candidates = _pick(raw, "candidates")
    if candidates is None:
        candidates = _pick(observation, "candidates")
    valid_skills = _pick(raw, "validSkills", "valid_skills")
    if valid_skills is None:
        valid_skills = _pick(allowed, "validSkills", "valid_skills", "skills")

    return {
        "schemaVersion": schema_version,
        "utc": _pick(raw, "utc", default=_pick(episode, "utc")),
        "episodeId": _pick(raw, "episodeId", default=_pick(episode, "episodeId", "episode_id")),
        "decisionId": _pick(raw, "decisionId", default=_pick(label, "decisionId", "decision_id")),
        "projectId": _pick(raw, "projectId", default=_pick(episode, "projectId", "project_id")),
        "sceneName": _pick(raw, "sceneName", default=_pick(episode, "sceneName", "scene_name", "scene")),
        "projectFamily": _pick(raw, "projectFamily", default=_pick(episode, "projectFamily", "project_family")),
        "sceneFamily": _pick(raw, "sceneFamily", default=_pick(episode, "sceneFamily", "scene_family")),
        "splitFamily": _pick(raw, "splitFamily", default=_pick(episode, "splitFamily", "split_family")),
        "seed": _pick(raw, "seed", default=_pick(episode, "seed")),
        "sourceKind": _pick(raw, "sourceKind", default=_pick(provenance, "sourceKind", "source_kind")),
        "humanDemonstration": _pick(raw, "humanDemonstration", default=_pick(provenance, "humanDemonstration", "human_demonstration")),
        "syntheticFixture": _pick(raw, "syntheticFixture", "synthetic_fixture", default=False),
        "simulatedTime": _pick(raw, "simulatedTime", default=_pick(global_state, "simulatedTime", "simulated_time", default=0.0)),
        "brainState": _pick(raw, "brainState", "state", default=_pick(global_state, "brainState", "brain_state", "state", default="Unknown")),
        "decisionKind": _pick(raw, "decisionKind", "decision_kind", default="target_selection"),
        "previousSkill": _pick(raw, "previousSkill", default=_pick(global_state, "previousSkill", "previous_skill", default="Unknown")),
        "bodyPosition": _pick(raw, "bodyPosition", default=_pick(global_state, "bodyPosition", "body_position")),
        "bodyYawDegrees": _pick(raw, "bodyYawDegrees", default=_pick(global_state, "bodyYawDegrees", "body_yaw_degrees", default=0.0)),
        "coverage": _pick(raw, "coverage", default=_pick(global_state, "coverage", default=0.0)),
        "replans": _pick(raw, "replans", default=_pick(global_state, "replans", default=0)),
        "plannerMode": _pick(raw, "plannerMode", default=_pick(global_state, "plannerMode", "planner_mode", default="unknown")),
        "candidates": candidates,
        "validSkills": valid_skills,
        "chosenTargetIndex": _pick(raw, "chosenTargetIndex", default=_pick(label, "targetIndex", "target_index")),
        "chosenTargetId": _pick(raw, "chosenTargetId", default=_pick(label, "targetId", "target_id")),
        "chosenSkill": _pick(raw, "chosenSkill", default=_pick(label, "skill")),
        "labelSource": _pick(raw, "labelSource", default=_pick(label, "labelSource", "label_source")),
        "humanAccepted": _pick(raw, "humanAccepted", default=_pick(label, "humanAccepted", "human_accepted")),
        "humanCorrectionTargetId": _pick(raw, "humanCorrectionTargetId", default=_pick(label, "humanCorrectionTargetId", "human_correction_target_id")),
        "humanCorrectionSkill": _pick(raw, "humanCorrectionSkill", default=_pick(label, "humanCorrectionSkill", "human_correction_skill")),
    }


def normalize_outcome(raw: dict[str, Any]) -> dict[str, Any]:
    outcome = _pick(raw, "outcome", default={})
    episode = _pick(raw, "episode", default={})
    merged = dict(outcome) if isinstance(outcome, dict) else {}
    merged.update({key: value for key, value in raw.items() if key != "outcome"})
    return {
        "schemaVersion": _pick(merged, "schemaVersion", "schema_version"),
        "utc": _pick(merged, "utc"),
        "episodeId": _pick(merged, "episodeId", "episode_id", default=_pick(episode, "episodeId", "episode_id")),
        "decisionId": _pick(merged, "decisionId", "decision_id"),
        "projectId": _pick(merged, "projectId", "project_id", default=_pick(episode, "projectId", "project_id")),
        "sceneName": _pick(merged, "sceneName", "scene_name", default=_pick(episode, "sceneName", "scene_name")),
        "seed": _pick(merged, "seed", default=_pick(episode, "seed")),
        "simulatedTime": _pick(merged, "simulatedTime", "simulated_time", default=0.0),
        "chosenTargetId": _pick(merged, "chosenTargetId", "chosen_target_id", default=""),
        "chosenSkill": _pick(merged, "chosenSkill", "chosen_skill", default=""),
        "reached": _pick(merged, "reached", default=False),
        "visited": _pick(merged, "visited", default=False),
        "interactionAttempted": _pick(merged, "interactionAttempted", "interaction_attempted", default=False),
        "interactionVerified": _pick(merged, "interactionVerified", "interaction_verified", default=False),
        "timedOut": _pick(merged, "timedOut", "timed_out", default=False),
        "collisionRecoveriesDelta": _pick(merged, "collisionRecoveriesDelta", "collision_recoveries_delta", default=0),
        "coverageBefore": _pick(merged, "coverageBefore", "coverage_before", default=0.0),
        "coverageAfter": _pick(merged, "coverageAfter", "coverage_after", default=0.0),
        "coverageGain": _pick(merged, "coverageGain", "coverage_gain"),
        "elapsedSeconds": _pick(merged, "elapsedSeconds", "elapsed_seconds", default=0.0),
        "terminalState": _pick(merged, "terminalState", "terminal_state", default=""),
        "evidence": _pick(merged, "evidence", default=""),
        "syntheticFixture": _pick(merged, "syntheticFixture", "synthetic_fixture", default=False),
    }


def normalize_candidate(raw: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PreparationError(f"candidate[{index}] must be a JSON object")
    path_length_value = _pick(raw, "pathLength", "path_length")
    confidence_value = _pick(raw, "analysisConfidence", "analysis_confidence")
    return {
        "targetId": _pick(raw, "targetId", "target_id"),
        "semanticType": _pick(raw, "semanticType", "semantic_type", default="unknown"),
        "priority": _pick(raw, "priority", default=0.0),
        "relativeX": _pick(raw, "relativeX", "relative_x", default=0.0),
        "relativeY": _pick(raw, "relativeY", "relative_y", default=0.0),
        "relativeZ": _pick(raw, "relativeZ", "relative_z", default=0.0),
        "distance": _pick(raw, "distance", default=0.0),
        "bearingSin": _pick(raw, "bearingSin", "bearing_sin", default=0.0),
        "bearingCos": _pick(raw, "bearingCos", "bearing_cos", default=1.0),
        "resolved": _pick(raw, "resolved", default=False),
        "reachabilityKnown": _pick(raw, "reachabilityKnown", "reachability_known", default=False),
        "reachable": _pick(raw, "reachable", default=False),
        "visibilityKnown": _pick(raw, "visibilityKnown", "visibility_known", default=False),
        "visible": _pick(raw, "visible", default=False),
        "navMeshProjected": _pick(raw, "navMeshProjected", "nav_mesh_projected", default=False),
        "visited": _pick(raw, "visited", default=False),
        "cooldownSeconds": _pick(raw, "cooldownSeconds", "cooldown_seconds", default=0.0),
        "interactionDeclared": _pick(raw, "interactionDeclared", "interaction_declared", default=False),
        "interactionAvailable": _pick(raw, "interactionAvailable", "interaction_available", default=False),
        "interactionRequired": _pick(raw, "interactionRequired", "interaction_required", default=False),
        "interactionAttempted": _pick(raw, "interactionAttempted", "interaction_attempted", default=False),
        "interactionVerified": _pick(raw, "interactionVerified", "interaction_verified", default=False),
        "riskCandidate": _pick(raw, "riskCandidate", "risk_candidate", default=False),
        "selectable": _pick(raw, "selectable"),
        "baseUtility": _pick(raw, "baseUtility", "base_utility"),
        "randomJitter": _pick(raw, "randomJitter", "random_jitter"),
        "totalUtility": _pick(raw, "totalUtility", "total_utility"),
        "pathLength": path_length_value,
        "dynamicObject": _pick(raw, "dynamicObject", "dynamic_object", default=False),
        "analysisConfidence": confidence_value,
    }


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if value == candidate else 0.0 for candidate in vocabulary]


def encode_global(context: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    state = _enum_value(context["brainState"], config["brain_states"], "Unknown")
    previous = _enum_value(context["previousSkill"], config["skills"], "Unknown")
    if previous == "Unknown" and "Unknown" not in config["skills"]:
        previous_vocabulary = [*config["skills"], "Unknown"]
    else:
        previous_vocabulary = [*config["skills"], "Unknown"]
    planner = _enum_value(context["plannerMode"], config["planner_modes"], "unknown")
    values = [
        _finite_float(context["simulatedTime"], "simulatedTime"),
        _finite_float(context["coverage"], "coverage"),
        float(_integer(context["replans"], "replans")),
    ]
    if not 0.0 <= values[1] <= 1.0 + 1.0e-6:
        raise PreparationError("coverage must be in [0, 1]")
    values.extend(_one_hot(state, config["brain_states"]))
    values.extend(_one_hot(previous, previous_vocabulary))
    values.extend(_one_hot(planner, config["planner_modes"]))
    array = np.asarray(values, dtype=np.float32)
    expected = len(global_feature_names(config))
    if array.shape != (expected,):
        raise AssertionError(f"Global encoder produced {array.shape}, expected {(expected,)}")
    return array


def encode_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    numeric = [
        _finite_float(candidate["priority"], "candidate.priority"),
        _finite_float(candidate["relativeX"], "candidate.relativeX"),
        _finite_float(candidate["relativeY"], "candidate.relativeY"),
        _finite_float(candidate["relativeZ"], "candidate.relativeZ"),
        _finite_float(candidate["distance"], "candidate.distance"),
        _finite_float(candidate["bearingSin"], "candidate.bearingSin"),
        _finite_float(candidate["bearingCos"], "candidate.bearingCos"),
        _finite_float(candidate["cooldownSeconds"], "candidate.cooldownSeconds"),
    ]
    boolean_values = [
        _boolean(candidate["resolved"], "candidate.resolved"),
        _boolean(candidate["reachabilityKnown"], "candidate.reachabilityKnown"),
        _boolean(candidate["reachable"], "candidate.reachable"),
        _boolean(candidate["visibilityKnown"], "candidate.visibilityKnown"),
        _boolean(candidate["visible"], "candidate.visible"),
        _boolean(candidate["navMeshProjected"], "candidate.navMeshProjected"),
        _boolean(candidate["visited"], "candidate.visited"),
        _boolean(candidate["interactionDeclared"], "candidate.interactionDeclared"),
        _boolean(candidate["interactionAvailable"], "candidate.interactionAvailable"),
        _boolean(candidate["interactionRequired"], "candidate.interactionRequired"),
        _boolean(candidate["interactionAttempted"], "candidate.interactionAttempted"),
        _boolean(candidate["interactionVerified"], "candidate.interactionVerified"),
        _boolean(candidate["riskCandidate"], "candidate.riskCandidate"),
    ]
    semantic = _enum_value(candidate["semanticType"], config["semantic_types"], "unknown")
    values = [*numeric, *(float(value) for value in boolean_values)]
    values.extend(_one_hot(semantic, config["semantic_types"]))
    array = np.asarray(values, dtype=np.float32)
    expected = len(candidate_feature_names(config))
    if array.shape != (expected,):
        raise AssertionError(f"Candidate encoder produced {array.shape}, expected {(expected,)}")
    return array


def stop_candidate() -> dict[str, Any]:
    return {
        "index": -1,
        "targetId": "__STOP__",
        "semanticType": "__stop__",
        "priority": 0.0,
        "relativeX": 0.0,
        "relativeY": 0.0,
        "relativeZ": 0.0,
        "distance": 0.0,
        "bearingSin": 0.0,
        "bearingCos": 1.0,
        "resolved": True,
        "reachabilityKnown": True,
        "reachable": True,
        "visibilityKnown": True,
        "visible": True,
        "navMeshProjected": False,
        "visited": False,
        "cooldown": False,
        "cooldownSeconds": 0.0,
        "interactionDeclared": False,
        "interactionAvailable": False,
        "interactionRequired": False,
        "interactionAttempted": False,
        "interactionVerified": False,
        "riskCandidate": False,
        "selectable": True,
        "ineligibleReason": "",
        "utilityIncludesNewJitter": False,
        "baseUtility": None,
        "randomJitter": None,
        "totalUtility": None,
        "pathLength": None,
        "dynamicObject": False,
        "analysisConfidence": None,
    }


def candidate_is_legal(candidate: dict[str, Any]) -> bool:
    if candidate.get("selectable") is not None:
        return _boolean(candidate["selectable"], "candidate.selectable")
    if _boolean(candidate["visited"], "candidate.visited"):
        return False
    if _finite_float(candidate["cooldownSeconds"], "candidate.cooldownSeconds") > 0.0:
        return False
    known = _boolean(candidate["reachabilityKnown"], "candidate.reachabilityKnown")
    reachable = _boolean(candidate["reachable"], "candidate.reachable")
    return known and reachable


def preprocessing_spec(config: dict[str, Any]) -> dict[str, Any]:
    stop = stop_candidate()
    return {
        "schema_version": "unity-decision-preprocessing-1.0",
        "append_stop_candidate": bool(config.get("append_stop_candidate", True)),
        "global_numeric": [
            {"feature": "simulated_time", "field": "simulatedTime"},
            {"feature": "coverage", "field": "coverage"},
            {"feature": "replans", "field": "replans"},
        ],
        "global_one_hot": [
            {
                "field": "state",
                "feature_prefix": "brain_state::",
                "vocabulary": list(config["brain_states"]),
                "unknown": "Unknown",
            },
            {
                "field": "previousSkill",
                "feature_prefix": "previous_skill::",
                "vocabulary": [*config["skills"], "Unknown"],
                "unknown": "Unknown",
            },
            {
                "field": "plannerMode",
                "feature_prefix": "planner_mode::",
                "vocabulary": list(config["planner_modes"]),
                "unknown": "unknown",
            },
        ],
        "candidate_numeric": [
            {"feature": feature, "field": field}
            for feature, field in (
                ("priority", "priority"),
                ("relative_x", "relativeX"),
                ("relative_y", "relativeY"),
                ("relative_z", "relativeZ"),
                ("distance", "distance"),
                ("bearing_sin", "bearingSin"),
                ("bearing_cos", "bearingCos"),
                ("cooldown_seconds", "cooldownSeconds"),
            )
        ],
        "candidate_boolean": [
            {"feature": feature, "field": field}
            for feature, field in (
                ("resolved", "resolved"),
                ("reachability_known", "reachabilityKnown"),
                ("reachable", "reachable"),
                ("visibility_known", "visibilityKnown"),
                ("visible", "visible"),
                ("navmesh_projected", "navMeshProjected"),
                ("visited", "visited"),
                ("interaction_declared", "interactionDeclared"),
                ("interaction_available", "interactionAvailable"),
                ("interaction_required", "interactionRequired"),
                ("interaction_attempted", "interactionAttempted"),
                ("interaction_verified", "interactionVerified"),
                ("risk_candidate", "riskCandidate"),
            )
        ],
        "candidate_one_hot": {
            "field": "semanticType",
            "feature_prefix": "semantic_type::",
            "vocabulary": list(config["semantic_types"]),
            "unknown": "unknown",
        },
        "candidate_mask": "true for a present candidate; false only for padding",
        "target_mask": {
            "preferred": "candidate.selectable emitted by Unity",
            "fallback": (
                "not visited AND cooldownSeconds<=0 AND reachabilityKnown AND reachable"
            ),
            "runtime_requirement": "must be applied before target execution",
        },
        "skill_mask": {
            "source": "validSkills",
            "runtime_requirement": "must be applied before skill execution",
        },
        "decision_kind_routing": {
            "target_selection": {
                "target_rank_loss": True,
                "skill_loss": True,
                "skill_conditioning_target": "model-ranked legal target at inference",
            },
            "skill_transition": {
                "target_rank_loss": False,
                "skill_loss": True,
                "skill_conditioning_target": "logged/runtime active target index",
            },
        },
        "virtual_stop_candidate": {
            "enabled": bool(config.get("append_stop_candidate", True)),
            "reason": "provide a candidate embedding when the Stop skill has no scene target",
            "raw_template": stop,
            "encoded_feature_names": candidate_feature_names(config),
            "encoded_feature_vector": encode_candidate(stop, config).tolist(),
            "target_mask_rule": "legal only when Stop is present in validSkills",
        },
        "excluded_from_model": [
            "targetId",
            "projectId",
            "sceneName",
            "bodyPosition",
            "bodyYawDegrees",
            "baseUtility",
            "randomJitter",
            "totalUtility",
            "pathLength",
            "analysisConfidence",
            "dynamicObject",
        ],
    }


def parse_human_acceptance(value: Any, label: str) -> bool | None:
    if value is None or value == "":
        return None
    if type(value) is bool:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "accepted", "accept", "yes", "1"}:
        return True
    if normalized in {"false", "rejected", "reject", "corrected", "no", "0"}:
        return False
    raise PreparationError(f"{label} must be empty, accepted/true, or rejected/false")


def _group_id(context: dict[str, Any]) -> str:
    def canonical(value: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
        return " ".join(normalized.split())

    explicit = canonical(context.get("splitFamily"))
    if explicit:
        return explicit
    project_family = canonical(context.get("projectFamily"))
    scene_family = canonical(context.get("sceneFamily"))
    if project_family and scene_family:
        project, scene = project_family, scene_family
    else:
        project = canonical(context["projectId"])
        scene = canonical(context["sceneName"])
    if not project or not scene:
        raise PreparationError("project/scene family cannot be empty")
    return project + "::" + scene


def validate_outcome(
    outcome: dict[str, Any], config: dict[str, Any], location: str
) -> dict[str, Any]:
    if outcome["schemaVersion"] != config["outcome_schema_version"]:
        raise PreparationError(
            f"{location}: expected {config['outcome_schema_version']}, got {outcome['schemaVersion']}"
        )
    result = dict(outcome)
    result["utc"] = _required_text(outcome["utc"], f"{location}.utc")
    result["episodeId"] = _required_text(outcome["episodeId"], f"{location}.episodeId")
    result["decisionId"] = _required_text(outcome["decisionId"], f"{location}.decisionId")
    result["projectId"] = _required_text(outcome["projectId"], f"{location}.projectId")
    result["sceneName"] = _required_text(outcome["sceneName"], f"{location}.sceneName")
    result["seed"] = _integer(outcome["seed"], f"{location}.seed")
    result["simulatedTime"] = _finite_float(
        outcome["simulatedTime"], f"{location}.simulatedTime"
    )
    result["chosenTargetId"] = str(outcome["chosenTargetId"] or "").strip()
    result["chosenSkill"] = _required_text(
        outcome["chosenSkill"], f"{location}.chosenSkill"
    )
    for name in ("reached", "visited", "interactionAttempted", "interactionVerified", "timedOut"):
        result[name] = _boolean(outcome[name], f"{location}.{name}")
    result["collisionRecoveriesDelta"] = _integer(
        outcome["collisionRecoveriesDelta"], f"{location}.collisionRecoveriesDelta"
    )
    if result["collisionRecoveriesDelta"] < 0:
        raise PreparationError(f"{location}.collisionRecoveriesDelta cannot be negative")
    before = _finite_float(outcome["coverageBefore"], f"{location}.coverageBefore")
    after = _finite_float(outcome["coverageAfter"], f"{location}.coverageAfter")
    gain_raw = outcome["coverageGain"]
    gain = after - before if gain_raw is None else _finite_float(gain_raw, f"{location}.coverageGain")
    if abs(gain - (after - before)) > 1.0e-4:
        raise PreparationError(f"{location}: coverageGain disagrees with before/after")
    if not (0.0 <= before <= 1.0 + 1.0e-6 and 0.0 <= after <= 1.0 + 1.0e-6):
        raise PreparationError(f"{location}: coverage before/after must be in [0, 1]")
    result["coverageBefore"] = before
    result["coverageAfter"] = after
    result["coverageGain"] = gain
    result["elapsedSeconds"] = _finite_float(
        outcome["elapsedSeconds"], f"{location}.elapsedSeconds"
    )
    if result["elapsedSeconds"] < 0.0:
        raise PreparationError(f"{location}.elapsedSeconds cannot be negative")
    result["terminalState"] = _required_text(
        outcome["terminalState"], f"{location}.terminalState"
    )
    result["evidence"] = _required_text(outcome["evidence"], f"{location}.evidence")
    return result


def build_example(
    context_raw: dict[str, Any],
    outcome: dict[str, Any],
    config: dict[str, Any],
    location: str,
    *,
    allow_synthetic_fixture: bool,
) -> PreparedExample:
    validate_flat_context_shape(context_raw, location)
    context = normalize_context(context_raw)
    if context["schemaVersion"] != config["context_schema_version"]:
        raise PreparationError(
            f"{location}: expected {config['context_schema_version']}, got {context['schemaVersion']}"
        )
    episode_id = _required_text(context["episodeId"], f"{location}.episodeId")
    decision_id = _required_text(context["decisionId"], f"{location}.decisionId")
    project_id = _required_text(context["projectId"], f"{location}.projectId")
    scene_name = _required_text(context["sceneName"], f"{location}.sceneName")
    _required_text(context["utc"], f"{location}.utc")
    context_seed = _integer(context["seed"], f"{location}.seed")
    if isinstance(context_raw.get("bodyPosition"), dict):
        for axis in ("x", "y", "z"):
            _finite_float(
                context_raw["bodyPosition"].get(axis),
                f"{location}.bodyPosition.{axis}",
            )
    elif "schemaVersion" in context_raw:
        raise PreparationError(f"{location}.bodyPosition must be a Vector3 JSON object")
    _finite_float(context["bodyYawDegrees"], f"{location}.bodyYawDegrees")
    source_kind = _required_text(context["sourceKind"], f"{location}.sourceKind")
    if source_kind not in config["allowed_source_kinds"]:
        raise PreparationError(f"{location}: unsupported sourceKind {source_kind!r}")
    label_source = _required_text(context["labelSource"], f"{location}.labelSource")
    if label_source not in config["allowed_source_kinds"]:
        raise PreparationError(f"{location}: unsupported labelSource {label_source!r}")
    if label_source != source_kind:
        raise PreparationError(
            f"{location}: labelSource={label_source!r} must match "
            f"sourceKind={source_kind!r}; mixed provenance is not accepted"
        )
    human_demonstration = _boolean(
        context["humanDemonstration"], f"{location}.humanDemonstration"
    )
    should_be_human = source_kind in set(config["human_source_kinds"])
    if human_demonstration != should_be_human:
        raise PreparationError(
            f"{location}: sourceKind={source_kind} requires "
            f"humanDemonstration={str(should_be_human).lower()}"
        )
    decision_kind = str(context.get("decisionKind") or "target_selection").strip()
    if decision_kind not in {"target_selection", "skill_transition"}:
        raise PreparationError(f"{location}: unsupported decisionKind {decision_kind!r}")
    synthetic = _boolean(context["syntheticFixture"], f"{location}.syntheticFixture")
    synthetic = synthetic or _boolean(
        outcome.get("syntheticFixture", False), f"{location}.outcome.syntheticFixture"
    )
    if synthetic and not allow_synthetic_fixture:
        raise PreparationError(
            f"{location}: explicit synthetic fixture rejected; use --allow-synthetic-fixture only for tests"
        )
    if outcome["episodeId"] != episode_id or outcome["decisionId"] != decision_id:
        raise PreparationError(f"{location}: context/outcome key mismatch")
    if outcome.get("projectId") and str(outcome["projectId"]) != project_id:
        raise PreparationError(f"{location}: context/outcome projectId mismatch")
    if outcome.get("sceneName") and str(outcome["sceneName"]) != scene_name:
        raise PreparationError(f"{location}: context/outcome sceneName mismatch")
    if int(outcome["seed"]) != context_seed:
        raise PreparationError(f"{location}: context/outcome seed mismatch")

    raw_candidates = context["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise PreparationError(f"{location}.candidates must be a non-empty array")
    candidates = [normalize_candidate(value, index) for index, value in enumerate(raw_candidates)]
    target_ids = [_required_text(value["targetId"], f"{location}.candidates.targetId") for value in candidates]
    if len(target_ids) != len(set(target_ids)):
        raise PreparationError(f"{location}: candidate targetId values must be unique")
    existing_stop = next(
        (index for index, value in enumerate(candidates) if value["targetId"] == "__STOP__"),
        None,
    )
    if config.get("append_stop_candidate", True) and existing_stop is None:
        candidates.append(stop_candidate())
        target_ids.append("__STOP__")
        stop_index = len(candidates) - 1
    else:
        stop_index = existing_stop

    valid_skills = context["validSkills"]
    if not isinstance(valid_skills, list) or not valid_skills:
        raise PreparationError(f"{location}.validSkills must be a non-empty array")
    skill_lookup = {_normal_key(name): index for index, name in enumerate(config["skills"])}
    valid_skill_indices: set[int] = set()
    for raw_skill in valid_skills:
        key = _normal_key(raw_skill)
        if key not in skill_lookup:
            raise PreparationError(f"{location}: unsupported valid skill {raw_skill!r}")
        valid_skill_indices.add(skill_lookup[key])
    skill_mask = np.asarray(
        [index in valid_skill_indices for index in range(len(config["skills"]))],
        dtype=bool,
    )

    chosen_skill_raw = context["chosenSkill"]
    chosen_target_id = str(context["chosenTargetId"] or "").strip()
    chosen_target_index_raw = context["chosenTargetIndex"]
    correction_target = str(context["humanCorrectionTargetId"] or "").strip()
    correction_skill = str(context["humanCorrectionSkill"] or "").strip()
    human_accepted = parse_human_acceptance(
        context["humanAccepted"], f"{location}.humanAccepted"
    )
    if source_kind == "unity_human" and human_accepted is None:
        raise PreparationError(
            f"{location}: unity_human labels require an explicit humanAccepted annotation"
        )
    if not should_be_human and (
        human_accepted is not None or correction_target or correction_skill
    ):
        raise PreparationError(
            f"{location}: bootstrap records cannot contain human acceptance/corrections"
        )
    if human_accepted is True and (correction_target or correction_skill):
        raise PreparationError(
            f"{location}: accepted human proposal cannot also contain corrections"
        )
    if human_accepted is None and (correction_target or correction_skill):
        raise PreparationError(
            f"{location}: corrections require humanAccepted=false"
        )
    if should_be_human and human_accepted is False and not (
        correction_target or correction_skill
    ):
        raise PreparationError(
            f"{location}: rejected human proposal needs at least one explicit correction"
        )
    if correction_target:
        chosen_target_id = correction_target
        chosen_target_index_raw = None
    if correction_skill:
        chosen_skill_raw = correction_skill
    chosen_skill_key = _normal_key(chosen_skill_raw)
    if chosen_skill_key not in skill_lookup:
        raise PreparationError(f"{location}: unsupported chosenSkill {chosen_skill_raw!r}")
    skill_index = skill_lookup[chosen_skill_key]
    if not skill_mask[skill_index]:
        raise PreparationError(f"{location}: chosenSkill is absent from validSkills")
    if _normal_key(outcome["chosenSkill"]) != chosen_skill_key:
        raise PreparationError(f"{location}: context/outcome chosenSkill mismatch")
    if str(outcome["chosenTargetId"] or "").strip() != chosen_target_id:
        raise PreparationError(f"{location}: context/outcome chosenTargetId mismatch")

    if chosen_skill_key == _normal_key("Stop") and not chosen_target_id:
        if stop_index is None:
            raise PreparationError(f"{location}: Stop action has no virtual STOP candidate")
        target_index = int(stop_index)
    elif chosen_target_index_raw is not None and not correction_target:
        target_index = _integer(chosen_target_index_raw, f"{location}.chosenTargetIndex")
        if target_index < 0 or target_index >= len(raw_candidates):
            raise PreparationError(f"{location}: chosenTargetIndex is outside emitted candidates")
        if chosen_target_id and target_ids[target_index] != chosen_target_id:
            raise PreparationError(f"{location}: chosenTargetId/index disagree")
    else:
        if not chosen_target_id:
            raise PreparationError(f"{location}: chosen target is missing")
        try:
            target_index = target_ids.index(chosen_target_id)
        except ValueError as exc:
            raise PreparationError(f"{location}: chosen target is absent from candidates") from exc

    candidate_mask = np.ones(len(candidates), dtype=bool)
    target_mask_values = [candidate_is_legal(value) for value in candidates]
    if stop_index is not None:
        target_mask_values[stop_index] = skill_lookup[_normal_key("Stop")] in valid_skill_indices
    target_mask = np.asarray(target_mask_values, dtype=bool)
    if not target_mask.any():
        raise PreparationError(f"{location}: no legal candidate remains after masks")
    target_label_available = decision_kind == "target_selection"
    if target_label_available and not target_mask[target_index]:
        raise PreparationError(f"{location}: chosen target is illegal under the emitted state")

    features = np.stack([encode_candidate(value, config) for value in candidates])
    utilities = []
    for candidate in candidates:
        total = _optional_float(candidate["totalUtility"])
        base = _optional_float(candidate["baseUtility"])
        utilities.append(total if math.isfinite(total) else base)
    source_weight = float(config["source_weights"][source_kind])
    if not math.isfinite(source_weight) or source_weight <= 0:
        raise PreparationError(f"Invalid source weight for {source_kind}")
    return PreparedExample(
        episode_id=episode_id,
        decision_id=decision_id,
        group_id=_group_id(context),
        source_kind=source_kind,
        human_demonstration=human_demonstration,
        decision_kind=decision_kind,
        global_features=encode_global(context, config),
        candidate_features=features,
        candidate_mask=candidate_mask,
        target_mask=target_mask,
        target_label_available=target_label_available,
        skill_mask=skill_mask,
        target_index=target_index,
        skill_index=skill_index,
        teacher_utilities=np.asarray(utilities, dtype=np.float32),
        sample_weight=source_weight,
        outcome_reached=bool(outcome["reached"]),
        outcome_interaction_verified=bool(outcome["interactionVerified"]),
        outcome_coverage_gain=float(outcome["coverageGain"]),
    )


def grouped_split(
    examples: list[PreparedExample], config: dict[str, Any], *, allow_smoke: bool
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    by_group: dict[str, list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_group[example.group_id].append(index)
    groups = sorted(by_group)
    minimum_groups = int(config["minimum_production_groups"])
    minimum_samples = int(config["minimum_production_samples"])
    if (len(groups) < minimum_groups or len(examples) < minimum_samples) and not allow_smoke:
        raise PreparationError(
            "Insufficient production data: "
            f"samples={len(examples)} (need {minimum_samples}), "
            f"groups={len(groups)} (need {minimum_groups}). "
            "Use --allow-smoke only for an explicitly labelled pipeline check."
        )
    if len(groups) < 3:
        raise PreparationError(
            "At least three scene/project-family groups are required for leakage-resistant train/val/test splits"
        )
    fractions = config["split"]
    train_fraction = float(fractions["train_fraction"])
    validation_fraction = float(fractions["validation_fraction"])
    test_fraction = float(fractions["test_fraction"])
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1.0e-6:
        raise PreparationError("Split fractions must sum to one")
    rng = np.random.default_rng(int(fractions["seed"]))
    shuffled = [groups[index] for index in rng.permutation(len(groups))]
    validation_count = max(1, int(round(len(groups) * validation_fraction)))
    test_count = max(1, int(round(len(groups) * test_fraction)))
    if validation_count + test_count >= len(groups):
        validation_count = 1
        test_count = 1
    train_count = len(groups) - validation_count - test_count
    if train_count < 1:
        raise PreparationError("Grouped split left no training groups")
    group_splits = {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + validation_count]),
        "test": sorted(shuffled[train_count + validation_count :]),
    }
    if any(set(group_splits[left]) & set(group_splits[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Group leakage across prepared splits")
    row_splits = {
        split: sorted(index for group in split_groups for index in by_group[group])
        for split, split_groups in group_splits.items()
    }
    if any(not rows for rows in row_splits.values()):
        raise PreparationError("Grouped split produced an empty data partition")
    missing_target_labels = [
        split
        for split, rows in row_splits.items()
        if not any(examples[index].target_label_available for index in rows)
    ]
    if missing_target_labels:
        raise PreparationError(
            "Grouped split has no target_selection labels in: "
            + ", ".join(missing_target_labels)
        )
    return row_splits, group_splits


def save_split(
    path: Path,
    examples: list[PreparedExample],
    indices: list[int],
    maximum_candidates: int,
    config: dict[str, Any],
) -> None:
    selected = [examples[index] for index in indices]
    global_dim = len(global_feature_names(config))
    candidate_dim = len(candidate_feature_names(config))
    skill_count = len(config["skills"])
    count = len(selected)
    global_values = np.zeros((count, global_dim), dtype=np.float32)
    candidate_values = np.zeros(
        (count, maximum_candidates, candidate_dim), dtype=np.float32
    )
    candidate_masks = np.zeros((count, maximum_candidates), dtype=bool)
    target_masks = np.zeros((count, maximum_candidates), dtype=bool)
    target_label_available = np.zeros(count, dtype=bool)
    skill_masks = np.zeros((count, skill_count), dtype=bool)
    teacher_utilities = np.full((count, maximum_candidates), np.nan, dtype=np.float32)
    target_indices = np.zeros(count, dtype=np.int64)
    skill_indices = np.zeros(count, dtype=np.int64)
    sample_weights = np.ones(count, dtype=np.float32)
    outcome_reached = np.zeros(count, dtype=bool)
    outcome_interaction_verified = np.zeros(count, dtype=bool)
    outcome_coverage_gain = np.zeros(count, dtype=np.float32)
    for row, example in enumerate(selected):
        candidate_count = len(example.candidate_features)
        global_values[row] = example.global_features
        candidate_values[row, :candidate_count] = example.candidate_features
        candidate_masks[row, :candidate_count] = example.candidate_mask
        target_masks[row, :candidate_count] = example.target_mask
        target_label_available[row] = example.target_label_available
        teacher_utilities[row, :candidate_count] = example.teacher_utilities
        skill_masks[row] = example.skill_mask
        target_indices[row] = example.target_index
        skill_indices[row] = example.skill_index
        sample_weights[row] = example.sample_weight
        outcome_reached[row] = example.outcome_reached
        outcome_interaction_verified[row] = example.outcome_interaction_verified
        outcome_coverage_gain[row] = example.outcome_coverage_gain
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        global_features=global_values,
        candidate_features=candidate_values,
        candidate_mask=candidate_masks,
        target_mask=target_masks,
        target_label_available=target_label_available,
        skill_mask=skill_masks,
        target_index=target_indices,
        skill_index=skill_indices,
        teacher_utilities=teacher_utilities,
        sample_weight=sample_weights,
        source_kind=np.asarray([value.source_kind for value in selected], dtype=str),
        decision_kind=np.asarray([value.decision_kind for value in selected], dtype=str),
        group_id=np.asarray([value.group_id for value in selected], dtype=str),
        episode_id=np.asarray([value.episode_id for value in selected], dtype=str),
        decision_id=np.asarray([value.decision_id for value in selected], dtype=str),
        outcome_reached=outcome_reached,
        outcome_interaction_verified=outcome_interaction_verified,
        outcome_coverage_gain=outcome_coverage_gain,
    )
    temporary.replace(path)


def sha256_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.expanduser().resolve() for path in paths):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    *,
    context_paths: list[Path],
    outcome_paths: list[Path],
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    allow_smoke: bool = False,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    context_records = _read_jsonl(context_paths, "context")
    outcome_records = _read_jsonl(outcome_paths, "outcome")
    outcomes: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    synthetic_seen = False
    for raw, location in outcome_records:
        validate_flat_outcome_shape(raw, location)
        normalized = validate_outcome(normalize_outcome(raw), config, location)
        key = (normalized["episodeId"], normalized["decisionId"])
        if key in outcomes:
            raise PreparationError(f"Duplicate outcome key {key}")
        outcomes[key] = (normalized, location)
        synthetic_seen = synthetic_seen or bool(normalized.get("syntheticFixture"))

    examples: list[PreparedExample] = []
    seen_context_keys: set[tuple[str, str]] = set()
    for raw, location in context_records:
        normalized_preview = normalize_context(raw)
        key = (
            _required_text(normalized_preview["episodeId"], f"{location}.episodeId"),
            _required_text(normalized_preview["decisionId"], f"{location}.decisionId"),
        )
        if key in seen_context_keys:
            raise PreparationError(f"Duplicate context key {key}")
        seen_context_keys.add(key)
        if key not in outcomes:
            raise PreparationError(f"No outcome matches context key {key}")
        outcome, _ = outcomes[key]
        example = build_example(
            raw,
            outcome,
            config,
            location,
            allow_synthetic_fixture=allow_synthetic_fixture,
        )
        examples.append(example)
        synthetic_seen = synthetic_seen or bool(normalized_preview["syntheticFixture"])
    unused_outcomes = set(outcomes) - seen_context_keys
    if unused_outcomes:
        preview = sorted(unused_outcomes)[:5]
        raise PreparationError(f"Outcomes without contexts: {preview}")

    row_splits, group_splits = grouped_split(examples, config, allow_smoke=allow_smoke)
    maximum_candidates = max(len(example.candidate_features) for example in examples)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, str] = {}
    split_hashes: dict[str, str] = {}
    split_sizes: dict[str, int] = {}
    for split, indices in row_splits.items():
        path = output_dir / f"unity_decision_{split}.npz"
        save_split(path, examples, indices, maximum_candidates, config)
        split_paths[split] = str(path)
        split_hashes[split] = sha256_file(path)
        split_sizes[split] = path.stat().st_size

    source_counts = collections.Counter(value.source_kind for value in examples)
    decision_kind_counts = collections.Counter(value.decision_kind for value in examples)
    skill_label_counts = collections.Counter(
        config["skills"][value.skill_index] for value in examples
    )
    human_count = count_human_decisions(
        (value.source_kind for value in examples), config
    )
    smoke = (
        len(examples) < int(config["minimum_production_samples"])
        or len({value.group_id for value in examples})
        < int(config["minimum_production_groups"])
    )
    split_manifest_path = output_dir / "split_manifest.json"
    split_manifest = {
        "schema_version": "unity-decision-group-split-1.0",
        "seed": int(config["split"]["seed"]),
        "group_definition": config["split"]["group_definition"],
        "groups": group_splits,
    }
    write_json_atomic(split_manifest_path, split_manifest)
    preprocess = preprocessing_spec(config)
    manifest = {
        "schema_version": "unity-decision-prepared-data-1.0",
        "generated_at_utc": utc_now(),
        "status": "smoke_only" if smoke or synthetic_seen else "production_candidate",
        "context_schema_version": config["context_schema_version"],
        "outcome_schema_version": config["outcome_schema_version"],
        "config_schema_version": config["schema_version"],
        "config_sha256": sha256_file(config_path),
        "split_seed": int(config["split"]["seed"]),
        "synthetic_fixture": bool(synthetic_seen),
        "synthetic_fixture_authorized_for_this_run": bool(allow_synthetic_fixture),
        "samples": len(examples),
        "groups": len({value.group_id for value in examples}),
        "maximum_candidates": maximum_candidates,
        "source_counts": dict(sorted(source_counts.items())),
        "decision_kind_counts": dict(sorted(decision_kind_counts.items())),
        "target_label_samples": int(
            sum(value.target_label_available for value in examples)
        ),
        "skill_label_coverage": {
            "counts": {
                skill: int(skill_label_counts.get(skill, 0))
                for skill in config["skills"]
            },
            "observed_skills": [
                skill for skill in config["skills"] if skill_label_counts.get(skill, 0) > 0
            ],
            "all_six_skills_observed": all(
                skill_label_counts.get(skill, 0) > 0 for skill in config["skills"]
            ),
        },
        "human_decision_samples": human_count,
        "human_like_claim_allowed": False,
        "split_method": "scene/project-family grouped; no row-random split",
        "split_groups": group_splits,
        "split_samples": {name: len(indices) for name, indices in row_splits.items()},
        "global_feature_names": global_feature_names(config),
        "candidate_feature_names": candidate_feature_names(config),
        "forbidden_model_inputs": [
            "targetId",
            "projectId",
            "sceneName",
            "bodyPosition absolute coordinates",
            "baseUtility",
            "randomJitter",
            "totalUtility",
        ],
        "teacher_utility_role": "evaluation-only regret; not a model input",
        "preprocessing": preprocess,
        "data_fingerprint_sha256": sha256_paths(
            [*context_paths, *outcome_paths, config_path]
        ),
        "artifacts": split_paths,
        "artifact_sha256": split_hashes,
        "artifact_size_bytes": split_sizes,
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
    }
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--outcomes", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow a clearly labelled undersized dataset; never upgrades it to production.",
    )
    parser.add_argument(
        "--allow-synthetic-fixture",
        action="store_true",
        help="Test-only: accept records explicitly marked syntheticFixture=true.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    try:
        manifest = prepare(
            context_paths=arguments.contexts,
            outcome_paths=arguments.outcomes,
            output_dir=arguments.output_dir,
            config_path=arguments.config,
            allow_smoke=arguments.allow_smoke,
            allow_synthetic_fixture=arguments.allow_synthetic_fixture,
        )
    except Exception as exc:
        print(f"Decision data preparation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Prepared {manifest['samples']} decisions across {manifest['groups']} groups "
        f"({manifest['status']}); human decisions={manifest['human_decision_samples']}"
    )
    print(Path(arguments.output_dir).expanduser().resolve() / "dataset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
