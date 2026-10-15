#!/usr/bin/env python3
"""Convert the evidence compiler schema into the small Unity runtime schema.

The conversion is deterministic.  LLM text can rank or describe targets, but it
cannot authorize a method call.  Interactions enter the runtime allowlist only
through a separate human-reviewed bindings file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "scene-agent-contract-1.0"
RUNTIME_SCHEMA = "generic-scene-agent-contract-1.0"


def vector(values: Any, fallback: list[float] | None = None) -> dict[str, float]:
    values = values if isinstance(values, list) and len(values) >= 3 else (fallback or [0.0, 0.0, 0.0])
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewed_binding_map(path: Path | None, source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "scene-agent-reviewed-bindings-1.0":
        raise ValueError("reviewed bindings use an unsupported schema")
    if payload.get("approvalStatus") != "approved":
        raise ValueError("reviewed bindings are not approved")
    project_root = Path(str((source.get("project") or {}).get("root") or "")).resolve()
    runtime_scene = source.get("runtime_scene") or {}
    scene_relative = str(runtime_scene.get("path") or "").replace("\\", "/")
    if Path(str(payload.get("projectPath") or "")).resolve() != project_root:
        raise ValueError("reviewed bindings belong to a different project")
    if str(payload.get("scenePath") or "").replace("\\", "/") != scene_relative:
        raise ValueError("reviewed bindings belong to a different scene")
    scene_path = (project_root / scene_relative).resolve()
    if (
        not scene_path.is_file()
        or project_root not in scene_path.parents
        or sha256_file(scene_path).casefold() != str(payload.get("sourceSceneSha256") or "").casefold()
    ):
        raise ValueError("reviewed bindings pin a missing or changed scene")
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("reviewed bindings must contain a bindings array")
    result: dict[str, dict[str, Any]] = {}
    for item in bindings:
        required = {
            "target_node_id", "component_type", "method", "success_member", "expected_value",
            "script_path", "script_sha256",
        }
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("each reviewed binding requires " + ", ".join(sorted(required)))
        if not item.get("reviewed_by") or not item.get("reviewed_at_utc"):
            raise ValueError("each binding must record reviewed_by and reviewed_at_utc")
        script_relative = str(item["script_path"]).replace("\\", "/")
        script_path = (project_root / script_relative).resolve()
        if (
            not script_relative.startswith("Assets/")
            or not script_relative.casefold().endswith(".cs")
            or not script_path.is_file()
            or project_root not in script_path.parents
            or sha256_file(script_path).casefold() != str(item["script_sha256"]).casefold()
        ):
            raise ValueError(f"reviewed binding pins a missing or changed script: {script_relative}")
        result[str(item["target_node_id"])] = item
    return result


def choose_spawn(source: dict[str, Any]) -> tuple[list[float], str]:
    candidates = source.get("spawn_candidates") or []
    for item in candidates:
        position = item.get("position")
        if item.get("executable_without_runtime_resolution") and isinstance(position, list) and len(position) >= 3:
            evidence = "; ".join(str(value) for value in item.get("constraints") or [])
            return [float(value) for value in position[:3]], evidence
    return [0.0, 0.0, 0.0], "No executable spawn candidate; runtime must validate origin."


def target_position(node: dict[str, Any]) -> list[float] | None:
    position = node.get("position")
    if node.get("world_position_executable") and isinstance(position, list) and len(position) >= 3:
        return [float(value) for value in position[:3]]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-bindings", type=Path)
    parser.add_argument("--disable-behaviour", action="append", default=[])
    parser.add_argument("--maximum-targets", type=int, default=8)
    arguments = parser.parse_args()

    source_path = arguments.compiled_contract.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"expected {SOURCE_SCHEMA}, got {source.get('schema_version')!r}")
    runtime_scene = source.get("runtime_scene") or {}
    if not runtime_scene.get("exists"):
        raise ValueError("compiled contract runtime_scene is not present")
    nodes = {item.get("id"): item for item in source.get("semantic_nodes") or [] if isinstance(item, dict)}
    bindings = reviewed_binding_map(arguments.reviewed_bindings, source)
    spawn, spawn_evidence = choose_spawn(source)
    advisory = source.get("llm_advisory") or {}
    advisory_by_target = {
        item.get("target_id"): item
        for item in advisory.get("planning_advice") or []
        if isinstance(item, dict)
    }

    deduplicated: dict[str, dict[str, Any]] = {}
    objectives = sorted(
        (item for item in source.get("objectives") or [] if isinstance(item, dict)),
        key=lambda item: (-float(item.get("priority") or 0), int(item.get("order_hint") or 9999)),
    )
    maximum_priority = max((float(item.get("priority") or 0) for item in objectives), default=1.0)
    for objective in objectives:
        node_id = objective.get("target_node_id")
        node = nodes.get(node_id)
        if not node:
            continue
        position = target_position(node)
        if position is None:
            continue
        physical_key = "|".join([
            str(node.get("label") or "").casefold(),
            *(f"{value:.3f}" for value in position),
        ])
        existing = deduplicated.get(physical_key)
        if existing:
            existing["reason"] += " " + str(objective.get("reason") or "")
            existing["priority"] = max(existing["priority"], 0.5 + float(objective.get("priority") or 0) / maximum_priority)
            continue
        dx, dz = spawn[0] - position[0], spawn[2] - position[2]
        magnitude = math.hypot(dx, dz)
        approach = {"x": 0.0, "y": 0.0, "z": -1.2}
        if magnitude > 0.001:
            approach = {"x": dx / magnitude * 1.2, "y": 0.0, "z": dz / magnitude * 1.2}
        advice = advisory_by_target.get(objective.get("target_source_id")) or {}
        reason = str(advice.get("rationale") or objective.get("reason") or "Inspect semantic target.")
        entry: dict[str, Any] = {
            "id": str(node_id),
            "label": str(node.get("label") or node_id),
            "semanticType": str(node.get("kind") or objective.get("kind") or "unknown"),
            "objectNameContains": str(node.get("label") or ""),
            "positionHint": vector(position),
            "approachOffset": approach,
            "arrivalRadius": 0.9,
            "observeSeconds": 2.4 if objective.get("kind") == "observe_and_measure" else 1.8,
            "priority": 0.5 + float(objective.get("priority") or 0) / maximum_priority,
            "reason": reason,
        }
        binding = bindings.get(str(node_id))
        if binding:
            method_key = f"{binding['component_type']}.{binding['method']}"
            public_method_candidate = any(
                item.get("target_node_id") == node_id
                and item.get("script_path") == binding["script_path"]
                and item.get("declaring_type") == binding["component_type"]
                and item.get("method") == binding["method"]
                and item.get("access") == "public"
                and item.get("parameter_count") == 0
                and not item.get("is_static")
                for item in source.get("method_candidates") or []
            )
            # Unity UI Button has no source-declared `Press` method: Press is a
            # deliberately narrow runtime adapter for its public onClick event.
            # Still require a separate reviewed binding, an affordance on this
            # exact target and static component evidence containing "Button".
            component_type = str(binding["component_type"])
            is_ui_button_press = (
                component_type in {"Button", "UnityEngine.UI.Button"}
                and binding["method"] == "Press"
            )
            evidence_text = " ".join(
                str(ref.get("observation") or "")
                for ref in node.get("evidence") or []
                if isinstance(ref, dict)
            ).casefold()
            reviewed_ui_button = (
                is_ui_button_press
                and "button" in evidence_text
                and any(
                    item.get("target_node_id") == node_id
                    for item in source.get("affordances") or []
                    if isinstance(item, dict)
                )
            )
            candidate_ok = public_method_candidate or reviewed_ui_button
            if not candidate_ok:
                raise ValueError(f"reviewed binding is absent from evidence candidates: {method_key} on {node_id}")
            entry["interaction"] = {
                "verb": str(binding.get("verb") or "interact"),
                "componentType": str(binding["component_type"]),
                "method": str(binding["method"]),
                "successMember": str(binding["success_member"]),
                "expectedValue": str(binding["expected_value"]),
                "required": bool(binding.get("required", True)),
            }
        deduplicated[physical_key] = entry
        if len(deduplicated) >= max(1, arguments.maximum_targets):
            break

    # A scene can contain useful semantic nodes without explicit objectives.
    if not deduplicated:
        for node_id, node in nodes.items():
            position = target_position(node)
            if position is None:
                continue
            deduplicated[str(node_id)] = {
                "id": str(node_id),
                "label": str(node.get("label") or node_id),
                "semanticType": str(node.get("kind") or "unknown"),
                "objectNameContains": str(node.get("label") or ""),
                "positionHint": vector(position),
                "approachOffset": {"x": 0.0, "y": 0.0, "z": -1.2},
                "arrivalRadius": 0.9,
                "observeSeconds": 1.8,
                "priority": 0.8,
                "reason": "Coverage fallback from an evidence-backed semantic node.",
            }
            if len(deduplicated) >= max(1, arguments.maximum_targets):
                break
    if not deduplicated:
        raise ValueError("compiled contract has no world-position target that can be materialized")

    interactions = [item.get("interaction") for item in deduplicated.values() if item.get("interaction")]
    semantics = sorted({str(item.get("kind") or "unknown") for item in nodes.values()})
    source_hints = source.get("runtime_discovery_hints") or advisory.get("runtime_discovery_hints") or []
    runtime_hints = []
    for hint in source_hints:
        if not isinstance(hint, dict):
            continue
        if hint.get("authorization") != "discovery_only" or hint.get("call_authorized") is True:
            continue
        runtime_hints.append({
            "hintId": str(hint.get("hint_id") or ""),
            "targetId": str(hint.get("target_id") or ""),
            "objectNamePatterns": [str(value) for value in hint.get("object_name_patterns") or []],
            "requiredComponents": [str(value) for value in hint.get("required_components") or []],
            "sourcePath": str(hint.get("source_path") or ""),
            "locator": str(hint.get("locator") or ""),
            "rationale": str(hint.get("rationale") or "Runtime object must be grounded."),
            "confidence": float(hint.get("confidence") or 0.0),
            "discoveryOnly": True,
        })
    source_finding_payload = source.get("source_inspection_audit") or advisory.get("source_findings") or []
    runtime_findings = []
    for finding in source_finding_payload:
        if not isinstance(finding, dict):
            continue
        runtime_findings.append({
            "path": str(finding.get("path") or ""),
            "kind": str(finding.get("kind") or "unknown"),
            "locator": str(finding.get("locator") or ""),
            "finding": str(finding.get("finding") or ""),
            "confidence": float(finding.get("confidence") or 0.0),
        })
    runtime = {
        "schemaVersion": RUNTIME_SCHEMA,
        "projectId": str((source.get("project") or {}).get("name") or "unity-project"),
        "projectName": str((source.get("project") or {}).get("name") or "Unity project"),
        "sceneName": Path(str(runtime_scene.get("path"))).stem,
        "generatedAtUtc": str(source.get("generated_at_utc") or ""),
        "generator": "materialize_generic_runtime_contract.py from deterministic evidence + advisory-only LLM report",
        "sourceReportPath": str(source_path),
        "reportSummary": str(advisory.get("scene_summary") or (
            f"{len(nodes)} semantic nodes and {len(objectives)} objectives were compiled from the selected Unity scene."
        )),
        "sceneSemantics": semantics,
        "safeInteractionMethods": sorted({
            f"{item['componentType']}.{item['method']}" for item in interactions
        }),
        "disabledBehaviourTypes": sorted(set(arguments.disable_behaviour)),
        "runtimeDiscoveryHints": runtime_hints,
        "sourceFindings": runtime_findings,
        "spawn": {"position": vector(spawn), "yawDegrees": 0.0, "evidence": spawn_evidence},
        "targets": list(deduplicated.values()),
        "declaredLimitations": [
            "LLM analysis is advisory and cannot authorize interactions.",
            "Every target must be grounded against the live scene and pass a runtime reachability check.",
            "Only separately reviewed, zero-argument bindings enter safeInteractionMethods.",
            "EyeNavGS trains the HMD motion prior; the semantic brain and route planner are separate rule-based systems.",
            "The exposure proxy is an uncalibrated engineering signal, not a VR.net probability or clinical score.",
            "Without a baked NavMesh, navigation is collision-checked direct steering with bounded recovery, not global path planning.",
            "Uneven-ground following uses downward raycasts and keeps pitch/roll at zero to preserve the horizon; unrecognised surfaces hold the last safe height.",
        ],
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "source_schema": SOURCE_SCHEMA,
        "runtime_schema": RUNTIME_SCHEMA,
        "scene": runtime_scene.get("path"),
        "targets": len(runtime["targets"]),
        "reviewed_interactions": len(interactions),
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
