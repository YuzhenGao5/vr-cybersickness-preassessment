#!/usr/bin/env python3
"""Tests for the semantic-report to Agent-contract compiler."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compile_scene_agent_contract import (  # noqa: E402
    ContractError,
    canonicalize_advisory_targets,
    collect_llm_source_documents,
    compile_contract,
    deterministic_advisory,
    extract_csharp_methods,
    render_report,
    validate_advisory,
)


class SceneAgentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = REPO_ROOT / "results" / "unity_semantic_fixture"
        cls.semantic_path = cls.fixture / "parse_result.json"
        cls.semantic = json.loads(cls.semantic_path.read_text(encoding="utf-8-sig"))

    def compile_fixture(self) -> dict:
        return compile_contract(
            self.fixture,
            self.semantic_path,
            self.semantic,
            "Assets/Scenes/SemanticFixture.unity",
        )

    def test_compiles_scene_nodes_spawn_objectives_and_gate(self) -> None:
        contract = self.compile_fixture()
        self.assertEqual("scene-agent-contract-1.0", contract["schema_version"])
        self.assertEqual("Assets/Scenes/SemanticFixture.unity", contract["runtime_scene"]["path"])
        self.assertTrue(contract["runtime_scene"]["exists"])
        self.assertGreaterEqual(len(contract["semantic_nodes"]), 1)
        self.assertTrue(any(item["label"] == "XROrigin_PlayerSpawn" for item in contract["spawn_candidates"]))
        self.assertGreaterEqual(len(contract["objectives"]), 1)
        self.assertEqual("needs_runtime_binding", contract["activation_gate"]["status"])
        self.assertFalse(contract["activation_gate"]["can_execute_in_unmodified_project"])

    def test_method_candidates_are_evidence_backed_and_never_authorized(self) -> None:
        contract = self.compile_fixture()
        methods = contract["method_candidates"]
        self.assertTrue(any(item["method"] == "Interact" for item in methods))
        interact = next(item for item in methods if item["method"] == "Interact")
        self.assertEqual("Assets/Scripts/DoorButton.cs", interact["script_path"])
        self.assertEqual(8, interact["line"])
        self.assertFalse(interact["call_authorized"])
        self.assertEqual("candidate_requires_runtime_review", interact["binding_status"])

    def test_hashed_source_excerpts_include_scene_and_referenced_script(self) -> None:
        contract = self.compile_fixture()
        documents = collect_llm_source_documents(self.fixture, contract)
        by_path = {item["path"]: item for item in documents}
        self.assertIn("Assets/Scenes/SemanticFixture.unity", by_path)
        self.assertIn("Assets/Scripts/DoorButton.cs", by_path)
        self.assertIn("public void Interact", by_path["Assets/Scripts/DoorButton.cs"]["content"])
        self.assertEqual(64, len(by_path["Assets/Scripts/DoorButton.cs"]["content_sha256"]))

    def test_source_excerpts_include_csharp_not_directly_referenced_by_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir)
            scene = project / "Assets" / "Scenes" / "Scene.unity"
            direct = project / "Assets" / "Scripts" / "DirectBootstrap.cs"
            indirect = project / "Assets" / "Scripts" / "RuntimeAgentController.cs"
            scene.parent.mkdir(parents=True)
            direct.parent.mkdir(parents=True)
            scene.write_text("--- !u!1 &1\nGameObject:\n", encoding="utf-8")
            direct.write_text("class DirectBootstrap {}\n", encoding="utf-8")
            indirect.write_text("class RuntimeAgentController {}\n", encoding="utf-8")
            contract = {
                "runtime_scene": {"path": "Assets/Scenes/Scene.unity"},
                "semantic_nodes": [{
                    "evidence": [{"path": "Assets/Scripts/DirectBootstrap.cs"}],
                }],
            }

            documents = collect_llm_source_documents(project, contract)
            paths = [item["path"] for item in documents]

            self.assertEqual("Assets/Scenes/Scene.unity", paths[0])
            self.assertEqual("Assets/Scripts/DirectBootstrap.cs", paths[1])
            self.assertIn("Assets/Scripts/RuntimeAgentController.cs", paths)
            self.assertIn(
                "class RuntimeAgentController",
                next(
                    item["content"]
                    for item in documents
                    if item["path"] == "Assets/Scripts/RuntimeAgentController.cs"
                ),
            )

    def test_known_node_alias_is_canonicalized_but_unknown_is_not(self) -> None:
        contract = self.compile_fixture()
        objective = contract["objectives"][0]
        payload = {
            "planning_advice": [{"target_id": objective["target_node_id"]}],
            "runtime_discovery_hints": [],
        }
        canonicalize_advisory_targets(payload, contract)
        self.assertEqual(objective["target_source_id"], payload["planning_advice"][0]["target_id"])
        payload["planning_advice"][0]["target_id"] = "unknown"
        canonicalize_advisory_targets(payload, contract)
        self.assertEqual("unknown", payload["planning_advice"][0]["target_id"])

    def test_csharp_scanner_does_not_escape_project(self) -> None:
        with self.assertRaisesRegex(ContractError, "project-relative"):
            extract_csharp_methods(self.fixture, "../outside.cs")

    def test_requested_scene_must_exist_in_semantic_evidence(self) -> None:
        with self.assertRaisesRegex(ContractError, "not included"):
            compile_contract(
                self.fixture,
                self.semantic_path,
                self.semantic,
                "Assets/Scenes/NotInEvidence.unity",
            )

    def test_deterministic_advisory_and_report_are_explicitly_labelled(self) -> None:
        contract = self.compile_fixture()
        contract["compiler"]["narrative_backend_requested"] = "deterministic"
        contract["compiler"]["narrative_backend_used"] = "deterministic_template"
        contract["compiler"]["narrative_fallback_reason"] = "explicit test"
        contract["llm_advisory"] = deterministic_advisory(contract, self.semantic)
        report = render_report(contract)
        self.assertIn("确定性模板回退", report)
        self.assertIn("EyeNavGS", report)
        self.assertIn("任何方法候选都没有被自动授权调用", report)

    def test_llm_advice_cannot_reference_unknown_target_or_skill(self) -> None:
        payload = {
            "scene_summary": "summary",
            "likely_player_loop": [],
            "planning_advice": [{
                "target_id": "invented-target",
                "rationale": "invented",
                "recommended_skills": ["navigate_to"],
                "confidence": 0.5,
            }],
            "source_findings": [{
                "path": "Assets/Scenes/SemanticFixture.unity",
                "kind": "scene",
                "locator": "fileID=1",
                "finding": "test",
                "confidence": 0.5,
            }],
            "runtime_discovery_hints": [],
            "important_unknowns": [],
            "rendering_checks": [],
        }
        with self.assertRaisesRegex(ContractError, "unknown target"):
            validate_advisory(payload, {"real-target"})
        payload["planning_advice"][0]["target_id"] = "real-target"
        payload["planning_advice"][0]["recommended_skills"] = ["arbitrary_method_call"]
        with self.assertRaisesRegex(ContractError, "unsupported skill"):
            validate_advisory(payload, {"real-target"})


if __name__ == "__main__":
    unittest.main()
