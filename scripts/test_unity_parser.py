#!/usr/bin/env python3
"""Regression tests for the deterministic Unity semantic parser.

The checked fixture is synthetic and intentionally contains isolated evidence
for Unity YAML relationships, script GUID resolution, semantic risk cues, and
natural-language exploration instructions.  These tests do not claim that the
fixture or its inferred zones are empirical cybersickness observations.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIXTURE = REPO_ROOT / "results" / "unity_semantic_fixture"
PARSER_CLI = SCRIPT_DIR / "unity_parser.py"

sys.path.insert(0, str(SCRIPT_DIR))

from unity_parser import UnityProjectParser  # noqa: E402


class UnitySemanticParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE.is_dir():
            raise AssertionError(f"Synthetic Unity fixture is missing: {FIXTURE}")

    def parse_fixture(self) -> dict:
        parser = UnityProjectParser(FIXTURE)
        result = parser.parse(
            "\n".join(
                [
                    "这是一个恐怖游戏",
                    "重点关注RotatingPlatform",
                    "忽略DoorButton",
                ]
            ),
            use_llm=False,
        )
        parser.validate_result(result)
        return result

    def test_file_inventory_and_rule_output_schema(self) -> None:
        parser = UnityProjectParser(FIXTURE)
        files = parser.extract_scene_files()

        self.assertEqual(1, len(files["scenes"]))
        self.assertEqual(1, len(files["prefabs"]))
        self.assertEqual(3, len(files["scripts"]))
        self.assertEqual(2, len(files["settings"]))

        result = parser.parse(use_llm=False)
        parser.validate_result(result)
        self.assertEqual("rules", result["analysis_scope"]["backend"])
        self.assertTrue(result["analysis_scope"]["synthetic_input"])
        self.assertEqual(
            {"scene": 1, "prefab": 1, "script": 3, "settings": 2},
            dict(Counter(item["kind"] for item in result["analysis_scope"]["files_received"])),
        )
        self.assertEqual("1.1.0", result["schema_version"])
        self.assertEqual(9, len(result))

    def test_yaml_components_transforms_scripts_and_risk_words(self) -> None:
        result = self.parse_fixture()
        objects = {item["name"]: item for item in result["objects"]}

        self.assertEqual([2.0, 1.0, 4.0], objects["DoorButton"]["position"])
        self.assertTrue(objects["DoorButton"]["interactable"])
        self.assertEqual("interactive", objects["DoorButton"]["type"])
        door_evidence = " ".join(ref["observation"] for ref in objects["DoorButton"]["source_refs"])
        self.assertIn("BoxCollider", door_evidence)
        self.assertIn("DoorButton", door_evidence)

        # Console_01 has only a collider plus a GUID-resolved DoorButton script.
        # Its world position also exercises the serialized parent Transform.
        self.assertEqual([3.0, 1.0, 5.0], objects["Console_01"]["position"])
        self.assertEqual("world", objects["Console_01"]["position_space"])
        self.assertTrue(objects["Console_01"]["interactable"])
        self.assertEqual("interactive", objects["Console_01"]["type"])

        self.assertIn("HighSpeedElevator", objects)
        self.assertTrue(objects["HighSpeedElevator"]["interactable"])
        self.assertEqual("prefab_local", objects["HighSpeedElevator"]["position_space"])
        interactions = {item["object_name"]: item for item in result["interaction_points"]}
        self.assertEqual("press", interactions["DoorButton"]["interaction_type"])
        self.assertEqual("prefab_local", interactions["HighSpeedElevator"]["position_space"])

        risk_pairs = {
            (item["zone_name"], item["risk_type"])
            for item in result["risk_zones"]
        }
        expected_pairs = {
            ("FastCoasterZone", "fast_motion"),
            # This cue depends on resolving ForcedMotionZone.cs through its GUID.
            ("FastCoasterZone", "high_rotation"),
            ("RotatingPlatform", "high_rotation"),
            ("DenseParticleStorm", "high_complexity"),
            ("NarrowCorridor", "narrow_space"),
            ("CliffEdge_DropZone", "height_change"),
            ("HighSpeedElevator", "height_change"),
        }
        self.assertTrue(expected_pairs.issubset(risk_pairs))
        self.assertNotIn(("HighSpeedElevator", "high_rotation"), risk_pairs)
        self.assertTrue(all(item["severity"] == "unknown" for item in result["risk_zones"]))
        self.assertEqual(
            len(result["risk_zones"]), len({item["id"] for item in result["risk_zones"]})
        )

    def test_user_instruction_changes_priorities_and_records_context(self) -> None:
        result = self.parse_fixture()

        self.assertEqual("game", result["project_type"]["category"])
        self.assertEqual(
            ["RotatingPlatform"], result["user_instructions"]["focus_areas"]
        )
        self.assertEqual(["DoorButton"], result["user_instructions"]["avoid_areas"])

        priorities = {
            (item["target_kind"], item["target_name"]): item
            for item in result["exploration_priority"]
        }
        rotating = priorities[("risk_zone", "RotatingPlatform")]
        self.assertEqual(10, rotating["priority"])
        self.assertIn("matched user focus", rotating["reason"].casefold())
        self.assertNotIn(("interaction_point", "DoorButton"), priorities)
        elevator_interaction = priorities[("interaction_point", "HighSpeedElevator")]
        self.assertIn("world-space pose", " ".join(elevator_interaction["prerequisites"]))

    def test_cli_writes_only_valid_json_result_and_status(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as directory:
            output = Path(directory) / "result.json"
            status = Path(directory) / "status.json"
            command = [
                sys.executable,
                str(PARSER_CLI),
                "--project",
                str(FIXTURE),
                "--output",
                str(output),
                "--status",
                str(status),
                "--instruction",
                "重点关注RotatingPlatform",
                "--instruction",
                "忽略DoorButton",
            ]
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", env=environment, check=False
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            UnityProjectParser.validate_result(payload)
            status_payload = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual("success", status_payload["status"])
            self.assertTrue(status_payload["schema_valid"])

    def test_llm_request_without_api_key_stops_without_fake_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as directory:
            output = Path(directory) / "must_not_exist.json"
            status = Path(directory) / "status.json"
            output.write_text("stale-result-must-be-preserved", encoding="utf-8")
            environment = os.environ.copy()
            environment.pop("OPENAI_API_KEY", None)
            environment["PYTHONUTF8"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PARSER_CLI),
                    "--project",
                    str(FIXTURE),
                    "--output",
                    str(output),
                    "--status",
                    str(status),
                    "--use-llm",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
                check=False,
            )

            self.assertEqual(2, completed.returncode)
            self.assertEqual("stale-result-must-be-preserved", output.read_text(encoding="utf-8"))
            status_payload = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual("not_run", status_payload["status"])
            self.assertEqual("llm", status_payload["backend"])
            self.assertIn("OPENAI_API_KEY", status_payload["reason"])
            self.assertFalse(status_payload["output_written"])
            self.assertTrue(status_payload["preexisting_output_detected"])

    def test_validator_rejects_invalid_nested_values_and_references(self) -> None:
        valid = self.parse_fixture()
        mutations = []

        bad = copy.deepcopy(valid)
        bad["analysis_scope"] = "not-an-object"
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["project_type"]["confidence"] = True
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["objects"][0]["position"] = [float("nan"), "bad", {}]
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["objects"][0]["source_refs"][0]["path"] = "../escape.unity"
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["risk_zones"][0]["object_ids"] = ["missing-object"]
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["exploration_priority"][0]["target_id"] = "missing-target"
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["objects"][0]["name"] = 123
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["interaction_points"][0]["runtime_validation_required"] = "oops"
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["risk_zones"][0]["reasoning"] = {}
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["exploration_priority"][0]["objectives"] = 123
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["uncertainties"][0]["reason"] = False
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["analysis_scope"]["files_received"][0]["complete"] = 1
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["objects"][0]["interactable"] = 0
        mutations.append(bad)

        bad = copy.deepcopy(valid)
        bad["objects"][0]["source_refs"][0]["kind"] = "unknown-kind"
        mutations.append(bad)

        for payload in mutations:
            with self.subTest(payload=payload.get("schema_version")):
                with self.assertRaises(Exception):
                    UnityProjectParser.validate_result(payload)

    def test_cli_rejects_same_result_and_status_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "results") as directory:
            shared = Path(directory) / "shared.json"
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PARSER_CLI),
                    "--project",
                    str(FIXTURE),
                    "--output",
                    str(shared),
                    "--status",
                    str(shared),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
                check=False,
            )
            self.assertEqual(1, completed.returncode)
            self.assertFalse(shared.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
