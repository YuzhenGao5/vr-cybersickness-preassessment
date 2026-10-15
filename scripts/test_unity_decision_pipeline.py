#!/usr/bin/env python3
"""Regression tests for the Unity Decision Brain v0 data/training contract."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_unity_decision_data import (  # noqa: E402
    PreparationError,
    _group_id,
    encode_candidate,
    normalize_candidate,
    normalize_context,
    prepare,
)
from export_unity_decision_brain import (  # noqa: E402
    build_arg_parser as build_export_arg_parser,
    export as export_existing_checkpoint,
)
from train_unity_decision_brain import (  # noqa: E402
    build_arg_parser,
    train,
    weighted_mean,
)
from unity_decision_policy import (  # noqa: E402
    DEFAULT_CONFIG,
    UNITY_FLAT_SCHEMA_VERSION,
    UnityDecisionRanker,
    apply_normalization,
    assert_unity_flat_payload,
    candidate_feature_names,
    evaluate_predictions,
    global_feature_names,
    load_config,
    predict_numpy_from_unity_flat_export,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def candidate(target: str, priority: float, distance: float, *, risk: bool) -> dict:
    return {
        "targetId": target,
        "semanticType": "risk_candidate" if risk else "landmark",
        "priority": priority,
        "relativeX": distance * 0.6,
        "relativeY": 0.0,
        "relativeZ": distance * 0.8,
        "distance": distance,
        "bearingSin": 0.6,
        "bearingCos": 0.8,
        "resolved": True,
        "reachabilityKnown": True,
        "reachable": True,
        "visibilityKnown": True,
        "visible": distance < 3.0,
        "navMeshProjected": True,
        "visited": False,
        "cooldownSeconds": 0.0,
        "interactionAvailable": False,
        "interactionRequired": False,
        "riskCandidate": risk,
        "selectable": True,
        "baseUtility": priority * 2.0 - distance * 0.055,
        "randomJitter": 0.0,
        "totalUtility": priority * 2.0 - distance * 0.055,
    }


def fixture_records(group_count: int = 9, decisions_per_group: int = 6) -> tuple[list[dict], list[dict]]:
    contexts: list[dict] = []
    outcomes: list[dict] = []
    for group in range(group_count):
        for decision in range(decisions_per_group):
            episode_id = f"synthetic-episode-{group}"
            decision_id = f"decision-{decision}"
            items = [
                candidate(f"g{group}-near", 0.7 + group * 0.01, 1.2, risk=False),
                candidate(f"g{group}-risk", 1.4 + decision * 0.02, 3.0, risk=True),
                candidate(f"g{group}-far", 0.9, 7.5, risk=False),
            ]
            chosen = int(np.argmax([item["totalUtility"] for item in items]))
            is_stop = decision == decisions_per_group - 1
            chosen_skill = "Stop" if is_stop else "Navigate"
            contexts.append(
                {
                    "schemaVersion": "unity-decision-context-1.0",
                    "utc": "2026-10-14T00:00:00Z",
                    "episodeId": episode_id,
                    "decisionId": decision_id,
                    "projectId": f"SyntheticProject{group}",
                    "sceneName": f"SyntheticScene{group}",
                    "projectFamily": f"fixture-family-{group}",
                    "sceneFamily": f"layout-{group}",
                    "seed": 1000 + group,
                    "sourceKind": "rule_bootstrap",
                    "humanDemonstration": False,
                    "syntheticFixture": True,
                    "simulatedTime": float(decision * 4),
                    "state": "Replan",
                    "previousSkill": "Navigate" if decision else "Unknown",
                    "bodyPosition": {"x": 100.0 + group, "y": 0.0, "z": -50.0},
                    "bodyYawDegrees": 20.0,
                    "coverage": decision / decisions_per_group,
                    "replans": decision,
                    "plannerMode": "unity-navmesh-corner-following",
                    "decisionKind": "skill_transition" if decision == 1 else "target_selection",
                    "candidates": items,
                    "validSkills": ["Navigate", "Stop"],
                    "chosenTargetIndex": -1 if is_stop else chosen,
                    "chosenTargetId": "" if is_stop else items[chosen]["targetId"],
                    "chosenSkill": chosen_skill,
                    "labelSource": "rule_bootstrap",
                    "humanAccepted": "",
                    "humanCorrectionTargetId": "",
                    "humanCorrectionSkill": "",
                }
            )
            before = decision / decisions_per_group
            after = min(1.0, before + (0.0 if is_stop else 0.1))
            outcomes.append(
                {
                    "schemaVersion": "unity-decision-outcome-1.0",
                    "utc": "2026-10-14T00:00:01Z",
                    "episodeId": episode_id,
                    "decisionId": decision_id,
                    "projectId": f"SyntheticProject{group}",
                    "sceneName": f"SyntheticScene{group}",
                    "seed": 1000 + group,
                    "simulatedTime": float(decision * 4 + 2),
                    "chosenTargetId": "" if is_stop else items[chosen]["targetId"],
                    "chosenSkill": chosen_skill,
                    "reached": not is_stop,
                    "visited": not is_stop,
                    "interactionAttempted": False,
                    "interactionVerified": False,
                    "timedOut": False,
                    "collisionRecoveriesDelta": 0,
                    "coverageBefore": before,
                    "coverageAfter": after,
                    "coverageGain": after - before,
                    "elapsedSeconds": 2.0,
                    "terminalState": "Complete" if is_stop else "Observe",
                    "evidence": "explicit synthetic fixture",
                    "syntheticFixture": True,
                }
            )
    return contexts, outcomes


class FeatureContractTests(unittest.TestCase):
    def test_config_freezes_both_seeds_at_42(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            payload["split"]["seed"] = 20261014
            path = Path(directory) / "bad_seed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freezes split.seed and training.seed at 42"):
                load_config(path)

    def test_config_cannot_redefine_human_provenance_invariants(self) -> None:
        original = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        mutations = [
            ("human_sources", lambda value: value["human_source_kinds"].append("rule_bootstrap")),
            ("allowed_sources", lambda value: value["allowed_source_kinds"].remove("unity_human")),
            (
                "rule_claim",
                lambda value: value["claims"].__setitem__(
                    "rule_bootstrap_is_human_data", True
                ),
            ),
            (
                "risk_claim",
                lambda value: value["claims"].__setitem__(
                    "risk_score_is_a_decision_label", True
                ),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in mutations:
                with self.subTest(name=name):
                    payload = json.loads(json.dumps(original))
                    mutate(payload)
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_source_weight_is_absolute_per_sample_not_batch_normalized(self) -> None:
        loss = torch.tensor([2.0])
        bootstrap = weighted_mean(loss, torch.tensor([0.25]))
        human = weighted_mean(loss, torch.tensor([1.0]))
        self.assertAlmostEqual(0.5, float(bootstrap))
        self.assertAlmostEqual(2.0, float(human))

    def test_group_family_is_unicode_and_case_canonicalized(self) -> None:
        left = {
            "splitFamily": "  ＦＡＭＩＬＹ-A  ",
            "projectId": "ignored-a",
            "sceneName": "ignored-a",
        }
        right = {
            "splitFamily": "family-a",
            "projectId": "ignored-b",
            "sceneName": "ignored-b",
        }
        self.assertEqual(_group_id(left), _group_id(right))

    def test_nested_draft_is_normalized_to_canonical_fields(self) -> None:
        normalized = normalize_context(
            {
                "schema_version": "unity-decision-context-1.0",
                "episode": {
                    "episode_id": "episode-a",
                    "project_id": "project-a",
                    "scene_name": "scene-a",
                    "seed": 42,
                },
                "provenance": {
                    "source_kind": "rule_bootstrap",
                    "human_demonstration": False,
                },
                "observation": {
                    "global": {"brain_state": "Replan"},
                    "candidates": [candidate("target-a", 1.0, 2.0, risk=False)],
                },
                "allowed": {"valid_skills": ["Navigate", "Stop"]},
                "label": {
                    "decision_id": "decision-a",
                    "target_index": 0,
                    "target_id": "target-a",
                    "skill": "Navigate",
                },
            }
        )
        self.assertEqual("episode-a", normalized["episodeId"])
        self.assertEqual("decision-a", normalized["decisionId"])
        self.assertEqual("Replan", normalized["brainState"])
        self.assertEqual(["Navigate", "Stop"], normalized["validSkills"])

    def test_identifiers_and_absolute_position_are_not_model_features(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        names = [*global_feature_names(config), *candidate_feature_names(config)]
        lowered = {name.lower() for name in names}
        for forbidden in (
            "targetid", "target_id", "scene_id", "scene_name", "project_id",
            "project_name", "body_position", "absolute_x", "absolute_z",
        ):
            self.assertNotIn(forbidden, lowered)
        for unsupported in (
            "path_length",
            "analysis_confidence",
            "dynamic_object",
            "path_length_known",
            "analysis_confidence_known",
        ):
            self.assertNotIn(unsupported, lowered)

        left = candidate("object-a", 1.0, 2.0, risk=True)
        right = dict(left, targetId="completely-different-id")
        np.testing.assert_array_equal(
            encode_candidate(normalize_candidate(left, 0), config),
            encode_candidate(normalize_candidate(right, 0), config),
        )

    def test_ranker_is_candidate_permutation_equivariant(self) -> None:
        torch.manual_seed(42)
        model = UnityDecisionRanker(5, 7, 6, dropout=0.0).eval()
        global_values = torch.randn(2, 5)
        candidates = torch.randn(2, 4, 7)
        mask = torch.ones(2, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            original, _, _ = model(global_values, candidates, mask, mask)
            changed, _, _ = model(
                global_values, candidates[:, permutation], mask[:, permutation], mask[:, permutation]
            )
        np.testing.assert_allclose(
            original.numpy()[:, permutation], changed.numpy(), atol=1e-6, rtol=1e-6
        )

    def test_target_metrics_ignore_skill_transition_rows(self) -> None:
        metrics = evaluate_predictions(
            raw_target_logits=np.asarray([[2.0, 1.0], [9.0, 1.0]]),
            candidate_mask=np.asarray([[True, True], [False, True]]),
            target_index=np.asarray([0, 0]),
            raw_skill_logits=np.asarray([[2.0, 1.0], [2.0, 1.0]]),
            skill_mask=np.asarray([[True, True], [True, True]]),
            skill_index=np.asarray([0, 0]),
            teacher_utilities=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            skill_names=["Navigate", "Stop"],
            target_label_available=np.asarray([True, False]),
        )
        self.assertEqual(1, metrics["target"]["labelled_samples"])
        self.assertEqual(0.0, metrics["target"]["illegal_rate_before_mask"])
        self.assertEqual(
            1.0, metrics["target"]["baselines"]["first_legal_top1_accuracy"]
        )
        self.assertEqual(1.0, metrics["skill"]["baselines"]["first_legal_accuracy"])


class PreparationAndTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        contexts, outcomes = fixture_records()
        self.context_path = self.root / "contexts.jsonl"
        self.outcome_path = self.root / "outcomes.jsonl"
        write_jsonl(self.context_path, contexts)
        write_jsonl(self.outcome_path, outcomes)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def training_arguments(self, data_dir: Path, stem: str = "check"):
        return build_arg_parser().parse_args(
            [
                "--data-dir", str(data_dir),
                "--model", str(self.root / f"{stem}.pth"),
                "--export", str(self.root / f"{stem}.json"),
                "--metrics", str(self.root / f"{stem}_metrics.json"),
                "--history", str(self.root / f"{stem}_history.csv"),
                "--epochs", "2",
                "--threads", "1",
                "--seed", "42",
                "--allow-smoke",
                "--allow-synthetic-fixture",
            ]
        )

    def write_variant(
        self, name: str, contexts: list[dict], outcomes: list[dict]
    ) -> tuple[Path, Path]:
        context_path = self.root / f"{name}_contexts.jsonl"
        outcome_path = self.root / f"{name}_outcomes.jsonl"
        write_jsonl(context_path, contexts)
        write_jsonl(outcome_path, outcomes)
        return context_path, outcome_path

    def test_flat_schema_requires_decision_kind(self) -> None:
        contexts, outcomes = fixture_records(group_count=3, decisions_per_group=2)
        del contexts[0]["decisionKind"]
        context_path, outcome_path = self.write_variant("missing_kind", contexts, outcomes)
        with self.assertRaisesRegex(PreparationError, "decisionKind"):
            prepare(
                context_paths=[context_path],
                outcome_paths=[outcome_path],
                output_dir=self.root / "missing_kind",
                allow_smoke=True,
                allow_synthetic_fixture=True,
            )

    def test_outcome_seed_target_and_skill_must_match_context(self) -> None:
        mutations = {
            "seed": lambda outcome: outcome.__setitem__("seed", outcome["seed"] + 1),
            "chosenTargetId": lambda outcome: outcome.__setitem__(
                "chosenTargetId", "different-target"
            ),
            "chosenSkill": lambda outcome: outcome.__setitem__("chosenSkill", "Stop"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                contexts, outcomes = fixture_records(group_count=3, decisions_per_group=2)
                mutate(outcomes[0])
                context_path, outcome_path = self.write_variant(field, contexts, outcomes)
                with self.assertRaisesRegex(PreparationError, field):
                    prepare(
                        context_paths=[context_path],
                        outcome_paths=[outcome_path],
                        output_dir=self.root / f"bad_{field}",
                        allow_smoke=True,
                        allow_synthetic_fixture=True,
                    )

    def test_accepted_human_label_cannot_also_apply_a_correction(self) -> None:
        contexts, outcomes = fixture_records(group_count=3, decisions_per_group=2)
        contexts[0]["sourceKind"] = "unity_human"
        contexts[0]["labelSource"] = "unity_human"
        contexts[0]["humanDemonstration"] = True
        contexts[0]["humanAccepted"] = True
        contexts[0]["humanCorrectionTargetId"] = contexts[0]["chosenTargetId"]
        context_path, outcome_path = self.write_variant("human_conflict", contexts, outcomes)
        with self.assertRaisesRegex(PreparationError, "cannot also contain corrections"):
            prepare(
                context_paths=[context_path],
                outcome_paths=[outcome_path],
                output_dir=self.root / "human_conflict",
                allow_smoke=True,
                allow_synthetic_fixture=True,
            )

    def test_variable_candidate_counts_are_padded_and_masked(self) -> None:
        contexts, outcomes = fixture_records()
        for index, context in enumerate(contexts):
            if index % 2:
                context["candidates"] = context["candidates"][:-1]
        context_path, outcome_path = self.write_variant("variable", contexts, outcomes)
        data_dir = self.root / "variable_prepared"
        prepare(
            context_paths=[context_path],
            outcome_paths=[outcome_path],
            output_dir=data_dir,
            allow_smoke=True,
            allow_synthetic_fixture=True,
        )
        present_counts: set[int] = set()
        for split in ("train", "val", "test"):
            with np.load(data_dir / f"unity_decision_{split}.npz") as archive:
                present_counts.update(archive["candidate_mask"].sum(axis=1).tolist())
        self.assertEqual({3, 4}, present_counts)

    def test_training_rejects_manifest_feature_order_tampering(self) -> None:
        data_dir = self.root / "feature_tamper"
        prepare(
            context_paths=[self.context_path], outcome_paths=[self.outcome_path],
            output_dir=data_dir, allow_smoke=True, allow_synthetic_fixture=True,
        )
        manifest_path = data_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["global_feature_names"] = list(reversed(manifest["global_feature_names"]))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "global feature names/order"):
            train(self.training_arguments(data_dir, "feature_tamper"))

    def test_training_rejects_non_42_split_manifest(self) -> None:
        data_dir = self.root / "seed_tamper"
        prepare(
            context_paths=[self.context_path], outcome_paths=[self.outcome_path],
            output_dir=data_dir, allow_smoke=True, allow_synthetic_fixture=True,
        )
        split_path = data_dir / "split_manifest.json"
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        split_payload["seed"] = 99
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")
        manifest_path = data_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["split_manifest_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen seed 42"):
            train(self.training_arguments(data_dir, "seed_tamper"))

    def test_training_rejects_npz_hash_tampering(self) -> None:
        data_dir = self.root / "npz_tamper"
        prepare(
            context_paths=[self.context_path], outcome_paths=[self.outcome_path],
            output_dir=data_dir, allow_smoke=True, allow_synthetic_fixture=True,
        )
        split_path = data_dir / "unity_decision_train.npz"
        payload = bytearray(split_path.read_bytes())
        payload[len(payload) // 2] ^= 1
        split_path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "split hash"):
            train(self.training_arguments(data_dir, "npz_tamper"))

    def test_synthetic_fixture_is_rejected_without_explicit_test_flag(self) -> None:
        with self.assertRaisesRegex(PreparationError, "synthetic fixture rejected"):
            prepare(
                context_paths=[self.context_path],
                outcome_paths=[self.outcome_path],
                output_dir=self.root / "rejected",
                allow_smoke=True,
                allow_synthetic_fixture=False,
            )

    def test_undersized_non_synthetic_data_is_not_silently_promoted(self) -> None:
        contexts, outcomes = fixture_records()
        for record in contexts:
            record["syntheticFixture"] = False
        for record in outcomes:
            record["syntheticFixture"] = False
        context_path = self.root / "small_real_contexts.jsonl"
        outcome_path = self.root / "small_real_outcomes.jsonl"
        write_jsonl(context_path, contexts)
        write_jsonl(outcome_path, outcomes)
        with self.assertRaisesRegex(PreparationError, "Insufficient production data"):
            prepare(
                context_paths=[context_path],
                outcome_paths=[outcome_path],
                output_dir=self.root / "too_small",
                allow_smoke=False,
                allow_synthetic_fixture=False,
            )

    def test_provenance_cannot_call_rule_bootstrap_human(self) -> None:
        contexts, outcomes = fixture_records(group_count=3, decisions_per_group=2)
        contexts[0]["humanDemonstration"] = True
        bad_contexts = self.root / "bad_contexts.jsonl"
        bad_outcomes = self.root / "bad_outcomes.jsonl"
        write_jsonl(bad_contexts, contexts)
        write_jsonl(bad_outcomes, outcomes)
        with self.assertRaisesRegex(PreparationError, "requires humanDemonstration=false"):
            prepare(
                context_paths=[bad_contexts],
                outcome_paths=[bad_outcomes],
                output_dir=self.root / "bad",
                allow_smoke=True,
                allow_synthetic_fixture=True,
            )

    def test_label_source_cannot_hide_mixed_provenance(self) -> None:
        contexts, outcomes = fixture_records(group_count=3, decisions_per_group=2)
        contexts[0]["labelSource"] = "scripted_bootstrap"
        bad_contexts = self.root / "mixed_source_contexts.jsonl"
        bad_outcomes = self.root / "mixed_source_outcomes.jsonl"
        write_jsonl(bad_contexts, contexts)
        write_jsonl(bad_outcomes, outcomes)
        with self.assertRaisesRegex(PreparationError, "mixed provenance"):
            prepare(
                context_paths=[bad_contexts],
                outcome_paths=[bad_outcomes],
                output_dir=self.root / "mixed_source",
                allow_smoke=True,
                allow_synthetic_fixture=True,
            )

    def test_grouped_prepare_train_and_export(self) -> None:
        data_dir = self.root / "prepared"
        manifest = prepare(
            context_paths=[self.context_path],
            outcome_paths=[self.outcome_path],
            output_dir=data_dir,
            allow_smoke=True,
            allow_synthetic_fixture=True,
        )
        self.assertEqual("smoke_only", manifest["status"])
        self.assertEqual(45, manifest["target_label_samples"])
        self.assertEqual(0, manifest["human_decision_samples"])
        self.assertFalse(manifest["human_like_claim_allowed"])
        group_sets = [set(values) for values in manifest["split_groups"].values()]
        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        self.assertTrue(group_sets[0].isdisjoint(group_sets[2]))
        self.assertTrue(group_sets[1].isdisjoint(group_sets[2]))

        checkpoint = self.root / "brain.pth"
        export = self.root / "brain.json"
        unity_export = self.root / "brain_unity.json"
        metrics_path = self.root / "metrics.json"
        history = self.root / "history.csv"
        arguments = build_arg_parser().parse_args(
            [
                "--data-dir", str(data_dir),
                "--model", str(checkpoint),
                "--export", str(export),
                "--unity-export", str(unity_export),
                "--metrics", str(metrics_path),
                "--history", str(history),
                "--epochs", "4",
                "--batch-size", "16",
                "--threads", "1",
                "--seed", "42",
                "--allow-smoke",
                "--allow-synthetic-fixture",
            ]
        )
        metrics = train(arguments)
        self.assertEqual("smoke_only", metrics["status"])
        self.assertEqual(42, metrics["seed"])
        self.assertEqual(0, metrics["human_decision_samples"])
        self.assertEqual(
            "refused_no_human_decisions",
            metrics["metrics"]["test"]["human_metrics"]["status"],
        )
        self.assertEqual(
            0.0, metrics["metrics"]["test"]["target"]["illegal_rate_after_mask"]
        )
        self.assertEqual(
            0.0, metrics["metrics"]["test"]["skill"]["illegal_rate_after_mask"]
        )
        self.assertIn(
            "rule_base_utility_top1_accuracy",
            metrics["metrics"]["test"]["target"]["baselines"],
        )
        self.assertIn(
            "first_legal_accuracy",
            metrics["metrics"]["test"]["skill"]["baselines"],
        )
        coverage = metrics["metrics"]["test"]["skill_label_coverage"]
        self.assertFalse(coverage["all_six_skills_observed"])
        self.assertIsNotNone(coverage["warning"])
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(export.is_file())
        self.assertTrue(unity_export.is_file())
        self.assertTrue(metrics_path.is_file())
        exported = json.loads(export.read_text(encoding="utf-8"))
        unity_exported = json.loads(unity_export.read_text(encoding="utf-8"))
        self.assertEqual(UNITY_FLAT_SCHEMA_VERSION, unity_exported["schemaVersion"])
        assert_unity_flat_payload(unity_exported)
        self.assertEqual(metrics["status"], unity_exported["dataStatus"])
        self.assertEqual(
            metrics["artifacts"]["checkpoint_sha256"],
            unity_exported["modelSha256"],
        )
        self.assertEqual(
            metrics["data_fingerprint_sha256"],
            unity_exported["dataFingerprintSha256"],
        )
        for key, value in unity_exported.items():
            self.assertNotIsInstance(value, dict, key)
            if isinstance(value, list):
                self.assertFalse(any(isinstance(item, (dict, list)) for item in value), key)
        for prefix in (
            "globalEncoderLinear1", "globalEncoderLinear2",
            "candidateEncoderLinear1", "candidateEncoderLinear2",
            "targetHeadLinear1", "targetHeadLinear2",
            "skillHeadLinear1", "skillHeadLinear2",
        ):
            self.assertEqual(
                unity_exported[f"{prefix}InputSize"]
                * unity_exported[f"{prefix}OutputSize"],
                len(unity_exported[f"{prefix}Weights"]),
            )
            self.assertEqual(
                unity_exported[f"{prefix}OutputSize"],
                len(unity_exported[f"{prefix}Bias"]),
            )
        self.assertFalse(exported["training_provenance"]["human_like_claim_allowed"])
        self.assertEqual(0, exported["training_provenance"]["human_decision_samples"])
        self.assertIn("weights", exported)
        self.assertEqual("unity-decision-context-1.0", exported["context_schema_version"])
        self.assertTrue(exported["preprocessing"]["append_stop_candidate"])
        routing = exported["skill_conditioning_by_decision_kind"]
        self.assertTrue(routing["target_selection"]["target_rank_loss"])
        self.assertFalse(routing["skill_transition"]["target_rank_loss"])
        self.assertIn(
            "active target", routing["skill_transition"]["skill_conditioning_target"]
        )
        stop_contract = exported["preprocessing"]["virtual_stop_candidate"]
        self.assertEqual("__STOP__", stop_contract["raw_template"]["targetId"])
        self.assertEqual(
            len(exported["candidate_feature_names"]),
            len(stop_contract["encoded_feature_vector"]),
        )
        config = load_config(DEFAULT_CONFIG)
        np.testing.assert_allclose(
            encode_candidate(normalize_candidate(stop_contract["raw_template"], 0), config),
            np.asarray(stop_contract["encoded_feature_vector"], dtype=np.float32),
        )
        self.assertNotIn("target_id", " ".join(exported["candidate_feature_names"]).lower())

        with np.load(data_dir / "unity_decision_train.npz", allow_pickle=False) as archive:
            train_global = archive["global_features"]
            train_candidate = archive["candidate_features"]
            train_candidate_mask = archive["candidate_mask"].astype(bool)
        np.testing.assert_allclose(
            np.asarray(exported["normalization"]["global_mean"]),
            train_global.astype(np.float64).mean(axis=0),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(exported["normalization"]["candidate_mean"]),
            train_candidate[train_candidate_mask].astype(np.float64).mean(axis=0),
            atol=1e-7,
        )

        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        architecture = exported["architecture"]
        model_options = {
            "global_hidden_dim": architecture["global_hidden_dim"],
            "candidate_hidden_dim": architecture["candidate_hidden_dim"],
            "joint_hidden_dim": architecture["joint_hidden_dim"],
            "dropout": architecture["dropout"],
        }
        checkpoint_model = UnityDecisionRanker(
            architecture["global_dim"], architecture["candidate_dim"],
            architecture["skill_count"], **model_options,
        ).eval()
        checkpoint_model.load_state_dict(checkpoint_payload["state_dict"])
        json_model = UnityDecisionRanker(
            architecture["global_dim"], architecture["candidate_dim"],
            architecture["skill_count"], **model_options,
        ).eval()
        json_state = {
            name: torch.tensor(value["values"], dtype=torch.float32)
            for name, value in exported["weights"].items()
        }
        json_model.load_state_dict(json_state)
        normalized_global, normalized_candidate = apply_normalization(
            train_global, train_candidate, exported["normalization"]
        )
        global_tensor = torch.from_numpy(normalized_global[:3])
        candidate_tensor = torch.from_numpy(normalized_candidate[:3])
        present_tensor = torch.from_numpy(train_candidate_mask[:3])
        with np.load(data_dir / "unity_decision_train.npz", allow_pickle=False) as archive:
            target_tensor = torch.from_numpy(archive["target_mask"][:3].astype(bool))
        with torch.no_grad():
            checkpoint_outputs = checkpoint_model(
                global_tensor, candidate_tensor, present_tensor, target_tensor
            )
            json_outputs = json_model(
                global_tensor, candidate_tensor, present_tensor, target_tensor
            )
        for expected, actual in zip(checkpoint_outputs, json_outputs, strict=True):
            np.testing.assert_allclose(expected.numpy(), actual.numpy(), atol=1e-7)

        # The dictionary-free Unity asset must independently reproduce the
        # checkpoint for its held-out golden decision.  This protects layer
        # order, row-major flattening, normalization, pooling, masking, and
        # decision-kind skill conditioning in one small fixture.
        golden_candidate_count = unity_exported["goldenCandidateCount"]
        golden_global = np.asarray(
            unity_exported["goldenGlobalFeatures"], dtype=np.float32
        )
        golden_candidates = np.asarray(
            unity_exported["goldenCandidateFeatures"], dtype=np.float32
        ).reshape(golden_candidate_count, unity_exported["candidateDim"])
        golden_present = np.asarray(
            unity_exported["goldenCandidateMask"], dtype=bool
        )
        golden_legal = np.asarray(unity_exported["goldenTargetMask"], dtype=bool)
        selected_index = int(unity_exported["goldenSkillConditioningTargetIndex"])
        numpy_golden = predict_numpy_from_unity_flat_export(
            unity_exported,
            golden_global,
            golden_candidates,
            golden_present,
            golden_legal,
            selected_target_index=selected_index,
        )
        np.testing.assert_allclose(
            numpy_golden["raw_target_logits"],
            unity_exported["goldenExpectedRawTargetLogits"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            numpy_golden["raw_skill_logits"],
            unity_exported["goldenExpectedRawSkillLogits"],
            atol=1e-7,
        )
        with torch.no_grad():
            torch_golden = checkpoint_model(
                torch.from_numpy(
                    np.asarray(
                        unity_exported["goldenNormalizedGlobalFeatures"],
                        dtype=np.float32,
                    )[None, :]
                ),
                torch.from_numpy(
                    np.asarray(
                        unity_exported["goldenNormalizedCandidateFeatures"],
                        dtype=np.float32,
                    ).reshape(
                        1, golden_candidate_count, unity_exported["candidateDim"]
                    )
                ),
                torch.from_numpy(golden_present[None, :]),
                torch.from_numpy(golden_legal[None, :]),
                selected_target_indices=torch.tensor([selected_index]),
            )
        np.testing.assert_allclose(
            torch_golden[0][0].numpy(),
            unity_exported["goldenExpectedRawTargetLogits"],
            atol=2e-5,
        )
        np.testing.assert_allclose(
            torch_golden[2][0].numpy(),
            unity_exported["goldenExpectedRawSkillLogits"],
            atol=2e-5,
        )

        checkpoint_only_export = self.root / "brain_unity_checkpoint_only.json"
        export_result = export_existing_checkpoint(
            build_export_arg_parser().parse_args(
                [
                    "--data-dir", str(data_dir),
                    "--checkpoint", str(checkpoint),
                    "--python-export", str(export),
                    "--output", str(checkpoint_only_export),
                    "--allow-smoke",
                    "--allow-synthetic-fixture",
                ]
            )
        )
        self.assertEqual("success", export_result["status"])
        checkpoint_only_payload = json.loads(
            checkpoint_only_export.read_text(encoding="utf-8")
        )
        self.assertEqual(
            unity_exported["goldenExpectedRawTargetLogits"],
            checkpoint_only_payload["goldenExpectedRawTargetLogits"],
        )
        self.assertEqual(
            unity_exported["modelSha256"], checkpoint_only_payload["modelSha256"]
        )

        bad_seed_arguments = build_arg_parser().parse_args(
            [
                "--data-dir", str(data_dir),
                "--seed", "41",
                "--allow-smoke",
                "--allow-synthetic-fixture",
            ]
        )
        with self.assertRaisesRegex(ValueError, "override 41 rejected"):
            train(bad_seed_arguments)


if __name__ == "__main__":
    unittest.main()
