#!/usr/bin/env python3
"""Regression tests for the Habitat-Web decision-prior pipeline."""

from __future__ import annotations

import copy
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from habitat_decision_policy import (  # noqa: E402
    ACTIONS,
    ACTION_TO_INDEX,
    FEATURE_NAMES,
    HabitatDecisionMLP,
    export_runtime_json,
    predict_numpy_from_export,
)
from prepare_habitat_decision_data import (  # noqa: E402
    DecisionSample,
    Reservoir,
    extract_episode_samples,
    goal_waypoints_for_episode,
    load_config,
    prepare_dataset,
    split_scene_ids,
)
from train_habitat_decision_prior import (  # noqa: E402
    fit_train_normalization,
    geometric_rule_predictions,
    metric_bundle,
    repeat_previous_predictions,
    train,
)


def state(
    position: list[float],
    yaw_quaternion_y: float = 0.0,
    sensor_pitch_quaternion_x: float = 0.0,
    include_sensor: bool = True,
) -> dict:
    y = float(yaw_quaternion_y)
    w = float(np.sqrt(max(0.0, 1.0 - y * y)))
    result = {
        "position": position,
        "rotation": [0.0, y, 0.0, w],
    }
    if include_sensor:
        x = float(sensor_pitch_quaternion_x)
        sensor_w = float(np.sqrt(max(0.0, 1.0 - x * x)))
        result["sensor_data"] = {
            "rgb": {
                "position": [0.0, 0.88, 0.0],
                "rotation": [x, 0.0, 0.0, sensor_w],
            }
        }
    return result


def fixture_episode(scene: str, suffix: str = "0") -> dict:
    actions = [
        "STOP",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "LOOK_UP",
        "LOOK_DOWN",
        "MOVE_FORWARD",
        "STOP",
    ]
    states = [
        state([0.0, 0.0, 0.0]),
        state([0.0, 0.0, -0.25]),
        state([0.0, 0.0, -0.25], -0.25),
        state([0.0, 0.0, -0.25], 0.0),
        state([0.0, 0.0, -0.25]),
        state([0.0, 0.0, -0.25]),
        state([0.0, 0.0, -0.50]),
        state([0.0, 0.0, -0.50]),
    ]
    return {
        "episode_id": f"{scene}:{suffix}",
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "object_category": "chair",
        "goals": [],
        "reference_replay": [
            {"action": action, "agent_state": agent_state}
            for action, agent_state in zip(actions, states)
        ],
    }


def fixture_goals(scene: str) -> dict:
    return {
        f"{scene}.glb_chair": [
            {
                "position": [0.0, 0.0, -0.6],
                "view_points": [
                    {"agent_state": state([0.0, 0.0, -0.5])},
                    {"agent_state": state([0.2, 0.0, -0.5])},
                ],
            }
        ]
    }


def write_fixture_dataset(root: Path, scene_count: int = 6) -> None:
    content = root / "train" / "content"
    content.mkdir(parents=True)
    for index in range(scene_count):
        scene = f"scene{index}"
        document = {
            "goals_by_category": fixture_goals(scene),
            "episodes": [fixture_episode(scene, str(item)) for item in range(3)],
        }
        with gzip.open(content / f"{scene}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(document, handle)


class SceneSplitTests(unittest.TestCase):
    def test_scene_split_is_deterministic_disjoint_and_nonempty(self) -> None:
        scenes = [f"scene{index}" for index in range(11)]
        first = split_scene_ids(
            scenes, seed=42, ratios={"train": 0.7, "val": 0.15, "test": 0.15}
        )
        second = split_scene_ids(
            list(reversed(scenes)),
            seed=42,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        )
        self.assertEqual(first, second)
        self.assertTrue(all(first.values()))
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))
        self.assertEqual(set(scenes), set().union(*map(set, first.values())))


class GoalAndAlignmentTests(unittest.TestCase):
    def test_empty_episode_goals_resolve_from_top_level_category_key(self) -> None:
        episode = fixture_episode("abc")
        waypoints, source = goal_waypoints_for_episode(episode, fixture_goals("abc"))
        self.assertEqual("view_point", source)
        self.assertEqual(2, len(waypoints))
        np.testing.assert_allclose(waypoints[0], [0.0, 0.0, -0.5])

    def test_placeholder_stop_removed_but_final_stop_retained(self) -> None:
        samples, audit = extract_episode_samples(
            fixture_episode("abc"),
            goals_by_category=fixture_goals("abc"),
            history_run_clip=10,
        )
        labels = [ACTIONS[sample.label] for sample in samples]
        self.assertEqual(7, len(samples))
        self.assertEqual("MOVE_FORWARD", labels[0])
        self.assertEqual("STOP", labels[-1])
        self.assertEqual(1, labels.count("STOP"))
        self.assertEqual(1, audit["first_placeholder_stop_removed"])
        self.assertEqual(1, audit["final_stop_present"])

    def test_resultant_state_and_future_actions_cannot_leak_into_input(self) -> None:
        original = fixture_episode("abc")
        changed = copy.deepcopy(original)
        # action[1] is predicted from state[0].  Its resultant state[1] and all
        # later actions/states must not affect sample zero.
        changed["reference_replay"][1]["agent_state"] = state(
            [99.0, 7.0, -42.0], 0.8, sensor_pitch_quaternion_x=0.5
        )
        changed["reference_replay"][2]["action"] = "LOOK_DOWN"
        original_samples, _ = extract_episode_samples(
            original, goals_by_category=fixture_goals("abc"), history_run_clip=10
        )
        changed_samples, _ = extract_episode_samples(
            changed, goals_by_category=fixture_goals("abc"), history_run_clip=10
        )
        self.assertEqual(original_samples[0].label, changed_samples[0].label)
        np.testing.assert_array_equal(
            original_samples[0].features, changed_samples[0].features
        )
        # The following sample is allowed to differ: state[1] is then history.
        self.assertFalse(
            np.array_equal(original_samples[1].features, changed_samples[1].features)
        )

    def test_sensor_pitch_comes_from_pre_action_observation_with_identity_fallback(self) -> None:
        episode = fixture_episode("abc")
        episode["reference_replay"][0]["agent_state"] = state(
            [0.0, 0.0, 0.0], sensor_pitch_quaternion_x=0.2588190451
        )
        samples, audit = extract_episode_samples(
            episode, goals_by_category=fixture_goals("abc"), history_run_clip=10
        )
        forward_vertical = FEATURE_NAMES.index("current_forward_vertical")
        up_vertical = FEATURE_NAMES.index("current_up_vertical")
        self.assertGreater(abs(float(samples[0].features[forward_vertical])), 0.49)
        self.assertLess(float(samples[0].features[up_vertical]), 0.90)
        self.assertEqual(0, audit["sensor_rotation_fallback_states"])

        missing = fixture_episode("abc")
        missing["reference_replay"][0]["agent_state"] = state(
            [0.0, 0.0, 0.0], include_sensor=False
        )
        fallback_samples, fallback_audit = extract_episode_samples(
            missing, goals_by_category=fixture_goals("abc"), history_run_clip=10
        )
        self.assertEqual(0.0, fallback_samples[0].features[forward_vertical])
        self.assertEqual(1.0, fallback_samples[0].features[up_vertical])
        self.assertEqual(1, fallback_audit["sensor_rotation_fallback_states"])

    def test_label_is_not_encoded_as_previous_action(self) -> None:
        samples, _ = extract_episode_samples(
            fixture_episode("abc"),
            goals_by_category=fixture_goals("abc"),
            history_run_clip=10,
        )
        previous_start = FEATURE_NAMES.index("previous_action_MOVE_FORWARD")
        self.assertEqual(0.0, float(np.sum(samples[0].features[previous_start : previous_start + 6])))
        # Second label is TURN_LEFT, but previous-action input is MOVE_FORWARD.
        self.assertEqual(ACTION_TO_INDEX["TURN_LEFT"], samples[1].label)
        self.assertEqual(1.0, samples[1].features[previous_start])
        self.assertEqual(0.0, samples[1].features[previous_start + ACTION_TO_INDEX["TURN_LEFT"]])


class ReservoirAndPreparationTests(unittest.TestCase):
    def test_reservoir_is_bounded_and_reproducible(self) -> None:
        def populate(seed: int) -> Reservoir:
            reservoir = Reservoir(7, seed)
            for index in range(100):
                reservoir.add(
                    DecisionSample(
                        features=np.full(len(FEATURE_NAMES), index, dtype=np.float32),
                        label=index % len(ACTIONS),
                        scene_key="scene",
                        episode_hash=np.uint64(index),
                    )
                )
            return reservoir

        first = populate(42).arrays({"scene": 0})
        second = populate(42).arrays({"scene": 0})
        self.assertEqual(7, len(first["y"]))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_end_to_end_preparation_preserves_scene_groups(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "habitat_decision_prior_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "prepared"
            write_fixture_dataset(root)
            manifest = prepare_dataset(
                config,
                data_root=root,
                output_dir=output,
                max_samples_override={"train": 100, "val": 100, "test": 100},
                max_episodes_per_scene=2,
            )
            self.assertEqual("public_human", manifest["source_kind"])
            self.assertEqual("CC BY-NC 4.0", manifest["license"])
            self.assertEqual(
                "lowest sha256(seed:scene:episode_id), independently within every scene",
                manifest["episode_subset_policy"],
            )
            self.assertTrue(all(not value for value in manifest["scene_overlap_checks"].values()))
            for split_name in ("train", "val", "test"):
                with np.load(output / f"{split_name}.npz", allow_pickle=False) as data:
                    self.assertEqual(len(FEATURE_NAMES), data["X"].shape[1])
                    self.assertGreater(len(data["y"]), 0)


class TrainingUtilityTests(unittest.TestCase):
    def test_normalization_is_fitted_only_from_passed_training_array(self) -> None:
        train_features = np.vstack(
            (np.zeros(len(FEATURE_NAMES)), np.full(len(FEATURE_NAMES), 2.0))
        ).astype(np.float32)
        mean, std = fit_train_normalization(train_features)
        np.testing.assert_allclose(mean, 1.0)
        np.testing.assert_allclose(std, 1.0)
        # An extreme validation value was deliberately not passed and cannot alter stats.
        validation = np.full((1, len(FEATURE_NAMES)), 1e9, dtype=np.float32)
        self.assertGreater(float(np.max((validation - mean) / std)), 1e8)

    def test_metrics_and_baselines(self) -> None:
        labels = np.arange(len(ACTIONS), dtype=np.int64)
        probabilities = np.eye(len(ACTIONS), dtype=np.float64)
        metrics = metric_bundle(labels, labels.copy(), probabilities, ece_bins=6)
        self.assertEqual(1.0, metrics["accuracy"])
        self.assertEqual(1.0, metrics["balanced_accuracy"])
        self.assertEqual(1.0, metrics["macro_f1"])
        self.assertAlmostEqual(0.0, metrics["calibration"]["ece"])

        features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
        previous_start = FEATURE_NAMES.index("previous_action_MOVE_FORWARD")
        features[1, previous_start + ACTION_TO_INDEX["TURN_RIGHT"]] = 1.0
        np.testing.assert_array_equal(
            repeat_previous_predictions(features),
            [ACTION_TO_INDEX["MOVE_FORWARD"], ACTION_TO_INDEX["TURN_RIGHT"]],
        )
        features[:, FEATURE_NAMES.index("goal_distance_m")] = [0.1, 2.0]
        features[1, FEATURE_NAMES.index("goal_direction_right")] = -0.8
        predictions = geometric_rule_predictions(
            features,
            {
                "stop_distance_m": 0.2,
                "look_vertical_direction_threshold": 0.2,
                "turn_direction_threshold": 0.18,
            },
        )
        np.testing.assert_array_equal(
            predictions, [ACTION_TO_INDEX["STOP"], ACTION_TO_INDEX["TURN_LEFT"]]
        )

    def test_self_contained_export_matches_torch_golden_vector(self) -> None:
        torch.manual_seed(42)
        model = HabitatDecisionMLP(hidden_dims=(8, 5), dropout=0.0).eval()
        mean = np.linspace(-1.0, 1.0, len(FEATURE_NAMES), dtype=np.float32)
        std = np.linspace(0.5, 1.5, len(FEATURE_NAMES), dtype=np.float32)
        raw = np.linspace(-0.3, 0.7, len(FEATURE_NAMES), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            output = Path(directory) / "model.json"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            payload = export_runtime_json(
                model=model,
                normalization={"mean": mean, "std": std},
                output_path=output,
                checkpoint_path=checkpoint,
                metadata={"source_kind": "public_human"},
                golden_raw_features=raw,
            )
            logits, probabilities = predict_numpy_from_export(payload, raw.reshape(1, -1))
            np.testing.assert_allclose(logits[0], payload["golden_vector"]["logits"], atol=2e-5)
            np.testing.assert_allclose(
                probabilities[0], payload["golden_vector"]["probabilities"], atol=2e-5
            )


class TinyTrainingIntegrationTest(unittest.TestCase):
    def test_tiny_cpu_training_writes_all_artifacts(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "habitat_decision_prior_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            prepared = base / "prepared"
            write_fixture_dataset(source, scene_count=9)
            prepare_dataset(
                config,
                data_root=source,
                output_dir=prepared,
                max_samples_override={"train": 500, "val": 500, "test": 500},
            )
            local_config = copy.deepcopy(config)
            local_config["model"] = {"hidden_dims": [12, 8], "dropout": 0.0}
            local_config["training"].update(
                {"epochs": 1, "batch_size": 32, "patience": 1, "num_workers": 0}
            )
            local_config["outputs"] = {
                "checkpoint": str(base / "model.pth"),
                "runtime_json": str(base / "model.json"),
                "metrics": str(base / "metrics.json"),
                "history": str(base / "history.csv"),
                "metadata": str(base / "metadata.json"),
            }
            metadata = train(
                local_config,
                prepared_dir=prepared,
                device_name="cpu",
                epochs_override=1,
            )
            self.assertEqual("success", metadata["status"])
            for path in local_config["outputs"].values():
                self.assertTrue(Path(path).is_file())
            exported = json.loads((base / "model.json").read_text(encoding="utf-8"))
            self.assertEqual("public_human", exported["metadata"]["source_kind"])
            self.assertFalse(exported["metadata"]["complete_unity_brain"])


if __name__ == "__main__":
    unittest.main()
