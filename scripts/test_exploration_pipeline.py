#!/usr/bin/env python3
"""Fast regression tests for the EyeNavGS motion-imitation pipeline.

These tests use small in-memory or temporary synthetic fixtures.  They do not
train a model and do not report empirical EyeNavGS performance.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from exploration_policy import (  # noqa: E402
    FEATURE_NAMES,
    ExplorationLSTM,
    build_model,
    integrate_local_velocity,
    load_checkpoint,
    normalize_quaternions,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    relative_quaternions,
    rotate_vectors,
    rotation_vector_to_quaternion,
)
from prepare_exploration_data import (  # noqa: E402
    POSE_COLUMNS,
    pair_stereo_rows,
    participant_splits,
    trace_windows,
)
from train_exploration import load_split, metric_bundle  # noqa: E402


class QuaternionUtilityTests(unittest.TestCase):
    def test_rotation_vector_round_trip_including_zero_and_near_pi(self) -> None:
        vectors = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1e-12, -2e-12, 3e-12],
                [0.2, -0.4, 0.1],
                [0.0, np.pi - 1e-7, 0.0],
            ],
            dtype=np.float64,
        )
        reconstructed = quaternion_to_rotation_vector(
            rotation_vector_to_quaternion(vectors)
        )
        np.testing.assert_allclose(reconstructed, vectors, atol=1e-9, rtol=1e-9)

    def test_relative_rotation_is_local_and_sign_invariant(self) -> None:
        current = rotation_vector_to_quaternion(np.asarray([0.0, 0.0, np.pi / 2.0]))
        local_delta = rotation_vector_to_quaternion(np.asarray([0.25, 0.0, 0.0]))
        following = quaternion_multiply(current, local_delta)

        relative = relative_quaternions(current, -following)

        np.testing.assert_allclose(relative, local_delta, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            quaternion_to_rotation_vector(relative),
            np.asarray([0.25, 0.0, 0.0]),
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertGreaterEqual(float(relative[3]), 0.0)

    def test_local_velocity_integration_uses_current_orientation(self) -> None:
        orientation = rotation_vector_to_quaternion(
            np.asarray([0.0, 0.0, np.pi / 2.0])
        )
        next_position, next_orientation = integrate_local_velocity(
            position=np.zeros(3),
            orientation=orientation,
            local_linear_velocity=np.asarray([2.0, 0.0, 0.0]),
            local_angular_velocity=np.asarray([0.0, 0.0, 1.0]),
            dt_seconds=0.5,
        )

        np.testing.assert_allclose(next_position, [0.0, 1.0, 0.0], atol=1e-10)
        expected_orientation = quaternion_multiply(
            orientation, rotation_vector_to_quaternion(np.asarray([0.0, 0.0, 0.5]))
        )
        np.testing.assert_allclose(next_orientation, expected_orientation, atol=1e-10)

    def test_quaternion_validation_and_vector_rotation(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            normalize_quaternions(np.zeros(4))
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            normalize_quaternions(np.asarray([0.0, 0.0, np.nan, 1.0]))

        quarter_turn = rotation_vector_to_quaternion(
            np.asarray([0.0, 0.0, np.pi / 2.0])
        )
        rotated = rotate_vectors(quarter_turn, np.asarray([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-10)


class SequenceModelTests(unittest.TestCase):
    def test_lstm_consumes_true_sequence_and_returns_one_action(self) -> None:
        torch.manual_seed(7)
        model = ExplorationLSTM(
            input_dim=len(FEATURE_NAMES),
            hidden_dim=8,
            output_dim=len(FEATURE_NAMES),
            num_layers=1,
            dropout=0.5,
        ).eval()
        baseline = torch.zeros((2, 5, len(FEATURE_NAMES)), dtype=torch.float32)
        changed_history = baseline.clone()
        changed_history[:, 0, 0] = 1.0

        output = model(baseline)
        changed_output = model(changed_history)

        self.assertEqual((2, len(FEATURE_NAMES)), tuple(output.shape))
        self.assertFalse(torch.equal(output, changed_output))
        self.assertEqual(0.0, model.lstm.dropout)
        with self.assertRaisesRegex(ValueError, "batch, sequence, feature"):
            model(torch.zeros((2, 5 * len(FEATURE_NAMES))))

    def test_build_model_preserves_declared_dimensions(self) -> None:
        model = build_model(
            {
                "input_dim": 6,
                "hidden_dim": 11,
                "output_dim": 6,
                "num_layers": 2,
                "dropout": 0.2,
            }
        )
        output = model(torch.zeros((3, 8, 6), dtype=torch.float32))
        self.assertEqual((3, 6), tuple(output.shape))
        self.assertEqual(11, model.lstm.hidden_size)
        self.assertEqual(2, model.lstm.num_layers)


class CheckpointTests(unittest.TestCase):
    @staticmethod
    def valid_checkpoint() -> dict:
        model = ExplorationLSTM(input_dim=6, hidden_dim=4, output_dim=6, num_layers=1)
        return {
            "schema_version": "1.0.0",
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": 6,
                "hidden_dim": 4,
                "output_dim": 6,
                "num_layers": 1,
                "dropout": 0.0,
                "history_length": 8,
            },
            "normalization": {
                "x_mean": [0.0] * 6,
                "x_std": [1.0] * 6,
                "y_mean": [0.0] * 6,
                "y_std": [1.0] * 6,
            },
            "median_dt_seconds": 0.02,
            "action_clip_low": [-1.0] * 6,
            "action_clip_high": [1.0] * 6,
            "demo_seed_velocity": [[0.0] * 6 for _ in range(8)],
            "data_source": "EyeNavGS",
        }

    def test_checkpoint_loads_and_state_dict_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.pth"
            torch.save(self.valid_checkpoint(), path)
            checkpoint = load_checkpoint(str(path))

        model = build_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        output = model(torch.zeros((1, 8, 6), dtype=torch.float32))
        self.assertEqual((1, 6), tuple(output.shape))
        self.assertEqual("EyeNavGS", checkpoint["data_source"])

    def test_checkpoint_missing_required_field_fails_closed(self) -> None:
        checkpoint = self.valid_checkpoint()
        del checkpoint["normalization"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_checkpoint(str(path))

    def test_non_dictionary_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.pth"
            torch.save(torch.zeros(1), path)
            with self.assertRaisesRegex(ValueError, "must be a dictionary"):
                load_checkpoint(str(path))


class PreprocessingInvariantTests(unittest.TestCase):
    @staticmethod
    def stereo_frame(frame_count: int, timestamps_ms: list[float] | None = None) -> pd.DataFrame:
        timestamps = timestamps_ms or [float(index * 20) for index in range(frame_count)]
        rows: list[dict[str, float]] = []
        for index, timestamp in enumerate(timestamps):
            for view, eye_offset in ((0, -0.03), (1, 0.03)):
                rows.append(
                    {
                        "ViewIndex": view,
                        "PositionX": float(index) * 0.01 + eye_offset,
                        "PositionY": 0.0,
                        "PositionZ": 0.0,
                        "QuaternionX": 0.0,
                        "QuaternionY": 0.0,
                        "QuaternionZ": 0.0,
                        "QuaternionW": -1.0 if view else 1.0,
                        "Timestamp": timestamp,
                    }
                )
        return pd.DataFrame(rows, columns=[*POSE_COLUMNS, "Timestamp"])

    def test_stereo_pairing_averages_positions_and_aligns_quaternion_sign(self) -> None:
        frame = self.stereo_frame(3)
        positions, quaternions, timestamps, stats = pair_stereo_rows(frame, "Timestamp")

        np.testing.assert_allclose(positions[:, 0], [0.0, 0.01, 0.02], atol=1e-12)
        np.testing.assert_allclose(quaternions, [[0.0, 0.0, 0.0, 1.0]] * 3)
        np.testing.assert_allclose(timestamps, [0.0, 20.0, 40.0])
        self.assertEqual(3, stats["candidate_pairs"])
        self.assertEqual(3, stats["valid_pairs"])

    def test_participant_split_has_no_overlap_and_stratifies_each_site(self) -> None:
        records = [
            {"participant_group": f"{site}:user{index:02d}"}
            for site in ("Rutgers", "NTHU")
            for index in range(10)
        ]
        first = participant_splits(records, seed=42)
        second = participant_splits(records, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(20, len(first))
        for site in ("Rutgers", "NTHU"):
            site_assignments = {
                split for participant, split in first.items() if participant.startswith(site + ":")
            }
            self.assertEqual({"train", "val", "test"}, site_assignments)
        split_sets = {
            split: {participant for participant, assigned in first.items() if assigned == split}
            for split in ("train", "val", "test")
        }
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))

    def test_motion_windows_never_cross_timestamp_reset(self) -> None:
        # Two valid runs separated by a timestamp reset.  A history of two plus
        # one target requires three consecutive valid motion intervals.
        frame = self.stereo_frame(9, [0, 20, 40, 60, 5, 25, 45, 65, 85])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user01_alameda.csv"
            frame.to_csv(path, index=False)
            record = {"path": path, "time_column": "Timestamp"}
            x, y, dt, stats = trace_windows(
                record=record,
                scale=1.0,
                history=2,
                max_windows=0,
                min_dt=0.004,
                max_dt=0.1,
                max_linear_speed=4.0,
                max_angular_speed=15.0,
                rng=np.random.default_rng(1),
            )

        self.assertEqual((3, 2, 6), x.shape)
        self.assertEqual((3, 6), y.shape)
        np.testing.assert_allclose(dt, 0.02, atol=1e-8)
        self.assertEqual(1, stats["timestamp_resets_or_duplicates"])
        self.assertEqual(3, stats["available_windows"])


class EvaluationAndArchiveTests(unittest.TestCase):
    def test_prepared_archive_shape_and_finite_value_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.npz"
            np.savez_compressed(
                valid_path,
                X=np.zeros((4, 8, 6), dtype=np.float32),
                y=np.zeros((4, 6), dtype=np.float32),
                dt_seconds=np.full(4, 0.02, dtype=np.float32),
                trajectory_index=np.asarray([0, 0, 1, 1], dtype=np.int32),
                trajectory_ids=np.asarray(["a", "b"]),
            )
            payload = load_split(valid_path)
            self.assertEqual((4, 8, 6), payload["X"].shape)

            invalid_path = Path(directory) / "invalid.npz"
            np.savez_compressed(
                invalid_path,
                X=np.zeros((4, 48), dtype=np.float32),
                y=np.zeros((4, 6), dtype=np.float32),
                dt_seconds=np.full(4, 0.02, dtype=np.float32),
                trajectory_index=np.asarray([0, 0, 1, 1], dtype=np.int32),
                trajectory_ids=np.asarray(["a", "b"]),
            )
            with self.assertRaisesRegex(ValueError, "Unexpected X shape"):
                load_split(invalid_path)

    def test_perfect_prediction_metrics_are_zero(self) -> None:
        target = np.asarray(
            [
                [0.1, 0.0, 0.2, 0.01, -0.02, 0.03],
                [0.2, 0.1, 0.0, -0.03, 0.02, 0.01],
            ],
            dtype=np.float64,
        )
        metrics = metric_bundle(
            prediction=target.copy(),
            target=target,
            dt=np.asarray([0.02, 0.03]),
            trajectory_index=np.asarray([0, 1]),
        )
        self.assertEqual(0.0, metrics["linear_velocity_vector_mae_mps"])
        self.assertEqual(0.0, metrics["angular_velocity_vector_mae_degps"])
        self.assertEqual(0.0, metrics["position_step_error_mm"])
        self.assertAlmostEqual(
            0.0, metrics["orientation_step_geodesic_error_deg"], delta=1e-12
        )


if __name__ == "__main__":
    unittest.main()
