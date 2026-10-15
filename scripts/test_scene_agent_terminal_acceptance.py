import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "demo" / "scene_agent_terminal.py"
SPEC = importlib.util.spec_from_file_location("scene_agent_terminal_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
terminal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terminal
SPEC.loader.exec_module(terminal)


def passing_report() -> dict[str, object]:
    return {
        "status": "passed",
        "completionReason": "content_complete",
        "contentCompletionSatisfied": True,
        "simulatedSeconds": 75.0,
        "minimumEvidenceSeconds": 15.0,
        "telemetryRows": 2778,
        "targetsDeclared": 4,
        "targetsResolved": 4,
        "targetsVisited": 4,
        "unfinishedTargetCount": 0,
        "interactionsDeclared": 1,
        "interactionsAttempted": 1,
        "interactionsRequired": 1,
        "interactionsRequiredVerified": 1,
        "interactionMotionLockPassed": True,
        "interactionMotionLockAcquisitions": 1,
        "interactionMotionLockReleases": 1,
        "interactionMotionLockedTelemetryRows": 18,
        "interactionMotionLockViolations": 0,
        "interactionHeadMotionTelemetryRows": 14,
        "collisionSafetyMode": (
            "full-height-capsule-sweep+stable-side-steering+inertia-hard-stop"
        ),
        "collisionSafetyPassed": True,
        "collisionOverlapDetections": 0,
        "failures": [],
    }


class GenericAcceptanceGateTests(unittest.TestCase):
    def test_complete_safe_run_passes(self) -> None:
        self.assertTrue(terminal.completed_generic_acceptance_run(passing_report()))

    def test_legacy_partial_60_second_report_is_rejected(self) -> None:
        report = passing_report()
        report.update({
            "completionReason": "60_seconds_complete",
            "contentCompletionSatisfied": False,
            "targetsDeclared": 10,
            "targetsResolved": 10,
            "targetsVisited": 3,
        })
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_interaction_lock_failure_is_rejected(self) -> None:
        report = passing_report()
        report["interactionMotionLockPassed"] = False
        report["interactionMotionLockViolations"] = 99
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_interaction_requires_locked_rows_and_live_headlocal(self) -> None:
        report = passing_report()
        report["interactionMotionLockedTelemetryRows"] = 0
        report["interactionHeadMotionTelemetryRows"] = 0
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_collision_overlap_is_rejected(self) -> None:
        report = passing_report()
        report["collisionOverlapDetections"] = 1
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_collision_safety_failure_is_rejected(self) -> None:
        report = passing_report()
        report["collisionSafetyPassed"] = False
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_visited_position_hint_without_object_is_accepted(self) -> None:
        position_hint = passing_report()
        position_hint["targetsResolved"] = 3
        self.assertTrue(terminal.completed_generic_acceptance_run(position_hint))

    def test_invalid_resolved_count_or_unvisited_content_is_rejected(self) -> None:
        impossible = passing_report()
        impossible["targetsResolved"] = 5
        self.assertFalse(terminal.completed_generic_acceptance_run(impossible))
        unvisited = passing_report()
        unvisited["targetsVisited"] = 3
        unvisited["unfinishedTargetCount"] = 1
        self.assertFalse(terminal.completed_generic_acceptance_run(unvisited))

    def test_nonzero_unfinished_count_is_rejected(self) -> None:
        report = passing_report()
        report["unfinishedTargetCount"] = 1
        self.assertFalse(terminal.completed_generic_acceptance_run(report))

    def test_runtime_source_manifest_covers_collision_and_lock_implementation(self) -> None:
        hashes = terminal.repository_file_hashes(
            REPOSITORY, terminal.GENERIC_RUNTIME_SOURCE_FILES
        )
        self.assertIn(
            "demo/generic_scene_agent/GenericSceneAgentBrain.cs", hashes
        )
        self.assertIn(
            "demo/UnityNormalHmdAgentDemo/Assets/Scripts/NormalHmdAgentController.cs",
            hashes,
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
