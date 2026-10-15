from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "demo" / "generic_scene_agent" / "GenericSceneAgentBrain.cs"
CONTRACT = ROOT / "demo" / "generic_scene_agent" / "GenericSceneContract.cs"
CONTROLLER = (
    ROOT
    / "demo"
    / "UnityNormalHmdAgentDemo"
    / "Assets"
    / "Scripts"
    / "NormalHmdAgentController.cs"
)


class GenericSceneAgentInteractionLockSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = BRAIN.read_text(encoding="utf-8-sig")
        cls.contract = CONTRACT.read_text(encoding="utf-8-sig")
        cls.controller = CONTROLLER.read_text(encoding="utf-8-sig")

    def test_settle_gate_precedes_invocation(self) -> None:
        interact = self.brain.index("if (state == BrainState.Interact)")
        settle = self.brain.index(
            "time - interactionMotionLockStartedAt < InteractionSettleSeconds", interact
        )
        invoke = self.brain.index("TryInvokeInteraction(current, out evidence)", interact)
        self.assertLess(settle, invoke)
        self.assertIn("private const float InteractionSettleSeconds = 0.35f;", self.brain)

    def test_invocation_and_postcondition_polling_are_split_across_frames(self) -> None:
        invoke_method = self.brain.index("private bool TryInvokeInteraction(")
        poll_method = self.brain.index("private bool TryPollInteractionVerification(")
        self.assertLess(invoke_method, poll_method)
        self.assertNotIn("InvokeAndVerify", self.brain)
        self.assertIn("target.interactionInvokedFrame = Time.frameCount;", self.brain)
        self.assertIn(
            "if (Time.frameCount <= target.interactionInvokedFrame)", self.brain
        )
        observe = self.brain.index("if (state == BrainState.Observe)")
        poll_call = self.brain.index(
            "TryPollInteractionVerification(current, out verificationEvidence)", observe
        )
        mark_visited = self.brain.index("current.visited = true;", observe)
        self.assertLess(poll_call, mark_visited)

    def test_required_interaction_waits_unvisited_until_lock_deadline(self) -> None:
        observe = self.brain.index("if (state == BrainState.Observe)")
        pending = self.brain.index("bool requiredInteractionPending", observe)
        pending_return = self.brain.index("current.visited = false;", pending)
        mark_visited = self.brain.index("current.visited = true;", pending)
        self.assertLess(pending, pending_return)
        self.assertLess(pending_return, mark_visited)
        self.assertIn("time >= interactionMotionLockDeadline", self.brain)
        self.assertIn("private const float InteractionTransactionTimeoutSeconds = 30f;", self.brain)
        self.assertIn("Required interaction timed out:", self.brain)
        self.assertNotIn("Required interaction failed:", self.brain)

    def test_required_failure_stops_fail_closed_instead_of_reinvoking(self) -> None:
        interact = self.brain.index("if (state == BrainState.Interact)")
        observe = self.brain.index("if (state == BrainState.Observe)", interact)
        interact_block = self.brain[interact:observe]
        self.assertIn(
            "if (!invoked && current.contract.interaction.required)", interact_block
        )
        self.assertIn("MarkContentExhausted(time,", interact_block)

        timeout = self.brain.index("bool requiredInteractionFailed =")
        timeout_block = self.brain[timeout : timeout + 2200]
        exhaust = timeout_block.index("MarkContentExhausted(time,")
        optional_retry = timeout_block.index(
            "timedOutTarget.unreachableUntil = time + 6f;"
        )
        self.assertLess(exhaust, optional_retry)
        self.assertIn("potentially non-idempotent", timeout_block)

    def test_reflected_ienumerator_is_started_as_a_coroutine(self) -> None:
        invoke_method = self.brain.index("private bool TryInvokeInteraction(")
        poll_method = self.brain.index("private bool TryPollInteractionVerification(")
        invoke_block = self.brain[invoke_method:poll_method]
        self.assertIn("using System.Collections;", self.brain)
        self.assertIn("IEnumerator coroutine = returned as IEnumerator;", invoke_block)
        self.assertIn("MonoBehaviour coroutineHost = component as MonoBehaviour;", invoke_block)
        self.assertIn("coroutineHost.StartCoroutine(coroutine);", invoke_block)

    def test_failed_invocation_cannot_be_verified_by_preexisting_state(self) -> None:
        poll = self.brain.index("private bool TryPollInteractionVerification(")
        poll_block = self.brain[poll : poll + 2400]
        invocation_gate = poll_block.index(
            "if (!target.interactionInvocationSucceeded)"
        )
        state_read = poll_block.index(
            "object stateValue = ReadMemberPath(component, action.successMember);"
        )
        self.assertLess(invocation_gate, state_read)
        self.assertIn("postcondition cannot authorize success", poll_block)

    def test_root_commands_are_hard_zeroed_before_planning(self) -> None:
        guide = self.brain.index("public void GuideRootMotion(")
        lock = self.brain.index("if (interactionMotionLocked)", guide)
        planning = self.brain.index("Vector3 desiredWorldVelocity = Vector3.zero;", guide)
        self.assertLess(lock, planning)
        lock_block = self.brain[lock:planning]
        self.assertIn("smoothedVelocity = Vector3.zero;", lock_block)
        self.assertIn("smoothedYawRate = 0f;", lock_block)
        self.assertIn("executedRight = 0f;", lock_block)
        self.assertIn("executedForward = 0f;", lock_block)
        self.assertIn("executedYawRate = 0f;", lock_block)

    def test_post_interaction_observation_releases_only_after_final_zero_command(self) -> None:
        observe = self.brain.index("bool finishesInteractionTransaction")
        release_request = self.brain.index(
            "RequestInteractionMotionLockRelease(time,", observe
        )
        release = self.brain.index("private void ReleaseInteractionMotionLock")
        self.assertLess(observe, release_request)
        self.assertIn("final zero-root command emitted", self.brain)
        self.assertLess(release_request, release)

    def test_ground_offset_is_held_and_head_model_still_runs(self) -> None:
        platform = self.brain.index("public bool TryGetPlatformPose")
        platform_block = self.brain[platform : platform + 900]
        self.assertIn(
            "interactionMotionLocked || interactionMotionLockedForLastCommand",
            platform_block,
        )
        predict = self.controller.index("lastStep = runtime.PredictHeads();")
        root_driver = self.controller.index("motionDriver.GuideRootMotion(", predict)
        apply_head = self.controller.index("SetCurrentHeadState(lastApplied);", root_driver)
        self.assertLess(predict, root_driver)
        self.assertLess(root_driver, apply_head)

    def test_lock_has_runtime_evidence_fields(self) -> None:
        for field in (
            "interactionMotionLockAcquisitions",
            "interactionMotionLockReleases",
            "interactionMotionLockedTelemetryRows",
            "interactionMotionLockViolations",
            "interactionHeadMotionTelemetryRows",
            "interactionMotionLockPassed",
        ):
            self.assertIn(field, self.contract)
            self.assertIn(field, self.brain)
        self.assertIn("interaction_motion_locked,interaction_lock_target", self.brain)

    def test_generic_acceptance_cannot_pass_a_fixed_60_second_partial_run(self) -> None:
        self.assertNotIn('reason == "60_seconds_complete"', self.brain)
        self.assertNotIn("legacyPass", self.brain)
        self.assertIn(
            "bool passed = adaptivePass && failures.Count == 0;", self.brain
        )

    def test_runtime_log_suffix_is_bounded_for_windows_unity_file_apis(self) -> None:
        self.assertIn(
            "if (projectToken.Length > 24) projectToken = projectToken.Substring(0, 24);",
            self.brain,
        )


if __name__ == "__main__":
    unittest.main()
