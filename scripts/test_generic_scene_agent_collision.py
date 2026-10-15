from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRAIN = (
    ROOT / "demo" / "generic_scene_agent" / "GenericSceneAgentBrain.cs"
)
CONTRACT = ROOT / "demo" / "generic_scene_agent" / "GenericSceneContract.cs"


class GenericSceneAgentCollisionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brain = BRAIN.read_text(encoding="utf-8-sig")
        cls.contract = CONTRACT.read_text(encoding="utf-8-sig")

    def test_overlap_is_sampled_before_the_interaction_lock_early_return(self) -> None:
        guide = self.brain.index("public void GuideRootMotion(")
        sample = self.brain.index("SampleCollisionOverlapEpisode(bodyPosition);", guide)
        lock_return = self.brain.index("if (interactionMotionLocked)", guide)
        self.assertLess(sample, lock_return)

    def test_overlap_count_uses_entry_edges_and_fails_internal_acceptance(self) -> None:
        sampler = self.brain.index("private void SampleCollisionOverlapEpisode(")
        sampler_block = self.brain[sampler : sampler + 900]
        self.assertIn("overlapping && !collisionOverlapActive", sampler_block)
        self.assertIn("collisionOverlapDetections++;", sampler_block)
        self.assertIn("collisionOverlapActive = overlapping;", sampler_block)
        self.assertIn("if (collisionOverlapDetections > 0)", self.brain)
        self.assertIn("collisionSafetyPassed = collisionOverlapDetections == 0", self.brain)
        self.assertIn("public bool collisionSafetyPassed;", self.contract)

    def test_ground_projection_ignores_triggers_steep_hits_and_participant(self) -> None:
        projector = self.brain.index("private static Vector3 ProjectToWalkableSurface(")
        helper = self.brain.index("private static bool IsWalkableGroundHit(", projector)
        projector_block = self.brain[projector:helper]
        helper_block = self.brain[helper : helper + 1300]
        self.assertIn("QueryTriggerInteraction.Ignore", projector_block)
        self.assertIn("IsWalkableGroundHit(hit, colliders)", projector_block)
        self.assertIn("collider.isTrigger", helper_block)
        self.assertIn("hit.normal.y < 0.55f", helper_block)
        self.assertIn("GetComponentInParent<NormalHmdAgentController>()", helper_block)
        self.assertIn("Array.IndexOf(colliders, collider)", helper_block)

    def test_capsule_sweep_can_distinguish_walkable_foot_support(self) -> None:
        cast = self.brain.index("private bool CapsuleCastEnvironment(")
        support = self.brain.index("private bool IsWalkableSupportHit(", cast)
        cast_block = self.brain[cast:support]
        support_block = self.brain[support : support + 2800]
        self.assertIn("IsWalkableSupportHit(hit, bodyPosition, direction)", cast_block)
        self.assertIn("maximumStepHeight", support_block)
        self.assertIn("maximumSupportDrop", support_block)
        self.assertIn("hit.collider.Raycast(", support_block)
        self.assertIn("IsWalkableGroundHit(supportHit, null)", support_block)


if __name__ == "__main__":
    unittest.main()
