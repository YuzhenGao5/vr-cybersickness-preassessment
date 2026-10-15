using System;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Boundary between the trained Normal HMD motion prior and a task-level
    /// controller. Implementations may guide BodyRoot, but must not rewrite the
    /// trained HeadLocal channels or recurrent state.
    /// </summary>
    public interface IRootMotionDriver
    {
        void ResetForEpisode(int seed, out Vector3 spawn, out float yawDegrees);

        void GuideRootMotion(
            float simulatedTime,
            float dt,
            Vector3 bodyPosition,
            float bodyYawDegrees,
            float learnedRight,
            float learnedForward,
            float learnedYawRate,
            out float executedRight,
            out float executedForward,
            out float executedYawRate);

        void BeginEpisode(int seed, string normalHmdLogDirectory);
        void ObserveSimulationStep(SceneAwareStepSample sample);
        bool TryGetPlatformPose(
            float simulatedTime,
            out float verticalOffset,
            out float pitchDegrees,
            out float rollDegrees);
        void CompleteEpisode(string reason);
    }

    /// <summary>
    /// Optional lifecycle contract for scene-level drivers.  Standalone Normal
    /// HMD rollouts deliberately keep their fixed 60-second validation window;
    /// a scene-aware driver may instead finish when its declared content is
    /// exhausted, while still supplying a hard safety ceiling.
    /// </summary>
    public interface IAdaptiveEpisodeDriver
    {
        float SafetyTimeoutSeconds { get; }

        bool TryGetEpisodeCompletion(float simulatedTime, out string reason);
    }

    [Serializable]
    public struct SceneAwareStepSample
    {
        public int stepIndex;
        public float simulatedTime;
        public float deltaTime;
        public Vector3 bodyPosition;
        public float bodyYawDegrees;
        public Vector3 cameraPosition;
        public Quaternion cameraRotation;
        public Vector3 headLocalPosition;
        public Quaternion headLocalRotation;
        public float learnedRight;
        public float learnedForward;
        public float learnedYawRate;
        public float executedRight;
        public float executedForward;
        public float executedYawRate;
    }

    [Serializable]
    public sealed class SceneSemanticManifest
    {
        public string schemaVersion = "scene-aware-agent-semantic-1.0";
        public string scenarioId;
        public string scenarioName;
        public string description;
        public string generatedFrom;
        public string[] semanticTags;
        public SemanticNodeContract[] nodes;
        public SemanticEdgeContract[] edges;
        public SemanticObjectiveContract[] objectives;
        public SemanticAffordanceContract[] affordances;
    }

    [Serializable]
    public sealed class SemanticNodeContract
    {
        public string id;
        public string label;
        public string semanticType;
        public Vector3 position;
        public float arrivalRadius = 0.55f;
    }

    [Serializable]
    public sealed class SemanticEdgeContract
    {
        public string from;
        public string to;
        public float traversalCost = 1f;
        public string traversalSkill = "Navigate";
        public bool bidirectional = true;
    }

    [Serializable]
    public sealed class SemanticObjectiveContract
    {
        public string id;
        public string label;
        public string nodeId;
        public string targetAffordanceId;
        public string requiredSkill;
        public float observeSeconds = 1.5f;
    }

    [Serializable]
    public sealed class SemanticAffordanceContract
    {
        public string id;
        public string label;
        public string semanticType;
        public string verb;
        public string nodeId;
        public Vector3 approachPosition;
        public float activationDistance = 1.35f;
        public string successState;
        public string evidenceSignal;
    }

    [Serializable]
    public sealed class SceneAwareAcceptanceReport
    {
        public string status;
        public int scenarioIndex;
        public string scenarioName;
        public int seed;
        public float generatedSeconds;
        public int telemetryRowCount;
        public string telemetryPath;
        public int interactionsExpected;
        public int interactionsAttempted;
        public int interactionsSucceeded;
        public int objectivesExpected;
        public int objectivesCompleted;
        public float coverage;
        public float meanExposureProxy;
        public float maxExposureProxy;
        public int vehicleMotionRows;
        public float vehicleMaxSpeed;
        public float vehicleVerticalRange;
        public float cameraVerticalRangeDuringVehicle;
        public string exposureMeaning;
        public string plannerMeaning;
        public string completionReason;
        public string manifestPath;
        public string actionLedgerPath;
        public string[] failures;
    }
}
