using System;
using UnityEngine;

namespace GenericSceneAgent
{
    /// <summary>
    /// Machine-readable hand-off between the offline Unity/LLM analyser and the
    /// runtime agent.  The runtime never treats prose as executable code: the
    /// prose is retained for audit, while targets and interactions are explicit.
    /// </summary>
    [Serializable]
    public sealed class GenericSceneContract
    {
        public string schemaVersion;
        public string projectId;
        public string projectName;
        public string sceneName;
        public string generatedAtUtc;
        public string generator;
        public string sourceReportPath;
        public string reportSummary;
        public string[] sceneSemantics;
        public string[] safeInteractionMethods;
        public string[] disabledBehaviourTypes;
        public RuntimeDiscoveryHint[] runtimeDiscoveryHints;
        public ContractSourceFinding[] sourceFindings;
        public ContractSpawn spawn;
        public ContractTarget[] targets;
        public string[] declaredLimitations;
    }

    [Serializable]
    public sealed class RuntimeDiscoveryHint
    {
        public string hintId;
        public string targetId;
        public string[] objectNamePatterns;
        public string[] requiredComponents;
        public string sourcePath;
        public string locator;
        public string rationale;
        public float confidence;
        public bool discoveryOnly = true;
    }

    [Serializable]
    public sealed class ContractSourceFinding
    {
        public string path;
        public string kind;
        public string locator;
        public string finding;
        public float confidence;
    }

    [Serializable]
    public sealed class ContractSpawn
    {
        public Vector3 position;
        public float yawDegrees;
        public string evidence;
    }

    [Serializable]
    public sealed class ContractTarget
    {
        public string id;
        public string label;
        public string semanticType;
        public string objectNameContains;
        public Vector3 positionHint;
        public Vector3 approachOffset;
        public float arrivalRadius = 0.8f;
        public float observeSeconds = 1.8f;
        public float priority = 1f;
        public string reason;
        public ContractInteraction interaction;
    }

    [Serializable]
    public sealed class ContractInteraction
    {
        public string verb;
        public string componentType;
        public string method;
        public string successMember;
        public string expectedValue;
        public bool required;
    }

    [Serializable]
    public sealed class RuntimeSceneInventory
    {
        public string schemaVersion = "generic-scene-runtime-inventory-1.0";
        public string projectId;
        public string sceneName;
        public int rootObjectCount;
        public int activeObjectCount;
        public int rendererCount;
        public int colliderCount;
        public int behaviourCount;
        public string[] rootObjects;
        public string[] behaviourTypes;
        public RuntimeTargetResolution[] targetResolutions;
        public string walkabilityMode;
        public BoundsSnapshot inferredWalkableBounds;
        public string capturedAtUtc;
    }

    [Serializable]
    public sealed class RuntimeTargetResolution
    {
        public string targetId;
        public string requestedNameContains;
        public string resolvedObjectPath;
        public Vector3 resolvedWorldPosition;
        public Vector3 approachPosition;
        public bool objectResolved;
        public bool navMeshProjected;
        public string resolutionEvidence;
    }

    [Serializable]
    public sealed class BoundsSnapshot
    {
        public Vector3 center;
        public Vector3 size;
    }

    [Serializable]
    public sealed class GenericAgentAcceptance
    {
        public string status;
        public string generatedAtUtc;
        public string projectId;
        public string sceneName;
        public string episodeId;
        public int seed;
        public float simulatedSeconds;
        public int telemetryRows;
        public int decisions;
        public int targetsDeclared;
        public int targetsResolved;
        public int targetsVisited;
        public float coverage;
        public int interactionsDeclared;
        public int interactionsRequired;
        public int interactionsAttempted;
        public int interactionsVerified;
        public int interactionsRequiredVerified;
        public int interactionMotionLockAcquisitions;
        public int interactionMotionLockReleases;
        public int interactionMotionLockedTelemetryRows;
        public int interactionMotionLockViolations;
        public float interactionMotionLockMaxLinearSpeed;
        public float interactionMotionLockMaxAbsYawRate;
        public int interactionHeadMotionTelemetryRows;
        public float interactionHeadMotionMaxPositionDelta;
        public float interactionHeadMotionMaxAngularDelta;
        public bool interactionMotionLockPassed;
        public int replans;
        public int collisionRecoveries;
        public int collisionAvoidanceAdjustments;
        public int collisionHardStops;
        public int collisionOverlapDetections;
        public bool collisionSafetyPassed;
        public float maxCollisionBlockedSeconds;
        public string collisionSafetyMode;
        public int groundHitCount;
        public int groundMissCount;
        public int groundRejectedStepCount;
        public string plannerMode;
        public string movementSource;
        public string brainSource;
        public string contractPath;
        public string runtimeInventoryPath;
        public string decisionsPath;
        public string decisionContextsPath;
        public string decisionOutcomesPath;
        public string decisionBrainModeRequested;
        public string policyAuthority;
        public string decisionBrainModelStatus;
        public string decisionBrainModelDetail;
        public string decisionBrainModelSha256;
        public string decisionBrainModelSchemaVersion;
        public string decisionBrainDataStatus;
        public bool decisionBrainGoldenSelfTestPassed;
        public bool decisionBrainAllInferenceLogitsFinite;
        public string decisionBrainInferencePath;
        public int decisionBrainInferences;
        public int decisionBrainFallbacks;
        public int decisionBrainDisagreements;
        public bool decisionBrainResearchOverride;
        public string telemetryPath;
        public int decisionContexts;
        public int decisionOutcomes;
        public string cameraAuditPath;
        public string screenshotPath;
        public string completionReason;
        public bool contentCompletionSatisfied;
        public int unfinishedTargetCount;
        public float minimumEvidenceSeconds;
        public float noProgressTimeoutSeconds;
        public float safetyTimeoutSeconds;
        public float lastMeaningfulProgressAtSeconds;
        public string[] failures;
    }

    /// <summary>
    /// One candidate as it was known immediately before a high-level action was
    /// selected.  Unknown visibility and reachability are represented by their
    /// explicit *Known flags rather than by optimistic defaults.
    /// </summary>
    [Serializable]
    public sealed class UnityDecisionCandidate
    {
        public int index;
        public string targetId;
        public string semanticType;
        public float priority;
        public float relativeX;
        public float relativeY;
        public float relativeZ;
        public float distance;
        public float bearingSin;
        public float bearingCos;
        public bool resolved;
        public bool reachabilityKnown;
        public bool reachable;
        public bool visibilityKnown;
        public bool visible;
        public bool navMeshProjected;
        public bool visited;
        public bool cooldown;
        public float cooldownSeconds;
        public bool interactionDeclared;
        public bool interactionAvailable;
        public bool interactionRequired;
        public bool interactionAttempted;
        public bool interactionVerified;
        public bool riskCandidate;
        public bool selectable;
        public string ineligibleReason;
        public bool utilityIncludesNewJitter;
        public float baseUtility;
        public float randomJitter;
        public float totalUtility;
    }

    /// <summary>
    /// Leakage-safe pre-action record for future imitation/ranking training.
    /// It deliberately contains no success, elapsed-action or future-coverage
    /// fields; those are written separately to UnityDecisionOutcome.
    /// </summary>
    [Serializable]
    public sealed class UnityDecisionContext
    {
        public string schemaVersion = "unity-decision-context-1.0";
        public string utc;
        public string episodeId;
        public string decisionId;
        public string projectId;
        public string sceneName;
        public int seed;
        public string sourceKind = "rule_bootstrap";
        public bool humanDemonstration;
        public float simulatedTime;
        public string decisionKind;
        public string state;
        public string previousSkill;
        public Vector3 bodyPosition;
        public float bodyX;
        public float bodyY;
        public float bodyZ;
        public float bodyYawDegrees;
        public float coverage;
        public int replans;
        public string plannerMode;
        public UnityDecisionCandidate[] candidates;
        public string[] validSkills;
        public int chosenTargetIndex;
        public string chosenTargetId;
        public string chosenSkill;
        public string labelSource = "rule_bootstrap";
        // Empty strings are intentional annotation slots. JsonUtility does not
        // support nullable scalar fields consistently across Unity versions.
        public string humanAccepted;
        public string humanCorrectionTargetId;
        public string humanCorrectionSkill;
        public string humanCorrectionNotes;
    }

    /// <summary>
    /// Post-action record joined to UnityDecisionContext by decisionId.  This is
    /// the only record allowed to contain results that occur after selection.
    /// </summary>
    [Serializable]
    public sealed class UnityDecisionOutcome
    {
        public string schemaVersion = "unity-decision-outcome-1.0";
        public string utc;
        public string episodeId;
        public string decisionId;
        public string projectId;
        public string sceneName;
        public int seed;
        public float simulatedTime;
        public float completedSimulatedTime;
        public float elapsedSeconds;
        public string chosenTargetId;
        public string chosenSkill;
        public bool reached;
        public bool visited;
        public bool interactionAttempted;
        public bool interactionVerified;
        public bool timedOut;
        public int collisionRecoveriesDelta;
        public float coverageBefore;
        public float coverageAfter;
        public float coverageGain;
        public bool terminal;
        public string terminalState;
        public string evidence;
    }

    /// <summary>
    /// Auditable policy comparison.  Shadow records a model proposal while the
    /// rule remains authoritative; Model records both proposals and the actually
    /// executed, safety-masked choice.  Logits follow scene-candidate order with
    /// the virtual STOP candidate appended at the end.
    /// </summary>
    [Serializable]
    public sealed class UnityDecisionBrainInference
    {
        public string schemaVersion = "unity-decision-brain-inference-1.0";
        public string utc;
        public string episodeId;
        public string decisionId;
        public float simulatedTime;
        public string decisionKind;
        public string requestedMode;
        public string policyAuthority;
        public string modelStatus;
        public string modelSha256;
        public bool researchOverride;
        public bool modelEvaluated;
        public bool fallback;
        public string fallbackReason;
        public int ruleTargetIndex;
        public string ruleTargetId;
        public string ruleSkill;
        public int modelTargetIndex;
        public string modelTargetId;
        public string modelSkill;
        public int executedTargetIndex;
        public string executedTargetId;
        public string executedSkill;
        public bool targetDisagreement;
        public bool skillDisagreement;
        public string[] validSkills;
        public float[] rawTargetLogits;
        public float[] maskedTargetLogits;
        public float[] rawSkillLogits;
        public float[] maskedSkillLogits;
    }

    [Serializable]
    public sealed class GenericExperienceReport
    {
        public string schemaVersion = "generic-scene-experience-report-1.0";
        public string status;
        public string projectId;
        public string sceneName;
        public int seed;
        public float simulatedSeconds;
        public int targetsVisited;
        public int targetsDeclared;
        public float coverage;
        public int interactionsDeclared;
        public int interactionsVerified;
        public string interactionFinding;
        public float meanCameraSpeed;
        public float maxCameraSpeed;
        public float meanCameraAngularSpeed;
        public float maxCameraAngularSpeed;
        public float maxCameraAcceleration;
        public float meanEngineeringExposureProxy;
        public float maxEngineeringExposureProxy;
        public string riskMeaning;
        public string[] observedSceneSemantics;
        public string[] recommendations;
        public string[] limitations;
    }
}
