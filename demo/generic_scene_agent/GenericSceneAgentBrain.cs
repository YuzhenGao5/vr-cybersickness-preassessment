using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using NormalHmdAgentDemo;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

namespace GenericSceneAgent
{
    /// <summary>
    /// Scene-independent executive controller.  It consumes a contract produced
    /// by the offline analyser, grounds it against the live Unity scene, chooses
    /// targets by novelty/priority and invokes locomotion, look, interact and
    /// observe skills.  NormalHmdAgentController remains the learned HMD motion
    /// skill; this class only schedules skills and guides BodyRoot direction.
    /// </summary>
    public sealed class GenericSceneAgentBrain : MonoBehaviour, IRootMotionDriver,
        IAdaptiveEpisodeDriver
    {
        private const float MinimumEvidenceSeconds = 15f;
        private const float NoProgressTimeoutSeconds = 45f;
        private const float MaximumEpisodeSeconds = 900f;
        private const float NavigationNoProgressTimeoutSeconds = 18f;
        private const float NavigationProgressDistanceEpsilon = 0.02f;
        private const float InteractionSettleSeconds = 0.35f;
        private const float InteractionTransactionTimeoutSeconds = 30f;
        private const float InteractionRootCommandEpsilon = 0.00001f;

        private enum BrainState { Scan, Navigate, LookAt, Interact, Observe, Replan, Complete }

        [Serializable]
        private sealed class DecisionRecord
        {
            public string utc;
            public float simulatedTime;
            public string state;
            public string skill;
            public string targetId;
            public string intent;
            public string command;
            public bool engineConfirmed;
            public string evidence;
        }

        private sealed class GroundedTarget
        {
            public ContractTarget contract;
            public Transform resolvedObject;
            public Vector3 focusPosition;
            public Vector3 approachPosition;
            public bool navMeshProjected;
            public bool reachabilityKnown;
            public bool reachable;
            public bool visited;
            public bool interactionAttempted;
            public bool interactionInvocationSucceeded;
            public bool interactionVerified;
            public Component interactionComponent;
            public object interactionBeforeState;
            public int interactionInvokedFrame = -1;
            public string interactionInvocationEvidence = "";
            public string interactionVerificationEvidence = "";
            public float unreachableUntil;
        }

        private sealed class ActiveHighLevelDecision
        {
            public string decisionId;
            public string targetId;
            public string skill;
            public float startedAt;
            public float coverageBefore;
            public int collisionRecoveriesBefore;
        }

        private sealed class PolicyChoice
        {
            public GroundedTarget target;
            public int targetIndex;
            public string skill;
            public bool failClosed;
            public string evidence;
        }

        private NormalHmdAgentController motionSkill;
        private Camera observerCamera;
        private GenericSceneContract contract;
        private readonly List<GroundedTarget> targets = new List<GroundedTarget>();
        private readonly List<string> failures = new List<string>();
        private readonly List<string> pendingDecisions = new List<string>();
        private readonly List<string> pendingDecisionContexts = new List<string>();
        private readonly List<string> pendingDecisionOutcomes = new List<string>();
        private readonly List<string> pendingDecisionBrainInferences = new List<string>();
        private readonly List<string> groundingEvidence = new List<string>();
        private System.Random random;
        private BrainState state;
        private GroundedTarget current;
        private Vector3 smoothedVelocity;
        private float smoothedYawRate;
        private float stateEnteredAt;
        private float lastSimulatedTime;
        private float lastMeaningfulProgressAt;
        private float lastNavigationProgressAt;
        private float bestNavigationDistance;
        private string navigationProgressTargetId = "";
        private float nextPlanningRetryAt;
        private string incompleteTerminalReason = "";
        private int seed;
        private int telemetryRows;
        private int decisionCount;
        private int decisionContextCount;
        private int decisionOutcomeCount;
        private int decisionBrainInferenceCount;
        private int decisionBrainFallbackCount;
        private int decisionBrainDisagreementCount;
        private int decisionSequence;
        private int interactionsAttempted;
        private int interactionsVerified;
        private bool interactionMotionLocked;
        private bool interactionMotionUnlockAfterCommand;
        private bool interactionMotionLockedForLastCommand;
        private float interactionMotionLockStartedAt;
        private float interactionMotionLockDeadline;
        private string interactionMotionLockTargetId = "";
        private string interactionMotionLockTargetIdForLastCommand = "";
        private int interactionMotionLockAcquisitions;
        private int interactionMotionLockReleases;
        private int interactionMotionLockedTelemetryRows;
        private int interactionMotionLockViolations;
        private float interactionMotionLockMaxLinearSpeed;
        private float interactionMotionLockMaxAbsYawRate;
        private int interactionHeadMotionTelemetryRows;
        private float interactionHeadMotionMaxPositionDelta;
        private float interactionHeadMotionMaxAngularDelta;
        private int replans;
        private Bounds inferredWalkableBounds;
        private bool hasInferredWalkableBounds;
        private bool hasUsableNavMesh;
        private SceneAwareStepSample previousSample;
        private bool hasPreviousSample;
        private float previousCameraSpeed;
        private float cameraSpeedSum;
        private float cameraAngularSpeedSum;
        private float exposureProxySum;
        private float maxCameraSpeed;
        private float maxCameraAngularSpeed;
        private float maxCameraAcceleration;
        private float maxExposureProxy;
        private float collisionBlockedSeconds;
        private int collisionRecoveries;
        private int collisionAvoidanceAdjustments;
        private int collisionHardStops;
        private int collisionOverlapDetections;
        private bool collisionOverlapActive;
        private float maxCollisionBlockedSeconds;
        private int preferredAvoidanceSign = 1;
        private const string CollisionSafetyMode =
            "full-height-capsule-sweep+stable-side-steering+inertia-hard-stop";
        private Vector3 lastBodyPosition;
        private float spawnGroundY;
        private float groundVerticalOffset;
        private float lastDriverDt = 0.027f;
        private int groundHitCount;
        private int groundMissCount;
        private int groundRejectedStepCount;
        private string plannerMode = "collision-checked-direct-steering(no-NavMesh)";
        private Vector3[] activeNavCorners = new Vector3[0];
        private int activeNavCornerIndex;
        private float activeNavPathBuiltAt = -999f;
        private string activeNavTargetId = "";
        private bool episodeBegun;
        private bool episodeComplete;
        private bool testMode;
        private string requestedLogRoot;
        private string testOutputPath;
        private string requestedScreenshotPath;
        private string requestedCameraAuditPath;
        private string logDirectory;
        private string contractPath;
        private string inventoryPath;
        private string decisionsPath;
        private string decisionContextsPath;
        private string decisionOutcomesPath;
        private string decisionBrainInferencePath;
        private string telemetryPath;
        private StreamWriter decisionWriter;
        private StreamWriter decisionContextWriter;
        private StreamWriter decisionOutcomeWriter;
        private StreamWriter decisionBrainInferenceWriter;
        private StreamWriter telemetryWriter;
        private string episodeId;
        private string previousSkill = "";
        private ActiveHighLevelDecision activeHighLevelDecision;
        private UnityDecisionBrainRuntime decisionBrainRuntime;
        private GUIStyle titleStyle;
        private GUIStyle bodyStyle;
        private GUIStyle learnedStyle;
        private GUIStyle ruleStyle;
        private Texture2D panelTexture;

        public void Configure(NormalHmdAgentController controller, Camera observer)
        {
            motionSkill = controller;
            observerCamera = observer;
            testMode = HasArgument("--generic-agent-test");
            requestedLogRoot = ReadArgument("--generic-agent-log-root=");
            testOutputPath = ReadArgument("--generic-agent-test-output=");
            requestedScreenshotPath = ReadArgument("--generic-agent-capture-output=");
            requestedCameraAuditPath = ReadArgument("--generic-agent-camera-audit=");
            decisionBrainRuntime = new UnityDecisionBrainRuntime();
            decisionBrainRuntime.ConfigureFromCommandLine();
            LoadContract();
            InspectAndGroundScene();
            if (testMode && motionSkill != null) motionSkill.SetSceneAwareSimulationSpeed(12f);
        }

        private void LoadContract()
        {
            TextAsset asset = Resources.Load<TextAsset>("AgentSceneContract");
            if (asset == null) throw new InvalidOperationException(
                "Resources/AgentSceneContract.json is missing. Refusing to explore without analyser output.");
            contract = JsonUtility.FromJson<GenericSceneContract>(asset.text);
            if (contract == null || contract.schemaVersion != "generic-scene-agent-contract-1.0")
                throw new InvalidOperationException("Unsupported or invalid scene contract.");
            if (contract.targets == null || contract.targets.Length == 0)
                throw new InvalidOperationException("Scene contract declares no exploration targets.");
        }

        private void InspectAndGroundScene()
        {
            // Runtime-created primitives often moved/scaled their colliders in
            // Awake.  Sync before the first ground ray or a stale unit collider
            // can be mistaken for a one-metre-high floor at the spawn point.
            Physics.SyncTransforms();
            Transform[] sceneTransforms = FindObjectsByType<Transform>(
                FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            Collider[] colliders = FindObjectsByType<Collider>(
                FindObjectsInactive.Exclude, FindObjectsSortMode.None);

            NavMeshTriangulation triangulation = NavMesh.CalculateTriangulation();
            if (triangulation.vertices != null && triangulation.vertices.Length >= 3)
            {
                hasUsableNavMesh = true;
                plannerMode = "unity-navmesh-corner-following";
            }
            inferredWalkableBounds = InferWalkableBounds(
                FindObjectsByType<Renderer>(FindObjectsInactive.Exclude, FindObjectsSortMode.None));
            hasInferredWalkableBounds = inferredWalkableBounds.size.sqrMagnitude > 0.001f;

            foreach (ContractTarget declared in contract.targets)
            {
                GroundedTarget grounded = new GroundedTarget { contract = declared };
                Transform best = null;
                float bestScore = float.PositiveInfinity;
                foreach (Transform candidate in sceneTransforms)
                {
                    if (candidate == null || string.IsNullOrWhiteSpace(declared.objectNameContains)) continue;
                    if (candidate.name.IndexOf(
                        declared.objectNameContains, StringComparison.OrdinalIgnoreCase) < 0) continue;
                    float score = Vector3.SqrMagnitude(candidate.position - declared.positionHint);
                    if (score < bestScore) { best = candidate; bestScore = score; }
                }
                grounded.resolvedObject = best;
                grounded.focusPosition = best == null ? declared.positionHint : BoundsCentre(best);
                Vector3 proposed = declared.positionHint + declared.approachOffset;
                if (best != null && declared.positionHint == Vector3.zero)
                    proposed = best.position + declared.approachOffset;
                proposed.y = contract.spawn == null ? 0f : contract.spawn.position.y;

                NavMeshHit navHit;
                if (hasUsableNavMesh && NavMesh.SamplePosition(proposed, out navHit, 4f, NavMesh.AllAreas))
                {
                    grounded.approachPosition = navHit.position;
                    grounded.navMeshProjected = true;
                }
                else
                {
                    grounded.approachPosition = ProjectToWalkableSurface(proposed, colliders);
                }
                targets.Add(grounded);
                groundingEvidence.Add("contract_target=" + declared.id + "; object=" +
                    (best == null ? "unresolved-position-fallback" : HierarchyPath(best)) +
                    "; live_name=" + (best != null) + "; navmesh_projected=" + grounded.navMeshProjected);
            }
            AddRuntimeDiscoveredTargets(sceneTransforms, colliders);
        }

        private void AddRuntimeDiscoveredTargets(Transform[] sceneTransforms, Collider[] colliders)
        {
            string[] observationKeywords =
            {
                "button", "switch", "lever", "console", "door", "gate", "portal",
                "vehicle", "coaster", "ride", "moving", "dynamic", "kinetic", "landmark"
            };
            HashSet<int> existing = new HashSet<int>(
                targets.Where(item => item.resolvedObject != null).Select(item => item.resolvedObject.GetInstanceID()));
            int discovered = 0;

            // First consume the structured LLM/source-analysis hand-off.  A hint
            // earns an observation target only after both the live object name
            // and every required live component are confirmed.  It never earns
            // permission to invoke a method.
            foreach (RuntimeDiscoveryHint hint in contract.runtimeDiscoveryHints ?? new RuntimeDiscoveryHint[0])
            {
                if (hint == null || !hint.discoveryOnly) continue;
                foreach (Transform candidate in sceneTransforms.OrderBy(item => HierarchyPath(item)))
                {
                    if (targets.Count >= 10 || candidate == null) break;
                    bool nameMatches = hint.objectNamePatterns != null && hint.objectNamePatterns.Any(pattern =>
                        NameMatchesPattern(candidate.name, pattern));
                    if (!nameMatches) continue;
                    Component[] liveComponents = candidate.GetComponents<Component>();
                    bool componentsMatch = (hint.requiredComponents ?? new string[0]).All(required =>
                        liveComponents.Any(component => ComponentMatchesRequiredType(component, required)));
                    if (!componentsMatch)
                    {
                        groundingEvidence.Add("hint=" + hint.hintId + "; object=" + HierarchyPath(candidate) +
                            "; rejected=required component mismatch");
                        continue;
                    }
                    groundingEvidence.Add("hint=" + hint.hintId + "; object=" + HierarchyPath(candidate) +
                        "; live_name=true; live_components=true; authorization=discovery_only");
                    if (existing.Contains(candidate.GetInstanceID())) continue;
                    AddDiscoveredObservation(
                        candidate, colliders, existing, ref discovered,
                        "hint:" + hint.targetId,
                        Mathf.Lerp(0.72f, 1.25f, Mathf.Clamp01(hint.confidence)),
                        "Grounded analyser/LLM discovery hint: " + hint.rationale);
                }
            }

            // Explicitly labelled fallback for projects whose analysis could not
            // extract runtime-created names.  This is observation-only and is not
            // represented as an LLM finding or interaction capability.
            foreach (Transform candidate in sceneTransforms.OrderBy(item => HierarchyPath(item)))
            {
                if (targets.Count >= 4 || candidate == null) break;
                string lower = candidate.name.ToLowerInvariant();
                if (!observationKeywords.Any(lower.Contains)) continue;
                if (candidate.GetComponentsInChildren<Renderer>().Length == 0) continue;
                if (existing.Contains(candidate.GetInstanceID())) continue;
                if (candidate.parent != null && observationKeywords.Any(
                    keyword => candidate.parent.name.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) >= 0))
                    continue;
                AddDiscoveredObservation(
                    candidate, colliders, existing, ref discovered,
                    "fallback", lower.Contains("button") || lower.Contains("console") || lower.Contains("kinetic") ? 1.15f : 0.82f,
                    "Explicit observation-keyword fallback matched the live hierarchy; not an interaction authorization.");
            }

            if (targets.Count < 4 && hasInferredWalkableBounds)
            {
                Vector3 center = inferredWalkableBounds.center;
                Vector3 extents = inferredWalkableBounds.extents * 0.55f;
                Vector3[] samples =
                {
                    center + new Vector3(extents.x, 0f, extents.z),
                    center + new Vector3(-extents.x, 0f, extents.z),
                    center + new Vector3(extents.x, 0f, -extents.z),
                    center + new Vector3(-extents.x, 0f, -extents.z),
                };
                for (int index = 0; index < samples.Length && targets.Count < 4; index++)
                {
                    Vector3 point = samples[index];
                    point.y = contract.spawn == null ? 0f : contract.spawn.position.y;
                    point = ProjectToWalkableSurface(point, colliders);
                    ContractTarget sampled = new ContractTarget
                    {
                        id = "runtime:walkable-sample:" + index,
                        label = "Walkable coverage sample " + (index + 1),
                        semanticType = "runtime-walkable-sample",
                        objectNameContains = "",
                        positionHint = point,
                        approachOffset = Vector3.zero,
                        arrivalRadius = 0.9f,
                        observeSeconds = 1.2f,
                        priority = 0.6f,
                        reason = "Runtime coverage sample inside renderer-inferred walkable bounds."
                    };
                    targets.Add(new GroundedTarget
                    {
                        contract = sampled, focusPosition = point + Vector3.up,
                        approachPosition = point, reachable = true
                    });
                }
            }
        }

        private void AddDiscoveredObservation(
            Transform candidate, Collider[] colliders, HashSet<int> existing,
            ref int discovered, string idPrefix, float priority, string reason)
        {
            if (candidate.GetComponentsInChildren<Renderer>().Length == 0) return;
            Vector3 focus = BoundsCentre(candidate);
            Vector3 spawn = contract.spawn == null ? Vector3.zero : contract.spawn.position;
            Vector3 away = spawn - focus; away.y = 0f;
            if (away.sqrMagnitude < 0.01f) away = Vector3.back;
            Vector3 proposed = focus + away.normalized * 1.5f;
            proposed.y = spawn.y;
            bool navProjected = false;
            NavMeshHit navHit;
            if (hasUsableNavMesh && NavMesh.SamplePosition(proposed, out navHit, 4f, NavMesh.AllAreas))
            {
                proposed = navHit.position;
                navProjected = true;
            }
            else proposed = ProjectToWalkableSurface(proposed, colliders);
            ContractTarget runtimeContract = new ContractTarget
            {
                id = idPrefix + ":" + Safe(candidate.name).ToLowerInvariant() + ":" + discovered,
                label = candidate.name,
                semanticType = "runtime-discovered-observation",
                objectNameContains = candidate.name,
                positionHint = candidate.position,
                approachOffset = proposed - candidate.position,
                arrivalRadius = 0.9f,
                observeSeconds = 1.5f,
                priority = priority,
                reason = reason
            };
            targets.Add(new GroundedTarget
            {
                contract = runtimeContract,
                resolvedObject = candidate,
                focusPosition = focus,
                approachPosition = proposed,
                navMeshProjected = navProjected,
                reachable = true
            });
            existing.Add(candidate.GetInstanceID());
            discovered++;
        }

        private static bool NameMatchesPattern(string name, string pattern)
        {
            if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(pattern)) return false;
            string normalized = pattern.Trim().Trim('*');
            return normalized.Length > 0 && name.IndexOf(normalized, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static bool ComponentMatchesRequiredType(Component component, string required)
        {
            if (component == null || string.IsNullOrWhiteSpace(required)) return false;
            Type type = component.GetType();
            for (Type cursor = type; cursor != null; cursor = cursor.BaseType)
            {
                if (string.Equals(cursor.Name, required, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(cursor.FullName, required, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return type.GetInterfaces().Any(contractType =>
                string.Equals(contractType.Name, required, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(contractType.FullName, required, StringComparison.OrdinalIgnoreCase));
        }

        public void ResetForEpisode(int requestedSeed, out Vector3 spawn, out float yawDegrees)
        {
            CloseWriters();
            seed = requestedSeed;
            random = new System.Random(seed);
            telemetryRows = 0;
            decisionCount = 0;
            decisionContextCount = 0;
            decisionOutcomeCount = 0;
            decisionBrainInferenceCount = 0;
            decisionBrainFallbackCount = 0;
            decisionBrainDisagreementCount = 0;
            decisionSequence = 0;
            interactionsAttempted = 0;
            interactionsVerified = 0;
            interactionMotionLocked = false;
            interactionMotionUnlockAfterCommand = false;
            interactionMotionLockedForLastCommand = false;
            interactionMotionLockStartedAt = 0f;
            interactionMotionLockDeadline = 0f;
            interactionMotionLockTargetId = "";
            interactionMotionLockTargetIdForLastCommand = "";
            interactionMotionLockAcquisitions = 0;
            interactionMotionLockReleases = 0;
            interactionMotionLockedTelemetryRows = 0;
            interactionMotionLockViolations = 0;
            interactionMotionLockMaxLinearSpeed = 0f;
            interactionMotionLockMaxAbsYawRate = 0f;
            interactionHeadMotionTelemetryRows = 0;
            interactionHeadMotionMaxPositionDelta = 0f;
            interactionHeadMotionMaxAngularDelta = 0f;
            replans = 0;
            collisionBlockedSeconds = 0f;
            collisionRecoveries = 0;
            collisionAvoidanceAdjustments = 0;
            collisionHardStops = 0;
            collisionOverlapDetections = 0;
            collisionOverlapActive = false;
            maxCollisionBlockedSeconds = 0f;
            preferredAvoidanceSign = random.Next(0, 2) == 0 ? -1 : 1;
            groundVerticalOffset = 0f;
            groundHitCount = 0;
            groundMissCount = 0;
            groundRejectedStepCount = 0;
            activeNavCorners = new Vector3[0];
            activeNavCornerIndex = 0;
            activeNavPathBuiltAt = -999f;
            activeNavTargetId = "";
            hasPreviousSample = false;
            previousCameraSpeed = 0f;
            cameraSpeedSum = 0f;
            cameraAngularSpeedSum = 0f;
            exposureProxySum = 0f;
            maxCameraSpeed = 0f;
            maxCameraAngularSpeed = 0f;
            maxCameraAcceleration = 0f;
            maxExposureProxy = 0f;
            smoothedVelocity = Vector3.zero;
            smoothedYawRate = 0f;
            stateEnteredAt = 0f;
            lastSimulatedTime = 0f;
            lastMeaningfulProgressAt = 0f;
            lastNavigationProgressAt = 0f;
            bestNavigationDistance = float.PositiveInfinity;
            navigationProgressTargetId = "";
            nextPlanningRetryAt = 0f;
            incompleteTerminalReason = "";
            episodeBegun = false;
            episodeComplete = false;
            episodeId = "episode-" + Guid.NewGuid().ToString("N");
            previousSkill = "";
            activeHighLevelDecision = null;
            failures.Clear();
            pendingDecisions.Clear();
            pendingDecisionContexts.Clear();
            pendingDecisionOutcomes.Clear();
            pendingDecisionBrainInferences.Clear();
            foreach (GroundedTarget target in targets)
            {
                target.visited = false;
                target.reachabilityKnown = false;
                target.reachable = false;
                target.interactionAttempted = false;
                target.interactionInvocationSucceeded = false;
                target.interactionVerified = false;
                target.interactionComponent = null;
                target.interactionBeforeState = null;
                target.interactionInvokedFrame = -1;
                target.interactionInvocationEvidence = "";
                target.interactionVerificationEvidence = "";
                target.unreachableUntil = 0f;
            }
            spawn = contract.spawn == null ? Vector3.zero : contract.spawn.position;
            float sampledSpawnY;
            if (TrySampleGround(spawn, out sampledSpawnY))
            {
                spawn.y = sampledSpawnY;
                groundHitCount++;
            }
            else groundMissCount++;
            spawnGroundY = spawn.y;
            lastBodyPosition = spawn;
            yawDegrees = contract.spawn == null ? 0f : contract.spawn.yawDegrees;
            state = BrainState.Scan;
            current = null;
            RecordDecision(0f, "SceneUnderstanding", "", "consume_contract_and_runtime_inventory",
                "ground " + targets.Count + " semantic targets", true,
                "contract=" + contract.projectId + "; planner=" + plannerMode);
            foreach (string evidence in groundingEvidence)
                RecordDecision(0f, "SceneUnderstanding", "", "validate_runtime_grounding",
                    "observe-only discovery", true, evidence);
        }

        public void BeginEpisode(int episodeSeed, string normalHmdLogDirectory)
        {
            string root = string.IsNullOrWhiteSpace(requestedLogRoot)
                ? normalHmdLogDirectory : Path.GetFullPath(requestedLogRoot);
            if (string.IsNullOrWhiteSpace(root))
                root = Path.Combine(Application.persistentDataPath, "GenericSceneAgentLogs");
            string projectToken = Safe(contract.projectId);
            // The terminal already keeps its run root short.  Bound the
            // runtime-owned suffix as well because some Unity/Mono file APIs on
            // Windows still fail close to MAX_PATH even when Python/.NET can
            // create the parent directory.
            if (projectToken.Length > 24) projectToken = projectToken.Substring(0, 24);
            string suffix = string.Format(CultureInfo.InvariantCulture,
                "generic_{0}_{1}_{2}", projectToken, episodeSeed,
                DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture));
            logDirectory = Path.Combine(root, suffix);
            Directory.CreateDirectory(logDirectory);
            contractPath = Path.Combine(logDirectory, "agent_scene_contract.json");
            inventoryPath = Path.Combine(logDirectory, "runtime_scene_inventory.json");
            decisionsPath = Path.Combine(logDirectory, "brain_decisions.jsonl");
            decisionContextsPath = Path.Combine(logDirectory, "decision_contexts.jsonl");
            decisionOutcomesPath = Path.Combine(logDirectory, "decision_outcomes.jsonl");
            decisionBrainInferencePath = Path.Combine(logDirectory, "decision_brain_inference.jsonl");
            telemetryPath = Path.Combine(logDirectory, "generic_agent_telemetry.csv");
            File.WriteAllText(contractPath, JsonUtility.ToJson(contract, true), new UTF8Encoding(false));
            File.WriteAllText(inventoryPath, JsonUtility.ToJson(CreateInventory(), true), new UTF8Encoding(false));
            decisionWriter = new StreamWriter(decisionsPath, false, new UTF8Encoding(false));
            foreach (string line in pendingDecisions) decisionWriter.WriteLine(line);
            pendingDecisions.Clear();
            decisionContextWriter = new StreamWriter(
                decisionContextsPath, false, new UTF8Encoding(false));
            foreach (string line in pendingDecisionContexts) decisionContextWriter.WriteLine(line);
            pendingDecisionContexts.Clear();
            decisionContextWriter.Flush();
            decisionOutcomeWriter = new StreamWriter(
                decisionOutcomesPath, false, new UTF8Encoding(false));
            foreach (string line in pendingDecisionOutcomes) decisionOutcomeWriter.WriteLine(line);
            pendingDecisionOutcomes.Clear();
            decisionOutcomeWriter.Flush();
            decisionBrainInferenceWriter = new StreamWriter(
                decisionBrainInferencePath, false, new UTF8Encoding(false));
            foreach (string line in pendingDecisionBrainInferences)
                decisionBrainInferenceWriter.WriteLine(line);
            pendingDecisionBrainInferences.Clear();
            decisionBrainInferenceWriter.Flush();
            telemetryWriter = new StreamWriter(telemetryPath, false, new UTF8Encoding(false));
            telemetryWriter.WriteLine(
                "step,sim_time,brain_state,target_id,body_x,body_y,body_z,body_yaw," +
                "camera_x,camera_y,camera_z,learned_right,learned_forward,learned_yaw_rate," +
                "executed_right,executed_forward,executed_yaw_rate,target_distance,coverage," +
                "camera_speed,camera_angular_speed,camera_acceleration,engineering_exposure_proxy," +
                "interaction_motion_locked,interaction_lock_target," +
                "head_local_position_delta,head_local_angular_delta");
            episodeBegun = true;
        }

        public void GuideRootMotion(
            float simulatedTime, float dt, Vector3 bodyPosition, float bodyYawDegrees,
            float learnedRight, float learnedForward, float learnedYawRate,
            out float executedRight, out float executedForward, out float executedYawRate)
        {
            lastSimulatedTime = simulatedTime;
            lastBodyPosition = bodyPosition;
            lastDriverDt = Mathf.Max(0.001f, dt);
            SampleCollisionOverlapEpisode(bodyPosition);
            AdvanceBrain(simulatedTime, bodyPosition, bodyYawDegrees);

            // Interaction is a transaction, not another locomotion state.  The
            // HMD model continues to generate HeadLocal below the driver boundary,
            // while BodyRoot translation, planner yaw and ground-offset adaptation
            // are hard-locked from pre-invocation settling through verification.
            interactionMotionLockedForLastCommand = interactionMotionLocked;
            interactionMotionLockTargetIdForLastCommand = interactionMotionLocked
                ? interactionMotionLockTargetId : "";
            if (interactionMotionLocked)
            {
                smoothedVelocity = Vector3.zero;
                smoothedYawRate = 0f;
                collisionBlockedSeconds = 0f;
                executedRight = 0f;
                executedForward = 0f;
                executedYawRate = 0f;
                if (interactionMotionUnlockAfterCommand)
                    ReleaseInteractionMotionLock(simulatedTime,
                        "final zero-root command emitted after interaction verification");
                return;
            }

            Vector3 desiredWorldVelocity = Vector3.zero;
            float desiredRootYawRate = 0f;
            if (current != null && state != BrainState.Complete)
            {
                Vector3 aim = state == BrainState.Navigate
                    ? current.approachPosition : current.focusPosition;
                bool navigationReady = true;
                if (state == BrainState.Navigate && hasUsableNavMesh)
                {
                    string pathEvidence;
                    navigationReady = TryGetNavigationAim(
                        bodyPosition, simulatedTime, out aim, out pathEvidence);
                    if (!navigationReady)
                    {
                        current.unreachableUntil = simulatedTime + 6f;
                        RecordDecision(simulatedTime, "Recover", current.contract.id,
                            "navmesh_path_invalid", "stop and replan", false, pathEvidence);
                        CompleteActiveDecision(
                            simulatedTime, false, false, false, pathEvidence);
                        PolicyChoice recoveryChoice = StartHighLevelDecision(
                            simulatedTime, bodyPosition, bodyYawDegrees, state.ToString(),
                            current, targets.IndexOf(current), "Replan",
                            new[] { "Replan" }, null);
                        ApplyPolicyTransition(simulatedTime, recoveryChoice,
                            "NavMesh path became invalid");
                    }
                }
                Vector3 delta = aim - bodyPosition;
                delta.y = 0f;
                if (navigationReady && delta.sqrMagnitude > 0.0001f)
                {
                    float desiredYaw = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
                    float yawError = Mathf.DeltaAngle(bodyYawDegrees, desiredYaw);
                    desiredRootYawRate = -Mathf.Clamp(yawError * 2.15f, -58f, 58f) * Mathf.Deg2Rad;
                    if (state == BrainState.Navigate)
                    {
                        // The dataset-trained root channels supply the moment-to-moment
                        // speed envelope.  The brain supplies only the task heading.
                        float learnedSpeed = Mathf.Sqrt(
                            learnedRight * learnedRight + learnedForward * learnedForward);
                        float speed = Mathf.Clamp(learnedSpeed, 0.52f, 1.02f);
                        float distance = delta.magnitude;
                        if (distance < 1.4f) speed *= Mathf.Lerp(0.22f, 1f, distance / 1.4f);
                        float facingGate = Mathf.Lerp(0.12f, 1f,
                            Mathf.Clamp01(1f - Mathf.Abs(yawError) / 100f));
                        desiredWorldVelocity = delta.normalized * speed * facingGate;
                    }
                }
            }

            Vector3 candidateVelocity = Vector3.MoveTowards(
                smoothedVelocity, desiredWorldVelocity, 1.65f * dt);
            if (candidateVelocity.sqrMagnitude > 0.0001f)
            {
                Vector3 collisionSafeVelocity;
                bool adjusted;
                bool startedOverlapping;
                string collisionEvidence;
                bool maySteer = state == BrainState.Navigate;
                if (TryResolveCollisionSafeVelocity(
                    bodyPosition, candidateVelocity, maySteer,
                    out collisionSafeVelocity, out adjusted,
                    out startedOverlapping, out collisionEvidence))
                {
                    smoothedVelocity = collisionSafeVelocity;
                    if (adjusted)
                    {
                        collisionAvoidanceAdjustments++;
                        collisionBlockedSeconds = 0f;
                    }
                    else collisionBlockedSeconds = 0f;
                }
                else
                {
                    // Do not ease out of a blocked velocity: that old inertia is
                    // exactly what allowed the transform-integrated BodyRoot to
                    // leak into walls after the earlier look-ahead had stopped.
                    smoothedVelocity = Vector3.zero;
                    collisionHardStops++;
                    collisionBlockedSeconds += dt;
                    maxCollisionBlockedSeconds = Mathf.Max(
                        maxCollisionBlockedSeconds, collisionBlockedSeconds);
                    if (collisionBlockedSeconds >= 1.25f &&
                        state == BrainState.Navigate && current != null)
                    {
                        GroundedTarget blockedTarget = current;
                        blockedTarget.unreachableUntil = simulatedTime + 6f;
                        collisionRecoveries++;
                        RecordDecision(simulatedTime, "Recover", blockedTarget.contract.id,
                            "collision_watchdog", "hard stop and replan", false,
                            collisionEvidence + "; blocked_for=1.25s");
                        CompleteActiveDecision(simulatedTime, false, false, false,
                            collisionEvidence + "; blocked_for=1.25s");
                        PolicyChoice recoveryChoice = StartHighLevelDecision(
                            simulatedTime, bodyPosition, bodyYawDegrees, state.ToString(),
                            blockedTarget, targets.IndexOf(blockedTarget), "Replan",
                            new[] { "Replan" }, null);
                        ApplyPolicyTransition(simulatedTime, recoveryChoice,
                            "collision capsule sweep could not progress");
                        collisionBlockedSeconds = 0f;
                        preferredAvoidanceSign *= -1;
                    }
                }
            }
            else
            {
                smoothedVelocity = Vector3.zero;
                collisionBlockedSeconds = 0f;
            }
            smoothedYawRate = Mathf.MoveTowards(
                smoothedYawRate, desiredRootYawRate, 3.2f * dt);
            Vector3 localVelocity = Quaternion.Inverse(
                Quaternion.Euler(0f, bodyYawDegrees, 0f)) * smoothedVelocity;
            executedRight = localVelocity.x;
            executedForward = localVelocity.z;
            executedYawRate = smoothedYawRate;
        }

        private void AdvanceBrain(float time, Vector3 bodyPosition, float bodyYawDegrees)
        {
            if (state == BrainState.Complete) return;
            if (state == BrainState.Scan)
            {
                SelectNextTarget(time, bodyPosition, bodyYawDegrees);
                return;
            }
            if (state == BrainState.Replan)
            {
                if (time < nextPlanningRetryAt) return;
                nextPlanningRetryAt = 0f;
                CompleteActiveDecision(time, false, false, false,
                    "replanning cycle entered; candidate set will be evaluated again");
                replans++;
                SelectNextTarget(time, bodyPosition, bodyYawDegrees);
                return;
            }
            if (current == null)
            {
                CompleteActiveDecision(time, false, false, false, "current target became unavailable");
                PolicyChoice missingChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                    null, -1, "Replan", new[] { "Replan" }, null);
                ApplyPolicyTransition(time, missingChoice, "missing current target");
                return;
            }

            if (interactionMotionLocked &&
                (state == BrainState.Interact || state == BrainState.Observe) &&
                time >= interactionMotionLockDeadline)
            {
                GroundedTarget timedOutTarget = current;
                string timeoutEvidence = "interaction transaction exceeded " +
                    InteractionTransactionTimeoutSeconds.ToString("F1", CultureInfo.InvariantCulture) +
                    "s while BodyRoot remained locked";
                if (!string.IsNullOrWhiteSpace(timedOutTarget.interactionVerificationEvidence))
                    timeoutEvidence += "; last_verification=" +
                        timedOutTarget.interactionVerificationEvidence;
                RecordDecision(time, "Verify", timedOutTarget.contract.id,
                    "interaction_transaction_timeout", "release motion lock and stop fail-closed",
                    false, timeoutEvidence);
                bool requiredInteractionFailed = timedOutTarget.contract.interaction != null &&
                    timedOutTarget.contract.interaction.required &&
                    !timedOutTarget.interactionVerified;
                if (requiredInteractionFailed &&
                    !failures.Any(message => message.Contains(timedOutTarget.contract.id)))
                    failures.Add("Required interaction timed out: " +
                        timedOutTarget.contract.id + " — " + timeoutEvidence);
                timedOutTarget.visited = false;
                CompleteActiveDecision(time, false, true,
                    requiredInteractionFailed, timeoutEvidence);
                if (requiredInteractionFailed)
                {
                    // The approved invocation received its full verification
                    // window. Repeating a potentially non-idempotent gameplay
                    // method indefinitely is unsafe and used to stretch this
                    // deterministic failure to the 900-second safety ceiling.
                    MarkContentExhausted(time,
                        "required interaction postcondition was not verified: " +
                        timedOutTarget.contract.id + "; " + timeoutEvidence);
                    return;
                }
                timedOutTarget.unreachableUntil = time + 6f;
                RequestInteractionMotionLockRelease(time, timeoutEvidence);
                PolicyChoice timeoutChoice = StartHighLevelDecision(
                    time, bodyPosition, bodyYawDegrees, state.ToString(),
                    timedOutTarget, targets.IndexOf(timedOutTarget), "Replan",
                    new[] { "Replan" }, null);
                ApplyPolicyTransition(time, timeoutChoice, timeoutEvidence);
                return;
            }

            float arrival = Mathf.Max(0.35f, current.contract.arrivalRadius);
            Vector3 planarDelta = current.approachPosition - bodyPosition;
            planarDelta.y = 0f;
            if (state == BrainState.Navigate)
            {
                float distance = planarDelta.magnitude;
                if (!string.Equals(navigationProgressTargetId, current.contract.id,
                    StringComparison.Ordinal))
                {
                    navigationProgressTargetId = current.contract.id;
                    bestNavigationDistance = distance;
                    lastNavigationProgressAt = time;
                }
                else if (distance <= bestNavigationDistance -
                    NavigationProgressDistanceEpsilon)
                {
                    bestNavigationDistance = distance;
                    lastNavigationProgressAt = time;
                    lastMeaningfulProgressAt = time;
                }
            }
            if (state == BrainState.Navigate && planarDelta.magnitude <= arrival)
            {
                lastMeaningfulProgressAt = time;
                string arrivalEvidence = "arrived within " +
                    planarDelta.magnitude.ToString("F3", CultureInfo.InvariantCulture) + "m";
                CompleteActiveDecision(time, true, false, false, arrivalEvidence);
                PolicyChoice arrivalChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                    current, targets.IndexOf(current), "LookAt",
                    new[] { "LookAt" }, null);
                ApplyPolicyTransition(time, arrivalChoice, arrivalEvidence);
                return;
            }
            if (state == BrainState.Navigate &&
                time - lastNavigationProgressAt > NavigationNoProgressTimeoutSeconds)
            {
                RecordDecision(time, "Recover", current.contract.id, "recover_from_timeout",
                    "abandon target and replan", false,
                    "navigation_no_progress_timeout=" +
                    NavigationNoProgressTimeoutSeconds.ToString(
                        "F1", CultureInfo.InvariantCulture) +
                    "s; best_distance=" + bestNavigationDistance.ToString(
                        "F3", CultureInfo.InvariantCulture) + "m");
                current.visited = false;
                current.unreachableUntil = time + 6f;
                CompleteActiveDecision(time, false, true, false,
                    "navigation made no distance progress for " +
                    NavigationNoProgressTimeoutSeconds.ToString(
                        "F1", CultureInfo.InvariantCulture) + "s");
                PolicyChoice timeoutChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                    current, targets.IndexOf(current), "Replan", new[] { "Replan" }, null);
                ApplyPolicyTransition(time, timeoutChoice, "navigation no-progress timeout");
                return;
            }
            if (state == BrainState.LookAt)
            {
                Vector3 focusDelta = current.focusPosition - bodyPosition;
                focusDelta.y = 0f;
                float desiredYaw = focusDelta.sqrMagnitude < 0.0001f
                    ? bodyYawDegrees : Mathf.Atan2(focusDelta.x, focusDelta.z) * Mathf.Rad2Deg;
                if (Mathf.Abs(Mathf.DeltaAngle(bodyYawDegrees, desiredYaw)) < 10f || time - stateEnteredAt > 2.2f)
                {
                    CompleteActiveDecision(time, true, false, false,
                        "look alignment completed or bounded look timeout elapsed");
                    if (IsInteractionAvailable(current))
                    {
                        PolicyChoice nextChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                            current, targets.IndexOf(current), "Interact",
                            new[] { "Interact" }, null);
                        ApplyPolicyTransition(time, nextChoice, "declared and runtime-authorized affordance is in view");
                    }
                    else
                    {
                        PolicyChoice nextChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                            current, targets.IndexOf(current), "Observe",
                            new[] { "Observe" }, null);
                        ApplyPolicyTransition(time, nextChoice, "no runtime-authorized affordance declared");
                    }
                }
                return;
            }
            if (state == BrainState.Interact)
            {
                // The first part of Interact is an explicit zero-root settling
                // window.  Invocation cannot run until residual BodyRoot velocity
                // and planner yaw have been hard-zeroed for 350 ms. Verification
                // is deliberately deferred to Observe so coroutine and event
                // side effects get at least one rendered frame to become visible.
                if (time - interactionMotionLockStartedAt < InteractionSettleSeconds)
                    return;
                if (!current.interactionAttempted)
                {
                    lastMeaningfulProgressAt = time;
                    current.interactionAttempted = true;
                    interactionsAttempted++;
                    string evidence;
                    bool invoked = TryInvokeInteraction(current, out evidence);
                    current.interactionInvocationSucceeded = invoked;
                    current.interactionInvocationEvidence = evidence;
                    RecordDecision(time, "Interact", current.contract.id,
                        current.contract.interaction.verb,
                        current.contract.interaction.componentType + "." + current.contract.interaction.method + "()",
                        invoked, evidence);
                    CompleteActiveDecision(time, invoked, false, !invoked, evidence);
                    if (!invoked && current.contract.interaction.required)
                    {
                        if (!failures.Any(message => message.Contains(current.contract.id)))
                            failures.Add("Required interaction invocation failed: " +
                                current.contract.id + " — " + evidence);
                        current.visited = false;
                        MarkContentExhausted(time,
                            "required interaction invocation failed: " +
                            current.contract.id + "; " + evidence);
                        return;
                    }
                }
                PolicyChoice postInteractionChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                    current, targets.IndexOf(current), "Observe",
                    new[] { "Observe" }, null);
                ApplyPolicyTransition(time, postInteractionChoice, "interaction attempt completed");
                return;
            }
            if (state == BrainState.Observe)
            {
                if (current.interactionAttempted && !current.interactionVerified)
                {
                    string verificationEvidence;
                    if (TryPollInteractionVerification(current, out verificationEvidence))
                    {
                        current.interactionVerified = true;
                        current.interactionVerificationEvidence = verificationEvidence;
                        interactionsVerified++;
                        lastMeaningfulProgressAt = time;
                        RecordDecision(time, "Verify", current.contract.id,
                            "verify_declared_postcondition",
                            current.contract.interaction.successMember,
                            true, verificationEvidence);
                    }
                    else current.interactionVerificationEvidence = verificationEvidence;
                }

                if (time - stateEnteredAt < Mathf.Max(0.5f, current.contract.observeSeconds))
                    return;

                bool requiredInteractionPending = interactionMotionLocked &&
                    current.interactionAttempted && HasInteraction(current.contract) &&
                    current.contract.interaction.required && !current.interactionVerified;
                if (requiredInteractionPending)
                {
                    // The transaction timeout above owns the failure decision.
                    // Until then, keep observing without granting coverage and
                    // keep BodyRoot locked while HeadLocal generation continues.
                    current.visited = false;
                    return;
                }

                bool finishesInteractionTransaction = interactionMotionLocked;
                current.visited = true;
                lastMeaningfulProgressAt = time;
                RecordDecision(time, "Verify", current.contract.id, "verify_target_coverage",
                    "runtime arrival + timed observation", true,
                    "visited=true; position=" + Vec(bodyPosition));
                RecordDecision(time, "Observe", current.contract.id, current.contract.reason,
                    "mark semantic target visited", true,
                    "coverage=" + Coverage().ToString("F3", CultureInfo.InvariantCulture));
                CompleteActiveDecision(time, true, false, false, "timed observation completed");
                if (finishesInteractionTransaction)
                    RequestInteractionMotionLockRelease(time,
                        "post-interaction observation and verification completed");
                if (ContentCompletionSatisfied())
                {
                    PolicyChoice completeChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                        null, -1, "Stop", new[] { "Stop" }, null);
                    ApplyPolicyTransition(time, completeChoice,
                        "all targets visited and required interactions verified");
                }
                else if (AllVisited())
                {
                    MarkContentExhausted(time,
                        "all targets were visited but at least one required interaction was not verified");
                }
                else
                {
                    PolicyChoice replanChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                        current, targets.IndexOf(current), "Replan", new[] { "Replan" }, null);
                    ApplyPolicyTransition(time, replanChoice, "seek novel target");
                }
            }
        }

        private void SelectNextTarget(float time, Vector3 bodyPosition, float bodyYawDegrees)
        {
            GroundedTarget selected = null;
            int selectedIndex = -1;
            double bestUtility = double.NegativeInfinity;
            UnityDecisionCandidate[] candidateSnapshots = new UnityDecisionCandidate[targets.Count];
            for (int index = 0; index < targets.Count; index++)
            {
                GroundedTarget candidate = targets[index];
                bool cooldown = time < candidate.unreachableUntil;
                if (!candidate.visited && !cooldown)
                {
                    candidate.reachable = CanReach(bodyPosition, candidate.approachPosition);
                    candidate.reachabilityKnown = true;
                    if (!candidate.reachable)
                    {
                        RecordDecision(time, "Plan", candidate.contract.id, "test_runtime_reachability",
                            plannerMode, false, "target rejected as unreachable at current planning step");
                    }
                }
                float distance = Vector3.Distance(bodyPosition, candidate.approachPosition);
                double resolutionBonus = candidate.resolvedObject == null ? -0.35 : 0.20;
                double baseUtilityExact = candidate.contract.priority * 2.0 -
                    distance * 0.055 + resolutionBonus;
                bool selectable = !candidate.visited && !cooldown &&
                    candidate.reachabilityKnown && candidate.reachable;
                // Seeded jitter is sampled only for candidates the rule policy
                // can actually choose. Logging therefore does not alter policy RNG.
                double jitterExact = selectable ? random.NextDouble() * 0.38 : 0.0;
                double totalUtilityExact = baseUtilityExact + jitterExact;
                string ineligibleReason = candidate.visited ? "visited" :
                    cooldown ? "cooldown" :
                    !candidate.reachabilityKnown ? "reachability_unknown" :
                    !candidate.reachable ? "unreachable" : "";
                candidateSnapshots[index] = BuildCandidateSnapshot(
                    candidate, index, time, bodyPosition, bodyYawDegrees,
                    (float)baseUtilityExact, (float)jitterExact, (float)totalUtilityExact,
                    selectable, ineligibleReason, selectable);
                if (selectable && totalUtilityExact > bestUtility)
                {
                    bestUtility = totalUtilityExact;
                    selected = candidate;
                    selectedIndex = index;
                }
            }
            if (selected == null)
            {
                GroundedTarget[] unvisited = targets.Where(item => !item.visited).ToArray();
                if (unvisited.Length == 0)
                {
                    if (ContentCompletionSatisfied())
                    {
                        PolicyChoice stopChoice = StartHighLevelDecision(
                            time, bodyPosition, bodyYawDegrees, state.ToString(),
                            null, -1, "Stop", new[] { "Stop" }, candidateSnapshots);
                        ApplyPolicyTransition(time, stopChoice,
                            "all targets visited and required interactions verified");
                    }
                    else
                    {
                        MarkContentExhausted(time,
                            "all targets were visited but required interaction evidence is incomplete");
                    }
                    return;
                }

                GroundedTarget[] cooling = unvisited.Where(
                    item => time < item.unreachableUntil).ToArray();
                if (cooling.Length > 0)
                {
                    nextPlanningRetryAt = Mathf.Max(
                        time + 0.05f, cooling.Min(item => item.unreachableUntil));
                    current = null;
                    PolicyChoice retryChoice = StartHighLevelDecision(
                        time, bodyPosition, bodyYawDegrees, state.ToString(),
                        null, -1, "Replan", new[] { "Replan" }, null);
                    RecordDecision(time, "Recover", "", "wait_for_target_cooldown",
                        "hold position then replan", true,
                        "retry_at=" + nextPlanningRetryAt.ToString(
                            "F3", CultureInfo.InvariantCulture) +
                        "; cooling_targets=" + cooling.Length);
                    ApplyPolicyTransition(time, retryChoice,
                        "all currently viable remaining targets are cooling down");
                    return;
                }

                MarkContentExhausted(time,
                    "all remaining unvisited targets failed runtime reachability checks");
                return;
            }
            current = selected;
            PolicyChoice selectionChoice = StartHighLevelDecision(time, bodyPosition, bodyYawDegrees, state.ToString(),
                current, selectedIndex, "Navigate", new[] { "Navigate" }, candidateSnapshots);
            current = selectionChoice.target;
            if (selectionChoice.skill != "Navigate" || current == null)
            {
                ApplyPolicyTransition(time, selectionChoice, "target-selection policy decision");
                return;
            }
            if (hasUsableNavMesh)
            {
                string pathEvidence;
                if (!BuildActiveNavPath(bodyPosition, current.approachPosition, time, out pathEvidence))
                {
                    current.unreachableUntil = time + 6f;
                    RecordDecision(time, "Recover", current.contract.id, "build_navmesh_path",
                        "stop and replan", false, pathEvidence);
                    CompleteActiveDecision(time, false, false, false, pathEvidence);
                    PolicyChoice pathFailureChoice = StartHighLevelDecision(
                        time, bodyPosition, bodyYawDegrees, state.ToString(),
                        current, selectionChoice.targetIndex, "Replan",
                        new[] { "Replan" }, null);
                    ApplyPolicyTransition(time, pathFailureChoice, "NavMesh path build failed");
                    return;
                }
                RecordDecision(time, "Plan", current.contract.id, "build_navmesh_path",
                    "follow cached NavMesh corners", true, pathEvidence);
            }
            string selectionEvidence = decisionBrainRuntime.PolicyAuthority == "model"
                ? "model_ranked_target; rule_reference_utility=" +
                    bestUtility.ToString("F3", CultureInfo.InvariantCulture)
                : "rule_novelty_priority_utility=" +
                    bestUtility.ToString("F3", CultureInfo.InvariantCulture);
            ApplyPolicyTransition(time, selectionChoice, selectionEvidence);
            RecordDecision(time, "Plan", current.contract.id, "choose_next_semantic_target",
                plannerMode + " to " + Vec(current.approachPosition), true,
                "reason=" + current.contract.reason + "; resolved=" + (current.resolvedObject != null));
        }

        private static float BaseUtility(GroundedTarget candidate, float distance)
        {
            float resolutionBonus = candidate.resolvedObject == null ? -0.35f : 0.20f;
            return candidate.contract.priority * 2.0f - distance * 0.055f + resolutionBonus;
        }

        private bool IsInteractionAvailable(GroundedTarget target)
        {
            if (target == null || !HasInteraction(target.contract) || target.resolvedObject == null)
                return false;
            ContractInteraction action = target.contract.interaction;
            string whitelistKey = action.componentType + "." + action.method;
            if (contract.safeInteractionMethods == null || !contract.safeInteractionMethods.Any(item =>
                string.Equals(item, whitelistKey, StringComparison.Ordinal)))
                return false;
            Component component = FindNamedComponent(target.resolvedObject, action.componentType);
            if (component == null) return false;
            if (component.GetType().FullName == "UnityEngine.UI.Button" && action.method == "Press")
            {
                PropertyInfo onClick = component.GetType().GetProperty("onClick");
                object eventObject = onClick == null ? null : onClick.GetValue(component, null);
                return eventObject != null &&
                    eventObject.GetType().GetMethod("Invoke", Type.EmptyTypes) != null;
            }
            return component.GetType().GetMethod(
                action.method, BindingFlags.Instance | BindingFlags.Public,
                null, Type.EmptyTypes, null) != null;
        }

        private UnityDecisionCandidate BuildCandidateSnapshot(
            GroundedTarget candidate, int index, float time, Vector3 bodyPosition,
            float bodyYawDegrees, float baseUtility, float jitter, float totalUtility,
            bool selectable, string ineligibleReason, bool utilityIncludesNewJitter)
        {
            Vector3 worldDelta = candidate.approachPosition - bodyPosition;
            Vector3 relative = Quaternion.Inverse(Quaternion.Euler(0f, bodyYawDegrees, 0f)) * worldDelta;
            float planar = Mathf.Sqrt(relative.x * relative.x + relative.z * relative.z);
            bool interactionDeclared = HasInteraction(candidate.contract);
            bool interactionAvailable = IsInteractionAvailable(candidate);
            string semanticType = candidate.contract.semanticType ?? "";
            return new UnityDecisionCandidate
            {
                index = index,
                targetId = candidate.contract.id ?? "",
                semanticType = semanticType,
                priority = candidate.contract.priority,
                relativeX = relative.x,
                relativeY = relative.y,
                relativeZ = relative.z,
                distance = worldDelta.magnitude,
                bearingSin = planar > 0.0001f ? relative.x / planar : 0f,
                bearingCos = planar > 0.0001f ? relative.z / planar : 1f,
                resolved = candidate.resolvedObject != null,
                reachabilityKnown = candidate.reachabilityKnown,
                reachable = candidate.reachabilityKnown && candidate.reachable,
                // No current runtime sensor proves visibility with occlusion and
                // camera-frustum semantics, so both fields remain explicitly unknown.
                visibilityKnown = false,
                visible = false,
                navMeshProjected = candidate.navMeshProjected,
                visited = candidate.visited,
                cooldown = time < candidate.unreachableUntil,
                cooldownSeconds = Mathf.Max(0f, candidate.unreachableUntil - time),
                interactionDeclared = interactionDeclared,
                interactionAvailable = interactionAvailable,
                interactionRequired = interactionDeclared && candidate.contract.interaction.required,
                interactionAttempted = candidate.interactionAttempted,
                interactionVerified = candidate.interactionVerified,
                riskCandidate = semanticType.IndexOf(
                    "risk", StringComparison.OrdinalIgnoreCase) >= 0,
                selectable = selectable,
                ineligibleReason = ineligibleReason ?? "",
                utilityIncludesNewJitter = utilityIncludesNewJitter,
                baseUtility = baseUtility,
                randomJitter = jitter,
                totalUtility = totalUtility,
            };
        }

        private UnityDecisionCandidate[] SnapshotCandidates(
            float time, Vector3 bodyPosition, float bodyYawDegrees)
        {
            UnityDecisionCandidate[] snapshots = new UnityDecisionCandidate[targets.Count];
            for (int index = 0; index < targets.Count; index++)
            {
                GroundedTarget candidate = targets[index];
                float distance = Vector3.Distance(bodyPosition, candidate.approachPosition);
                float baseUtility = BaseUtility(candidate, distance);
                bool cooldown = time < candidate.unreachableUntil;
                bool selectable = !candidate.visited && !cooldown &&
                    candidate.reachabilityKnown && candidate.reachable;
                string ineligibleReason = candidate.visited ? "visited" :
                    cooldown ? "cooldown" :
                    !candidate.reachabilityKnown ? "reachability_unknown" :
                    !candidate.reachable ? "unreachable" : "";
                snapshots[index] = BuildCandidateSnapshot(
                    candidate, index, time, bodyPosition, bodyYawDegrees,
                    baseUtility, 0f, baseUtility, selectable, ineligibleReason, false);
            }
            return snapshots;
        }

        private PolicyChoice StartHighLevelDecision(
            float time, Vector3 bodyPosition, float bodyYawDegrees, string stateName,
            GroundedTarget chosenTarget, int chosenIndex, string chosenSkill,
            string[] validSkills, UnityDecisionCandidate[] candidateSnapshots)
        {
            if (activeHighLevelDecision != null)
            {
                CompleteActiveDecision(time, false, false, false,
                    "previous macro action was superseded by " + chosenSkill);
            }
            bool targetSelection = candidateSnapshots != null;
            UnityDecisionCandidate[] snapshots = candidateSnapshots ??
                SnapshotCandidates(time, bodyPosition, bodyYawDegrees);
            decisionSequence++;
            string decisionId = episodeId + ":decision:" +
                decisionSequence.ToString("D6", CultureInfo.InvariantCulture);
            string decisionKind = targetSelection ? "target_selection" : "skill_transition";
            PolicyChoice choice = new PolicyChoice
            {
                target = chosenTarget,
                targetIndex = chosenIndex,
                skill = chosenSkill ?? "",
                failClosed = false,
                evidence = "rule choice"
            };
            UnityDecisionPrediction prediction;
            string inferenceError;
            bool evaluated = decisionBrainRuntime.TryPredict(
                time, Coverage(), replans, stateName ?? "", previousSkill ?? "", plannerMode,
                decisionKind, snapshots, validSkills, chosenIndex,
                out prediction, out inferenceError);
            bool fallback = false;
            if (evaluated)
            {
                decisionBrainInferenceCount++;
                if (decisionBrainRuntime.RequestedMode == UnityDecisionBrainMode.Model)
                {
                    if (prediction.skill == "Stop")
                    {
                        choice.target = null;
                        choice.targetIndex = -1;
                    }
                    else if (targetSelection)
                    {
                        if (prediction.targetIndex < 0 || prediction.targetIndex >= targets.Count)
                        {
                            choice.target = null;
                            choice.targetIndex = -1;
                            choice.skill = "Stop";
                            choice.failClosed = true;
                            choice.evidence = "fail-closed: model target index is invalid";
                        }
                        else
                        {
                            choice.target = targets[prediction.targetIndex];
                            choice.targetIndex = prediction.targetIndex;
                        }
                    }
                    choice.skill = choice.failClosed ? "Stop" : prediction.skill;
                    if (!choice.failClosed) choice.evidence = "safety-masked model choice";
                }
            }
            else if (decisionBrainRuntime.RequestedMode == UnityDecisionBrainMode.Shadow)
            {
                fallback = true;
                decisionBrainFallbackCount++;
                choice.evidence = "shadow unavailable; rule remains authoritative: " + inferenceError;
            }
            else if (decisionBrainRuntime.RequestedMode == UnityDecisionBrainMode.Model)
            {
                fallback = true;
                decisionBrainFallbackCount++;
                choice.target = null;
                choice.targetIndex = -1;
                choice.skill = "Stop";
                choice.failClosed = true;
                choice.evidence = "fail-closed: " + inferenceError;
                if (!failures.Any(message => message.StartsWith("Decision Brain fail-closed", StringComparison.Ordinal)))
                    failures.Add("Decision Brain fail-closed: " + inferenceError);
            }
            bool targetDisagreement = evaluated && prediction.targetIndex != chosenIndex;
            bool skillDisagreement = evaluated && prediction.skill != chosenSkill;
            if (targetDisagreement || skillDisagreement) decisionBrainDisagreementCount++;
            WriteDecisionBrainInference(new UnityDecisionBrainInference
            {
                utc = DateTime.UtcNow.ToString("O"),
                episodeId = episodeId,
                decisionId = decisionId,
                simulatedTime = time,
                decisionKind = decisionKind,
                requestedMode = decisionBrainRuntime.RequestedModeName,
                policyAuthority = decisionBrainRuntime.PolicyAuthority,
                modelStatus = decisionBrainRuntime.ModelStatus,
                modelSha256 = decisionBrainRuntime.ModelSha256,
                researchOverride = decisionBrainRuntime.ResearchOverride,
                modelEvaluated = evaluated,
                fallback = fallback || choice.failClosed,
                fallbackReason = fallback || choice.failClosed ? choice.evidence : "",
                ruleTargetIndex = chosenIndex,
                ruleTargetId = chosenTarget == null ? "" : chosenTarget.contract.id,
                ruleSkill = chosenSkill ?? "",
                modelTargetIndex = evaluated ? prediction.targetIndex : -1,
                modelTargetId = evaluated ? prediction.targetId : "",
                modelSkill = evaluated ? prediction.skill : "",
                executedTargetIndex = choice.targetIndex,
                executedTargetId = choice.target == null ? "" : choice.target.contract.id,
                executedSkill = choice.skill,
                targetDisagreement = targetDisagreement,
                skillDisagreement = skillDisagreement,
                validSkills = validSkills ?? new string[0],
                rawTargetLogits = evaluated ? prediction.rawTargetLogits : new float[0],
                maskedTargetLogits = evaluated ? prediction.maskedTargetLogits : new float[0],
                rawSkillLogits = evaluated ? prediction.rawSkillLogits : new float[0],
                maskedSkillLogits = evaluated ? prediction.maskedSkillLogits : new float[0],
            });
            string provenance = decisionBrainRuntime.RequestedMode == UnityDecisionBrainMode.Model
                ? "model_rollout" : "rule_bootstrap";
            UnityDecisionContext context = new UnityDecisionContext
            {
                utc = DateTime.UtcNow.ToString("O"),
                episodeId = episodeId,
                decisionId = decisionId,
                projectId = contract.projectId ?? "",
                sceneName = contract.sceneName ?? "",
                seed = seed,
                sourceKind = provenance,
                humanDemonstration = false,
                simulatedTime = time,
                decisionKind = decisionKind,
                state = stateName ?? "",
                previousSkill = previousSkill ?? "",
                bodyPosition = bodyPosition,
                bodyX = bodyPosition.x,
                bodyY = bodyPosition.y,
                bodyZ = bodyPosition.z,
                bodyYawDegrees = bodyYawDegrees,
                coverage = Coverage(),
                replans = replans,
                plannerMode = plannerMode,
                candidates = snapshots,
                validSkills = validSkills ?? new string[0],
                chosenTargetIndex = choice.targetIndex,
                chosenTargetId = choice.target == null ? "" : choice.target.contract.id,
                chosenSkill = choice.skill,
                labelSource = provenance,
                humanAccepted = "",
                humanCorrectionTargetId = "",
                humanCorrectionSkill = "",
                humanCorrectionNotes = "",
            };
            string line = JsonUtility.ToJson(context);
            decisionContextCount++;
            if (decisionContextWriter == null) pendingDecisionContexts.Add(line);
            else { decisionContextWriter.WriteLine(line); decisionContextWriter.Flush(); }
            activeHighLevelDecision = new ActiveHighLevelDecision
            {
                decisionId = decisionId,
                targetId = context.chosenTargetId,
                skill = context.chosenSkill,
                startedAt = time,
                coverageBefore = context.coverage,
                collisionRecoveriesBefore = collisionRecoveries,
            };
            previousSkill = context.chosenSkill;
            return choice;
        }

        private void WriteDecisionBrainInference(UnityDecisionBrainInference inference)
        {
            string line = JsonUtility.ToJson(inference);
            if (decisionBrainInferenceWriter == null) pendingDecisionBrainInferences.Add(line);
            else
            {
                decisionBrainInferenceWriter.WriteLine(line);
                decisionBrainInferenceWriter.Flush();
            }
        }

        private void ApplyPolicyTransition(float time, PolicyChoice choice, string evidence)
        {
            current = choice.target;
            string policyEvidence = evidence + "; policy=" + choice.evidence;
            if (choice.failClosed)
            {
                current = null;
                Transition(time, BrainState.Complete, "Stop", policyEvidence);
                return;
            }
            if (choice.skill == "Stop")
            {
                if (ContentCompletionSatisfied())
                {
                    current = null;
                    Transition(time, BrainState.Complete, "Stop", policyEvidence);
                }
                else
                {
                    if (!failures.Any(message => message.StartsWith(
                        "Premature Stop", StringComparison.Ordinal)))
                        failures.Add("Premature Stop was blocked before declared scene content completed.");
                    if (interactionMotionLocked)
                        RequestInteractionMotionLockRelease(time,
                            "premature Stop was rejected by the lifecycle safety gate");
                    current = null;
                    Transition(time, BrainState.Replan, "Replan",
                        policyEvidence + "; premature Stop rejected");
                }
                return;
            }
            BrainState next;
            if (!Enum.TryParse(choice.skill, true, out next) || next == BrainState.Scan || next == BrainState.Complete)
            {
                current = null;
                failures.Add("Decision Brain emitted unsupported executable skill: " + choice.skill);
                Transition(time, BrainState.Complete, "Stop", policyEvidence + "; unsupported skill fail-closed");
                return;
            }
            if ((next == BrainState.Navigate || next == BrainState.LookAt ||
                 next == BrainState.Interact || next == BrainState.Observe) && current == null)
            {
                failures.Add("Decision Brain emitted " + choice.skill + " without an active target.");
                Transition(time, BrainState.Complete, "Stop", policyEvidence + "; missing target fail-closed");
                return;
            }
            if (next == BrainState.Interact && !IsInteractionAvailable(current))
            {
                failures.Add("Decision Brain interaction was blocked by runtime authorization.");
                Transition(time, BrainState.Replan, "Replan", policyEvidence + "; interaction authorization blocked");
                return;
            }
            Transition(time, next, choice.skill, policyEvidence);
        }

        private void CompleteActiveDecision(
            float time, bool reached, bool timedOut, bool terminal, string evidence)
        {
            if (activeHighLevelDecision == null) return;
            ActiveHighLevelDecision completed = activeHighLevelDecision;
            GroundedTarget target = targets.FirstOrDefault(item => string.Equals(
                item.contract.id, completed.targetId, StringComparison.Ordinal));
            float coverageAfter = Coverage();
            UnityDecisionOutcome outcome = new UnityDecisionOutcome
            {
                utc = DateTime.UtcNow.ToString("O"),
                episodeId = episodeId,
                decisionId = completed.decisionId,
                projectId = contract.projectId ?? "",
                sceneName = contract.sceneName ?? "",
                seed = seed,
                simulatedTime = time,
                completedSimulatedTime = time,
                elapsedSeconds = Mathf.Max(0f, time - completed.startedAt),
                chosenTargetId = completed.targetId ?? "",
                chosenSkill = completed.skill ?? "",
                reached = reached,
                visited = target != null && target.visited,
                interactionAttempted = target != null && target.interactionAttempted,
                interactionVerified = target != null && target.interactionVerified,
                timedOut = timedOut,
                collisionRecoveriesDelta = Mathf.Max(
                    0, collisionRecoveries - completed.collisionRecoveriesBefore),
                coverageBefore = completed.coverageBefore,
                coverageAfter = coverageAfter,
                coverageGain = coverageAfter - completed.coverageBefore,
                terminal = terminal,
                terminalState = state.ToString(),
                evidence = evidence ?? "",
            };
            string line = JsonUtility.ToJson(outcome);
            decisionOutcomeCount++;
            if (decisionOutcomeWriter == null) pendingDecisionOutcomes.Add(line);
            else { decisionOutcomeWriter.WriteLine(line); decisionOutcomeWriter.Flush(); }
            activeHighLevelDecision = null;
        }

        private void Transition(float time, BrainState next, string skill, string evidence)
        {
            bool enteringInteraction = next == BrainState.Interact && state != BrainState.Interact;
            state = next;
            if (next != BrainState.Navigate)
            {
                navigationProgressTargetId = "";
                bestNavigationDistance = float.PositiveInfinity;
                lastNavigationProgressAt = time;
            }
            stateEnteredAt = time;
            RecordDecision(time, next.ToString(), current == null ? "" : current.contract.id,
                "brain_state_transition", skill, true, evidence);
            if (enteringInteraction) AcquireInteractionMotionLock(time);
        }

        private void AcquireInteractionMotionLock(float time)
        {
            interactionMotionLocked = true;
            interactionMotionUnlockAfterCommand = false;
            interactionMotionLockStartedAt = time;
            interactionMotionLockDeadline = time + InteractionTransactionTimeoutSeconds;
            interactionMotionLockTargetId = current == null ? "" : current.contract.id;
            interactionMotionLockAcquisitions++;
            // Discard locomotion inertia immediately.  MoveTowards is intentionally
            // not used here because even a short decay can slide through a button,
            // lever, seat or trigger while its interaction is being verified.
            smoothedVelocity = Vector3.zero;
            smoothedYawRate = 0f;
            collisionBlockedSeconds = 0f;
            RecordDecision(time, "Interact", interactionMotionLockTargetId,
                "interaction_motion_lock_acquired", "BodyRoot XZ/yaw/ground offset = locked",
                true, "settle_before_invoke=" +
                    InteractionSettleSeconds.ToString("F2", CultureInfo.InvariantCulture) +
                    "s; HeadLocal generation remains active");
        }

        private void RequestInteractionMotionLockRelease(float time, string evidence)
        {
            if (!interactionMotionLocked || interactionMotionUnlockAfterCommand) return;
            interactionMotionUnlockAfterCommand = true;
            RecordDecision(time, "Verify", interactionMotionLockTargetId,
                "interaction_motion_lock_release_scheduled",
                "emit one final zero-root command, then unlock", true, evidence);
        }

        private void ReleaseInteractionMotionLock(float time, string evidence)
        {
            if (!interactionMotionLocked) return;
            string targetId = interactionMotionLockTargetId;
            interactionMotionLocked = false;
            interactionMotionUnlockAfterCommand = false;
            interactionMotionLockTargetId = "";
            interactionMotionLockReleases++;
            RecordDecision(time, "Verify", targetId,
                "interaction_motion_lock_released", "resume planner-controlled BodyRoot",
                true, evidence);
        }

        private bool TryInvokeInteraction(GroundedTarget target, out string evidence)
        {
            ContractInteraction action = target.contract.interaction;
            target.interactionComponent = null;
            target.interactionBeforeState = null;
            target.interactionInvokedFrame = Time.frameCount;
            target.interactionVerificationEvidence = "";
            string whitelistKey = action.componentType + "." + action.method;
            if (contract.safeInteractionMethods == null ||
                !contract.safeInteractionMethods.Any(item => string.Equals(
                    item, whitelistKey, StringComparison.Ordinal)))
            {
                evidence = "blocked: method is absent from contract safeInteractionMethods: " + whitelistKey;
                return false;
            }
            if (target.resolvedObject == null)
            {
                evidence = "target object was not resolved";
                return false;
            }
            Component component = FindNamedComponent(target.resolvedObject, action.componentType);
            if (component == null)
            {
                evidence = "component not found: " + action.componentType;
                return false;
            }
            target.interactionComponent = component;
            target.interactionBeforeState = string.IsNullOrWhiteSpace(action.successMember)
                ? null : ReadMemberPath(component, action.successMember);
            string invocationKind;
            if (component.GetType().FullName == "UnityEngine.UI.Button" && action.method == "Press")
            {
                PropertyInfo onClick = component.GetType().GetProperty("onClick");
                object eventObject = onClick == null ? null : onClick.GetValue(component, null);
                MethodInfo invoke = eventObject == null ? null : eventObject.GetType().GetMethod("Invoke", Type.EmptyTypes);
                if (invoke == null) { evidence = "UI Button onClick.Invoke unavailable"; return false; }
                try { invoke.Invoke(eventObject, null); }
                catch (Exception exception)
                {
                    evidence = "invocation exception: " + exception.GetBaseException().Message;
                    return false;
                }
                invocationKind = "UI Button onClick invoked";
            }
            else
            {
                MethodInfo method = component.GetType().GetMethod(
                    action.method, BindingFlags.Instance | BindingFlags.Public,
                    null, Type.EmptyTypes, null);
                if (method == null)
                {
                    evidence = "zero-argument method not found: " + action.method;
                    return false;
                }
                object returned;
                try { returned = method.Invoke(component, null); }
                catch (Exception exception)
                {
                    evidence = "invocation exception: " + exception.GetBaseException().Message;
                    return false;
                }
                IEnumerator coroutine = returned as IEnumerator;
                if (coroutine != null)
                {
                    MonoBehaviour coroutineHost = component as MonoBehaviour;
                    if (coroutineHost == null)
                    {
                        evidence = "IEnumerator result requires a MonoBehaviour coroutine host";
                        return false;
                    }
                    try { coroutineHost.StartCoroutine(coroutine); }
                    catch (Exception exception)
                    {
                        evidence = "coroutine start exception: " +
                            exception.GetBaseException().Message;
                        return false;
                    }
                    invocationKind = "IEnumerator started with MonoBehaviour.StartCoroutine";
                }
                else invocationKind = "method invoked";
                if (returned is bool && !(bool)returned)
                {
                    evidence = "method returned false";
                    return false;
                }
            }
            evidence = invocationKind + "; postcondition polling begins on a later frame";
            return true;
        }

        private bool TryPollInteractionVerification(GroundedTarget target, out string evidence)
        {
            ContractInteraction action = target.contract.interaction;
            if (!target.interactionAttempted || target.interactionInvokedFrame < 0)
            {
                evidence = "interaction invocation was not recorded";
                return false;
            }
            if (Time.frameCount <= target.interactionInvokedFrame)
            {
                evidence = "pending: waiting until the frame after invocation";
                return false;
            }
            if (!target.interactionInvocationSucceeded)
            {
                evidence = "interaction invocation failed; postcondition cannot authorize success: " +
                    target.interactionInvocationEvidence;
                return false;
            }
            if (string.IsNullOrWhiteSpace(action.successMember))
            {
                evidence = "invocation completed; contract supplied no state member for verification";
                return !action.required;
            }
            Component component = target.interactionComponent;
            if (component == null)
            {
                evidence = "pending: interaction component unavailable after invocation; " +
                    target.interactionInvocationEvidence;
                return false;
            }
            object stateValue = ReadMemberPath(component, action.successMember);
            string before = target.interactionBeforeState == null
                ? "null"
                : Convert.ToString(target.interactionBeforeState, CultureInfo.InvariantCulture);
            string actual = stateValue == null ? "null" : Convert.ToString(stateValue, CultureInfo.InvariantCulture);
            bool verified = string.Equals(action.expectedValue, "changed", StringComparison.OrdinalIgnoreCase)
                ? !string.Equals(before, actual, StringComparison.Ordinal)
                : string.Equals(actual, action.expectedValue, StringComparison.OrdinalIgnoreCase);
            evidence = "state " + action.successMember + ": before=" + before + "; after=" + actual +
                "; expected=" + action.expectedValue + (verified ? "; verified" : "; pending");
            return verified;
        }

        public void ObserveSimulationStep(SceneAwareStepSample sample)
        {
            telemetryRows++;
            float cameraSpeed = 0f;
            float angularSpeed = 0f;
            float acceleration = 0f;
            float headLocalPositionDelta = 0f;
            float headLocalAngularDelta = 0f;
            if (hasPreviousSample && sample.deltaTime > 0.00001f)
            {
                cameraSpeed = Vector3.Distance(sample.cameraPosition, previousSample.cameraPosition) / sample.deltaTime;
                angularSpeed = Quaternion.Angle(sample.cameraRotation, previousSample.cameraRotation) / sample.deltaTime;
                acceleration = Mathf.Abs(cameraSpeed - previousCameraSpeed) / sample.deltaTime;
                headLocalPositionDelta = Vector3.Distance(
                    sample.headLocalPosition, previousSample.headLocalPosition);
                headLocalAngularDelta = Quaternion.Angle(
                    sample.headLocalRotation, previousSample.headLocalRotation);
            }
            if (interactionMotionLockedForLastCommand)
            {
                interactionMotionLockedTelemetryRows++;
                float rootLinearSpeed = Mathf.Sqrt(
                    sample.executedRight * sample.executedRight +
                    sample.executedForward * sample.executedForward);
                float rootAbsYawRate = Mathf.Abs(sample.executedYawRate);
                interactionMotionLockMaxLinearSpeed = Mathf.Max(
                    interactionMotionLockMaxLinearSpeed, rootLinearSpeed);
                interactionMotionLockMaxAbsYawRate = Mathf.Max(
                    interactionMotionLockMaxAbsYawRate, rootAbsYawRate);
                if (rootLinearSpeed > InteractionRootCommandEpsilon ||
                    rootAbsYawRate > InteractionRootCommandEpsilon)
                    interactionMotionLockViolations++;
                interactionHeadMotionMaxPositionDelta = Mathf.Max(
                    interactionHeadMotionMaxPositionDelta, headLocalPositionDelta);
                interactionHeadMotionMaxAngularDelta = Mathf.Max(
                    interactionHeadMotionMaxAngularDelta, headLocalAngularDelta);
                if (headLocalPositionDelta > 0.000001f || headLocalAngularDelta > 0.0001f)
                    interactionHeadMotionTelemetryRows++;
            }
            float exposureProxy = 100f * (
                0.34f * Mathf.Clamp01(cameraSpeed / 2f) +
                0.46f * Mathf.Clamp01(angularSpeed / 120f) +
                0.20f * Mathf.Clamp01(acceleration / 6f));
            cameraSpeedSum += cameraSpeed;
            cameraAngularSpeedSum += angularSpeed;
            exposureProxySum += exposureProxy;
            maxCameraSpeed = Mathf.Max(maxCameraSpeed, cameraSpeed);
            maxCameraAngularSpeed = Mathf.Max(maxCameraAngularSpeed, angularSpeed);
            maxCameraAcceleration = Mathf.Max(maxCameraAcceleration, acceleration);
            maxExposureProxy = Mathf.Max(maxExposureProxy, exposureProxy);
            previousSample = sample;
            previousCameraSpeed = cameraSpeed;
            hasPreviousSample = true;
            if (telemetryWriter == null) return;
            float distance = current == null ? 0f : Vector3.Distance(sample.bodyPosition, current.approachPosition);
            StringBuilder line = new StringBuilder(320);
            line.Append(sample.stepIndex.ToString(CultureInfo.InvariantCulture));
            Csv(line, sample.simulatedTime);
            line.Append(',').Append(state);
            line.Append(',').Append(CsvText(current == null ? "" : current.contract.id));
            Csv(line, sample.bodyPosition.x); Csv(line, sample.bodyPosition.y); Csv(line, sample.bodyPosition.z);
            Csv(line, sample.bodyYawDegrees);
            Csv(line, sample.cameraPosition.x); Csv(line, sample.cameraPosition.y); Csv(line, sample.cameraPosition.z);
            Csv(line, sample.learnedRight); Csv(line, sample.learnedForward); Csv(line, sample.learnedYawRate);
            Csv(line, sample.executedRight); Csv(line, sample.executedForward); Csv(line, sample.executedYawRate);
            Csv(line, distance); Csv(line, Coverage());
            Csv(line, cameraSpeed); Csv(line, angularSpeed); Csv(line, acceleration); Csv(line, exposureProxy);
            line.Append(',').Append(interactionMotionLockedForLastCommand ? '1' : '0');
            line.Append(',').Append(CsvText(interactionMotionLockTargetIdForLastCommand));
            Csv(line, headLocalPositionDelta); Csv(line, headLocalAngularDelta);
            telemetryWriter.WriteLine(line.ToString());
            if (telemetryRows % 100 == 0) telemetryWriter.Flush();
        }

        public bool TryGetPlatformPose(float simulatedTime, out float verticalOffset,
            out float pitchDegrees, out float rollDegrees)
        {
            if (interactionMotionLocked || interactionMotionLockedForLastCommand)
            {
                // Keep the BodyRoot/ground baseline fixed during the transaction.
                // Natural HeadLocal pose is still generated by NormalHmdAgentController.
                verticalOffset = groundVerticalOffset;
                pitchDegrees = 0f;
                rollDegrees = 0f;
                return true;
            }
            float sampledY;
            if (TrySampleGround(lastBodyPosition, out sampledY))
            {
                float requested = sampledY - spawnGroundY;
                if (Mathf.Abs(requested - groundVerticalOffset) <= 0.75f)
                {
                    groundVerticalOffset = Mathf.MoveTowards(
                        groundVerticalOffset, requested, 1.5f * lastDriverDt);
                    groundHitCount++;
                }
                else groundRejectedStepCount++;
            }
            else groundMissCount++;
            verticalOffset = groundVerticalOffset;
            pitchDegrees = 0f;
            rollDegrees = 0f;
            return true;
        }

        public float SafetyTimeoutSeconds { get { return MaximumEpisodeSeconds; } }

        /// <summary>
        /// A generic scene run ends when the analyser-declared content has
        /// actually been covered, not when an arbitrary 60-second timer expires.
        /// Required interactions are part of completion.  A short evidence floor
        /// gives the render probe and motion logger time to produce useful output;
        /// the separate stall and safety limits prevent an unbounded player.
        /// </summary>
        public bool TryGetEpisodeCompletion(float simulatedTime, out string reason)
        {
            reason = "";
            if (!episodeBegun || episodeComplete) return false;

            bool contentReady = ContentCompletionSatisfied();
            if (contentReady)
            {
                if (simulatedTime >= MinimumEvidenceSeconds)
                {
                    reason = "content_complete";
                    return true;
                }
                return false;
            }
            if (simulatedTime >= MaximumEpisodeSeconds)
            {
                reason = "safety_timeout";
                return true;
            }
            if (string.Equals(incompleteTerminalReason, "content_exhausted",
                StringComparison.Ordinal))
            {
                reason = "content_exhausted";
                return true;
            }
            if (state == BrainState.Complete ||
                simulatedTime - lastMeaningfulProgressAt >= NoProgressTimeoutSeconds)
            {
                reason = "stalled";
                return true;
            }
            return false;
        }

        public void CompleteEpisode(string reason)
        {
            if (episodeComplete) return;
            episodeComplete = true;
            CompleteActiveDecision(lastSimulatedTime, false, false, true,
                "episode completed: " + reason);
            if (interactionMotionLocked)
                ReleaseInteractionMotionLock(lastSimulatedTime,
                    "episode ended while interaction transaction was locked: " + reason);
            CloseWriters();
            int resolved = targets.Count(item => item.resolvedObject != null);
            int visited = targets.Count(item => item.visited);
            int declaredInteractions = targets.Count(item => HasInteraction(item.contract));
            int requiredInteractions = targets.Count(item =>
                HasInteraction(item.contract) && item.contract.interaction.required);
            int requiredInteractionsVerified = targets.Count(item =>
                HasInteraction(item.contract) && item.contract.interaction.required &&
                item.interactionVerified);
            bool contentCompletionSatisfied = ContentCompletionSatisfied();
            if (visited == 0) failures.Add("No exploration target was completed.");
            foreach (GroundedTarget item in targets)
                if (HasInteraction(item.contract) && item.contract.interaction.required && !item.interactionVerified)
                    if (!failures.Any(message => message.Contains(item.contract.id)))
                        failures.Add("Required interaction was not verified: " + item.contract.id);
            if (reason == "safety_timeout" &&
                !failures.Any(message => message.StartsWith("Safety timeout", StringComparison.Ordinal)))
                failures.Add("Safety timeout reached before all declared scene content was completed.");
            if (reason == "stalled" &&
                !failures.Any(message => message.StartsWith("Exploration stalled", StringComparison.Ordinal)))
                failures.Add("Exploration stalled before all declared scene content was completed.");
            if (reason == "content_exhausted" &&
                !failures.Any(message => message.StartsWith("Content exhausted", StringComparison.Ordinal)))
                failures.Add("Content exhausted before all declared scene content was completed.");
            if (interactionMotionLockViolations > 0)
                failures.Add("Interaction motion lock emitted non-zero BodyRoot commands on " +
                    interactionMotionLockViolations + " telemetry row(s).");
            if (interactionsAttempted > interactionMotionLockAcquisitions)
                failures.Add("At least one interaction was attempted without first acquiring the motion lock.");
            if (interactionMotionLockReleases != interactionMotionLockAcquisitions)
                failures.Add("Interaction motion lock acquire/release counts do not match.");
            if (collisionOverlapDetections > 0)
                failures.Add("BodyRoot collision capsule entered an environment overlap on " +
                    collisionOverlapDetections + " distinct episode(s).");

            bool modelEvidenceRequired = decisionBrainRuntime.RequestedMode != UnityDecisionBrainMode.Rule;
            if (modelEvidenceRequired &&
                (decisionBrainRuntime.ModelStatus != "ready" ||
                 decisionBrainInferenceCount < 1 || decisionBrainFallbackCount != 0 ||
                 !decisionBrainRuntime.GoldenSelfTestPassed ||
                 !decisionBrainRuntime.AllInferenceLogitsFinite))
            {
                failures.Add("Decision Brain evidence gate failed: status=" +
                    decisionBrainRuntime.ModelStatus + "; inferences=" + decisionBrainInferenceCount +
                    "; fallbacks=" + decisionBrainFallbackCount + "; golden=" +
                    decisionBrainRuntime.GoldenSelfTestPassed + "; finite=" +
                    decisionBrainRuntime.AllInferenceLogitsFinite);
            }

            bool adaptivePass = reason == "content_complete" &&
                contentCompletionSatisfied &&
                lastSimulatedTime >= MinimumEvidenceSeconds && telemetryRows > 0;
            // Generic exploration is content-driven.  A fixed-duration partial
            // rollout must never manufacture a passing acceptance report.  The
            // standalone Normal HMD demo retains its separate 60-second test.
            bool passed = adaptivePass && failures.Count == 0;
            GenericAgentAcceptance report = new GenericAgentAcceptance
            {
                status = passed ? "passed" : "failed",
                generatedAtUtc = DateTime.UtcNow.ToString("O"),
                projectId = contract.projectId,
                sceneName = contract.sceneName,
                episodeId = episodeId,
                seed = seed,
                simulatedSeconds = lastSimulatedTime,
                telemetryRows = telemetryRows,
                decisions = decisionCount,
                targetsDeclared = targets.Count,
                targetsResolved = resolved,
                targetsVisited = visited,
                coverage = Coverage(),
                interactionsDeclared = declaredInteractions,
                interactionsRequired = requiredInteractions,
                interactionsAttempted = interactionsAttempted,
                interactionsVerified = interactionsVerified,
                interactionsRequiredVerified = requiredInteractionsVerified,
                interactionMotionLockAcquisitions = interactionMotionLockAcquisitions,
                interactionMotionLockReleases = interactionMotionLockReleases,
                interactionMotionLockedTelemetryRows = interactionMotionLockedTelemetryRows,
                interactionMotionLockViolations = interactionMotionLockViolations,
                interactionMotionLockMaxLinearSpeed = interactionMotionLockMaxLinearSpeed,
                interactionMotionLockMaxAbsYawRate = interactionMotionLockMaxAbsYawRate,
                interactionHeadMotionTelemetryRows = interactionHeadMotionTelemetryRows,
                interactionHeadMotionMaxPositionDelta = interactionHeadMotionMaxPositionDelta,
                interactionHeadMotionMaxAngularDelta = interactionHeadMotionMaxAngularDelta,
                interactionMotionLockPassed = interactionMotionLockViolations == 0 &&
                    interactionsAttempted <= interactionMotionLockAcquisitions &&
                    interactionMotionLockReleases == interactionMotionLockAcquisitions,
                replans = replans,
                collisionRecoveries = collisionRecoveries,
                collisionAvoidanceAdjustments = collisionAvoidanceAdjustments,
                collisionHardStops = collisionHardStops,
                collisionOverlapDetections = collisionOverlapDetections,
                collisionSafetyPassed = collisionOverlapDetections == 0,
                maxCollisionBlockedSeconds = maxCollisionBlockedSeconds,
                collisionSafetyMode = CollisionSafetyMode,
                groundHitCount = groundHitCount,
                groundMissCount = groundMissCount,
                groundRejectedStepCount = groundRejectedStepCount,
                plannerMode = plannerMode,
                movementSource = "EyeNavGS-trained Normal HMD v4 supplies HeadLocal and learned root speed envelope; brain redirects BodyRoot heading",
                brainSource = decisionBrainRuntime.PolicyAuthority == "model"
                    ? "Unity Decision Brain v0 smoke-only learned target/skill policy under explicit research override; runtime safety masks remain authoritative"
                    : decisionBrainRuntime.PolicyAuthority.StartsWith("rule_shadow", StringComparison.Ordinal)
                        ? "RULE-BOOTSTRAP remains authoritative; Unity Decision Brain v0 proposals are shadow-only"
                        : decisionBrainRuntime.PolicyAuthority == "fail_closed"
                            ? "Decision model requested but unavailable or unauthorized; execution stopped fail-closed"
                            : "RULE-BOOTSTRAP contract-grounded novelty/priority finite-state executive",
                contractPath = contractPath,
                runtimeInventoryPath = inventoryPath,
                decisionsPath = decisionsPath,
                decisionContextsPath = decisionContextsPath,
                decisionOutcomesPath = decisionOutcomesPath,
                decisionBrainModeRequested = decisionBrainRuntime.RequestedModeName,
                policyAuthority = decisionBrainRuntime.PolicyAuthority,
                decisionBrainModelStatus = decisionBrainRuntime.ModelStatus,
                decisionBrainModelDetail = decisionBrainRuntime.ModelDetail,
                decisionBrainModelSha256 = decisionBrainRuntime.ModelSha256,
                decisionBrainModelSchemaVersion = decisionBrainRuntime.ModelSchemaVersion,
                decisionBrainDataStatus = decisionBrainRuntime.ModelDataStatus,
                decisionBrainGoldenSelfTestPassed = decisionBrainRuntime.GoldenSelfTestPassed,
                decisionBrainAllInferenceLogitsFinite = decisionBrainRuntime.AllInferenceLogitsFinite,
                decisionBrainInferencePath = decisionBrainInferencePath,
                decisionBrainInferences = decisionBrainInferenceCount,
                decisionBrainFallbacks = decisionBrainFallbackCount,
                decisionBrainDisagreements = decisionBrainDisagreementCount,
                decisionBrainResearchOverride = decisionBrainRuntime.ResearchOverride,
                telemetryPath = telemetryPath,
                decisionContexts = decisionContextCount,
                decisionOutcomes = decisionOutcomeCount,
                cameraAuditPath = requestedCameraAuditPath,
                screenshotPath = requestedScreenshotPath,
                completionReason = reason,
                contentCompletionSatisfied = contentCompletionSatisfied,
                unfinishedTargetCount = Mathf.Max(0, targets.Count - visited),
                minimumEvidenceSeconds = MinimumEvidenceSeconds,
                noProgressTimeoutSeconds = NoProgressTimeoutSeconds,
                safetyTimeoutSeconds = MaximumEpisodeSeconds,
                lastMeaningfulProgressAtSeconds = lastMeaningfulProgressAt,
                failures = failures.ToArray(),
            };
            string local = Path.Combine(logDirectory, "generic_agent_acceptance.json");
            File.WriteAllText(local, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            WriteExperienceReports(report, declaredInteractions);
            if (!string.IsNullOrWhiteSpace(testOutputPath))
            {
                string full = Path.GetFullPath(testOutputPath);
                Directory.CreateDirectory(Path.GetDirectoryName(full));
                File.WriteAllText(full, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            }
            Debug.Log("GENERIC_SCENE_AGENT_RESULT " + report.status + " visited=" + visited + "/" + targets.Count);
            if (testMode) Application.Quit(passed ? 0 : 3);
        }

        private void WriteExperienceReports(GenericAgentAcceptance acceptance, int declaredInteractions)
        {
            float divisor = Mathf.Max(1, telemetryRows);
            List<string> recommendations = new List<string>();
            if (maxCameraAngularSpeed > 90f)
                recommendations.Add("Review rapid viewpoint rotation; consider turn-rate limits or snap-turn options.");
            if (maxCameraAcceleration > 5f)
                recommendations.Add("Ease locomotion starts/stops to reduce camera acceleration peaks.");
            if (contract.sceneSemantics != null && contract.sceneSemantics.Any(item =>
                item.IndexOf("dynamic", StringComparison.OrdinalIgnoreCase) >= 0))
                recommendations.Add("Keep a stable horizon/reference frame near the moving lattice and test its motion intensity.");
            if (declaredInteractions == 0)
                recommendations.Add("No verified gameplay method was exposed; add an explicit AgentAffordance contract before claiming interaction coverage.");
            if (groundMissCount > 0 || groundRejectedStepCount > 0)
                recommendations.Add("Ground following used a conservative raycast fallback and held the last safe height on " +
                    (groundMissCount + groundRejectedStepCount) + " samples; add/bake NavMesh or explicit walkable surfaces for terrain confidence.");
            if (recommendations.Count == 0)
                recommendations.Add("No threshold alert in this rollout; retain telemetry and compare multiple seeds before design changes.");

            GenericExperienceReport report = new GenericExperienceReport
            {
                status = acceptance.status,
                projectId = contract.projectId,
                sceneName = contract.sceneName,
                seed = seed,
                simulatedSeconds = lastSimulatedTime,
                targetsVisited = acceptance.targetsVisited,
                targetsDeclared = acceptance.targetsDeclared,
                coverage = acceptance.coverage,
                interactionsDeclared = declaredInteractions,
                interactionsVerified = interactionsVerified,
                interactionFinding = declaredInteractions == 0
                    ? "No callable gameplay affordance was declared by analysis; the agent did not invent one."
                    : interactionsVerified + "/" + declaredInteractions + " declared interactions verified by runtime state.",
                meanCameraSpeed = cameraSpeedSum / divisor,
                maxCameraSpeed = maxCameraSpeed,
                meanCameraAngularSpeed = cameraAngularSpeedSum / divisor,
                maxCameraAngularSpeed = maxCameraAngularSpeed,
                maxCameraAcceleration = maxCameraAcceleration,
                meanEngineeringExposureProxy = exposureProxySum / divisor,
                maxEngineeringExposureProxy = maxExposureProxy,
                riskMeaning = "Uncalibrated engineering signal from generated camera motion; not a VR.net probability, clinical score or causal estimate.",
                observedSceneSemantics = contract.sceneSemantics ?? new string[0],
                recommendations = recommendations.ToArray(),
                limitations = contract.declaredLimitations ?? new string[0]
            };
            string jsonPath = Path.Combine(logDirectory, "experience_report.json");
            string markdownPath = Path.Combine(logDirectory, "experience_report.md");
            File.WriteAllText(jsonPath, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            StringBuilder markdown = new StringBuilder();
            markdown.AppendLine("# Agent experience report").AppendLine();
            markdown.AppendLine("- Project: `" + report.projectId + "`");
            markdown.AppendLine("- Scene: `" + report.sceneName + "`");
            markdown.AppendLine("- Seed: `" + report.seed + "`");
            markdown.AppendLine("- Result: **" + report.status + "**");
            markdown.AppendLine("- Coverage: **" + report.targetsVisited + "/" + report.targetsDeclared +
                " (" + (report.coverage * 100f).ToString("F1", CultureInfo.InvariantCulture) + "%)**").AppendLine();
            markdown.AppendLine("## What the agent did").AppendLine();
            markdown.AppendLine("The brain consumed the analysis contract, grounded named targets against live Unity objects, " +
                PolicyReportSentence() + " It then scheduled Plan, Navigate, Stop, Look, Observe, Verify and Recover. " +
                "Interact is scheduled only when a separately reviewed method appears in the contract allowlist.").AppendLine();
            markdown.AppendLine("Interaction: " + report.interactionFinding).AppendLine();
            markdown.AppendLine("## Engineering motion signals").AppendLine();
            markdown.AppendLine("| Signal | Mean | Maximum |");
            markdown.AppendLine("|---|---:|---:|");
            markdown.AppendLine("| Camera speed | " + report.meanCameraSpeed.ToString("F3", CultureInfo.InvariantCulture) +
                " | " + report.maxCameraSpeed.ToString("F3", CultureInfo.InvariantCulture) + " |");
            markdown.AppendLine("| Camera angular speed | " + report.meanCameraAngularSpeed.ToString("F3", CultureInfo.InvariantCulture) +
                " | " + report.maxCameraAngularSpeed.ToString("F3", CultureInfo.InvariantCulture) + " |");
            markdown.AppendLine("| Exposure proxy | " + report.meanEngineeringExposureProxy.ToString("F2", CultureInfo.InvariantCulture) +
                " | " + report.maxEngineeringExposureProxy.ToString("F2", CultureInfo.InvariantCulture) + " |").AppendLine();
            markdown.AppendLine("> " + report.riskMeaning).AppendLine();
            markdown.AppendLine("## Recommendations").AppendLine();
            foreach (string item in report.recommendations) markdown.AppendLine("- " + item);
            markdown.AppendLine().AppendLine("## Declared limitations").AppendLine();
            foreach (string item in report.limitations) markdown.AppendLine("- " + item);
            File.WriteAllText(markdownPath, markdown.ToString(), new UTF8Encoding(false));
        }

        private RuntimeSceneInventory CreateInventory()
        {
            GameObject[] roots = SceneManager.GetActiveScene().GetRootGameObjects();
            Transform[] transforms = FindObjectsByType<Transform>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            Renderer[] renderers = FindObjectsByType<Renderer>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            Collider[] colliders = FindObjectsByType<Collider>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            MonoBehaviour[] behaviours = FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            HashSet<string> behaviourTypes = new HashSet<string>(StringComparer.Ordinal);
            foreach (MonoBehaviour behaviour in behaviours)
                if (behaviour != null) behaviourTypes.Add(behaviour.GetType().FullName);
            Bounds walkable = InferWalkableBounds(renderers);
            RuntimeTargetResolution[] resolutions = targets.Select(target => new RuntimeTargetResolution
            {
                targetId = target.contract.id,
                requestedNameContains = target.contract.objectNameContains,
                resolvedObjectPath = target.resolvedObject == null ? "" : HierarchyPath(target.resolvedObject),
                resolvedWorldPosition = target.focusPosition,
                approachPosition = target.approachPosition,
                objectResolved = target.resolvedObject != null,
                navMeshProjected = target.navMeshProjected,
                resolutionEvidence = target.resolvedObject == null
                    ? "positionHint fallback; no name match"
                    : "live object name + nearest positionHint match"
            }).ToArray();
            return new RuntimeSceneInventory
            {
                projectId = contract.projectId,
                sceneName = SceneManager.GetActiveScene().name,
                rootObjectCount = roots.Length,
                activeObjectCount = transforms.Length,
                rendererCount = renderers.Length,
                colliderCount = colliders.Length,
                behaviourCount = behaviours.Length,
                rootObjects = roots.Select(item => item.name).OrderBy(item => item).ToArray(),
                behaviourTypes = behaviourTypes.OrderBy(item => item).ToArray(),
                targetResolutions = resolutions,
                walkabilityMode = plannerMode,
                inferredWalkableBounds = new BoundsSnapshot { center = walkable.center, size = walkable.size },
                capturedAtUtc = DateTime.UtcNow.ToString("O")
            };
        }

        private bool CanReach(Vector3 from, Vector3 to)
        {
            if (hasUsableNavMesh)
            {
                NavMeshHit startHit;
                NavMeshHit endHit;
                if (!NavMesh.SamplePosition(from, out startHit, 2.5f, NavMesh.AllAreas) ||
                    !NavMesh.SamplePosition(to, out endHit, 4f, NavMesh.AllAreas))
                    return false;
                NavMeshPath path = new NavMeshPath();
                return NavMesh.CalculatePath(startHit.position, endHit.position, NavMesh.AllAreas, path) &&
                    path.status == NavMeshPathStatus.PathComplete && path.corners.Length >= 2;
            }
            if (!hasInferredWalkableBounds) return true;
            Bounds padded = inferredWalkableBounds;
            padded.Expand(new Vector3(0.5f, 20f, 0.5f));
            Vector3 flattened = new Vector3(to.x, padded.center.y, to.z);
            return padded.Contains(flattened);
        }

        private bool BuildActiveNavPath(
            Vector3 from, Vector3 to, float time, out string evidence)
        {
            activeNavCorners = new Vector3[0];
            activeNavCornerIndex = 0;
            activeNavTargetId = current == null ? "" : current.contract.id;
            NavMeshHit startHit;
            NavMeshHit endHit;
            if (!NavMesh.SamplePosition(from, out startHit, 2.5f, NavMesh.AllAreas))
            {
                evidence = "NavMesh.SamplePosition failed for live BodyRoot at " + Vec(from);
                return false;
            }
            if (!NavMesh.SamplePosition(to, out endHit, 4f, NavMesh.AllAreas))
            {
                evidence = "NavMesh.SamplePosition failed for approach target at " + Vec(to);
                return false;
            }
            NavMeshPath path = new NavMeshPath();
            if (!NavMesh.CalculatePath(startHit.position, endHit.position, NavMesh.AllAreas, path) ||
                path.status != NavMeshPathStatus.PathComplete || path.corners.Length < 2)
            {
                evidence = "NavMesh.CalculatePath did not return a complete path";
                return false;
            }
            activeNavCorners = path.corners.ToArray();
            activeNavCornerIndex = 1;
            activeNavPathBuiltAt = time;
            float length = 0f;
            for (int index = 1; index < activeNavCorners.Length; index++)
                length += Vector3.Distance(activeNavCorners[index - 1], activeNavCorners[index]);
            evidence = "planner=unity-navmesh-corner-following; corners=" +
                activeNavCorners.Length + "; length=" +
                length.ToString("F2", CultureInfo.InvariantCulture) + "m; start=" +
                Vec(startHit.position) + "; end=" + Vec(endHit.position);
            return true;
        }

        private bool TryGetNavigationAim(
            Vector3 bodyPosition, float time, out Vector3 aim, out string evidence)
        {
            aim = current == null ? bodyPosition : current.approachPosition;
            evidence = "no active NavMesh target";
            if (current == null) return false;
            bool refresh = activeNavCorners.Length < 2 ||
                !string.Equals(activeNavTargetId, current.contract.id, StringComparison.Ordinal) ||
                time - activeNavPathBuiltAt >= 3f;
            if (refresh)
            {
                if (!BuildActiveNavPath(bodyPosition, current.approachPosition, time, out evidence))
                    return false;
                RecordDecision(time, "Plan", current.contract.id, "refresh_navmesh_path",
                    "follow refreshed NavMesh corners", true, evidence);
            }
            while (activeNavCornerIndex < activeNavCorners.Length - 1)
            {
                Vector3 planar = activeNavCorners[activeNavCornerIndex] - bodyPosition;
                planar.y = 0f;
                if (planar.magnitude > 0.55f) break;
                activeNavCornerIndex++;
                RecordDecision(time, "Navigate", current.contract.id,
                    "advance_navmesh_corner", "corner " + activeNavCornerIndex,
                    true, "BodyRoot reached cached path corner");
            }
            if (activeNavCornerIndex < 0 || activeNavCornerIndex >= activeNavCorners.Length)
            {
                evidence = "cached NavMesh corner index escaped path bounds";
                return false;
            }
            aim = activeNavCorners[activeNavCornerIndex];
            evidence = "corner=" + activeNavCornerIndex + "/" +
                (activeNavCorners.Length - 1) + "; aim=" + Vec(aim);
            return true;
        }

        private bool TryResolveCollisionSafeVelocity(
            Vector3 bodyPosition, Vector3 desiredVelocity, bool maySteer,
            out Vector3 safeVelocity, out bool adjusted,
            out bool startedOverlapping, out string evidence)
        {
            float speed = desiredVelocity.magnitude;
            safeVelocity = Vector3.zero;
            adjusted = false;
            startedOverlapping = CapsuleOverlapsEnvironment(bodyPosition);
            if (startedOverlapping)
            {
                evidence = "BodyRoot capsule already overlaps a non-trigger environment collider";
                return false;
            }
            Vector3 direct = desiredVelocity / Mathf.Max(0.0001f, speed);
            float lookAhead = Mathf.Max(0.48f, speed * 0.72f + 0.22f);
            RaycastHit directHit;
            if (!CapsuleCastEnvironment(bodyPosition, direct, lookAhead, out directHit))
            {
                safeVelocity = desiredVelocity;
                evidence = "capsule sweep clear";
                return true;
            }
            if (!maySteer)
            {
                evidence = "capsule sweep blocked while stopping by " +
                    ColliderPath(directHit.collider);
                return false;
            }
            float side = preferredAvoidanceSign;
            float[] alternatives = { side * 34f, side * 62f, -side * 34f, -side * 62f, side * 88f, -side * 88f };
            foreach (float angle in alternatives)
            {
                Vector3 candidate = Quaternion.Euler(0f, angle, 0f) * direct;
                RaycastHit alternativeHit;
                if (!CapsuleCastEnvironment(bodyPosition, candidate, lookAhead, out alternativeHit))
                {
                    safeVelocity = candidate * speed * 0.68f;
                    adjusted = true;
                    preferredAvoidanceSign = angle < 0f ? -1 : 1;
                    evidence = "capsule sweep avoided " + ColliderPath(directHit.collider) +
                        " at " + angle.ToString("F0", CultureInfo.InvariantCulture) + "deg";
                    return true;
                }
            }
            evidence = "capsule sweep blocked direct and six stable-side steering candidates by " +
                ColliderPath(directHit.collider);
            return false;
        }

        private bool CapsuleOverlapsEnvironment(Vector3 bodyPosition)
        {
            Vector3 bottom;
            Vector3 top;
            CollisionCapsule(bodyPosition, out bottom, out top);
            Collider[] overlaps = Physics.OverlapCapsule(
                bottom, top, 0.28f, Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Ignore);
            foreach (Collider collider in overlaps)
            {
                if (!IsBlockingEnvironmentCollider(collider)) continue;
                if (IsWalkableSupportOverlap(collider, bottom, top)) continue;
                return true;
            }
            return false;
        }

        private static bool IsWalkableSupportOverlap(
            Collider collider, Vector3 bottom, Vector3 top)
        {
            if (!IsWalkableGroundCollider(collider, null)) return false;
            const float radius = 0.28f;
            Vector3 supportPoint = collider.ClosestPoint(bottom);
            Vector3 separation = bottom - supportPoint;
            float supportDistance = separation.magnitude;
            // ClosestPoint returns the query point when it is inside a collider.
            // That is a real penetration, never a floor-contact false positive.
            if (supportDistance <= 0.0001f ||
                separation.y / supportDistance < 0.55f)
                return false;
            float footY = bottom.y - radius;
            if (supportPoint.y < footY - 0.20f ||
                supportPoint.y > footY + 0.48f)
                return false;
            Vector3 middle = (bottom + top) * 0.5f;
            // Only the lower sphere may touch this collider.  A dynamic box,
            // steep wall, or overhang reaching the torso/top remains blocking
            // even when its name happens to contain a ground keyword.
            if (Vector3.Distance(middle, collider.ClosestPoint(middle)) <= radius ||
                Vector3.Distance(top, collider.ClosestPoint(top)) <= radius)
                return false;
            return true;
        }

        private void SampleCollisionOverlapEpisode(Vector3 bodyPosition)
        {
            bool overlapping = CapsuleOverlapsEnvironment(bodyPosition);
            if (overlapping && !collisionOverlapActive)
            {
                collisionOverlapDetections++;
                RecordDecision(lastSimulatedTime, "CollisionSafety",
                    current == null ? "" : current.contract.id,
                    "collision_overlap_entry", "fail acceptance", false,
                    "BodyRoot capsule entered a non-trigger environment collider");
            }
            collisionOverlapActive = overlapping;
        }

        private bool CapsuleCastEnvironment(
            Vector3 bodyPosition, Vector3 direction, float distance, out RaycastHit nearest)
        {
            Vector3 bottom;
            Vector3 top;
            CollisionCapsule(bodyPosition, out bottom, out top);
            RaycastHit[] hits = Physics.CapsuleCastAll(
                bottom, top, 0.28f, direction, distance,
                Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
            nearest = default(RaycastHit);
            bool found = false;
            float nearestDistance = float.PositiveInfinity;
            foreach (RaycastHit hit in hits)
            {
                if (!IsBlockingEnvironmentCollider(hit.collider) ||
                    IsWalkableSupportHit(hit, bodyPosition, direction) ||
                    hit.distance >= nearestDistance)
                    continue;
                nearest = hit;
                nearestDistance = hit.distance;
                found = true;
            }
            return found;
        }

        private bool IsWalkableSupportHit(
            RaycastHit hit, Vector3 bodyPosition, Vector3 direction)
        {
            if (!IsWalkableGroundCollider(hit.collider, null)) return false;
            Vector3 basePosition = bodyPosition + Vector3.up * groundVerticalOffset;
            float footY = basePosition.y + 0.03f;
            const float maximumStepHeight = 0.48f;
            const float maximumSupportDrop = 0.20f;
            if (hit.normal.y >= 0.55f &&
                hit.point.y >= footY - maximumSupportDrop &&
                hit.point.y <= footY + maximumStepHeight)
                return true;

            // CapsuleCastAll can return a zero-distance/zero-normal hit when the
            // lower sphere is already resting against a slope.  Reuse the
            // lower-only support geometry test; torso/top contact still blocks.
            if (hit.distance <= 0.001f)
            {
                Vector3 bottom;
                Vector3 top;
                CollisionCapsule(bodyPosition, out bottom, out top);
                if (IsWalkableSupportOverlap(hit.collider, bottom, top))
                    return true;
            }

            // A BoxCollider ramp exposes an artificial vertical nose even when
            // its adjacent top face is a legal slope.  Only a near-foot, nearby
            // nose may be reclassified, and only after a forward down-probe hits
            // this exact collider with a walkable normal inside the step window.
            if (hit.distance > 0.55f ||
                hit.point.y < footY - maximumSupportDrop ||
                hit.point.y > footY + maximumStepHeight)
                return false;
            Vector3 planarDirection = direction;
            planarDirection.y = 0f;
            if (planarDirection.sqrMagnitude <= 0.0001f) return false;
            planarDirection.Normalize();
            float probeAhead = Mathf.Clamp(hit.distance + 0.24f, 0.24f, 0.58f);
            Vector3 probeOrigin = bodyPosition + planarDirection * probeAhead;
            probeOrigin.y = footY + maximumStepHeight + 0.24f;
            RaycastHit supportHit;
            if (!hit.collider.Raycast(
                new Ray(probeOrigin, Vector3.down), out supportHit,
                maximumStepHeight + maximumSupportDrop + 0.72f))
                return false;
            return IsWalkableGroundHit(supportHit, null) &&
                supportHit.point.y >= footY - maximumSupportDrop &&
                supportHit.point.y <= footY + maximumStepHeight;
        }

        private void CollisionCapsule(Vector3 bodyPosition, out Vector3 bottom, out Vector3 top)
        {
            // groundVerticalOffset is presentation/world height maintained by
            // TryGetPlatformPose; include its last safe value so the collision
            // proxy follows ramps instead of sweeping from the spawn elevation.
            Vector3 basePosition = bodyPosition + Vector3.up * groundVerticalOffset;
            bottom = basePosition + Vector3.up * 0.31f;
            top = basePosition + Vector3.up * 1.37f;
        }

        private bool IsBlockingEnvironmentCollider(Collider collider)
        {
            if (collider == null || collider.isTrigger || !collider.enabled)
                return false;
            return motionSkill == null || !collider.transform.IsChildOf(motionSkill.transform);
        }

        private static string ColliderPath(Collider collider)
        {
            return collider == null ? "unknown collider" : HierarchyPath(collider.transform);
        }

        private static bool TrySampleGround(Vector3 position, out float groundY)
        {
            RaycastHit[] hits = Physics.RaycastAll(
                new Vector3(position.x, position.y + 20f, position.z),
                Vector3.down, 45f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
            Array.Sort(hits, (left, right) => left.distance.CompareTo(right.distance));
            foreach (RaycastHit hit in hits)
            {
                if (!IsWalkableGroundHit(hit, null)) continue;
                groundY = hit.point.y;
                return true;
            }
            groundY = position.y;
            return false;
        }

        private static Vector3 ProjectToWalkableSurface(Vector3 proposed, Collider[] colliders)
        {
            RaycastHit[] hits = Physics.RaycastAll(
                proposed + Vector3.up * 30f, Vector3.down, 60f,
                Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore);
            Array.Sort(hits, (left, right) => left.distance.CompareTo(right.distance));
            foreach (RaycastHit hit in hits)
            {
                if (IsWalkableGroundHit(hit, colliders))
                    return new Vector3(proposed.x, hit.point.y, proposed.z);
            }
            return proposed;
        }

        private static bool IsWalkableGroundHit(RaycastHit hit, Collider[] colliders)
        {
            Collider collider = hit.collider;
            if (!IsWalkableGroundCollider(collider, colliders) || hit.normal.y < 0.55f)
                return false;
            return true;
        }

        private static bool IsWalkableGroundCollider(Collider collider, Collider[] colliders)
        {
            if (collider == null || !collider.enabled || collider.isTrigger)
                return false;
            if (colliders != null && Array.IndexOf(colliders, collider) < 0)
                return false;
            // The Normal HMD visual body temporarily retains its primitive
            // collider until end-of-frame after Destroy().  Its diagnostic
            // name contains "Ground Truth", so it must never become terrain.
            if (collider.GetComponentInParent<NormalHmdAgentController>() != null)
                return false;
            string name = collider.name.ToLowerInvariant();
            bool declaredGround = name.Contains("floor") || name.Contains("ground") ||
                name.Contains("walkway") || name.Contains("path") || name.Contains("terrain") ||
                name.Contains("ramp") || name.Contains("slope") || name.Contains("platform");
            return declaredGround || collider is TerrainCollider;
        }

        private static Bounds InferWalkableBounds(Renderer[] renderers)
        {
            Bounds combined = new Bounds(Vector3.zero, Vector3.zero);
            bool hasBounds = false;
            foreach (Renderer renderer in renderers)
            {
                string name = renderer.name.ToLowerInvariant();
                if (!name.Contains("floor") && !name.Contains("ground") &&
                    !name.Contains("walkway") && !name.Contains("path")) continue;
                if (!hasBounds) { combined = renderer.bounds; hasBounds = true; }
                else combined.Encapsulate(renderer.bounds);
            }
            return combined;
        }

        private static Vector3 BoundsCentre(Transform target)
        {
            Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) return target.position;
            Bounds bounds = renderers[0].bounds;
            for (int index = 1; index < renderers.Length; index++) bounds.Encapsulate(renderers[index].bounds);
            return bounds.center;
        }

        private static Component FindNamedComponent(Transform target, string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName)) return null;
            Component[] components = target.GetComponentsInChildren<Component>(true);
            foreach (Component component in components)
                if (component != null && (component.GetType().Name == typeName || component.GetType().FullName == typeName))
                    return component;
            components = target.GetComponentsInParent<Component>(true);
            foreach (Component component in components)
                if (component != null && (component.GetType().Name == typeName || component.GetType().FullName == typeName))
                    return component;
            return null;
        }

        private static object ReadMemberPath(object instance, string path)
        {
            object current = instance;
            foreach (string name in path.Split('.'))
            {
                if (current == null) return null;
                Type type = current.GetType();
                PropertyInfo property = type.GetProperty(name,
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (property != null && property.GetIndexParameters().Length == 0)
                {
                    current = property.GetValue(current, null);
                    continue;
                }
                FieldInfo field = type.GetField(name,
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (field == null) return null;
                current = field.GetValue(current);
            }
            return current;
        }

        private void RecordDecision(float time, string stateName, string targetId,
            string intent, string command, bool confirmed, string evidence)
        {
            DecisionRecord record = new DecisionRecord
            {
                utc = DateTime.UtcNow.ToString("O"), simulatedTime = time,
                state = stateName, skill = command, targetId = targetId,
                intent = intent, command = command, engineConfirmed = confirmed,
                evidence = evidence
            };
            string line = JsonUtility.ToJson(record);
            decisionCount++;
            if (decisionWriter == null) pendingDecisions.Add(line);
            else { decisionWriter.WriteLine(line); decisionWriter.Flush(); }
        }

        private float Coverage()
        {
            return targets.Count == 0 ? 0f : (float)targets.Count(item => item.visited) / targets.Count;
        }

        private bool RequiredInteractionsComplete()
        {
            return targets.All(item =>
                !HasInteraction(item.contract) ||
                !item.contract.interaction.required || item.interactionVerified);
        }

        private bool ContentCompletionSatisfied()
        {
            return AllVisited() && RequiredInteractionsComplete();
        }

        private void MarkContentExhausted(float time, string evidence)
        {
            incompleteTerminalReason = "content_exhausted";
            CompleteActiveDecision(time, false, false, true, evidence);
            if (!failures.Any(message => message.StartsWith(
                "Content exhausted", StringComparison.Ordinal)))
                failures.Add("Content exhausted before completion: " + evidence);
            if (interactionMotionLocked)
                RequestInteractionMotionLockRelease(time,
                    "content exhausted while interaction motion lock was active");
            current = null;
            Transition(time, BrainState.Complete, "Stop",
                "content_exhausted; " + evidence);
        }

        private bool AllVisited() { return targets.Count > 0 && targets.All(item => item.visited); }
        private static bool HasInteraction(ContractTarget target)
        {
            return target != null && target.interaction != null &&
                !string.IsNullOrWhiteSpace(target.interaction.method);
        }

        private void OnGUI()
        {
            if (contract == null) return;
            EnsureStyles();
            float panelWidth = testMode
                ? Mathf.Clamp(Screen.width * 0.29f, 300f, 380f)
                : Mathf.Min(480f, Screen.width * 0.42f);
            float panelHeight = testMode ? 205f : 340f;
            Rect panelRect = new Rect(14f, 14f, panelWidth, panelHeight);
            GUI.DrawTexture(panelRect, panelTexture, ScaleMode.StretchToFill, true);
            GUILayout.BeginArea(panelRect);
            GUILayout.Space(14f); GUILayout.BeginHorizontal(); GUILayout.Space(15f); GUILayout.BeginVertical();
            GUILayout.Label(testMode ? PolicyTitle() : "GENERIC SCENE AGENT BRAIN", titleStyle);
            GUILayout.Label(contract.projectName + " / " + contract.sceneName, bodyStyle);
            GUILayout.Space(5f);
            GUILayout.Label(PolicyStatusLine(), PolicyUsesModel() ? learnedStyle : ruleStyle);
            GUILayout.Label("Brain: " + state + "  ·  Skill: " + CurrentSkillLabel(), ruleStyle);
            GUILayout.Label("Target: " + (current == null ? "none" : current.contract.label), bodyStyle);
            GUILayout.Label("Coverage: " + Mathf.RoundToInt(Coverage() * 100f) + "%  ·  Decisions: " + decisionCount, bodyStyle);
            if (testMode)
            {
                GUILayout.Label("Planner: " + plannerMode, bodyStyle);
                GUILayout.Label("HEAD learned · " + PolicyShortDescription(), learnedStyle);
                GUILayout.EndVertical(); GUILayout.Space(15f); GUILayout.EndHorizontal(); GUILayout.EndArea();
                return;
            }
            GUILayout.Space(7f);
            GUILayout.Label("LEARNED · EyeNavGS Normal HMD v4 HeadLocal + root speed envelope", learnedStyle);
            GUILayout.Label("BRAIN · " + PolicyShortDescription(), PolicyUsesModel() ? learnedStyle : ruleStyle);
            GUILayout.Space(7f);
            GUILayout.Label("Analysis handed to agent:", bodyStyle);
            GUILayout.Label(contract.reportSummary, bodyStyle);
            GUILayout.Space(5f);
            GUILayout.Label("Planner: " + plannerMode + " · Seed: " + seed, bodyStyle);
            GUILayout.Label("C toggles participant/observer camera · Space pauses", bodyStyle);
            GUILayout.EndVertical(); GUILayout.Space(15f); GUILayout.EndHorizontal(); GUILayout.EndArea();
        }

        private string CurrentSkillLabel()
        {
            if (state == BrainState.Navigate) return "Navigate (data-shaped movement)";
            if (state == BrainState.LookAt) return "LookAt";
            if (state == BrainState.Interact) return "Interact";
            if (state == BrainState.Observe) return "Observe";
            return state.ToString();
        }

        private bool PolicyUsesModel()
        {
            return decisionBrainRuntime != null &&
                decisionBrainRuntime.PolicyAuthority == "model";
        }

        private string PolicyTitle()
        {
            if (PolicyUsesModel()) return "BRAIN  ·  MODEL-CONTROLLED";
            if (decisionBrainRuntime != null &&
                decisionBrainRuntime.PolicyAuthority.StartsWith("rule_shadow", StringComparison.Ordinal))
                return "BRAIN  ·  SHADOW-COMPARED";
            if (decisionBrainRuntime != null && decisionBrainRuntime.PolicyAuthority == "fail_closed")
                return "BRAIN  ·  FAIL-CLOSED";
            return "BRAIN  ·  RULE-CONTROLLED";
        }

        private string PolicyStatusLine()
        {
            if (decisionBrainRuntime == null) return "Policy: unavailable";
            string status = "Policy: " + decisionBrainRuntime.RequestedModeName.ToUpperInvariant() +
                "  ·  Authority: " + decisionBrainRuntime.PolicyAuthority;
            if (PolicyUsesModel() && decisionBrainRuntime.ResearchOverride)
                status += "  ·  smoke-only research";
            return status;
        }

        private string PolicyShortDescription()
        {
            if (PolicyUsesModel())
                return "Decision Brain v0 ranks targets/skills; safety masks remain authoritative";
            if (decisionBrainRuntime != null &&
                decisionBrainRuntime.PolicyAuthority.StartsWith("rule_shadow", StringComparison.Ordinal))
                return "Rule executes; Decision Brain v0 proposals are logged in shadow";
            if (decisionBrainRuntime != null && decisionBrainRuntime.PolicyAuthority == "fail_closed")
                return "requested model is unavailable or unauthorized; motion stopped safely";
            return "contract-grounded deterministic rule selects targets and skills";
        }

        private string PolicyReportSentence()
        {
            if (PolicyUsesModel())
                return "The safety-masked Unity Decision Brain v0 selected unvisited reachable targets and macro skills.";
            if (decisionBrainRuntime != null &&
                decisionBrainRuntime.PolicyAuthority.StartsWith("rule_shadow", StringComparison.Ordinal))
                return "The deterministic rule selected unvisited reachable targets while model proposals were logged without control.";
            return "The deterministic rule selected unvisited reachable targets by priority, distance and seeded variation.";
        }

        private void EnsureStyles()
        {
            if (titleStyle != null) return;
            panelTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            panelTexture.SetPixel(0, 0, new Color(0.018f, 0.032f, 0.065f, 0.94f)); panelTexture.Apply();
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 18, fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.20f, 0.96f, 0.86f) } };
            bodyStyle = new GUIStyle(GUI.skin.label) { fontSize = 12, wordWrap = true,
                normal = { textColor = new Color(0.84f, 0.91f, 0.96f) } };
            learnedStyle = new GUIStyle(bodyStyle) { fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.25f, 0.72f, 1f) } };
            ruleStyle = new GUIStyle(bodyStyle) { fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(1f, 0.70f, 0.27f) } };
        }

        private void CloseWriters()
        {
            if (decisionWriter != null) { decisionWriter.Flush(); decisionWriter.Dispose(); decisionWriter = null; }
            if (decisionContextWriter != null)
            {
                decisionContextWriter.Flush();
                decisionContextWriter.Dispose();
                decisionContextWriter = null;
            }
            if (decisionOutcomeWriter != null)
            {
                decisionOutcomeWriter.Flush();
                decisionOutcomeWriter.Dispose();
                decisionOutcomeWriter = null;
            }
            if (decisionBrainInferenceWriter != null)
            {
                decisionBrainInferenceWriter.Flush();
                decisionBrainInferenceWriter.Dispose();
                decisionBrainInferenceWriter = null;
            }
            if (telemetryWriter != null) { telemetryWriter.Flush(); telemetryWriter.Dispose(); telemetryWriter = null; }
        }

        private void OnApplicationQuit()
        {
            if (!episodeComplete && episodeBegun) CompleteEpisode("application_quit");
            CloseWriters();
        }

        private static string HierarchyPath(Transform transform)
        {
            string path = transform.name;
            while (transform.parent != null) { transform = transform.parent; path = transform.name + "/" + path; }
            return path;
        }
        private static string Vec(Vector3 value) { return string.Format(CultureInfo.InvariantCulture, "({0:F2},{1:F2},{2:F2})", value.x, value.y, value.z); }
        private static void Csv(StringBuilder builder, float value) { builder.Append(',').Append(value.ToString("G9", CultureInfo.InvariantCulture)); }
        private static string CsvText(string value) { return string.IsNullOrEmpty(value) ? "" : value.Replace(',', '_'); }
        private static string Safe(string value)
        {
            value = string.IsNullOrWhiteSpace(value) ? "scene" : value;
            foreach (char invalid in Path.GetInvalidFileNameChars()) value = value.Replace(invalid, '_');
            return value;
        }
        private static bool HasArgument(string value)
        {
            return Environment.GetCommandLineArgs().Any(argument =>
                string.Equals(argument, value, StringComparison.OrdinalIgnoreCase));
        }
        private static string ReadArgument(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return argument.Substring(prefix.Length).Trim('"');
            return null;
        }
    }
}
