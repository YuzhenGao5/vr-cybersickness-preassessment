using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// A transparent high-level explorer for the Normal HMD Agent.
    ///
    /// Trained: NormalHmdAgentController's recurrent HMD motion prior.
    /// Rule based: this semantic graph, objective planner, skills and BodyRoot
    /// route guidance. The exposure number is an uncalibrated engineering proxy,
    /// not a diagnosis and not a VR.net probability.
    /// </summary>
    public sealed class SceneAwareExplorerController : MonoBehaviour, IRootMotionDriver
    {
        private enum SkillState
        {
            Navigate,
            LookAt,
            Activate,
            Observe,
            Complete
        }

        [Serializable]
        private sealed class ActionLedgerEntry
        {
            public string utc;
            public float simulatedTime;
            public string scenarioId;
            public string objectiveId;
            public string skill;
            public string target;
            public string intent;
            public string command;
            public bool engineConfirmed;
            public string result;
            public string evidence;
        }

        private NormalHmdAgentController normalHmdController;
        private Camera observerCamera;
        private GameObject scenarioRoot;
        private SceneSemanticManifest manifest;
        private readonly Dictionary<string, SemanticNodeContract> nodeById =
            new Dictionary<string, SemanticNodeContract>(StringComparer.Ordinal);
        private readonly Dictionary<string, SceneAwareAffordance> affordanceById =
            new Dictionary<string, SceneAwareAffordance>(StringComparer.Ordinal);
        private readonly HashSet<string> visitedNodes = new HashSet<string>(StringComparer.Ordinal);
        private readonly List<string> route = new List<string>();
        private readonly List<string> pendingLedgerLines = new List<string>();

        private int scenarioIndex;
        private int episodeSeed;
        private int objectiveIndex;
        private int routeCursor;
        private int objectivesCompleted;
        private int interactionsAttempted;
        private int interactionsSucceeded;
        private int telemetryRows;
        private SkillState skillState;
        private float stateEnteredAt;
        private float lastSimulatedTime;
        private float exposureSum;
        private float maxExposure;
        private float liveExposure;
        private int vehicleMotionRows;
        private float vehicleMaxSpeed;
        private float vehicleMinY;
        private float vehicleMaxY;
        private float vehicleCameraMinY;
        private float vehicleCameraMaxY;
        private bool activationAttemptedForObjective;
        private bool interactionSuccessRecorded;
        private bool interactionCommandAccepted;
        private Vector3 smoothedWorldVelocity;
        private float smoothedYawRate;
        private bool configured;
        private bool episodeBegun;
        private bool episodeCompleted;
        private bool testMode;
        private bool plannerHudExpanded;
        private string testOutputPath;
        private string requestedLogRoot;
        private string logDirectory;
        private string telemetryPath;
        private string manifestPath;
        private string actionLedgerPath;
        private StreamWriter telemetryWriter;
        private StreamWriter ledgerWriter;
        private SceneAwareStepSample previousSample;
        private bool hasPreviousSample;
        private float previousCameraSpeed;
        private SceneAwareDynamicMotionRig activeMotionRig;
        private SceneAwareRollerCoasterRig activeRollerCoaster;
        private GUIStyle titleStyle;
        private GUIStyle labelStyle;
        private GUIStyle smallStyle;
        private GUIStyle learnedStyle;
        private GUIStyle ruleStyle;
        private GUIStyle warningStyle;
        private Texture2D panelTexture;

        public int ScenarioIndex { get { return scenarioIndex; } }
        public string ScenarioName { get { return manifest == null ? "not built" : manifest.scenarioName; } }
        public float Coverage
        {
            get
            {
                int count = manifest == null || manifest.objectives == null ? 0 : manifest.objectives.Length;
                return count == 0 ? 0f : (float)objectivesCompleted / count;
            }
        }

        public void Configure(NormalHmdAgentController controller, Camera observer)
        {
            normalHmdController = controller;
            if (normalHmdController != null)
                normalHmdController.SetHudManagedExternally(true);
            observerCamera = observer;
            scenarioIndex = Mathf.Clamp(ReadIntArgument("--scenario=", 0), 0, 3);
            testMode = HasArgument("--scene-aware-test");
            testOutputPath = ReadArgumentValue("--scene-aware-test-output=");
            requestedLogRoot = ReadArgumentValue("--scene-log-root=");
            BuildScenario(scenarioIndex);
            configured = true;
            if (testMode && normalHmdController != null)
            {
                normalHmdController.SetSceneAwareSimulationSpeed(12f);
            }
        }

        private void Update()
        {
            if (!configured || testMode || normalHmdController == null) return;
            if (Input.GetKeyDown(KeyCode.Alpha1)) SwitchScenario(0);
            if (Input.GetKeyDown(KeyCode.Alpha2)) SwitchScenario(1);
            if (Input.GetKeyDown(KeyCode.Alpha3)) SwitchScenario(2);
            if (Input.GetKeyDown(KeyCode.Alpha4)) SwitchScenario(3);
            if (Input.GetKeyDown(KeyCode.R)) RestartCurrentScenario();
            if (Input.GetKeyDown(KeyCode.P)) plannerHudExpanded = !plannerHudExpanded;
        }

        public void ResetForEpisode(int seed, out Vector3 spawn, out float yawDegrees)
        {
            if (manifest == null) BuildScenario(scenarioIndex);
            CloseWriters();
            episodeSeed = seed;
            objectiveIndex = 0;
            routeCursor = 0;
            objectivesCompleted = 0;
            interactionsAttempted = 0;
            interactionsSucceeded = 0;
            telemetryRows = 0;
            exposureSum = 0f;
            maxExposure = 0f;
            liveExposure = 0f;
            vehicleMotionRows = 0;
            vehicleMaxSpeed = 0f;
            vehicleMinY = float.PositiveInfinity;
            vehicleMaxY = float.NegativeInfinity;
            vehicleCameraMinY = float.PositiveInfinity;
            vehicleCameraMaxY = float.NegativeInfinity;
            lastSimulatedTime = 0f;
            stateEnteredAt = 0f;
            activationAttemptedForObjective = false;
            interactionSuccessRecorded = false;
            interactionCommandAccepted = false;
            smoothedWorldVelocity = Vector3.zero;
            smoothedYawRate = 0f;
            episodeBegun = false;
            episodeCompleted = false;
            hasPreviousSample = false;
            previousCameraSpeed = 0f;
            visitedNodes.Clear();
            pendingLedgerLines.Clear();
            foreach (SceneAwareAffordance affordance in affordanceById.Values) affordance.ResetState();

            SemanticNodeContract start = manifest.nodes[0];
            spawn = start.position;
            yawDegrees = InitialYawForScenario(scenarioIndex);
            visitedNodes.Add(start.id);
            skillState = SkillState.Navigate;
            PlanRoute(start.id, CurrentObjective().nodeId);
            RecordLedger(0f, "Plan", CurrentObjective().id, "select_objective",
                "shortest_path(" + start.id + " -> " + CurrentObjective().nodeId + ")",
                true, "planned", RouteSummary());
        }

        public void GuideRootMotion(
            float simulatedTime,
            float dt,
            Vector3 bodyPosition,
            float bodyYawDegrees,
            float learnedRight,
            float learnedForward,
            float learnedYawRate,
            out float executedRight,
            out float executedForward,
            out float executedYawRate)
        {
            lastSimulatedTime = simulatedTime;
            foreach (SceneAwareAffordance affordance in affordanceById.Values)
                affordance.AdvanceSimulation(dt);
            UpdateSkillState(simulatedTime, bodyPosition);

            if (activeRollerCoaster != null && activeRollerCoaster.Boarded)
            {
                GuideBoardedVehicle(
                    simulatedTime,
                    dt,
                    bodyPosition,
                    bodyYawDegrees,
                    out executedRight,
                    out executedForward,
                    out executedYawRate);
                return;
            }

            Vector3 desiredWorldVelocity = Vector3.zero;
            float desiredRootYawRate = 0f;
            if (skillState != SkillState.Complete && manifest != null)
            {
                Vector3 target = TargetPosition();
                Vector3 delta = target - bodyPosition;
                delta.y = 0f;
                float distance = delta.magnitude;
                if (distance >= 0.001f)
                {
                    float desiredYaw = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
                    float yawError = Mathf.DeltaAngle(bodyYawDegrees, desiredYaw);
                    float desiredUnityYawDegreesPerSecond =
                        Mathf.Clamp(yawError * 2.2f, -62f, 62f);
                    // NormalHmdAgentController integrates Unity yaw as -rootYawRate.
                    desiredRootYawRate = -desiredUnityYawDegreesPerSecond * Mathf.Deg2Rad;

                    if (skillState == SkillState.Navigate)
                    {
                        float learnedSpeed = Mathf.Sqrt(
                            learnedRight * learnedRight + learnedForward * learnedForward);
                        float speed = Mathf.Clamp(learnedSpeed, 0.58f, 1.05f);
                        if (distance < 1.25f)
                            speed *= Mathf.Lerp(0.34f, 1f, distance / 1.25f);
                        float facing = Mathf.Lerp(
                            0.20f, 1f, Mathf.Clamp01(1f - Mathf.Abs(yawError) / 95f));
                        desiredWorldVelocity = delta.normalized * speed * facing;
                    }
                }
            }

            // Comfort filter: keep velocity and turn rate continuous at corners
            // and skill boundaries. It is explicitly rule-based; raw learned and
            // executed commands remain separate in scene_telemetry.csv.
            const float maximumPlanarAcceleration = 1.8f;
            const float maximumYawAcceleration = 3.5f;
            smoothedWorldVelocity = Vector3.MoveTowards(
                smoothedWorldVelocity,
                desiredWorldVelocity,
                maximumPlanarAcceleration * dt);
            smoothedYawRate = Mathf.MoveTowards(
                smoothedYawRate,
                desiredRootYawRate,
                maximumYawAcceleration * dt);
            if (smoothedWorldVelocity.sqrMagnitude < 0.000001f)
                smoothedWorldVelocity = Vector3.zero;
            if (Mathf.Abs(smoothedYawRate) < 0.0001f) smoothedYawRate = 0f;

            Vector3 localVelocity = Quaternion.Inverse(
                Quaternion.Euler(0f, bodyYawDegrees, 0f)) * smoothedWorldVelocity;
            executedRight = localVelocity.x;
            executedForward = localVelocity.z;
            executedYawRate = smoothedYawRate;
        }

        private void GuideBoardedVehicle(
            float simulatedTime,
            float dt,
            Vector3 bodyPosition,
            float bodyYawDegrees,
            out float executedRight,
            out float executedForward,
            out float executedYawRate)
        {
            activeRollerCoaster.Evaluate(simulatedTime, dt);
            Vector3 positionError = activeRollerCoaster.CurrentWorldPosition - bodyPosition;
            positionError.y = 0f;
            Vector3 desiredWorldVelocity = activeRollerCoaster.CurrentWorldVelocity;
            desiredWorldVelocity.y = 0f;
            desiredWorldVelocity += positionError * 3.8f;
            desiredWorldVelocity = Vector3.ClampMagnitude(desiredWorldVelocity, 3.2f);
            smoothedWorldVelocity = Vector3.MoveTowards(
                smoothedWorldVelocity,
                desiredWorldVelocity,
                4.5f * dt);

            Vector3 vehicleForward = activeRollerCoaster.CurrentWorldRotation * Vector3.forward;
            vehicleForward.y = 0f;
            float desiredYaw = vehicleForward.sqrMagnitude < 0.000001f
                ? bodyYawDegrees
                : Mathf.Atan2(vehicleForward.x, vehicleForward.z) * Mathf.Rad2Deg;
            float yawError = Mathf.DeltaAngle(bodyYawDegrees, desiredYaw);
            float targetRootYawRate = -Mathf.Clamp(yawError * 2.8f, -90f, 90f) * Mathf.Deg2Rad;
            smoothedYawRate = Mathf.MoveTowards(smoothedYawRate, targetRootYawRate, 5f * dt);

            Vector3 localVelocity = Quaternion.Inverse(
                Quaternion.Euler(0f, bodyYawDegrees, 0f)) * smoothedWorldVelocity;
            executedRight = localVelocity.x;
            executedForward = localVelocity.z;
            executedYawRate = smoothedYawRate;
        }

        public bool TryGetPlatformPose(
            float simulatedTime,
            out float verticalOffset,
            out float pitchDegrees,
            out float rollDegrees)
        {
            if (activeRollerCoaster != null && activeRollerCoaster.Boarded)
            {
                verticalOffset = activeRollerCoaster.CurrentVerticalOffset;
                pitchDegrees = activeRollerCoaster.CurrentPitchDegrees;
                rollDegrees = activeRollerCoaster.CurrentRollDegrees;
                return true;
            }
            verticalOffset = 0f;
            pitchDegrees = 0f;
            rollDegrees = 0f;
            return false;
        }

        public void BeginEpisode(int seed, string normalHmdLogDirectory)
        {
            episodeSeed = seed;
            string baseDirectory = string.IsNullOrEmpty(requestedLogRoot)
                ? normalHmdLogDirectory
                : Path.GetFullPath(requestedLogRoot);
            if (string.IsNullOrEmpty(baseDirectory))
                baseDirectory = Path.Combine(Application.persistentDataPath, "NormalHmdAgentLogs");

            string suffix = string.Format(
                CultureInfo.InvariantCulture,
                "scene_{0}_{1}_seed_{2}_{3}",
                scenarioIndex,
                SafeFileName(manifest.scenarioId),
                seed,
                DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture));
            logDirectory = Path.Combine(baseDirectory, suffix);
            Directory.CreateDirectory(logDirectory);
            manifestPath = Path.Combine(logDirectory, "scene_manifest.json");
            actionLedgerPath = Path.Combine(logDirectory, "action_ledger.jsonl");
            telemetryPath = Path.Combine(logDirectory, "scene_telemetry.csv");

            File.WriteAllText(manifestPath, JsonUtility.ToJson(manifest, true), new UTF8Encoding(false));
            ledgerWriter = new StreamWriter(actionLedgerPath, false, new UTF8Encoding(false));
            telemetryWriter = new StreamWriter(telemetryPath, false, new UTF8Encoding(false));
            telemetryWriter.WriteLine(
                "step,sim_time,scenario_index,scenario_id,objective,skill," +
                "body_x,body_y,body_z,body_yaw," +
                "camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw," +
                "head_x,head_y,head_z,head_qx,head_qy,head_qz,head_qw," +
                "learned_right,learned_forward,learned_yaw_rate," +
                "executed_right,executed_forward,executed_yaw_rate," +
                "camera_speed,camera_angular_speed,camera_acceleration,dynamic_visual_intensity," +
                "semantic_object_count,uncalibrated_exposure_proxy," +
                "vehicle_state,vehicle_x,vehicle_y,vehicle_z,vehicle_speed,vehicle_angular_speed");
            foreach (string line in pendingLedgerLines) ledgerWriter.WriteLine(line);
            pendingLedgerLines.Clear();
            ledgerWriter.Flush();
            episodeBegun = true;
        }

        public void ObserveSimulationStep(SceneAwareStepSample sample)
        {
            lastSimulatedTime = sample.simulatedTime;
            float dt = sample.deltaTime > 0f ? sample.deltaTime : 0.027f;
            float cameraSpeed = 0f;
            float cameraAngularSpeed = 0f;
            float cameraAcceleration = 0f;
            if (hasPreviousSample)
            {
                cameraSpeed = Vector3.Distance(sample.cameraPosition, previousSample.cameraPosition) / dt;
                cameraAngularSpeed = Quaternion.Angle(sample.cameraRotation, previousSample.cameraRotation) / dt;
                cameraAcceleration = Mathf.Abs(cameraSpeed - previousCameraSpeed) / dt;
            }
            float dynamicIntensity = activeMotionRig == null ? 0f : activeMotionRig.NormalizedIntensity;
            if (activeRollerCoaster != null)
                dynamicIntensity = Mathf.Max(dynamicIntensity, activeRollerCoaster.NormalizedIntensity);
            if (activeRollerCoaster != null && activeRollerCoaster.Running)
            {
                vehicleMotionRows++;
                vehicleMaxSpeed = Mathf.Max(vehicleMaxSpeed, activeRollerCoaster.CurrentSpeed);
                vehicleMinY = Mathf.Min(vehicleMinY, activeRollerCoaster.CurrentWorldPosition.y);
                vehicleMaxY = Mathf.Max(vehicleMaxY, activeRollerCoaster.CurrentWorldPosition.y);
                vehicleCameraMinY = Mathf.Min(vehicleCameraMinY, sample.cameraPosition.y);
                vehicleCameraMaxY = Mathf.Max(vehicleCameraMaxY, sample.cameraPosition.y);
            }
            liveExposure = 100f * (
                0.20f * Mathf.Clamp01(cameraSpeed / 2.0f) +
                0.42f * Mathf.Clamp01(cameraAngularSpeed / 120f) +
                0.23f * Mathf.Clamp01(cameraAcceleration / 6f) +
                0.15f * dynamicIntensity);
            exposureSum += liveExposure;
            maxExposure = Mathf.Max(maxExposure, liveExposure);

            if (telemetryWriter != null)
            {
                StringBuilder line = new StringBuilder(512);
                line.Append(sample.stepIndex.ToString(CultureInfo.InvariantCulture));
                Csv(line, sample.simulatedTime);
                line.Append(',').Append(scenarioIndex.ToString(CultureInfo.InvariantCulture));
                line.Append(',').Append(CsvText(manifest.scenarioId));
                line.Append(',').Append(CsvText(CurrentObjectiveId()));
                line.Append(',').Append(skillState.ToString());
                Csv(line, sample.bodyPosition.x); Csv(line, sample.bodyPosition.y); Csv(line, sample.bodyPosition.z);
                Csv(line, sample.bodyYawDegrees);
                Csv(line, sample.cameraPosition.x); Csv(line, sample.cameraPosition.y); Csv(line, sample.cameraPosition.z);
                Csv(line, sample.cameraRotation.x); Csv(line, sample.cameraRotation.y);
                Csv(line, sample.cameraRotation.z); Csv(line, sample.cameraRotation.w);
                Csv(line, sample.headLocalPosition.x); Csv(line, sample.headLocalPosition.y); Csv(line, sample.headLocalPosition.z);
                Csv(line, sample.headLocalRotation.x); Csv(line, sample.headLocalRotation.y);
                Csv(line, sample.headLocalRotation.z); Csv(line, sample.headLocalRotation.w);
                Csv(line, sample.learnedRight); Csv(line, sample.learnedForward); Csv(line, sample.learnedYawRate);
                Csv(line, sample.executedRight); Csv(line, sample.executedForward); Csv(line, sample.executedYawRate);
                Csv(line, cameraSpeed); Csv(line, cameraAngularSpeed); Csv(line, cameraAcceleration);
                Csv(line, dynamicIntensity);
                line.Append(',').Append((manifest.nodes.Length + manifest.affordances.Length).ToString(CultureInfo.InvariantCulture));
                Csv(line, liveExposure);
                if (activeRollerCoaster == null)
                {
                    line.Append(",none");
                    Csv(line, 0f); Csv(line, 0f); Csv(line, 0f); Csv(line, 0f); Csv(line, 0f);
                }
                else
                {
                    line.Append(',').Append(activeRollerCoaster.StateLabel);
                    Csv(line, activeRollerCoaster.CurrentWorldPosition.x);
                    Csv(line, activeRollerCoaster.CurrentWorldPosition.y);
                    Csv(line, activeRollerCoaster.CurrentWorldPosition.z);
                    Csv(line, activeRollerCoaster.CurrentSpeed);
                    Csv(line, activeRollerCoaster.CurrentAngularSpeed);
                }
                telemetryWriter.WriteLine(line.ToString());
                telemetryRows++;
                if (telemetryRows % 180 == 0) telemetryWriter.Flush();
            }
            previousSample = sample;
            previousCameraSpeed = cameraSpeed;
            hasPreviousSample = true;
        }

        public void CompleteEpisode(string reason)
        {
            if (episodeCompleted) return;
            episodeCompleted = true;
            CloseWriters();

            int expectedInteractions = manifest == null || manifest.affordances == null
                ? 0 : manifest.affordances.Length;
            int expectedObjectives = manifest == null || manifest.objectives == null
                ? 0 : manifest.objectives.Length;
            List<string> failures = new List<string>();
            if (lastSimulatedTime < 59f) failures.Add("generated duration below 59 seconds");
            if (telemetryRows < 1000) failures.Add("fewer than 1000 telemetry rows");
            if (interactionsSucceeded < expectedInteractions) failures.Add("not every declared interaction succeeded");
            if (objectivesCompleted < expectedObjectives) failures.Add("not every semantic objective completed");
            if (string.IsNullOrEmpty(telemetryPath) || !File.Exists(telemetryPath)) failures.Add("telemetry file missing");
            if (activeRollerCoaster != null)
            {
                if (vehicleMotionRows < 100) failures.Add("roller coaster produced too few moving vehicle samples");
                if (vehicleMaxY - vehicleMinY < 0.5f) failures.Add("roller coaster vertical track range was not observed");
                if (vehicleCameraMaxY - vehicleCameraMinY < 0.5f) failures.Add("camera did not follow roller coaster vertical motion");
            }

            SceneAwareAcceptanceReport report = new SceneAwareAcceptanceReport
            {
                status = failures.Count == 0 ? "passed" : "failed",
                scenarioIndex = scenarioIndex,
                scenarioName = manifest == null ? "unknown" : manifest.scenarioName,
                seed = episodeSeed,
                generatedSeconds = lastSimulatedTime,
                telemetryRowCount = telemetryRows,
                telemetryPath = telemetryPath,
                interactionsExpected = expectedInteractions,
                interactionsAttempted = interactionsAttempted,
                interactionsSucceeded = interactionsSucceeded,
                objectivesExpected = expectedObjectives,
                objectivesCompleted = objectivesCompleted,
                coverage = Coverage,
                meanExposureProxy = telemetryRows == 0 ? 0f : exposureSum / telemetryRows,
                maxExposureProxy = maxExposure,
                vehicleMotionRows = vehicleMotionRows,
                vehicleMaxSpeed = vehicleMaxSpeed,
                vehicleVerticalRange = vehicleMotionRows == 0 ? 0f : vehicleMaxY - vehicleMinY,
                cameraVerticalRangeDuringVehicle = vehicleMotionRows == 0 ? 0f : vehicleCameraMaxY - vehicleCameraMinY,
                exposureMeaning = "Uncalibrated engineering exposure index; not a VR.net probability or medical diagnosis.",
                plannerMeaning = "Semantic graph, skills and BodyRoot guidance are deterministic rules; HeadLocal motion is generated by the trained Normal HMD model.",
                completionReason = reason,
                manifestPath = manifestPath,
                actionLedgerPath = actionLedgerPath,
                failures = failures.ToArray()
            };

            string json = JsonUtility.ToJson(report, true);
            if (!string.IsNullOrEmpty(logDirectory))
                File.WriteAllText(Path.Combine(logDirectory, "scene_acceptance.json"), json, new UTF8Encoding(false));
            if (!string.IsNullOrEmpty(testOutputPath))
            {
                string full = Path.GetFullPath(testOutputPath);
                string parent = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(parent)) Directory.CreateDirectory(parent);
                File.WriteAllText(full, json, new UTF8Encoding(false));
            }

            if (testMode)
            {
                Debug.Log("SCENE_AWARE_ACCEPTANCE " + report.status + " " + json);
                Application.Quit(report.status == "passed" ? 0 : 2);
            }
        }

        private void UpdateSkillState(float simulatedTime, Vector3 bodyPosition)
        {
            if (skillState == SkillState.Complete || manifest == null || manifest.objectives.Length == 0) return;
            SemanticObjectiveContract objective = CurrentObjective();
            if (skillState == SkillState.Navigate)
            {
                if (objective.requiredSkill == "RideObserve" &&
                    activeRollerCoaster != null &&
                    activeRollerCoaster.Boarded)
                {
                    EnterSkill(SkillState.Observe, simulatedTime, "vehicle_observation_started");
                    return;
                }
                int safety = 0;
                while (routeCursor < route.Count && safety++ < 8)
                {
                    SemanticNodeContract node = nodeById[route[routeCursor]];
                    Vector3 flat = node.position - bodyPosition;
                    flat.y = 0f;
                    if (flat.magnitude > node.arrivalRadius) break;
                    visitedNodes.Add(node.id);
                    routeCursor++;
                }
                if (routeCursor >= route.Count)
                {
                    EnterSkill(SkillState.LookAt, simulatedTime, "face_target");
                }
            }
            else if (skillState == SkillState.LookAt)
            {
                if (simulatedTime - stateEnteredAt >= 0.8f)
                {
                    EnterSkill(string.IsNullOrEmpty(objective.targetAffordanceId)
                        ? SkillState.Observe : SkillState.Activate, simulatedTime, "target_acquired");
                }
            }
            else if (skillState == SkillState.Activate)
            {
                if (!activationAttemptedForObjective)
                {
                    activationAttemptedForObjective = true;
                    interactionsAttempted++;
                    SceneAwareAffordance affordance;
                    bool found = affordanceById.TryGetValue(objective.targetAffordanceId, out affordance);
                    string evidence = "affordance_not_found";
                    interactionCommandAccepted = found && affordance.Activate(out evidence);
                    RecordLedger(simulatedTime, "Activate", objective.targetAffordanceId,
                        "execute declared interaction verb", found ? affordance.Contract.verb : "missing",
                        interactionCommandAccepted,
                        interactionCommandAccepted ? "accepted" : "failed", evidence);
                }
                SceneAwareAffordance activeAffordance = null;
                bool success = interactionCommandAccepted &&
                    affordanceById.TryGetValue(
                        objective.targetAffordanceId, out activeAffordance) &&
                    activeAffordance.SuccessSatisfied;
                if (success && !interactionSuccessRecorded)
                {
                    interactionSuccessRecorded = true;
                    interactionsSucceeded++;
                    RecordLedger(simulatedTime, "VerifyInteraction",
                        objective.targetAffordanceId, "verify declared success state",
                        "query_engine_state", true, "success", activeAffordance.Evidence);
                    EnterSkill(SkillState.Observe, simulatedTime, "interaction_success_confirmed");
                }
                else if (!interactionCommandAccepted || simulatedTime - stateEnteredAt >= 5f)
                {
                    RecordLedger(simulatedTime, "VerifyInteraction",
                        objective.targetAffordanceId, "verify declared success state",
                        "query_engine_state", false, "timeout",
                        interactionCommandAccepted ? "success_state_not_reached" : "command_rejected");
                    EnterSkill(SkillState.Observe, simulatedTime, "interaction_failed_or_timed_out");
                }
            }
            else if (skillState == SkillState.Observe)
            {
                if (simulatedTime - stateEnteredAt >= objective.observeSeconds)
                {
                    objectivesCompleted++;
                    RecordLedger(simulatedTime, "Verify", objective.id, "verify objective completion",
                        "query_engine_state", true, "objective_complete", ObjectiveEvidence(objective));
                    objectiveIndex++;
                    if (objectiveIndex >= manifest.objectives.Length)
                    {
                        EnterSkill(SkillState.Complete, simulatedTime, "all_objectives_complete");
                    }
                    else
                    {
                        activationAttemptedForObjective = false;
                        interactionSuccessRecorded = false;
                        interactionCommandAccepted = false;
                        string nearest = FindNearestNode(bodyPosition);
                        PlanRoute(nearest, CurrentObjective().nodeId);
                        EnterSkill(SkillState.Navigate, simulatedTime, "next_objective");
                        RecordLedger(simulatedTime, "Plan", CurrentObjective().id, "select_objective",
                            "shortest_path(" + nearest + " -> " + CurrentObjective().nodeId + ")",
                            true, "planned", RouteSummary());
                    }
                }
            }
        }

        private void EnterSkill(SkillState next, float simulatedTime, string evidence)
        {
            skillState = next;
            stateEnteredAt = simulatedTime;
            RecordLedger(simulatedTime, next.ToString(), CurrentObjectiveId(), "advance_skill_state",
                next.ToString(), true, "entered", evidence);
        }

        private Vector3 TargetPosition()
        {
            if (skillState == SkillState.Navigate && routeCursor < route.Count)
                return nodeById[route[routeCursor]].position;
            SemanticObjectiveContract objective = CurrentObjective();
            SceneAwareAffordance affordance;
            if (!string.IsNullOrEmpty(objective.targetAffordanceId) &&
                affordanceById.TryGetValue(objective.targetAffordanceId, out affordance))
                return affordance.transform.position;
            return nodeById[objective.nodeId].position;
        }

        private SemanticObjectiveContract CurrentObjective()
        {
            int index = manifest == null || manifest.objectives == null || manifest.objectives.Length == 0
                ? 0 : Mathf.Clamp(objectiveIndex, 0, manifest.objectives.Length - 1);
            return manifest.objectives[index];
        }

        private string CurrentObjectiveId()
        {
            if (manifest == null || manifest.objectives == null || manifest.objectives.Length == 0)
                return "none";
            if (objectiveIndex >= manifest.objectives.Length) return "all_complete";
            return manifest.objectives[objectiveIndex].id;
        }

        private void PlanRoute(string from, string to)
        {
            route.Clear();
            routeCursor = 0;
            Dictionary<string, float> distance = new Dictionary<string, float>(StringComparer.Ordinal);
            Dictionary<string, string> previous = new Dictionary<string, string>(StringComparer.Ordinal);
            HashSet<string> open = new HashSet<string>(StringComparer.Ordinal);
            foreach (SemanticNodeContract node in manifest.nodes)
            {
                distance[node.id] = float.PositiveInfinity;
                open.Add(node.id);
            }
            if (!distance.ContainsKey(from) || !distance.ContainsKey(to)) return;
            distance[from] = 0f;
            while (open.Count > 0)
            {
                string current = null;
                float best = float.PositiveInfinity;
                foreach (string candidate in open)
                {
                    if (distance[candidate] < best)
                    {
                        best = distance[candidate];
                        current = candidate;
                    }
                }
                if (current == null || current == to) break;
                open.Remove(current);
                foreach (SemanticEdgeContract edge in manifest.edges)
                {
                    string neighbor = null;
                    if (edge.from == current) neighbor = edge.to;
                    else if (edge.bidirectional && edge.to == current) neighbor = edge.from;
                    if (neighbor == null || !open.Contains(neighbor)) continue;
                    float candidateDistance = distance[current] + Mathf.Max(0.001f, edge.traversalCost);
                    if (candidateDistance < distance[neighbor])
                    {
                        distance[neighbor] = candidateDistance;
                        previous[neighbor] = current;
                    }
                }
            }
            List<string> reverse = new List<string>();
            string cursor = to;
            reverse.Add(cursor);
            while (cursor != from && previous.ContainsKey(cursor))
            {
                cursor = previous[cursor];
                reverse.Add(cursor);
            }
            reverse.Reverse();
            route.AddRange(reverse);
            if (route.Count > 0 && route[0] == from) routeCursor = 1;
        }

        private string FindNearestNode(Vector3 position)
        {
            string bestId = manifest.nodes[0].id;
            float best = float.PositiveInfinity;
            foreach (SemanticNodeContract node in manifest.nodes)
            {
                Vector3 delta = node.position - position;
                delta.y = 0f;
                float square = delta.sqrMagnitude;
                if (square < best) { best = square; bestId = node.id; }
            }
            return bestId;
        }

        private string RouteSummary()
        {
            return route.Count == 0 ? "no_route" : string.Join(" -> ", route.ToArray());
        }

        private string ObjectiveEvidence(SemanticObjectiveContract objective)
        {
            if (objective.requiredSkill == "RideObserve" && activeRollerCoaster != null)
                return activeRollerCoaster.Evidence;
            if (string.IsNullOrEmpty(objective.targetAffordanceId)) return "dwell_timer_satisfied";
            SceneAwareAffordance affordance;
            return affordanceById.TryGetValue(objective.targetAffordanceId, out affordance)
                ? affordance.Evidence : "affordance_missing";
        }

        private void RecordLedger(
            float simulatedTime, string skill, string target, string intent, string command,
            bool confirmed, string result, string evidence)
        {
            ActionLedgerEntry entry = new ActionLedgerEntry
            {
                utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                simulatedTime = simulatedTime,
                scenarioId = manifest == null ? "unknown" : manifest.scenarioId,
                objectiveId = CurrentObjectiveId(),
                skill = skill,
                target = target,
                intent = intent,
                command = command,
                engineConfirmed = confirmed,
                result = result,
                evidence = evidence
            };
            string json = JsonUtility.ToJson(entry, false);
            if (ledgerWriter != null)
            {
                ledgerWriter.WriteLine(json);
                ledgerWriter.Flush();
            }
            else pendingLedgerLines.Add(json);
        }

        private void SwitchScenario(int requested)
        {
            if (requested == scenarioIndex && !episodeCompleted) return;
            CompleteEpisode("scenario_switch");
            scenarioIndex = Mathf.Clamp(requested, 0, 3);
            BuildScenario(scenarioIndex);
            normalHmdController.RestartSceneAwareEpisode(RandomSeed());
        }

        private void RestartCurrentScenario()
        {
            CompleteEpisode("manual_restart");
            normalHmdController.RestartSceneAwareEpisode(RandomSeed());
        }

        private void BuildScenario(int index)
        {
            if (scenarioRoot != null) Destroy(scenarioRoot);
            scenarioRoot = new GameObject("SceneAware Scenario Root");
            nodeById.Clear();
            affordanceById.Clear();
            activeMotionRig = null;
            activeRollerCoaster = null;
            if (index == 0) BuildCalmDoorScenario();
            else if (index == 1) BuildTurnCorridorScenario();
            else if (index == 2) BuildMotionLabScenario();
            else BuildRollerCoasterScenario();
            foreach (SemanticNodeContract node in manifest.nodes) nodeById[node.id] = node;
        }

        private void BuildCalmDoorScenario()
        {
            Material ground = Mat("Calm floor", new Color(0.08f, 0.18f, 0.18f));
            Material path = Mat("Stable path", new Color(0.22f, 0.36f, 0.38f));
            Material structure = Mat("Stable structure", new Color(0.08f, 0.12f, 0.16f));
            Material teal = Mat("Teal guide", new Color(0.12f, 0.88f, 0.72f), true);
            Material amber = Mat("Amber control", new Color(1f, 0.57f, 0.16f), true);
            Primitive("Calm Ground", PrimitiveType.Plane, Vector3.zero, new Vector3(2.2f, 1f, 2.2f), ground);
            Primitive("Stable Walkway", PrimitiveType.Cube, new Vector3(0f, 0.03f, 0f), new Vector3(3f, 0.06f, 16f), path);
            for (int i = -3; i <= 3; i++)
            {
                Primitive("Horizon Post L " + i, PrimitiveType.Cube, new Vector3(-4f, 1.2f, i * 2.5f), new Vector3(0.2f, 2.4f, 0.2f), structure);
                Primitive("Horizon Post R " + i, PrimitiveType.Cube, new Vector3(4f, 1.2f, i * 2.5f), new Vector3(0.2f, 2.4f, 0.2f), structure);
            }
            GameObject door = Primitive("Semantic Door", PrimitiveType.Cube, new Vector3(0f, 1.45f, 5.6f), new Vector3(2.5f, 2.9f, 0.22f), structure);
            GameObject button = Primitive("Door Button", PrimitiveType.Cube, new Vector3(0f, 1.0f, 1.2f), new Vector3(0.45f, 0.45f, 0.18f), amber);
            Primitive("Door Header", PrimitiveType.Cube, new Vector3(0f, 3.2f, 5.6f), new Vector3(4.8f, 0.25f, 0.3f), teal);

            SemanticAffordanceContract affordance = Affordance(
                "calm.open_door", "Door control", "button", "press", "button",
                button.transform.position, "door=open", "door_open_fraction");
            SceneAwareAffordance runtime = button.AddComponent<SceneAwareAffordance>();
            runtime.Configure(affordance, SceneAwareAffordance.EffectKind.OpenDoor, door.transform, button.GetComponent<Renderer>());
            affordanceById[affordance.id] = runtime;

            manifest = Manifest(
                "calm-door", "1 · Calm Door Baseline",
                "Stable horizon references, one explicit button affordance and a slowly opening door.",
                new[] { "baseline", "stable-horizon", "door", "button" },
                new[]
                {
                    Node("start", "Entry", "spawn", new Vector3(0f, 0f, -5.8f)),
                    Node("button", "Door button approach", "interaction", new Vector3(0f, 0f, 0.15f)),
                    Node("door", "Open doorway", "transition", new Vector3(0f, 0f, 4.4f))
                },
                new[] { Edge("start", "button", 5.9f), Edge("button", "door", 4.3f) },
                new[]
                {
                    Objective("open_door", "Open the semantic door", "button", affordance.id, "Activate", 1.2f),
                    Objective("inspect_doorway", "Inspect the opened doorway", "door", "", "Observe", 1.5f)
                },
                new[] { affordance });
        }

        private void BuildTurnCorridorScenario()
        {
            Material floor = Mat("Corridor floor", new Color(0.16f, 0.20f, 0.24f));
            Material wall = Mat("Corridor wall", new Color(0.075f, 0.10f, 0.15f));
            Material blue = Mat("Blue route", new Color(0.18f, 0.58f, 1f), true);
            Material amber = Mat("Turn switch", new Color(1f, 0.62f, 0.15f), true);
            Primitive("Turn Lab Ground", PrimitiveType.Plane, Vector3.zero, new Vector3(2.3f, 1f, 2.3f), floor);
            // Ground strips make the intended 90-degree route visible without relying on NavMesh.
            Segment("Route A", new Vector3(-5f, 0.04f, -2.5f), new Vector3(2.5f, 0.08f, 8f), blue);
            Segment("Route B", new Vector3(-1.25f, 0.045f, 1.5f), new Vector3(10f, 0.08f, 2.5f), blue);
            Segment("Route C", new Vector3(3.75f, 0.05f, 5f), new Vector3(2.5f, 0.08f, 9.5f), blue);
            Primitive("Wall A", PrimitiveType.Cube, new Vector3(-7f, 1.35f, -2.5f), new Vector3(0.25f, 2.7f, 10f), wall);
            Primitive("Wall B", PrimitiveType.Cube, new Vector3(-1.5f, 1.35f, 3.2f), new Vector3(9f, 2.7f, 0.25f), wall);
            Primitive("Wall C", PrimitiveType.Cube, new Vector3(2.0f, 1.35f, 7.8f), new Vector3(0.25f, 2.7f, 9f), wall);
            GameObject gate = Primitive("Exit Gate", PrimitiveType.Cube, new Vector3(3.8f, 1.4f, 8.2f), new Vector3(2.4f, 2.8f, 0.22f), wall);
            GameObject turnSwitch = Primitive("Turn Corridor Switch", PrimitiveType.Cube, new Vector3(3.8f, 1.0f, 6.4f), new Vector3(0.48f, 0.48f, 0.18f), amber);

            SemanticAffordanceContract affordance = Affordance(
                "turn.open_gate", "Exit switch", "button", "press", "exit_switch",
                turnSwitch.transform.position, "exit_gate=open", "door_open_fraction");
            SceneAwareAffordance runtime = turnSwitch.AddComponent<SceneAwareAffordance>();
            runtime.Configure(affordance, SceneAwareAffordance.EffectKind.OpenDoor, gate.transform, turnSwitch.GetComponent<Renderer>());
            affordanceById[affordance.id] = runtime;

            manifest = Manifest(
                "turn-corridor", "2 · Multi-turn Corridor",
                "A graph-constrained route with three deliberate 90-degree turns and a verified exit switch.",
                new[] { "locomotion", "turning", "corridor", "interaction" },
                new[]
                {
                    Node("start", "South entry", "spawn", new Vector3(-5f, 0f, -6f)),
                    Node("turn_1", "First left/right turn", "corner", new Vector3(-5f, 0f, 1.4f)),
                    Node("turn_2", "Cross corridor turn", "corner", new Vector3(3.7f, 0f, 1.4f)),
                    Node("exit_switch", "Exit switch approach", "interaction", new Vector3(3.7f, 0f, 5.25f)),
                    Node("exit", "Exit observation", "transition", new Vector3(3.7f, 0f, 7.2f))
                },
                new[]
                {
                    Edge("start", "turn_1", 7.4f), Edge("turn_1", "turn_2", 8.7f),
                    Edge("turn_2", "exit_switch", 3.9f), Edge("exit_switch", "exit", 2f)
                },
                new[]
                {
                    Objective("open_exit", "Traverse turns and open exit", "exit_switch", affordance.id, "Activate", 1.1f),
                    Objective("inspect_exit", "Inspect the opened exit", "exit", "", "Observe", 1.3f)
                },
                new[] { affordance });
        }

        private void BuildMotionLabScenario()
        {
            Material floor = Mat("Lab floor", new Color(0.10f, 0.13f, 0.18f));
            Material path = Mat("Lab path", new Color(0.22f, 0.31f, 0.39f));
            Material cyan = Mat("Lab cyan", new Color(0.10f, 0.82f, 0.92f), true);
            Material magenta = Mat("Lab magenta", new Color(0.94f, 0.24f, 0.68f), true);
            Material dark = Mat("Lab frame", new Color(0.055f, 0.075f, 0.12f));
            Primitive("Motion Lab Ground", PrimitiveType.Plane, Vector3.zero, new Vector3(2.1f, 1f, 2.1f), floor);
            Primitive("Lab Walkway", PrimitiveType.Cube, new Vector3(0f, 0.035f, 0f), new Vector3(3f, 0.07f, 16f), path);
            Primitive("Stable Horizon Bar", PrimitiveType.Cube, new Vector3(0f, 1.65f, 7.2f), new Vector3(12f, 0.12f, 0.12f), cyan);

            GameObject rig = new GameObject("Dynamic Visual Rig");
            rig.transform.SetParent(scenarioRoot.transform, false);
            rig.transform.position = new Vector3(0f, 2.1f, 5.2f);
            for (int i = 0; i < 12; i++)
            {
                float angle = i * Mathf.PI * 2f / 12f;
                GameObject stripe = Primitive("Moving Stripe " + i, PrimitiveType.Cube,
                    rig.transform.position + new Vector3(Mathf.Sin(angle) * 2.3f, 0f, Mathf.Cos(angle) * 2.3f),
                    new Vector3(0.18f, 2.8f, 0.42f), i % 2 == 0 ? cyan : magenta);
                stripe.transform.SetParent(rig.transform, true);
                stripe.transform.LookAt(rig.transform.position + Vector3.up * 1.0f);
            }
            activeMotionRig = rig.AddComponent<SceneAwareDynamicMotionRig>();
            activeMotionRig.SetMotionEnabled(false);
            GameObject switchObject = Primitive("Motion Start Switch", PrimitiveType.Cube,
                new Vector3(0f, 1f, 0.8f), new Vector3(0.52f, 0.52f, 0.18f), magenta);
            Primitive("Observation Frame L", PrimitiveType.Cube, new Vector3(-2.8f, 1.6f, 3.9f), new Vector3(0.2f, 3.2f, 0.2f), dark);
            Primitive("Observation Frame R", PrimitiveType.Cube, new Vector3(2.8f, 1.6f, 3.9f), new Vector3(0.2f, 3.2f, 0.2f), dark);

            SemanticAffordanceContract affordance = Affordance(
                "motion.start_rig", "Visual motion switch", "button", "press", "motion_switch",
                switchObject.transform.position, "dynamic_rig=running", "dynamic_rig_running");
            SceneAwareAffordance runtime = switchObject.AddComponent<SceneAwareAffordance>();
            runtime.Configure(affordance, SceneAwareAffordance.EffectKind.StartDynamicMotion, rig.transform, switchObject.GetComponent<Renderer>());
            affordanceById[affordance.id] = runtime;

            manifest = Manifest(
                "motion-lab", "3 · Dynamic Visual Motion Lab",
                "A switch starts a rotating and vertically oscillating visual rig while the agent observes from a fixed point.",
                new[] { "visual-motion", "vertical-motion", "rotation", "scene-change" },
                new[]
                {
                    Node("start", "Lab entry", "spawn", new Vector3(0f, 0f, -5.8f)),
                    Node("motion_switch", "Motion switch approach", "interaction", new Vector3(0f, 0f, -0.15f)),
                    Node("observation", "Controlled observation point", "observation", new Vector3(0f, 0f, 3.2f))
                },
                new[] { Edge("start", "motion_switch", 5.7f), Edge("motion_switch", "observation", 3.4f) },
                new[]
                {
                    Objective("start_motion", "Start the visual motion rig", "motion_switch", affordance.id, "Activate", 1.2f),
                    Objective("observe_motion", "Observe the changed scene", "observation", "", "Observe", 3.5f)
                },
                new[] { affordance });
        }

        private void BuildRollerCoasterScenario()
        {
            Material ground = Mat("Coaster ground", new Color(0.055f, 0.10f, 0.13f));
            Material station = Mat("Coaster station", new Color(0.18f, 0.24f, 0.29f));
            Material rail = Mat("Coaster rail", new Color(0.12f, 0.80f, 0.92f), true);
            Material support = Mat("Coaster support", new Color(0.10f, 0.14f, 0.18f));
            Material cartMaterial = Mat("Coaster cart", new Color(0.95f, 0.28f, 0.20f), true);
            Material boardMaterial = Mat("Board control", new Color(0.22f, 0.70f, 1f), true);
            Material startMaterial = Mat("Start control", new Color(1f, 0.62f, 0.12f), true);

            Primitive("Coaster Ground", PrimitiveType.Plane, Vector3.zero, new Vector3(2.2f, 1f, 2.2f), ground);
            Primitive("Station Walkway", PrimitiveType.Cube,
                new Vector3(-2.2f, 0.08f, -0.5f), new Vector3(5.6f, 0.16f, 2.6f), station);
            Primitive("Station Horizon", PrimitiveType.Cube,
                new Vector3(-2.2f, 2.6f, 0.65f), new Vector3(5.8f, 0.18f, 0.18f), rail);

            Vector3 trackOrigin = new Vector3(0f, 0.48f, -0.5f);
            GameObject vehicleRoot = new GameObject("Semantic Roller Coaster Vehicle");
            vehicleRoot.transform.SetParent(scenarioRoot.transform, false);
            activeRollerCoaster = vehicleRoot.AddComponent<SceneAwareRollerCoasterRig>();
            activeRollerCoaster.Configure(trackOrigin, 12f);
            CreateVehicleChild("Cart Chassis", PrimitiveType.Cube, vehicleRoot.transform,
                new Vector3(0f, 0.14f, 0f), new Vector3(1.45f, 0.24f, 1.05f), cartMaterial);
            CreateVehicleChild("Cart Seat", PrimitiveType.Cube, vehicleRoot.transform,
                new Vector3(0f, 0.39f, -0.34f), new Vector3(0.86f, 0.46f, 0.12f), support);
            CreateVehicleChild("Cart Safety Bar", PrimitiveType.Cube, vehicleRoot.transform,
                new Vector3(0f, 0.58f, 0.22f), new Vector3(0.94f, 0.08f, 0.08f), rail);

            const int trackSegments = 48;
            for (int index = 0; index < trackSegments; index++)
            {
                float t0 = (float)index / trackSegments;
                float t1 = (float)(index + 1) / trackSegments;
                Vector3 p0 = activeRollerCoaster.SamplePosition(t0);
                Vector3 p1 = activeRollerCoaster.SamplePosition(t1);
                Vector3 tangent = (p1 - p0).normalized;
                Vector3 right = Vector3.Cross(Vector3.up, tangent).normalized * 0.31f;
                CreateTrackBeam("Left Rail " + index, p0 - right, p1 - right, rail, 0.085f);
                CreateTrackBeam("Right Rail " + index, p0 + right, p1 + right, rail, 0.085f);
                if (index % 6 == 0)
                {
                    Vector3 point = (p0 + p1) * 0.5f;
                    Primitive("Track Support " + index, PrimitiveType.Cube,
                        new Vector3(point.x, point.y * 0.5f, point.z),
                        new Vector3(0.13f, Mathf.Max(0.2f, point.y), 0.13f), support);
                }
            }

            GameObject boardControl = Primitive("Board Vehicle Control", PrimitiveType.Cube,
                new Vector3(-0.82f, 0.92f, -1.18f), new Vector3(0.30f, 0.30f, 0.14f), boardMaterial);
            GameObject startControl = Primitive("Start Ride Control", PrimitiveType.Cube,
                new Vector3(0.82f, 0.92f, -1.18f), new Vector3(0.30f, 0.30f, 0.14f), startMaterial);

            SemanticAffordanceContract boardAffordance = Affordance(
                "coaster.board_vehicle", "Board and latch into cart", "vehicle-seat", "board",
                "boarding", boardControl.transform.position,
                "vehicle=boarded", "seat_latch");
            SceneAwareAffordance boardRuntime = boardControl.AddComponent<SceneAwareAffordance>();
            boardRuntime.Configure(
                boardAffordance,
                SceneAwareAffordance.EffectKind.BoardVehicle,
                vehicleRoot.transform,
                boardControl.GetComponent<Renderer>());
            affordanceById[boardAffordance.id] = boardRuntime;

            SemanticAffordanceContract startAffordance = Affordance(
                "coaster.start_vehicle", "Start coaster after safety check", "vehicle-control", "press",
                "boarding", startControl.transform.position,
                "vehicle=running", "safety_interlock");
            SceneAwareAffordance startRuntime = startControl.AddComponent<SceneAwareAffordance>();
            startRuntime.Configure(
                startAffordance,
                SceneAwareAffordance.EffectKind.StartVehicle,
                vehicleRoot.transform,
                startControl.GetComponent<Renderer>());
            affordanceById[startAffordance.id] = startRuntime;

            manifest = Manifest(
                "roller-coaster", "4 · Semantic Roller Coaster",
                "The agent approaches a station, boards a declared seat, starts the vehicle, follows a closed track, observes the ride and verifies return to station.",
                new[] { "vehicle", "boarding", "acceleration", "vertical-motion", "curved-track" },
                new[]
                {
                    Node("start", "Station entrance", "spawn", new Vector3(-5.2f, 0f, -0.5f)),
                    Node("boarding", "Vehicle boarding point", "vehicle-seat", new Vector3(0f, 0f, -0.5f)),
                    Node("ride_observation", "Moving vehicle observation", "vehicle-route", new Vector3(0f, 0f, -0.5f))
                },
                new[]
                {
                    Edge("start", "boarding", 5.2f),
                    Edge("boarding", "ride_observation", 0.1f)
                },
                new[]
                {
                    Objective("board_cart", "Board and latch into the vehicle", "boarding", boardAffordance.id, "Activate", 0.8f),
                    Objective("start_cart", "Start the coaster after boarding", "boarding", startAffordance.id, "Activate", 0.8f),
                    Objective("observe_ride", "Observe one complete vehicle circuit", "ride_observation", "", "RideObserve", 14.2f)
                },
                new[] { boardAffordance, startAffordance });
        }

        private void OnGUI()
        {
            if (!configured || manifest == null || testMode) return;
            EnsureStyles();
            float width = plannerHudExpanded
                ? Mathf.Min(465f, Screen.width * 0.42f)
                : Mathf.Clamp(Screen.width * 0.25f, 220f, 310f);
            float height = plannerHudExpanded
                ? Screen.height - 28f
                : Mathf.Min(activeRollerCoaster == null ? 286f : 314f, Screen.height - 28f);
            Rect panelRect = new Rect(Screen.width - width - 14f, 14f, width, height);
            GUI.DrawTexture(panelRect, panelTexture, ScaleMode.StretchToFill, true);
            GUILayout.BeginArea(panelRect);
            GUILayout.Space(15f);
            GUILayout.BeginHorizontal(); GUILayout.Space(17f);
            GUILayout.BeginVertical();
            if (plannerHudExpanded && GUILayout.Button("P  Collapse planner details", GUILayout.Height(25f)))
                plannerHudExpanded = false;
            GUILayout.Label("HEAD  ·  EyeNavGS HMD v4", learnedStyle);
            GUILayout.Label(
                normalHmdController == null
                    ? "starting"
                    : normalHmdController.SimulatedSeconds.ToString("00.0", CultureInfo.InvariantCulture) +
                      " / 60.0 s  ·  seed " + normalHmdController.EpisodeSeed,
                smallStyle);
            GUILayout.Space(5f);
            GUILayout.Label("BRAIN  ·  SCENE EXPLORER", titleStyle);
            GUILayout.Label(manifest.scenarioName, labelStyle);
            GUILayout.Label("Goal: " + CurrentObjectiveId(), labelStyle);
            GUILayout.Label("Skill: " + skillState + "  ·  " + RouteSummary(), smallStyle);
            GUILayout.Label("Coverage: " + Mathf.RoundToInt(Coverage * 100f) + "%   Interactions: " +
                interactionsSucceeded + "/" + manifest.affordances.Length, labelStyle);
            if (activeRollerCoaster != null)
            {
                GUILayout.Label(
                    "Vehicle: " + activeRollerCoaster.StateLabel +
                    "  ·  " + activeRollerCoaster.CurrentSpeed.ToString("F2", CultureInfo.InvariantCulture) + " m/s" +
                    "  ·  y " + activeRollerCoaster.CurrentWorldPosition.y.ToString("F2", CultureInfo.InvariantCulture),
                    labelStyle);
            }
            if (!plannerHudExpanded)
            {
                GUILayout.Label("P: details  ·  1–4: scene  ·  R: rerun", smallStyle);
                if (GUILayout.Button("P  Show planner details", GUILayout.Height(25f)))
                    plannerHudExpanded = true;
                GUILayout.EndVertical(); GUILayout.Space(17f);
                GUILayout.EndHorizontal();
                GUILayout.EndArea();
                return;
            }
            GUILayout.Space(7f);
            GUILayout.Label("Semantic scene graph: " + manifest.nodes.Length + " nodes · " +
                manifest.edges.Length + " edges · " + manifest.affordances.Length + " affordance", smallStyle);
            GUILayout.Space(8f);
            GUILayout.Label("LEARNED  ·  Normal HMD HeadLocal motion prior", learnedStyle);
            GUILayout.Label("RULE-BASED  ·  semantic planner, skills and BodyRoot route guidance", ruleStyle);
            if (activeRollerCoaster != null)
                GUILayout.Label("VEHICLE  ·  deterministic track pose is composed outside HeadLocal", ruleStyle);
            GUILayout.Space(8f);
            GUILayout.Label("Exposure proxy: " + liveExposure.ToString("F1", CultureInfo.InvariantCulture) + " / 100", warningStyle);
            GUILayout.Label("Engineering signal only — uncalibrated, not VR.net probability or medical risk.", smallStyle);
            GUILayout.Space(10f);
            GUILayout.Label("Skills: Navigate → LookAt → Activate → Observe → Verify", smallStyle);
            GUILayout.Label("Engine evidence is written to action_ledger.jsonl; per-step motion to scene_telemetry.csv.", smallStyle);
            GUILayout.Space(10f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("1  Calm", GUILayout.Height(30f))) SwitchScenario(0);
            if (GUILayout.Button("2  Turns", GUILayout.Height(30f))) SwitchScenario(1);
            if (GUILayout.Button("3  Motion", GUILayout.Height(30f))) SwitchScenario(2);
            if (GUILayout.Button("4  Coaster", GUILayout.Height(30f))) SwitchScenario(3);
            GUILayout.EndHorizontal();
            if (GUILayout.Button("R  New seeded run", GUILayout.Height(28f))) RestartCurrentScenario();
            GUILayout.EndVertical(); GUILayout.Space(17f);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void EnsureStyles()
        {
            if (titleStyle != null) return;
            panelTexture = Solid(new Color(0.025f, 0.045f, 0.068f, 0.94f));
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 18, fontStyle = FontStyle.Bold, normal = { textColor = new Color(0.35f, 0.95f, 0.86f) } };
            labelStyle = new GUIStyle(GUI.skin.label) { fontSize = 13, wordWrap = true, normal = { textColor = new Color(0.90f, 0.95f, 0.98f) } };
            smallStyle = new GUIStyle(labelStyle) { fontSize = 11, normal = { textColor = new Color(0.68f, 0.78f, 0.84f) } };
            learnedStyle = new GUIStyle(labelStyle) { fontStyle = FontStyle.Bold, normal = { textColor = new Color(0.23f, 0.72f, 1f) } };
            ruleStyle = new GUIStyle(labelStyle) { fontStyle = FontStyle.Bold, normal = { textColor = new Color(0.97f, 0.70f, 0.25f) } };
            warningStyle = new GUIStyle(labelStyle) { fontStyle = FontStyle.Bold, normal = { textColor = new Color(1f, 0.72f, 0.30f) } };
        }

        private void CloseWriters()
        {
            if (telemetryWriter != null) { telemetryWriter.Flush(); telemetryWriter.Dispose(); telemetryWriter = null; }
            if (ledgerWriter != null) { ledgerWriter.Flush(); ledgerWriter.Dispose(); ledgerWriter = null; }
        }

        private void OnApplicationQuit()
        {
            if (!episodeCompleted && episodeBegun) CompleteEpisode("application_quit");
            CloseWriters();
        }

        private static SceneSemanticManifest Manifest(
            string id, string name, string description, string[] tags, SemanticNodeContract[] nodes,
            SemanticEdgeContract[] edges, SemanticObjectiveContract[] objectives, SemanticAffordanceContract[] affordances)
        {
            return new SceneSemanticManifest
            {
                scenarioId = id,
                scenarioName = name,
                description = description,
                generatedFrom = "Procedural Unity objects plus explicit semantic authoring contracts",
                semanticTags = tags,
                nodes = nodes,
                edges = edges,
                objectives = objectives,
                affordances = affordances
            };
        }

        private static SemanticNodeContract Node(string id, string label, string type, Vector3 position)
        {
            return new SemanticNodeContract { id = id, label = label, semanticType = type, position = position, arrivalRadius = 0.58f };
        }

        private static SemanticEdgeContract Edge(string from, string to, float cost)
        {
            return new SemanticEdgeContract { from = from, to = to, traversalCost = cost, traversalSkill = "Navigate", bidirectional = true };
        }

        private static SemanticObjectiveContract Objective(
            string id, string label, string node, string affordance, string skill, float observeSeconds)
        {
            return new SemanticObjectiveContract
            {
                id = id, label = label, nodeId = node, targetAffordanceId = affordance,
                requiredSkill = skill, observeSeconds = observeSeconds
            };
        }

        private static SemanticAffordanceContract Affordance(
            string id, string label, string type, string verb, string node, Vector3 approach,
            string success, string evidence)
        {
            return new SemanticAffordanceContract
            {
                id = id, label = label, semanticType = type, verb = verb, nodeId = node,
                approachPosition = approach, activationDistance = 1.35f,
                successState = success, evidenceSignal = evidence
            };
        }

        private Material Mat(string name, Color color, bool emissive = false)
        {
            return NormalHmdDemoBootstrap.MakeMaterial(name, color, emissive ? 0.03f : 0f, 0.52f,
                emissive ? color * 0.24f : default(Color));
        }

        private GameObject Primitive(string name, PrimitiveType type, Vector3 position, Vector3 scale, Material material)
        {
            GameObject instance = GameObject.CreatePrimitive(type);
            instance.name = name;
            instance.transform.SetParent(scenarioRoot.transform, false);
            instance.transform.position = position;
            instance.transform.localScale = scale;
            Renderer renderer = instance.GetComponent<Renderer>();
            if (renderer != null) renderer.sharedMaterial = material;
            Collider collider = instance.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            return instance;
        }

        private static GameObject CreateVehicleChild(
            string name,
            PrimitiveType type,
            Transform parent,
            Vector3 localPosition,
            Vector3 localScale,
            Material material)
        {
            GameObject instance = GameObject.CreatePrimitive(type);
            instance.name = name;
            instance.transform.SetParent(parent, false);
            instance.transform.localPosition = localPosition;
            instance.transform.localScale = localScale;
            Renderer renderer = instance.GetComponent<Renderer>();
            if (renderer != null) renderer.sharedMaterial = material;
            Collider collider = instance.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            return instance;
        }

        private void CreateTrackBeam(
            string name,
            Vector3 from,
            Vector3 to,
            Material material,
            float width)
        {
            Vector3 direction = to - from;
            float length = direction.magnitude;
            GameObject beam = Primitive(
                name,
                PrimitiveType.Cube,
                (from + to) * 0.5f,
                new Vector3(width, width, length),
                material);
            if (length > 0.0001f) beam.transform.rotation = Quaternion.LookRotation(direction / length, Vector3.up);
        }

        private void Segment(string name, Vector3 position, Vector3 scale, Material material)
        {
            Primitive(name, PrimitiveType.Cube, position, scale, material);
        }

        private static float InitialYawForScenario(int index)
        {
            return index == 1 ? 0f : 0f;
        }

        private static void Csv(StringBuilder builder, float value)
        {
            builder.Append(',').Append(value.ToString("G9", CultureInfo.InvariantCulture));
        }

        private static string CsvText(string value)
        {
            if (value == null) return "";
            return value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0
                ? value : "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static bool HasArgument(string exact)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (string.Equals(argument, exact, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        private static string ReadArgumentValue(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return argument.Substring(prefix.Length).Trim('"');
            return null;
        }

        private static int ReadIntArgument(string prefix, int fallback)
        {
            int value;
            return int.TryParse(ReadArgumentValue(prefix), NumberStyles.Integer, CultureInfo.InvariantCulture, out value)
                ? value : fallback;
        }

        private static int RandomSeed()
        {
            unchecked { return ((Environment.TickCount * 397) ^ Guid.NewGuid().GetHashCode()) & int.MaxValue; }
        }

        private static string SafeFileName(string value)
        {
            foreach (char invalid in Path.GetInvalidFileNameChars()) value = value.Replace(invalid, '_');
            return value;
        }

        private static Texture2D Solid(Color color)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }
    }
}
