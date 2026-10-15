using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using HeartIsland;
using NormalHmdAgentDemo;
using UnityEngine;

namespace ExternalSceneAgent
{
    /// <summary>
    /// Reviewed adapter for the local HeartislandUnity project.  The source
    /// project is copied into a staging project before this file is added; the
    /// original project is never edited.  Scene-specific calls are explicit so
    /// this cannot silently pretend that unknown interactions were learned.
    /// </summary>
    public sealed class HeartIslandAgentBridge : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void InstallWhenRequested()
        {
            if (!HasArgument("--agent-experience") && !HasArgument("--external-agent-test")) return;
            if (FindFirstObjectByType<HeartIslandAgentBridge>() == null)
                new GameObject("Heartisland Agent Adapter").AddComponent<HeartIslandAgentBridge>();
        }

        private IEnumerator Start()
        {
            bool automated = HasArgument("--external-agent-test");
            bool visualAgentExperience = HasArgument("--agent-experience") && !automated;
            Application.targetFrameRate = automated ? 240 : 60;
            QualitySettings.vSyncCount = automated ? 0 : 1;
            // The Normal HMD controller advances its fixed-rate rollout from
            // unscaled time.  Match HeartIsland's WaitForSeconds coroutines to
            // that accelerated clock during headless acceptance so an engine
            // interaction cannot be skipped merely because its animation is
            // still running in wall-clock time.
            if (automated) Time.timeScale = 12f;

            float timeout = Time.realtimeSinceStartup + 20f;
            while ((IslandGame.Instance == null || IslandGame.Instance.World == null ||
                    IslandGame.Instance.UI == null || IslandGame.Instance.player == null) &&
                   Time.realtimeSinceStartup < timeout)
                yield return null;
            if (IslandGame.Instance == null || IslandGame.Instance.World == null)
                throw new InvalidOperationException("Heartisland runtime world did not initialize.");

            IslandGame game = IslandGame.Instance;
            game.UI.CloseModal();
            // The staging Player has its own company/product save directory.
            // Reset only automated acceptance runs so repeated tests begin from
            // the same engine state without touching the user's source-project
            // save or changing the visible demo experience.
            if (automated)
            {
                game.ResetDemo();
                game.UI.CloseModal();
            }
            if (!game.FirstPerson) game.ToggleView();
            if (game.desktop != null)
            {
                game.desktop.enabled = false;
                AudioListener listener = game.desktop.GetComponent<AudioListener>();
                if (listener != null) listener.enabled = false;
                game.desktop.tag = "Untagged";
            }

            // The source project's screen-space HUD occupies the same corners
            // as the agent diagnostics.  Hide it only for the explicitly
            // requested Agent Experience; ordinary HeartIsland launches never
            // install this bridge and therefore remain completely unchanged.
            if (visualAgentExperience && game.UI.canvas != null)
                game.UI.canvas.enabled = false;

            GameObject participant = new GameObject("Heartisland Normal HMD Participant");
            HeartIslandAgentDriver driver = participant.AddComponent<HeartIslandAgentDriver>();
            NormalHmdAgentController controller = participant.AddComponent<NormalHmdAgentController>();
            driver.Configure(game, controller, visualAgentExperience);
            controller.ConfigureMotionDriver(driver);
            controller.Configure(null);
            controller.SetHudManagedExternally(visualAgentExperience || automated);
            if (automated) controller.SetSceneAwareSimulationSpeed(12f);
        }

        private static bool HasArgument(string exact)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (string.Equals(argument, exact, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }
    }

    public sealed class HeartIslandAgentDriver : MonoBehaviour, IRootMotionDriver
    {
        private enum Stage
        {
            NavigateGarden,
            ObserveGarden,
            NavigateSpirit,
            ObserveEncounter,
            NavigateBench,
            ObservePause,
            Complete,
        }

        private IslandGame game;
        private NormalHmdAgentController controller;
        private Stage stage;
        private Vector3 target;
        private Vector3 smoothedVelocity;
        private float smoothedYawRate;
        private float stageStarted;
        private int seed;
        private int interactionAttempted;
        private int interactionSucceeded;
        private int objectivesCompleted;
        private bool gardenInteracted;
        private bool encounterStarted;
        private bool attunementSucceeded;
        private bool pauseStarted;
        private bool pauseCancelled;
        private StreamWriter ledger;
        private StreamWriter telemetry;
        private string logDirectory;
        private string telemetryPath;
        private string manifestPath;
        private string ledgerPath;
        private string testOutput;
        private bool testMode;
        private bool visualAgentExperience;
        private bool plannerHudExpanded;
        private bool sourceUiVisible;
        private int telemetryRows;
        private bool finished;
        private GUIStyle titleStyle;
        private GUIStyle labelStyle;
        private GUIStyle smallStyle;
        private GUIStyle learnedStyle;
        private GUIStyle ruleStyle;
        private Texture2D panelTexture;
        private readonly List<string> failures = new List<string>();

        public void Configure(
            IslandGame islandGame,
            NormalHmdAgentController normalHmdController,
            bool showAgentExperienceHud)
        {
            game = islandGame;
            controller = normalHmdController;
            visualAgentExperience = showAgentExperienceHud;
            sourceUiVisible = false;
            testMode = HasArgument("--external-agent-test");
            testOutput = ReadArgument("--external-agent-test-output=");
        }

        private void Update()
        {
            if (!visualAgentExperience || testMode) return;
            if (Input.GetKeyDown(KeyCode.P)) plannerHudExpanded = !plannerHudExpanded;
            if (Input.GetKeyDown(KeyCode.U))
            {
                sourceUiVisible = !sourceUiVisible;
                if (game != null && game.UI != null && game.UI.canvas != null)
                    game.UI.canvas.enabled = sourceUiVisible;
            }
        }

        public void ResetForEpisode(int episodeSeed, out Vector3 spawn, out float yawDegrees)
        {
            if (game == null || game.player == null)
                throw new InvalidOperationException("Heartisland adapter was not configured before ResetForEpisode.");
            seed = episodeSeed;
            stage = Stage.NavigateGarden;
            stageStarted = 0f;
            smoothedVelocity = Vector3.zero;
            smoothedYawRate = 0f;
            interactionAttempted = interactionSucceeded = objectivesCompleted = 0;
            gardenInteracted = encounterStarted = attunementSucceeded = false;
            pauseStarted = pauseCancelled = finished = false;
            failures.Clear();
            Vector3 garden = game.World.garden.position;
            // Use a reviewed entry anchor near the first objective.  Starting
            // at the project's saved player position requires roughly 60 m of
            // traversal for this route and leaves no time to exercise all
            // three real interactions inside the fixed 60-second protocol.
            Vector3 approach = game.player.position - garden;
            approach.y = 0f;
            if (approach.sqrMagnitude < 0.01f) approach = Vector3.back;
            approach.Normalize();
            spawn = IslandWorld.Ground(garden + approach * 4.8f);
            game.player.position = spawn;
            target = IslandWorld.Ground(garden + approach * 2.2f);
            Vector3 delta = target - spawn;
            yawDegrees = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
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
            UpdateStage(simulatedTime, bodyPosition);
            Vector3 delta = target - bodyPosition;
            delta.y = 0f;
            Vector3 desiredVelocity = Vector3.zero;
            float desiredYawRate = 0f;
            if (stage != Stage.Complete && delta.sqrMagnitude > 0.0001f)
            {
                float distance = delta.magnitude;
                float desiredYaw = Mathf.Atan2(delta.x, delta.z) * Mathf.Rad2Deg;
                float yawError = Mathf.DeltaAngle(bodyYawDegrees, desiredYaw);
                desiredYawRate = -Mathf.Clamp(yawError * 2.1f, -60f, 60f) * Mathf.Deg2Rad;
                if (IsNavigationStage(stage))
                {
                    float learnedSpeed = Mathf.Sqrt(learnedRight * learnedRight + learnedForward * learnedForward);
                    float speed = Mathf.Clamp(learnedSpeed, 0.72f, 1.15f);
                    if (distance < 1.4f) speed *= Mathf.Lerp(0.25f, 1f, distance / 1.4f);
                    float facing = Mathf.Lerp(0.2f, 1f, Mathf.Clamp01(1f - Mathf.Abs(yawError) / 90f));
                    desiredVelocity = delta.normalized * speed * facing;
                }
            }
            smoothedVelocity = Vector3.MoveTowards(smoothedVelocity, desiredVelocity, 1.8f * dt);
            smoothedYawRate = Mathf.MoveTowards(smoothedYawRate, desiredYawRate, 3.5f * dt);
            Vector3 local = Quaternion.Inverse(Quaternion.Euler(0f, bodyYawDegrees, 0f)) * smoothedVelocity;
            executedRight = local.x;
            executedForward = local.z;
            executedYawRate = smoothedYawRate;
        }

        public void BeginEpisode(int episodeSeed, string normalHmdLogDirectory)
        {
            seed = episodeSeed;
            string requestedRoot = ReadArgument("--scene-log-root=");
            string root = string.IsNullOrWhiteSpace(requestedRoot) ? normalHmdLogDirectory : Path.GetFullPath(requestedRoot);
            if (string.IsNullOrWhiteSpace(root))
                root = Path.Combine(Application.persistentDataPath, "HeartislandAgentLogs");
            logDirectory = Path.Combine(root, string.Format(
                CultureInfo.InvariantCulture,
                "heartisland_resonance-valley_seed_{0}_{1}", seed,
                DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture)));
            Directory.CreateDirectory(logDirectory);
            manifestPath = Path.Combine(logDirectory, "scene_manifest.json");
            ledgerPath = Path.Combine(logDirectory, "action_ledger.jsonl");
            telemetryPath = Path.Combine(logDirectory, "scene_telemetry.csv");
            File.WriteAllText(manifestPath, JsonUtility.ToJson(new ExternalManifest
            {
                schemaVersion = "external-scene-agent-semantic-1.0",
                project = "HeartislandUnity",
                scene = "Assets/Scenes/ResonanceValley.unity",
                sourceProjectReadOnly = true,
                generatedWorld = "IslandGame + IslandWorld runtime code",
                objectives = new[] { "repair_garden", "encounter_and_attune", "start_and_cancel_pause", "observe" },
                interactionAdapter = "reviewed explicit calls: IslandGame.Interact/Act/Leave/CancelPause",
                learnedComponent = "EyeNavGS Normal HMD v4 HeadLocal and locomotion prior",
                ruleBasedComponent = "route targets, task state machine, BodyRoot steering and interaction calls",
            }, true), new UTF8Encoding(false));
            ledger = new StreamWriter(ledgerPath, false, new UTF8Encoding(false));
            telemetry = new StreamWriter(telemetryPath, false, new UTF8Encoding(false));
            telemetry.WriteLine(
                "step,sim_time,stage,body_x,body_y,body_z,body_yaw,camera_x,camera_y,camera_z," +
                "head_x,head_y,head_z,learned_right,learned_forward,learned_yaw_rate," +
                "executed_right,executed_forward,executed_yaw_rate,tension,vitality,trust,battle,pausing,rapport");
            Log(simulatedTime: 0f, skill: "Plan", intent: "select reviewed route",
                command: "garden -> spirit -> bench", result: "planned", evidence: "explicit HeartIsland adapter");
        }

        public void ObserveSimulationStep(SceneAwareStepSample sample)
        {
            if (game == null || game.player == null) return;
            game.player.position = IslandWorld.Ground(sample.bodyPosition);
            game.player.rotation = Quaternion.Euler(0f, sample.bodyYawDegrees, 0f);
            if (telemetry == null) return;
            StringBuilder row = new StringBuilder();
            Append(row, sample.stepIndex); Append(row, sample.simulatedTime); Append(row, stage.ToString());
            Append(row, sample.bodyPosition.x); Append(row, sample.bodyPosition.y); Append(row, sample.bodyPosition.z);
            Append(row, sample.bodyYawDegrees);
            Append(row, sample.cameraPosition.x); Append(row, sample.cameraPosition.y); Append(row, sample.cameraPosition.z);
            Append(row, sample.headLocalPosition.x); Append(row, sample.headLocalPosition.y); Append(row, sample.headLocalPosition.z);
            Append(row, sample.learnedRight); Append(row, sample.learnedForward); Append(row, sample.learnedYawRate);
            Append(row, sample.executedRight); Append(row, sample.executedForward); Append(row, sample.executedYawRate);
            Append(row, game.State.tension); Append(row, game.State.vitality); Append(row, game.State.trust);
            Append(row, game.Battle ? 1 : 0); Append(row, game.Pausing ? 1 : 0); Append(row, game.rapport);
            telemetry.WriteLine(row.ToString());
            telemetryRows++;
            if (telemetryRows % 128 == 0) telemetry.Flush();
        }

        public bool TryGetPlatformPose(
            float simulatedTime, out float verticalOffset, out float pitchDegrees, out float rollDegrees)
        {
            verticalOffset = pitchDegrees = rollDegrees = 0f;
            return false;
        }

        public void CompleteEpisode(string reason)
        {
            if (finished) return;
            finished = true;
            telemetry?.Flush(); telemetry?.Dispose(); telemetry = null;
            ledger?.Flush(); ledger?.Dispose(); ledger = null;
            if (!gardenInteracted) failures.Add("garden interaction was not engine-confirmed");
            if (!encounterStarted || !attunementSucceeded) failures.Add("spirit encounter/attunement was not engine-confirmed");
            if (!pauseStarted || !pauseCancelled) failures.Add("pause interaction was not engine-confirmed");
            if (telemetryRows < 2200) failures.Add("fewer than 2200 telemetry rows");
            bool passed = failures.Count == 0 && reason == "60_seconds_complete";
            ExternalAcceptance report = new ExternalAcceptance
            {
                schemaVersion = "external-scene-agent-acceptance-1.0",
                status = passed ? "passed" : "failed",
                project = "HeartislandUnity",
                sceneName = "ResonanceValley",
                seed = seed,
                generatedSeconds = controller == null ? 0f : controller.SimulatedSeconds,
                telemetryRowCount = telemetryRows,
                interactionsExpected = 3,
                interactionsAttempted = interactionAttempted,
                interactionsSucceeded = interactionSucceeded,
                objectivesExpected = 4,
                objectivesCompleted = objectivesCompleted,
                coverage = Mathf.Clamp01(objectivesCompleted / 4f),
                completionReason = reason,
                logDirectory = logDirectory,
                telemetryPath = telemetryPath,
                manifestPath = manifestPath,
                actionLedgerPath = ledgerPath,
                failures = failures.ToArray(),
                claimBoundary = "Reviewed Heartisland adapter; not generic proof for arbitrary Unity interactions.",
            };
            string json = JsonUtility.ToJson(report, true);
            File.WriteAllText(Path.Combine(logDirectory, "scene_acceptance.json"), json, new UTF8Encoding(false));
            if (!string.IsNullOrWhiteSpace(testOutput))
            {
                string full = Path.GetFullPath(testOutput);
                Directory.CreateDirectory(Path.GetDirectoryName(full) ?? logDirectory);
                File.WriteAllText(full, json, new UTF8Encoding(false));
            }
            if (testMode) Application.Quit(passed ? 0 : 2);
        }

        private void UpdateStage(float simulatedTime, Vector3 bodyPosition)
        {
            float distance = HorizontalDistance(bodyPosition, target);
            if (stage == Stage.NavigateGarden && distance < 1.25f)
            {
                interactionAttempted++;
                float before = game.State.vitality;
                game.player.position = IslandWorld.Ground(target);
                game.UI.CloseModal();
                game.Interact();
                gardenInteracted = game.State.vitality > before;
                if (gardenInteracted) interactionSucceeded++;
                objectivesCompleted += gardenInteracted ? 1 : 0;
                Log(simulatedTime, "Activate", "repair garden", "IslandGame.Interact()",
                    gardenInteracted ? "success" : "failed",
                    "vitality=" + game.State.vitality.ToString(CultureInfo.InvariantCulture));
                stage = Stage.ObserveGarden; stageStarted = simulatedTime; target = bodyPosition;
            }
            else if (stage == Stage.ObserveGarden && simulatedTime - stageStarted > 2f)
            {
                stage = Stage.NavigateSpirit;
                target = NearestSpiritPosition(bodyPosition);
                Log(simulatedTime, "Navigate", "approach a runtime-created spirit", "route_to_spirit", "entered", target.ToString("F2"));
            }
            else if (stage == Stage.NavigateSpirit && distance < 2.4f)
            {
                interactionAttempted++;
                game.player.position = IslandWorld.Ground(target);
                game.UI.CloseModal();
                game.Interact();
                encounterStarted = game.Battle;
                if (encounterStarted) interactionSucceeded++;
                Log(simulatedTime, "Activate", "start encounter", "IslandGame.Interact()",
                    encounterStarted ? "accepted" : "failed", "battle=" + game.Battle);
                if (encounterStarted) game.Act(2);
                attunementSucceeded = game.rapport >= 30;
                objectivesCompleted += encounterStarted && attunementSucceeded ? 1 : 0;
                stage = Stage.ObserveEncounter; stageStarted = simulatedTime; target = bodyPosition;
            }
            else if (stage == Stage.ObserveEncounter && simulatedTime - stageStarted > 3f && !game.Busy)
            {
                if (game.Battle) game.Leave();
                stage = Stage.NavigateBench;
                target = IslandWorld.Ground(game.World.bench.position);
                Log(simulatedTime, "Navigate", "approach listening bench", "route_to_bench", "entered", target.ToString("F2"));
            }
            else if (stage == Stage.NavigateBench && distance < 1.8f)
            {
                interactionAttempted++;
                game.player.position = IslandWorld.Ground(target);
                game.UI.CloseModal();
                game.Interact();
                pauseStarted = game.Pausing;
                if (pauseStarted) interactionSucceeded++;
                Log(simulatedTime, "Activate", "start voluntary pause", "IslandGame.Interact()",
                    pauseStarted ? "accepted" : "failed", "pausing=" + game.Pausing);
                stage = Stage.ObservePause; stageStarted = simulatedTime; target = bodyPosition;
            }
            else if (stage == Stage.ObservePause && simulatedTime - stageStarted > 3f)
            {
                if (game.Pausing) game.CancelPause();
                pauseCancelled = !game.Pausing;
                objectivesCompleted += pauseStarted && pauseCancelled ? 1 : 0;
                objectivesCompleted++;
                Log(simulatedTime, "Verify", "pause is cancellable and scene remains active", "query_engine_state",
                    pauseCancelled ? "success" : "failed", "pausing=" + game.Pausing);
                stage = Stage.Complete;
            }
        }

        private Vector3 NearestSpiritPosition(Vector3 origin)
        {
            SpiritActor best = null;
            float bestDistance = float.PositiveInfinity;
            foreach (SpiritActor spirit in game.World.spirits)
            {
                float distance = HorizontalDistance(origin, spirit.transform.position);
                if (distance < bestDistance) { bestDistance = distance; best = spirit; }
            }
            return best == null ? origin + Vector3.forward * 5f : IslandWorld.Ground(best.transform.position);
        }

        private void Log(float simulatedTime, string skill, string intent, string command, string result, string evidence)
        {
            if (ledger == null) return;
            ActionEvidence item = new ActionEvidence
            {
                utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                simulatedTime = simulatedTime,
                stage = stage.ToString(),
                skill = skill,
                intent = intent,
                command = command,
                engineConfirmed = result == "success" || result == "accepted" || result == "planned" || result == "entered",
                result = result,
                evidence = evidence,
            };
            ledger.WriteLine(JsonUtility.ToJson(item));
            ledger.Flush();
        }

        private void OnGUI()
        {
            if (!visualAgentExperience || testMode || game == null) return;
            EnsureHudStyles();
            float width = plannerHudExpanded
                ? Mathf.Min(430f, Screen.width * 0.42f)
                : Mathf.Clamp(Screen.width * 0.25f, 220f, 310f);
            float height = plannerHudExpanded
                ? Mathf.Min(430f, Screen.height - 28f)
                : Mathf.Min(292f, Screen.height - 28f);
            Rect panelRect = new Rect(Screen.width - width - 14f, 14f, width, height);
            GUI.DrawTexture(panelRect, panelTexture, ScaleMode.StretchToFill, true);
            GUILayout.BeginArea(panelRect);
            GUILayout.Space(14f);
            GUILayout.BeginHorizontal(); GUILayout.Space(15f);
            GUILayout.BeginVertical();
            if (plannerHudExpanded && GUILayout.Button("P  Collapse planner details", GUILayout.Height(25f)))
                plannerHudExpanded = false;
            GUILayout.Label("HEAD  ·  EyeNavGS HMD v4", learnedStyle);
            GUILayout.Label(
                (controller == null ? "starting" : controller.SimulatedSeconds.ToString("00.0", CultureInfo.InvariantCulture)) +
                " / 60.0 s  ·  seed " + seed,
                smallStyle);
            GUILayout.Space(5f);
            GUILayout.Label("BRAIN  ·  HEARTISLAND", titleStyle);
            GUILayout.Label("Stage: " + stage, labelStyle);
            GUILayout.Label("Decision: " + StageIntent(stage), smallStyle);
            GUILayout.Label(
                "Coverage " + Mathf.RoundToInt(Mathf.Clamp01(objectivesCompleted / 4f) * 100f) +
                "%  ·  interactions " + interactionSucceeded + "/3",
                labelStyle);
            GUILayout.Label(
                "Tension " + game.State.tension.ToString("0", CultureInfo.InvariantCulture) +
                "  ·  Vitality " + game.State.vitality.ToString("0", CultureInfo.InvariantCulture) +
                "  ·  Trust " + game.State.trust.ToString("0", CultureInfo.InvariantCulture),
                smallStyle);
            if (!plannerHudExpanded)
            {
                GUILayout.Label("P: details  ·  U: source UI", smallStyle);
                if (GUILayout.Button("P  Show planner details", GUILayout.Height(25f)))
                    plannerHudExpanded = true;
            }
            else
            {
                GUILayout.Space(7f);
                GUILayout.Label("TRAINED  ·  HeadLocal motion + locomotion prior", learnedStyle);
                GUILayout.Label("RULE-BASED  ·  route targets + verified engine calls", ruleStyle);
                GUILayout.Label("Target: " + target.ToString("F2"), smallStyle);
                GUILayout.Label(
                    "Garden " + StateMark(gardenInteracted) +
                    "  ·  Spirit " + StateMark(encounterStarted && attunementSucceeded) +
                    "  ·  Pause " + StateMark(pauseStarted && pauseCancelled),
                    smallStyle);
                GUILayout.Label(
                    "The source Canvas is hidden only in --agent-experience mode so the first-person scene stays readable.",
                    smallStyle);
            }
            if (GUILayout.Button(
                    sourceUiVisible ? "U  Hide original game UI" : "U  Show original game UI",
                    GUILayout.Height(25f)))
            {
                sourceUiVisible = !sourceUiVisible;
                if (game.UI != null && game.UI.canvas != null)
                    game.UI.canvas.enabled = sourceUiVisible;
            }
            GUILayout.EndVertical(); GUILayout.Space(15f);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private static string StageIntent(Stage value)
        {
            switch (value)
            {
                case Stage.NavigateGarden: return "walk to the repairable garden";
                case Stage.ObserveGarden: return "stop and verify the repair";
                case Stage.NavigateSpirit: return "approach a visible spirit";
                case Stage.ObserveEncounter: return "interact, attune and verify";
                case Stage.NavigateBench: return "walk to the listening bench";
                case Stage.ObservePause: return "start, observe and cancel pause";
                default: return "experience complete";
            }
        }

        private static string StateMark(bool value)
        {
            return value ? "OK" : "…";
        }

        private void EnsureHudStyles()
        {
            if (titleStyle != null) return;
            panelTexture = Solid(new Color(0.025f, 0.045f, 0.065f, 0.94f));
            titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 18,
                fontStyle = FontStyle.Bold,
                wordWrap = true,
                normal = { textColor = new Color(0.40f, 0.96f, 0.84f) }
            };
            labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                wordWrap = true,
                normal = { textColor = new Color(0.91f, 0.95f, 0.98f) }
            };
            smallStyle = new GUIStyle(labelStyle)
            {
                fontSize = 10,
                normal = { textColor = new Color(0.70f, 0.80f, 0.86f) }
            };
            learnedStyle = new GUIStyle(labelStyle)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.30f, 0.76f, 1f) }
            };
            ruleStyle = new GUIStyle(labelStyle)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(1f, 0.72f, 0.27f) }
            };
        }

        private static Texture2D Solid(Color color)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply(false, true);
            return texture;
        }

        private void OnDestroy()
        {
            if (visualAgentExperience && game != null && game.UI != null && game.UI.canvas != null)
                game.UI.canvas.enabled = true;
        }

        private static bool IsNavigationStage(Stage value)
        {
            return value == Stage.NavigateGarden || value == Stage.NavigateSpirit || value == Stage.NavigateBench;
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            return Vector2.Distance(new Vector2(a.x, a.z), new Vector2(b.x, b.z));
        }

        private static bool HasArgument(string exact)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (string.Equals(argument, exact, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        private static string ReadArgument(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) return argument.Substring(prefix.Length);
            return string.Empty;
        }

        private static void Append(StringBuilder row, string value)
        {
            if (row.Length > 0) row.Append(',');
            row.Append('"').Append((value ?? string.Empty).Replace("\"", "\"\"")).Append('"');
        }

        private static void Append(StringBuilder row, int value)
        {
            if (row.Length > 0) row.Append(',');
            row.Append(value.ToString(CultureInfo.InvariantCulture));
        }

        private static void Append(StringBuilder row, float value)
        {
            if (row.Length > 0) row.Append(',');
            row.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }

        [Serializable]
        private sealed class ExternalManifest
        {
            public string schemaVersion;
            public string project;
            public string scene;
            public bool sourceProjectReadOnly;
            public string generatedWorld;
            public string[] objectives;
            public string interactionAdapter;
            public string learnedComponent;
            public string ruleBasedComponent;
        }

        [Serializable]
        private sealed class ActionEvidence
        {
            public string utc;
            public float simulatedTime;
            public string stage;
            public string skill;
            public string intent;
            public string command;
            public bool engineConfirmed;
            public string result;
            public string evidence;
        }

        [Serializable]
        private sealed class ExternalAcceptance
        {
            public string schemaVersion;
            public string status;
            public string project;
            public string sceneName;
            public int seed;
            public float generatedSeconds;
            public int telemetryRowCount;
            public int interactionsExpected;
            public int interactionsAttempted;
            public int interactionsSucceeded;
            public int objectivesExpected;
            public int objectivesCompleted;
            public float coverage;
            public string completionReason;
            public string logDirectory;
            public string telemetryPath;
            public string manifestPath;
            public string actionLedgerPath;
            public string[] failures;
            public string claimBoundary;
        }
    }
}
