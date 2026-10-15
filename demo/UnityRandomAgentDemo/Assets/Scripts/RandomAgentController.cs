using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace RandomAgentDemo
{
    public sealed class RandomAgentController : MonoBehaviour
    {
        private const int EpisodeSteps = 240;
        private const float HeadBaseHeight = 1.64f;
        private const float MaxHeadVerticalOffset = 0.19f;
        private const string ProductLabel = "EyeNavGS Random Human Motion Demo";

        private Camera mainCamera;
        private float stageLimit = 8.6f;
        private EyeNavLstmRuntime runtime;
        private string fatalError;
        private bool configured;
        private bool paused;
        private int cameraMode;
        private float simulationSpeed = 1f;
        private float accumulator;
        private int stepIndex;
        private int inferenceCount;
        private int sessionSeed;
        private int episodeNumber;
        private float simulationTime;
        private float headVerticalOffset;
        private float[] history;
        private float[] lastRawAction = new float[6];
        private float[] lastExecutedAction = new float[6];
        private SeedWindowData activeSeedWindow;
        private Quaternion simulatedOrientation = Quaternion.identity;
        private Vector3 simulatedPosition;
        private Vector3 episodeStartPosition;
        private bool lastBoundaryReflection;
        private float distanceTravelled;
        private string currentLogDirectory = "not started";
        private StreamWriter telemetryWriter;
        private string telemetryPath;
        private DateTime episodeStartedUtc;

        private Transform bodyVisual;
        private Transform headPivot;
        private Renderer headRenderer;
        private Renderer visorRenderer;
        private Renderer directionRenderer;
        private LineRenderer trail;
        private readonly Vector3[] trailPoints = new Vector3[EpisodeSteps + 1];
        private int trailCount;

        private GUIStyle titleStyle;
        private GUIStyle sectionStyle;
        private GUIStyle labelStyle;
        private GUIStyle smallStyle;
        private GUIStyle goodStyle;
        private GUIStyle warningStyle;
        private GUIStyle errorStyle;
        private Texture2D panelTexture;
        private Texture2D accentTexture;

        public void Configure(Camera camera, float movementLimit)
        {
            mainCamera = camera;
            stageLimit = movementLimit;
            configured = true;
        }

        private void Start()
        {
            try
            {
                if (!configured)
                {
                    mainCamera = Camera.main;
                }
                runtime = EyeNavLstmRuntime.LoadVerified();
                BuildAvatar();

                int requestedSeed;
                bool hasRequestedSeed = TryReadSeedArgument(out requestedSeed);
                sessionSeed = hasRequestedSeed ? requestedSeed : CreateRandomSeed();

                if (HasCommandLineFlag("--self-test"))
                {
                    RunPlayerSelfTest(sessionSeed);
                    return;
                }

                StartEpisode(sessionSeed, false);
                string captureOutput = ReadArgumentValue("--capture-output=");
                if (!string.IsNullOrEmpty(captureOutput))
                {
                    StartCoroutine(CaptureAndQuit(captureOutput));
                }
            }
            catch (Exception exception)
            {
                fatalError = exception.ToString();
                Debug.LogError(fatalError);
            }
        }

        private void Update()
        {
            if (fatalError != null || runtime == null || paused)
            {
                return;
            }

            accumulator += Time.unscaledDeltaTime * simulationSpeed;
            float stepDuration = runtime.Data.medianDtSeconds;
            int guard = 0;
            while (accumulator >= stepDuration && guard < 12)
            {
                accumulator -= stepDuration;
                StepModel(true);
                guard++;
                if (stepIndex >= EpisodeSteps)
                {
                    episodeNumber++;
                    StartEpisode(CreateRandomSeed(), false);
                    break;
                }
            }

            UpdateAvatar();
        }

        private void LateUpdate()
        {
            if (mainCamera == null || headPivot == null)
            {
                return;
            }

            Vector3 focus = headPivot.position;
            bool firstPerson = cameraMode == 0;
            SetHeadVisualsVisible(!firstPerson);
            if (firstPerson)
            {
                // This is the actual simulated participant viewpoint: the trained
                // EyeNavGS model directly controls both Camera pose components.
                mainCamera.orthographic = false;
                mainCamera.transform.SetPositionAndRotation(focus, simulatedOrientation);
                return;
            }

            Vector3 desired;
            if (cameraMode == 2)
            {
                desired = new Vector3(focus.x, 15.5f, focus.z - 0.01f);
            }
            else if (cameraMode == 3)
            {
                float angle = Time.unscaledTime * 12f * Mathf.Deg2Rad;
                desired = focus + new Vector3(Mathf.Sin(angle) * 8.5f, 5.2f, Mathf.Cos(angle) * 8.5f);
            }
            else
            {
                Vector3 flatForward = simulatedOrientation * Vector3.forward;
                flatForward.y = 0f;
                if (flatForward.sqrMagnitude < 0.0001f)
                {
                    flatForward = Vector3.forward;
                }
                flatForward.Normalize();
                desired = focus - flatForward * 6.5f + Vector3.up * 3.8f;
            }

            float blend = 1f - Mathf.Exp(-5f * Time.unscaledDeltaTime);
            mainCamera.transform.position = Vector3.Lerp(mainCamera.transform.position, desired, blend);
            Quaternion look = Quaternion.LookRotation(focus - mainCamera.transform.position, Vector3.up);
            mainCamera.transform.rotation = Quaternion.Slerp(mainCamera.transform.rotation, look, blend);
            mainCamera.orthographic = cameraMode == 2;
            if (mainCamera.orthographic)
            {
                mainCamera.orthographicSize = 8.5f;
            }
        }

        private void BuildAvatar()
        {
            Material suit = DemoSceneBootstrap.MakeMaterial(
                "Agent Suit", new Color(0.12f, 0.22f, 0.36f), 0.28f, 0.78f);
            Material visor = DemoSceneBootstrap.MakeMaterial(
                "Agent Visor",
                new Color(0.15f, 0.92f, 0.90f),
                0.05f,
                0.88f,
                new Color(0.05f, 0.42f, 0.42f));
            Material accent = DemoSceneBootstrap.MakeMaterial(
                "Agent Accent",
                new Color(1.0f, 0.45f, 0.30f),
                0.02f,
                0.7f,
                new Color(0.34f, 0.06f, 0.02f));

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = "Simulated Participant Body";
            body.transform.SetParent(transform, false);
            body.transform.localPosition = new Vector3(0f, 0.78f, 0f);
            body.transform.localScale = new Vector3(0.54f, 0.72f, 0.54f);
            body.GetComponent<Renderer>().sharedMaterial = suit;
            bodyVisual = body.transform;

            GameObject head = new GameObject("Model-driven Head Rig");
            head.transform.SetParent(transform, false);
            head.transform.localPosition = new Vector3(0f, HeadBaseHeight, 0f);
            headPivot = head.transform;

            GameObject headShape = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            headShape.name = "Head";
            headShape.transform.SetParent(headPivot, false);
            headShape.transform.localScale = Vector3.one * 0.48f;
            headRenderer = headShape.GetComponent<Renderer>();
            headRenderer.sharedMaterial = suit;

            GameObject visorShape = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visorShape.name = "Forward Visor";
            visorShape.transform.SetParent(headPivot, false);
            visorShape.transform.localPosition = new Vector3(0f, 0.015f, 0.245f);
            visorShape.transform.localScale = new Vector3(0.32f, 0.18f, 0.055f);
            visorRenderer = visorShape.GetComponent<Renderer>();
            visorRenderer.sharedMaterial = visor;

            GameObject direction = GameObject.CreatePrimitive(PrimitiveType.Cube);
            direction.name = "Head Direction Ray";
            direction.transform.SetParent(headPivot, false);
            direction.transform.localPosition = new Vector3(0f, -0.04f, 0.72f);
            direction.transform.localScale = new Vector3(0.045f, 0.045f, 0.95f);
            directionRenderer = direction.GetComponent<Renderer>();
            directionRenderer.sharedMaterial = accent;

            trail = gameObject.AddComponent<LineRenderer>();
            trail.name = "Executed Motion Trail";
            trail.useWorldSpace = true;
            trail.widthMultiplier = 0.055f;
            trail.numCornerVertices = 3;
            trail.numCapVertices = 3;
            trail.material = accent;
            trail.startColor = new Color(1f, 0.45f, 0.30f, 0.95f);
            trail.endColor = new Color(0.12f, 0.88f, 0.86f, 0.20f);
        }

        private void StartEpisode(int seed, bool replay)
        {
            FinishLog(replay ? "replay_reset" : "episode_complete");
            sessionSeed = seed;
            System.Random random = new System.Random(seed);
            SeedWindowData[] bank = runtime.Data.seedWindows;
            activeSeedWindow = bank[random.Next(bank.Length)];
            history = (float[])activeSeedWindow.values.Clone();
            stepIndex = 0;
            inferenceCount = 0;
            simulationTime = 0f;
            accumulator = 0f;
            headVerticalOffset = 0f;
            distanceTravelled = 0f;
            lastBoundaryReflection = false;
            simulatedPosition = new Vector3(
                NextRange(random, -2.6f, 2.6f),
                0f,
                NextRange(random, -2.6f, 2.6f));
            episodeStartPosition = simulatedPosition;
            simulatedOrientation = Quaternion.Euler(0f, NextRange(random, 0f, 360f), 0f);
            Array.Clear(lastRawAction, 0, lastRawAction.Length);
            Array.Clear(lastExecutedAction, 0, lastExecutedAction.Length);
            trailCount = 1;
            trailPoints[0] = new Vector3(simulatedPosition.x, 0.11f, simulatedPosition.z);
            trail.positionCount = trailCount;
            trail.SetPosition(0, trailPoints[0]);
            UpdateAvatar();
            OpenLog(replay);
            Debug.Log(
                "RANDOM_AGENT_EPISODE_START seed=" + sessionSeed +
                " source=" + activeSeedWindow.trajectoryId +
                " window=" + activeSeedWindow.windowIndex +
                " model=" + runtime.Data.sourceModelSha256);
        }

        private void StepModel(bool writeLog)
        {
            float dt = runtime.Data.medianDtSeconds;
            float[] predicted = runtime.Predict(history);
            inferenceCount++;
            Array.Copy(predicted, lastRawAction, 6);

            Quaternion currentOrientation = simulatedOrientation;
            // EyeNavGS records OpenXR coordinates (right-handed, -Z forward).
            // Unity is +Z forward. Linear vectors reflect Z; axial angular
            // vectors transform with det(S)S, yielding (-wx,-wy,wz).
            Vector3 unityLocalVelocity = new Vector3(predicted[0], predicted[1], -predicted[2]);
            Vector3 unityAngularVelocity = new Vector3(-predicted[3], -predicted[4], predicted[5]);
            Vector3 worldVelocity = currentOrientation * unityLocalVelocity;
            Vector3 nextPosition = simulatedPosition + new Vector3(worldVelocity.x, 0f, worldVelocity.z) * dt;
            float nextHeadOffset = headVerticalOffset + worldVelocity.y * dt;
            lastBoundaryReflection = false;

            ReflectCoordinate(ref nextPosition.x, ref worldVelocity.x, stageLimit);
            ReflectCoordinate(ref nextPosition.z, ref worldVelocity.z, stageLimit);
            ReflectCoordinate(ref nextHeadOffset, ref worldVelocity.y, MaxHeadVerticalOffset);

            Vector3 executedUnityLocalVelocity = Quaternion.Inverse(currentOrientation) * worldVelocity;
            lastExecutedAction[0] = executedUnityLocalVelocity.x;
            lastExecutedAction[1] = executedUnityLocalVelocity.y;
            lastExecutedAction[2] = -executedUnityLocalVelocity.z;
            lastExecutedAction[3] = predicted[3];
            lastExecutedAction[4] = predicted[4];
            lastExecutedAction[5] = predicted[5];

            Vector3 displacement = nextPosition - simulatedPosition;
            distanceTravelled += displacement.magnitude;
            simulatedPosition = nextPosition;
            headVerticalOffset = nextHeadOffset;

            float angularMagnitude = unityAngularVelocity.magnitude;
            if (angularMagnitude > 0.000001f)
            {
                Quaternion delta = Quaternion.AngleAxis(
                    angularMagnitude * dt * Mathf.Rad2Deg,
                    unityAngularVelocity / angularMagnitude);
                simulatedOrientation = Normalize(currentOrientation * delta);
            }

            ShiftHistory(lastExecutedAction);
            stepIndex++;
            simulationTime = stepIndex * dt;
            if (trailCount < trailPoints.Length)
            {
                trailPoints[trailCount] = new Vector3(simulatedPosition.x, 0.11f, simulatedPosition.z);
                trailCount++;
                trail.positionCount = trailCount;
                trail.SetPosition(trailCount - 1, trailPoints[trailCount - 1]);
            }
            if (writeLog)
            {
                WriteTelemetry();
            }
        }

        private void ReflectCoordinate(ref float coordinate, ref float velocity, float limit)
        {
            if (coordinate > limit)
            {
                coordinate = limit - (coordinate - limit);
                velocity = -Mathf.Abs(velocity);
                lastBoundaryReflection = true;
            }
            else if (coordinate < -limit)
            {
                coordinate = -limit + (-limit - coordinate);
                velocity = Mathf.Abs(velocity);
                lastBoundaryReflection = true;
            }
        }

        private void ShiftHistory(float[] action)
        {
            int width = runtime.Data.inputDim;
            Array.Copy(history, width, history, 0, history.Length - width);
            Array.Copy(action, 0, history, history.Length - width, width);
        }

        private void UpdateAvatar()
        {
            transform.position = simulatedPosition;
            if (bodyVisual != null)
            {
                Vector3 forward = simulatedOrientation * Vector3.forward;
                forward.y = 0f;
                if (forward.sqrMagnitude > 0.00001f)
                {
                    bodyVisual.rotation = Quaternion.LookRotation(forward.normalized, Vector3.up);
                }
            }
            if (headPivot != null)
            {
                headPivot.position = simulatedPosition + Vector3.up * (HeadBaseHeight + headVerticalOffset);
                headPivot.rotation = simulatedOrientation;
            }
        }

        private void OpenLog(bool replay)
        {
            episodeStartedUtc = DateTime.UtcNow;
            string root = Path.Combine(Application.persistentDataPath, "Sessions");
            string folder = episodeStartedUtc.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture) +
                "_seed_" + sessionSeed.ToString(CultureInfo.InvariantCulture);
            currentLogDirectory = Path.Combine(root, folder);
            Directory.CreateDirectory(currentLogDirectory);
            telemetryPath = Path.Combine(currentLogDirectory, "telemetry.csv");
            telemetryWriter = new StreamWriter(telemetryPath, false, new UTF8Encoding(false));
            telemetryWriter.WriteLine(
                "step,sim_time_s,action_source,session_seed,seed_trajectory,seed_window," +
                "raw_vx_mps,raw_vy_mps,raw_vz_mps,raw_wx_radps,raw_wy_radps,raw_wz_radps," +
                "executed_vx_mps,executed_vy_mps,executed_vz_mps," +
                "position_x_m,position_y_m,position_z_m,rotation_x,rotation_y,rotation_z,rotation_w," +
                "boundary_reflection");

            RunManifest manifest = new RunManifest
            {
                product = ProductLabel,
                scene = "Aurora Motion Courtyard",
                runSeed = sessionSeed,
                replay = replay,
                seedTrajectory = activeSeedWindow.trajectoryId,
                seedWindowIndex = activeSeedWindow.windowIndex,
                recordedSeedFrames = runtime.Data.historyLength,
                episodeModelSteps = EpisodeSteps,
                fixedDtSeconds = runtime.Data.medianDtSeconds,
                actionSource = "trained_model",
                modelTask = runtime.Data.modelTask,
                sourceModelSha256 = runtime.Data.sourceModelSha256,
                exportedModelSha256 = runtime.ExportSha256,
                datasetFingerprintSha256 = runtime.Data.datasetFingerprintSha256,
                trainingData = runtime.Data.dataSource,
                syntheticTraining = runtime.Data.syntheticTraining,
                runtimeSelfTestMaxAbsError = runtime.SelfTestMaxAbsError,
                startedAtUtc = episodeStartedUtc.ToString("O", CultureInfo.InvariantCulture),
                note = "Only the first 8 velocity frames are recorded seed input; all following actions are closed-loop model predictions. Boundary reflection is an environment safety constraint, not learned navigation."
            };
            File.WriteAllText(
                Path.Combine(currentLogDirectory, "manifest.json"),
                JsonUtility.ToJson(manifest, true),
                new UTF8Encoding(false));
        }

        private void WriteTelemetry()
        {
            if (telemetryWriter == null)
            {
                return;
            }
            Quaternion rotation = simulatedOrientation;
            telemetryWriter.WriteLine(string.Join(",", new string[]
            {
                stepIndex.ToString(CultureInfo.InvariantCulture),
                simulationTime.ToString("F6", CultureInfo.InvariantCulture),
                "trained_model",
                sessionSeed.ToString(CultureInfo.InvariantCulture),
                CsvEscape(activeSeedWindow.trajectoryId),
                activeSeedWindow.windowIndex.ToString(CultureInfo.InvariantCulture),
                F(lastRawAction[0]), F(lastRawAction[1]), F(lastRawAction[2]),
                F(lastRawAction[3]), F(lastRawAction[4]), F(lastRawAction[5]),
                F(lastExecutedAction[0]), F(lastExecutedAction[1]), F(lastExecutedAction[2]),
                F(simulatedPosition.x), F(HeadBaseHeight + headVerticalOffset), F(simulatedPosition.z),
                F(rotation.x), F(rotation.y), F(rotation.z), F(rotation.w),
                lastBoundaryReflection ? "1" : "0"
            }));
            if (stepIndex % 37 == 0)
            {
                telemetryWriter.Flush();
            }
        }

        private void FinishLog(string reason)
        {
            if (telemetryWriter == null)
            {
                return;
            }
            telemetryWriter.Flush();
            telemetryWriter.Dispose();
            telemetryWriter = null;
            string telemetryHash = File.Exists(telemetryPath) ? Sha256File(telemetryPath) : "missing";
            RunSummary summary = new RunSummary
            {
                reason = reason,
                steps = stepIndex,
                inferenceCount = inferenceCount,
                simulatedSeconds = simulationTime,
                distanceMetres = distanceTravelled,
                telemetrySha256 = telemetryHash,
                endedAtUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture)
            };
            File.WriteAllText(
                Path.Combine(currentLogDirectory, "summary.json"),
                JsonUtility.ToJson(summary, true),
                new UTF8Encoding(false));
        }

        private void OnApplicationQuit()
        {
            FinishLog("application_quit");
        }

        private void OnDestroy()
        {
            FinishLog("controller_destroyed");
        }

        private void OnGUI()
        {
            BuildGuiStyles();
            float panelWidth = Mathf.Min(390f, Screen.width * 0.34f);
            Rect panel = new Rect(Screen.width - panelWidth - 18f, 18f, panelWidth, Screen.height - 36f);
            GUI.DrawTexture(panel, panelTexture);
            GUILayout.BeginArea(new Rect(panel.x + 18f, panel.y + 16f, panel.width - 36f, panel.height - 30f));
            GUILayout.Label("RANDOM HUMAN MOTION", titleStyle);
            GUILayout.Label("First-person Camera driven by a trained EyeNavGS LSTM", smallStyle);
            GUILayout.Space(12f);

            if (fatalError != null)
            {
                GUILayout.Label("MODEL LOAD FAILED — MOTION DISABLED", errorStyle);
                GUILayout.Space(8f);
                GUILayout.Label(fatalError, smallStyle);
                GUILayout.EndArea();
                return;
            }

            if (runtime == null)
            {
                GUILayout.Label("Loading verified model…", warningStyle);
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label("MODEL STATUS", sectionStyle);
            GUILayout.Label("● TRAINED MODEL VERIFIED", goodStyle);
            GUILayout.Label("Source: real EyeNavGS data (synthetic=false)", labelStyle);
            GUILayout.Label("Architecture: 2-layer LSTM, 52,102 parameters", labelStyle);
            GUILayout.Label("Checkpoint: " + ShortHash(runtime.Data.sourceModelSha256), labelStyle);
            GUILayout.Label("PyTorch↔Unity max error: " + runtime.SelfTestMaxAbsError.ToString("0.0e+0"), labelStyle);
            GUILayout.Label(
                "Held-out test MAE: " + runtime.Data.testLinearVelocityVectorMaeMps.ToString("F5") +
                " m/s • " + runtime.Data.testAngularVelocityVectorMaeDegps.ToString("F2") + "°/s",
                smallStyle);
            GUILayout.Space(12f);

            GUILayout.Label("CURRENT SIMULATED PARTICIPANT", sectionStyle);
            if (cameraMode == 0)
            {
                GUILayout.Label("● AI CONTROLS THIS FIRST-PERSON CAMERA", goodStyle);
            }
            else
            {
                GUILayout.Label("Observer camera — AI still drives the participant head pose", warningStyle);
            }
            GUILayout.Label("Run seed: " + sessionSeed, labelStyle);
            GUILayout.Label("Seed source only: " + (activeSeedWindow == null ? "—" : activeSeedWindow.trajectoryId), labelStyle);
            GUILayout.Label("Recorded seed frames: 8", labelStyle);
            GUILayout.Label("Generated model frames: " + inferenceCount + " / " + EpisodeSteps, goodStyle);
            GUILayout.Label("Episode: " + (episodeNumber + 1) + " • " + simulationTime.ToString("F2") + " s", labelStyle);
            GUILayout.Label("Head speed: " + LinearSpeed().ToString("F3") + " m/s", labelStyle);
            GUILayout.Label(
                "Angular speed: " + AngularSpeed().ToString("F3") + " rad/s  (" +
                (AngularSpeed() * Mathf.Rad2Deg).ToString("F1") + "°/s)",
                labelStyle);
            GUILayout.Label("Travel: " + distanceTravelled.ToString("F2") + " m", labelStyle);
            if (lastBoundaryReflection)
            {
                GUILayout.Label("Environment safety reflection applied", warningStyle);
            }
            GUILayout.Space(12f);

            GUILayout.Label("CONTROLS", sectionStyle);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Resume" : "Pause", GUILayout.Height(34f)))
            {
                paused = !paused;
            }
            if (GUILayout.Button("New random person", GUILayout.Height(34f)))
            {
                episodeNumber++;
                StartEpisode(CreateRandomSeed(), false);
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Replay same seed", GUILayout.Height(32f)))
            {
                StartEpisode(sessionSeed, true);
            }
            if (GUILayout.Button("Camera: " + CameraModeLabel(), GUILayout.Height(32f)))
            {
                cameraMode = (cameraMode + 1) % 4;
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Speed", GUILayout.Width(48f));
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 0.5f), "0.5×", "Button")) simulationSpeed = 0.5f;
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 1f), "1×", "Button")) simulationSpeed = 1f;
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 2f), "2×", "Button")) simulationSpeed = 2f;
            GUILayout.EndHorizontal();
            if (GUILayout.Button("Copy session log path", GUILayout.Height(28f)))
            {
                GUIUtility.systemCopyBuffer = currentLogDirectory;
            }
            GUILayout.Space(10f);
            GUILayout.Label("In First Person, the trained model directly controls Camera position and rotation. Each episode starts from a random held-out 8-frame history; later frames are generated online, not replayed.", smallStyle);
            GUILayout.Label("The LSTM is a motion prior, not scene-aware navigation or a cybersickness predictor.", warningStyle);
            GUILayout.EndArea();

            GUI.DrawTexture(new Rect(0f, Screen.height - 7f, Screen.width, 7f), accentTexture);
        }

        private void BuildGuiStyles()
        {
            if (titleStyle != null)
            {
                return;
            }
            panelTexture = SolidTexture(new Color(0.025f, 0.04f, 0.075f, 0.94f));
            accentTexture = SolidTexture(new Color(0.13f, 0.88f, 0.86f, 0.95f));
            titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 21,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.80f, 0.97f, 0.96f) }
            };
            sectionStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.50f, 0.67f, 0.82f) }
            };
            labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                wordWrap = true,
                normal = { textColor = new Color(0.88f, 0.92f, 0.97f) }
            };
            smallStyle = new GUIStyle(labelStyle)
            {
                fontSize = 10,
                normal = { textColor = new Color(0.66f, 0.73f, 0.82f) }
            };
            goodStyle = new GUIStyle(labelStyle)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.30f, 0.90f, 0.62f) }
            };
            warningStyle = new GUIStyle(labelStyle)
            {
                fontSize = 10,
                wordWrap = true,
                normal = { textColor = new Color(1.0f, 0.75f, 0.32f) }
            };
            errorStyle = new GUIStyle(labelStyle)
            {
                fontStyle = FontStyle.Bold,
                wordWrap = true,
                normal = { textColor = new Color(1.0f, 0.30f, 0.32f) }
            };
        }

        private void RunPlayerSelfTest(int seed)
        {
            episodeNumber = 0;
            StartEpisode(seed, false);
            for (int index = 0; index < 180; index++)
            {
                StepModel(true);
            }
            FinishLog("self_test");
            PlayerSelfTestReport report = new PlayerSelfTestReport
            {
                status = fatalError == null ? "success" : "failed",
                modelVerified = fatalError == null,
                modelSha256 = runtime.Data.sourceModelSha256,
                exportSha256 = runtime.ExportSha256,
                unityParityMaxAbsError = runtime.SelfTestMaxAbsError,
                sessionSeed = sessionSeed,
                seedTrajectory = activeSeedWindow.trajectoryId,
                generatedFrames = inferenceCount,
                finalPosition = new float[] { simulatedPosition.x, simulatedPosition.y, simulatedPosition.z },
                distanceMetres = distanceTravelled,
                logDirectory = currentLogDirectory
            };
            string output = ReadArgumentValue("--self-test-output=");
            if (string.IsNullOrEmpty(output))
            {
                output = Path.Combine(Application.persistentDataPath, "player_self_test.json");
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)) ?? Application.persistentDataPath);
            File.WriteAllText(output, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            Debug.Log("RANDOM_AGENT_PLAYER_SELF_TEST_OK output=" + output);
            Application.Quit(0);
        }

        private IEnumerator CaptureAndQuit(string output)
        {
            yield return new WaitForSecondsRealtime(2.4f);
            string fullPath = Path.GetFullPath(output);
            string directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            ScreenCapture.CaptureScreenshot(fullPath, 1);
            float deadline = Time.realtimeSinceStartup + 5f;
            while ((!File.Exists(fullPath) || new FileInfo(fullPath).Length == 0) &&
                   Time.realtimeSinceStartup < deadline)
            {
                yield return null;
            }
            FinishLog("visual_capture_complete");
            Debug.Log("RANDOM_AGENT_SCREENSHOT_OK output=" + fullPath);
            Application.Quit(File.Exists(fullPath) ? 0 : 3);
        }

        private static int CreateRandomSeed()
        {
            byte[] bytes = new byte[4];
            using (RandomNumberGenerator generator = RandomNumberGenerator.Create())
            {
                generator.GetBytes(bytes);
            }
            return BitConverter.ToInt32(bytes, 0) & int.MaxValue;
        }

        private static float NextRange(System.Random random, float minimum, float maximum)
        {
            return minimum + (float)random.NextDouble() * (maximum - minimum);
        }

        private static Quaternion Normalize(Quaternion value)
        {
            float magnitude = Mathf.Sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
            if (magnitude < 0.0000001f)
            {
                return Quaternion.identity;
            }
            return new Quaternion(value.x / magnitude, value.y / magnitude, value.z / magnitude, value.w / magnitude);
        }

        private static string F(float value)
        {
            return value.ToString("G9", CultureInfo.InvariantCulture);
        }

        private static string CsvEscape(string value)
        {
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static string ShortHash(string value)
        {
            return string.IsNullOrEmpty(value) || value.Length <= 16 ? value : value.Substring(0, 16) + "…";
        }

        private static Texture2D SolidTexture(Color color)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }

        private float LinearSpeed()
        {
            return new Vector3(lastRawAction[0], lastRawAction[1], lastRawAction[2]).magnitude;
        }

        private float AngularSpeed()
        {
            return new Vector3(lastRawAction[3], lastRawAction[4], lastRawAction[5]).magnitude;
        }

        private string CameraModeLabel()
        {
            return cameraMode == 0 ? "First Person" :
                cameraMode == 1 ? "Follow" :
                cameraMode == 2 ? "Top" : "Orbit";
        }

        private void SetHeadVisualsVisible(bool visible)
        {
            if (headRenderer != null) headRenderer.enabled = visible;
            if (visorRenderer != null) visorRenderer.enabled = visible;
            if (directionRenderer != null) directionRenderer.enabled = visible;
        }

        private static bool HasCommandLineFlag(string flag)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (string.Equals(argument, flag, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        private static string ReadArgumentValue(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    return argument.Substring(prefix.Length).Trim('"');
                }
            }
            return null;
        }

        private static bool TryReadSeedArgument(out int seed)
        {
            string value = ReadArgumentValue("--seed=");
            return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out seed) && seed >= 0;
        }

        private static string Sha256File(string path)
        {
            using (SHA256 sha = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
            {
                byte[] digest = sha.ComputeHash(stream);
                StringBuilder builder = new StringBuilder(digest.Length * 2);
                foreach (byte value in digest)
                {
                    builder.Append(value.ToString("x2"));
                }
                return builder.ToString();
            }
        }

        [Serializable]
        private sealed class RunManifest
        {
            public string product;
            public string scene;
            public int runSeed;
            public bool replay;
            public string seedTrajectory;
            public int seedWindowIndex;
            public int recordedSeedFrames;
            public int episodeModelSteps;
            public float fixedDtSeconds;
            public string actionSource;
            public string modelTask;
            public string sourceModelSha256;
            public string exportedModelSha256;
            public string datasetFingerprintSha256;
            public string trainingData;
            public bool syntheticTraining;
            public float runtimeSelfTestMaxAbsError;
            public string startedAtUtc;
            public string note;
        }

        [Serializable]
        private sealed class RunSummary
        {
            public string reason;
            public int steps;
            public int inferenceCount;
            public float simulatedSeconds;
            public float distanceMetres;
            public string telemetrySha256;
            public string endedAtUtc;
        }

        [Serializable]
        private sealed class PlayerSelfTestReport
        {
            public string status;
            public bool modelVerified;
            public string modelSha256;
            public string exportSha256;
            public float unityParityMaxAbsError;
            public int sessionSeed;
            public string seedTrajectory;
            public int generatedFrames;
            public float[] finalPosition;
            public float distanceMetres;
            public string logDirectory;
        }
    }
}
