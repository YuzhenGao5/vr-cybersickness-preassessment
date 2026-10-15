using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Runs an uninterrupted first-person BodyRoot/HeadLocal rollout. Standalone
    /// model validation remains fixed at 60 seconds; scene-aware drivers may use
    /// content-driven completion with an explicit safety ceiling.
    /// EyeNavGS supplies HMD pose only, so BodyRoot is explicitly a learned-data
    /// proxy constrained to ground-plane translation and yaw; it is not anatomy.
    /// </summary>
    public sealed class NormalHmdAgentController : MonoBehaviour
    {
        private const float EpisodeSeconds = 60f;
        private const float ProductionMaximumClipStepRate = 0.01f;
        private const float ProductionMinimumNormalizedDeltaRms = 0.01f;
        // Deceleration is not a freeze: a participant can walk during the
        // seed window, then stand and keep looking around.  The normalized
        // state-delta and camera-change gates detect actual freezing, while
        // total distance proves that locomotion occurred.
        private const float ProductionMinimumDistanceMetres = 0.5f;
        private const float ProductionMaximumDistanceMetres = 30f;
        private const float ProductionMaximumAbsoluteCameraRollDegrees = 10f;
        private const float ProductionMinimumHeadDeltaLag1Correlation = 0.70f;
        private const float ProductionMaximumHeadDirectionReversalsPerSecond = 8.5f;
        private const float ProductionMaximumRootInnovationPassthroughError = 0.0000001f;
        private const string ProductLabel = "Normal HMD Agent — EyeNavGS 60 s";

        private Camera observerCamera;
        private Camera participantCamera;
        private AudioListener observerAudio;
        private AudioListener participantAudio;
        private Transform trackingOrigin;
        private Transform headLocal;
        private Renderer bodyRenderer;
        private Renderer headRenderer;
        private Renderer visorRenderer;
        private LineRenderer trail;

        private NormalHmdLstmRuntime runtime;
        private SeededRootInnovation rootInnovation;
        private NormalHmdEventController horizontalController;
        private NormalHmdEventController verticalController;
        private NormalHmdV4Step lastStep;
        private NormalHmdSeedData activeSeed;
        private float[] lastRaw = new float[9];
        private float[] lastBaseFeedback = new float[9];
        private float[] lastStabilizedBase = new float[9];
        private float[] lastDriftCorrection = new float[9];
        private float[] lastRawInnovation = new float[9];
        private float[] lastAppliedInnovation = new float[9];
        private float[] lastApplied = new float[9];
        private float[] previousApplied = new float[9];
        private float[] firstDeltaSquareSum = new float[9];
        private float[] lastDeltaSquareSum = new float[9];
        private int firstDeltaCount;
        private int lastDeltaCount;
        private float firstPlanarSpeedEnergySum;
        private float lastPlanarSpeedEnergySum;
        private int firstPlanarSpeedEnergyCount;
        private int lastPlanarSpeedEnergyCount;
        private readonly float[] previousHeadDelta = new float[6];
        private readonly double[] headDeltaPreviousSum = new double[6];
        private readonly double[] headDeltaCurrentSum = new double[6];
        private readonly double[] headDeltaPreviousSquareSum = new double[6];
        private readonly double[] headDeltaCurrentSquareSum = new double[6];
        private readonly double[] headDeltaCrossSum = new double[6];
        private readonly int[] headDirectionReversalCount = new int[6];
        private int headDeltaPairCount;
        private bool hasPreviousHeadDelta;
        private float maxRootInnovationPassthroughError;
        private int verticalEventCount;
        private int completedVerticalEventCount;
        private int horizontalEventCount;
        private int completedHorizontalEventCount;
        private readonly float[] horizontalEventAbsAmplitudeSum = new float[4];
        private readonly float[] verticalEventAbsAmplitudeSum = new float[2];
        private readonly float[] maxHorizontalEventAbsAmplitude = new float[4];
        private readonly float[] maxVerticalEventAbsAmplitude = new float[2];
        private float maxHorizontalEventEndpointResidual;
        private float maxVerticalEventEndpointResidual;
        private float verticalEventAbsAmplitudeYSum;
        private float verticalEventAbsAmplitudePitchSum;
        private float maxVerticalEventAbsAmplitudeY;
        private float maxVerticalEventAbsAmplitudePitch;
        private float maxVerticalEventAnchorClosureResidualY;
        private float maxVerticalEventAnchorClosureResidualPitch;
        private float maxAbsoluteHeadInnovation;

        private Vector3 bodyPosition;
        private float bodyYawDegrees;
        private Vector3 previousBodyPosition;
        private float previousBodyYawDegrees;
        // Optional moving-platform pose (for example the semantic coaster).
        // It is presentation/world motion only: the trained six-channel
        // HeadLocal state and its recurrent feedback remain unchanged.
        private float platformVerticalOffset;
        private float previousPlatformVerticalOffset;
        private float platformPitchDegrees;
        private float previousPlatformPitchDegrees;
        private float platformRollDegrees;
        private float previousPlatformRollDegrees;
        private Vector3 currentHeadLocalPosition;
        private Vector3 previousHeadLocalPosition;
        private Quaternion currentHeadLocalRotation = Quaternion.identity;
        private Quaternion previousHeadLocalRotation = Quaternion.identity;
        private float accumulator;
        private float simulationSpeed = 1f;
        private float noiseScale = 1f;
        private int stepIndex;
        private int targetSteps;
        private int clippedStepCount;
        private int clippedValueCount;
        private int comfortEnvelopeActivationStepCount;
        private int comfortEnvelopeActivationValueCount;
        private int baseClippedStepCount;
        private int baseClippedValueCount;
        private int rootBaseClippedStepCount;
        private int rootBaseClippedValueCount;
        private int stabilizedBaseClippedStepCount;
        private int stabilizedBaseClippedValueCount;
        private int nonFiniteCount;
        private int episodeStartCount;
        private int episodeSeed;
        private float distanceMetres;
        private float maxBodyTiltDegrees;
        private float maxQuaternionNormError;
        private Vector3 previousSimulatedCameraPosition;
        private Quaternion previousSimulatedCameraRotation = Quaternion.identity;
        private int cameraPoseChangedStepCount;
        private float maxCameraHierarchyPositionErrorMetres;
        private float maxCameraHierarchyRotationErrorDegrees;
        private float maxAbsoluteCameraRollDegrees;
        private bool paused;
        private bool episodeComplete;
        private bool firstPerson = true;
        private bool hudManagedExternally;
        private bool hudExpanded;
        private string fatalError;

        private StreamWriter telemetry;
        private string logDirectory = "not started";
        private string telemetryPath;
        private DateTime episodeStartedUtc;

        // Optional task-level controller.  The learned Normal HMD model always
        // owns the recurrent state and all six HeadLocal channels.  A driver may
        // replace only the three executed BodyRoot commands so a semantic
        // planner can steer the participant through a scene.
        private IRootMotionDriver motionDriver;
        private IAdaptiveEpisodeDriver adaptiveEpisodeDriver;
        private bool motionDriverEpisodeBegan;
        private bool motionDriverEpisodeCompleted;
        private float episodeSafetyLimitSeconds = EpisodeSeconds;

        private GUIStyle titleStyle;
        private GUIStyle labelStyle;
        private GUIStyle smallStyle;
        private GUIStyle goodStyle;
        private GUIStyle warningStyle;
        private GUIStyle errorStyle;
        private Texture2D panelTexture;
        private Texture2D accentTexture;

        public void Configure(Camera externalObserver)
        {
            observerCamera = externalObserver;
        }

        public void ConfigureMotionDriver(IRootMotionDriver driver)
        {
            motionDriver = driver;
            adaptiveEpisodeDriver = driver as IAdaptiveEpisodeDriver;
        }

        /// <summary>
        /// Scene adapters can present one combined HEAD + BRAIN HUD instead of
        /// stacking this diagnostic panel on top of their own UI.  This changes
        /// presentation only; model execution and telemetry are unaffected.
        /// </summary>
        public void SetHudManagedExternally(bool managed)
        {
            hudManagedExternally = managed;
        }

        public void RestartSceneAwareEpisode(int seed)
        {
            episodeStartCount = 0;
            if (runtime == null)
            {
                episodeSeed = seed;
                return;
            }
            StartEpisode(seed, false);
        }

        public void SetSimulationSpeed(float speed)
        {
            simulationSpeed = Mathf.Max(0f, speed);
        }

        public void SetSceneAwareSimulationSpeed(float speed)
        {
            SetSimulationSpeed(speed);
        }

        public bool EpisodeComplete { get { return episodeComplete; } }
        public int EpisodeSeed { get { return episodeSeed; } }
        public float SimulatedSeconds
        {
            get
            {
                return runtime == null ? 0f : stepIndex * runtime.Data.fixedDtSeconds;
            }
        }
        public string LogDirectory { get { return logDirectory; } }

        private void Start()
        {
            try
            {
                runtime = NormalHmdLstmRuntime.LoadVerified();
                noiseScale = runtime.Data.defaultRootNoiseScale;
                hudExpanded = HasArgument("--expanded-hud");
                BuildParticipantRig();
                SetCameraMode(!HasArgument("--observer-camera"));
                int requestedSeed;
                episodeSeed = TryReadSeedArgument(out requestedSeed) ? requestedSeed : CreateRandomSeed();

                if (HasArgument("--rollout-test"))
                {
                    RunRolloutTest(episodeSeed);
                    return;
                }

                StartEpisode(episodeSeed, false);
                string screenCaptureOutput = ReadArgumentValue("--screen-capture-output=");
                string captureOutput = ReadArgumentValue("--capture-output=");
                if (!string.IsNullOrEmpty(screenCaptureOutput))
                {
                    StartCoroutine(CaptureScreenAndQuit(screenCaptureOutput));
                }
                else if (!string.IsNullOrEmpty(captureOutput))
                {
                    StartCoroutine(CaptureAndQuit(captureOutput));
                }
            }
            catch (Exception exception)
            {
                fatalError = exception.ToString();
                Debug.LogError(fatalError);
                WriteFailedRolloutReportIfRequested(exception.Message);
            }
        }

        private void Update()
        {
            if (fatalError != null || runtime == null)
            {
                return;
            }
            if (Input.GetKeyDown(KeyCode.C))
            {
                SetCameraMode(!firstPerson);
            }
            if (Input.GetKeyDown(KeyCode.Space))
            {
                paused = !paused;
            }
            if (!hudManagedExternally && Input.GetKeyDown(KeyCode.H))
            {
                hudExpanded = !hudExpanded;
            }
            if (paused || episodeComplete)
            {
                ApplyVisualPose(1f);
                return;
            }

            accumulator += Time.unscaledDeltaTime * simulationSpeed;
            int guard = 0;
            try
            {
                while (accumulator >= runtime.Data.fixedDtSeconds && guard < 16)
                {
                    accumulator -= runtime.Data.fixedDtSeconds;
                    StepModel();
                    guard++;
                    string adaptiveCompletionReason;
                    if (adaptiveEpisodeDriver != null &&
                        adaptiveEpisodeDriver.TryGetEpisodeCompletion(
                            stepIndex * runtime.Data.fixedDtSeconds,
                            out adaptiveCompletionReason))
                    {
                        CompleteEpisode(adaptiveCompletionReason);
                        break;
                    }
                    if (stepIndex >= targetSteps)
                    {
                        CompleteEpisode(adaptiveEpisodeDriver == null
                            ? "60_seconds_complete" : "safety_timeout");
                        break;
                    }
                }
            }
            catch (Exception exception)
            {
                fatalError = exception.ToString();
                Debug.LogError(fatalError);
                CompleteEpisode("runtime_error");
            }
            float interpolation = episodeComplete
                ? 1f
                : Mathf.Clamp01(accumulator / runtime.Data.fixedDtSeconds);
            ApplyVisualPose(interpolation);
        }

        private void LateUpdate()
        {
            if (observerCamera == null || firstPerson || headLocal == null)
            {
                return;
            }
            Vector3 focus = headLocal.position;
            Vector3 flatForward = transform.forward;
            flatForward.y = 0f;
            flatForward = flatForward.sqrMagnitude > 0.0001f ? flatForward.normalized : Vector3.forward;
            Vector3 desired = focus - flatForward * 7.2f + Vector3.up * 3.8f;
            float blend = 1f - Mathf.Exp(-4.5f * Time.unscaledDeltaTime);
            observerCamera.transform.position = Vector3.Lerp(observerCamera.transform.position, desired, blend);
            observerCamera.transform.rotation = Quaternion.Slerp(
                observerCamera.transform.rotation,
                Quaternion.LookRotation(focus - observerCamera.transform.position, Vector3.up),
                blend);
        }

        private void BuildParticipantRig()
        {
            gameObject.name = "ParticipantRoot (AI: XZ + Yaw Only)";

            GameObject origin = new GameObject("TrackingOrigin (Fixed Upright)");
            origin.transform.SetParent(transform, false);
            origin.transform.localPosition = new Vector3(0f, runtime.Data.nominalEyeHeightM, 0f);
            trackingOrigin = origin.transform;

            GameObject head = new GameObject("HeadLocal (AI HMD Pose)");
            head.transform.SetParent(trackingOrigin, false);
            headLocal = head.transform;

            GameObject cameraObject = new GameObject("ParticipantCamera (AI First Person)");
            cameraObject.tag = "MainCamera";
            cameraObject.transform.SetParent(headLocal, false);
            participantCamera = cameraObject.AddComponent<Camera>();
            participantCamera.fieldOfView = 62f;
            participantCamera.nearClipPlane = 0.06f;
            participantCamera.farClipPlane = 300f;
            participantCamera.clearFlags = CameraClearFlags.Skybox;
            participantAudio = cameraObject.AddComponent<AudioListener>();

            Material bodyMaterial = NormalHmdDemoBootstrap.MakeMaterial(
                "Participant Body", new Color(0.11f, 0.28f, 0.38f), 0.15f, 0.58f);
            Material visorMaterial = NormalHmdDemoBootstrap.MakeMaterial(
                "Participant Visor", new Color(0.15f, 0.92f, 0.82f), 0.05f, 0.8f,
                new Color(0.04f, 0.32f, 0.27f));

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = "Body Proxy (Not EyeNavGS Ground Truth)";
            body.transform.SetParent(transform, false);
            body.transform.localPosition = new Vector3(0f, 0.83f, 0f);
            body.transform.localScale = new Vector3(0.5f, 0.72f, 0.5f);
            bodyRenderer = body.GetComponent<Renderer>();
            bodyRenderer.sharedMaterial = bodyMaterial;
            Destroy(body.GetComponent<Collider>());

            GameObject headMesh = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            headMesh.name = "HMD Head Visual";
            headMesh.transform.SetParent(headLocal, false);
            headMesh.transform.localScale = Vector3.one * 0.42f;
            headRenderer = headMesh.GetComponent<Renderer>();
            headRenderer.sharedMaterial = bodyMaterial;
            Destroy(headMesh.GetComponent<Collider>());

            GameObject visor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visor.name = "HMD Forward Marker";
            visor.transform.SetParent(headLocal, false);
            visor.transform.localPosition = new Vector3(0f, 0f, 0.225f);
            visor.transform.localScale = new Vector3(0.30f, 0.16f, 0.07f);
            visorRenderer = visor.GetComponent<Renderer>();
            visorRenderer.sharedMaterial = visorMaterial;
            Destroy(visor.GetComponent<Collider>());

            trail = gameObject.AddComponent<LineRenderer>();
            trail.name = "60 s BodyRoot Trail";
            trail.useWorldSpace = true;
            trail.widthMultiplier = 0.045f;
            trail.numCornerVertices = 2;
            trail.numCapVertices = 2;
            trail.material = visorMaterial;
            trail.startColor = new Color(0.15f, 0.92f, 0.82f, 0.9f);
            trail.endColor = new Color(0.15f, 0.55f, 0.92f, 0.2f);
        }

        private void StartEpisode(int seed, bool replay)
        {
            if (telemetry != null)
            {
                CompleteEpisode(replay ? "replay_requested" : "new_person_requested");
            }
            motionDriverEpisodeBegan = false;
            motionDriverEpisodeCompleted = false;
            episodeStartCount++;
            episodeSeed = seed;
            uint windowSeed = PortableHmdRandom.DeriveSeed(
                seed, runtime.Data.seedWindowXorConstant);
            PortableHmdRandom windowRandom = new PortableHmdRandom(windowSeed);
            int seedIndex = (int)(windowRandom.NextUInt32() %
                (uint)runtime.Data.seedWindows.Length);
            activeSeed = runtime.Data.seedWindows[seedIndex];
            runtime.Warm(activeSeed.historyRaw);
            uint derivedRootSeed = PortableHmdRandom.DeriveSeed(
                seed, runtime.Data.rootSeedXorConstant);
            uint derivedHorizontalSeed = PortableHmdRandom.DeriveSeed(
                seed, runtime.Data.horizontalSeedXorConstant);
            uint derivedVerticalSeed = PortableHmdRandom.DeriveSeed(
                seed, runtime.Data.verticalSeedXorConstant);
            rootInnovation = new SeededRootInnovation(
                derivedRootSeed,
                runtime.Data.rootInnovationRho,
                runtime.Data.rootInnovationCholesky);

            stepIndex = 0;
            episodeSafetyLimitSeconds = adaptiveEpisodeDriver == null
                ? EpisodeSeconds
                : Mathf.Clamp(adaptiveEpisodeDriver.SafetyTimeoutSeconds,
                    EpisodeSeconds + runtime.Data.fixedDtSeconds, 1800f);
            targetSteps = Mathf.CeilToInt(
                episodeSafetyLimitSeconds / runtime.Data.fixedDtSeconds);
            clippedStepCount = 0;
            clippedValueCount = 0;
            comfortEnvelopeActivationStepCount = 0;
            comfortEnvelopeActivationValueCount = 0;
            baseClippedStepCount = 0;
            baseClippedValueCount = 0;
            rootBaseClippedStepCount = 0;
            rootBaseClippedValueCount = 0;
            stabilizedBaseClippedStepCount = 0;
            stabilizedBaseClippedValueCount = 0;
            nonFiniteCount = 0;
            distanceMetres = 0f;
            maxBodyTiltDegrees = 0f;
            maxQuaternionNormError = 0f;
            cameraPoseChangedStepCount = 0;
            maxCameraHierarchyPositionErrorMetres = 0f;
            maxCameraHierarchyRotationErrorDegrees = 0f;
            maxAbsoluteCameraRollDegrees = 0f;
            platformVerticalOffset = 0f;
            previousPlatformVerticalOffset = 0f;
            platformPitchDegrees = 0f;
            previousPlatformPitchDegrees = 0f;
            platformRollDegrees = 0f;
            previousPlatformRollDegrees = 0f;
            firstDeltaCount = 0;
            lastDeltaCount = 0;
            firstPlanarSpeedEnergySum = 0f;
            lastPlanarSpeedEnergySum = 0f;
            firstPlanarSpeedEnergyCount = 0;
            lastPlanarSpeedEnergyCount = 0;
            headDeltaPairCount = 0;
            hasPreviousHeadDelta = false;
            maxRootInnovationPassthroughError = 0f;
            verticalEventCount = 0;
            completedVerticalEventCount = 0;
            horizontalEventCount = 0;
            completedHorizontalEventCount = 0;
            Array.Clear(horizontalEventAbsAmplitudeSum, 0, horizontalEventAbsAmplitudeSum.Length);
            Array.Clear(verticalEventAbsAmplitudeSum, 0, verticalEventAbsAmplitudeSum.Length);
            Array.Clear(maxHorizontalEventAbsAmplitude, 0, maxHorizontalEventAbsAmplitude.Length);
            Array.Clear(maxVerticalEventAbsAmplitude, 0, maxVerticalEventAbsAmplitude.Length);
            maxHorizontalEventEndpointResidual = 0f;
            maxVerticalEventEndpointResidual = 0f;
            verticalEventAbsAmplitudeYSum = 0f;
            verticalEventAbsAmplitudePitchSum = 0f;
            maxVerticalEventAbsAmplitudeY = 0f;
            maxVerticalEventAbsAmplitudePitch = 0f;
            maxVerticalEventAnchorClosureResidualY = 0f;
            maxVerticalEventAnchorClosureResidualPitch = 0f;
            maxAbsoluteHeadInnovation = 0f;
            Array.Clear(firstDeltaSquareSum, 0, firstDeltaSquareSum.Length);
            Array.Clear(lastDeltaSquareSum, 0, lastDeltaSquareSum.Length);
            Array.Clear(previousHeadDelta, 0, previousHeadDelta.Length);
            Array.Clear(headDeltaPreviousSum, 0, headDeltaPreviousSum.Length);
            Array.Clear(headDeltaCurrentSum, 0, headDeltaCurrentSum.Length);
            Array.Clear(headDeltaPreviousSquareSum, 0, headDeltaPreviousSquareSum.Length);
            Array.Clear(headDeltaCurrentSquareSum, 0, headDeltaCurrentSquareSum.Length);
            Array.Clear(headDeltaCrossSum, 0, headDeltaCrossSum.Length);
            Array.Clear(headDirectionReversalCount, 0, headDirectionReversalCount.Length);
            accumulator = 0f;
            episodeComplete = false;
            paused = false;

            Vector3[] spawns =
            {
                new Vector3(-7f, 0f, -7f), new Vector3(7f, 0f, -7f),
                new Vector3(-7f, 0f, 7f), new Vector3(7f, 0f, 7f),
                Vector3.zero,
            };
            bodyPosition = spawns[(int)(windowRandom.NextUInt32() % (uint)spawns.Length)];
            bodyYawDegrees = (int)(windowRandom.NextUInt32() % 8u) * 45f;
            if (motionDriver != null)
            {
                motionDriver.ResetForEpisode(seed, out bodyPosition, out bodyYawDegrees);
            }
            previousBodyPosition = bodyPosition;
            previousBodyYawDegrees = bodyYawDegrees;

            int lastOffset = (runtime.Data.historyLength - 1) * 9;
            for (int feature = 0; feature < 9; feature++)
            {
                lastApplied[feature] = Mathf.Clamp(
                    activeSeed.historyRaw[lastOffset + feature],
                    runtime.Data.stateClipLow[feature],
                    runtime.Data.stateClipHigh[feature]);
                previousApplied[feature] = lastApplied[feature];
                lastBaseFeedback[feature] = lastApplied[feature];
            }
            horizontalController = runtime.CreateHorizontalController(
                lastApplied, derivedHorizontalSeed);
            verticalController = runtime.CreateVerticalController(
                lastApplied, derivedVerticalSeed);
            SetCurrentHeadState(lastApplied);
            previousHeadLocalPosition = currentHeadLocalPosition;
            previousHeadLocalRotation = currentHeadLocalRotation;
            ApplyVisualPose(1f);
            ComputeSimulatedCameraPose(
                out previousSimulatedCameraPosition,
                out previousSimulatedCameraRotation);
            trail.positionCount = 1;
            trail.SetPosition(0, bodyPosition + Vector3.up * 0.04f);
            OpenLog(replay);
            if (motionDriver != null)
            {
                motionDriver.BeginEpisode(seed, logDirectory);
                motionDriverEpisodeBegan = true;
            }
        }

        private void StepModel()
        {
            lastStep = runtime.PredictHeads();
            float[] composed = runtime.ComposeBeforeInnovation(
                lastApplied, lastStep, horizontalController, verticalController);
            if (horizontalController.TriggeredThisStep)
            {
                horizontalEventCount++;
                float[] amplitudes = horizontalController.Amplitude;
                for (int axis = 0; axis < amplitudes.Length; axis++)
                {
                    float absolute = Mathf.Abs(amplitudes[axis]);
                    horizontalEventAbsAmplitudeSum[axis] += absolute;
                    maxHorizontalEventAbsAmplitude[axis] = Mathf.Max(
                        maxHorizontalEventAbsAmplitude[axis], absolute);
                }
            }
            if (verticalController.TriggeredThisStep)
            {
                verticalEventCount++;
                float[] amplitudes = verticalController.Amplitude;
                float absY = Mathf.Abs(amplitudes[0]);
                float absPitch = Mathf.Abs(amplitudes[1]);
                verticalEventAbsAmplitudeSum[0] += absY;
                verticalEventAbsAmplitudeSum[1] += absPitch;
                maxVerticalEventAbsAmplitude[0] = Mathf.Max(
                    maxVerticalEventAbsAmplitude[0], absY);
                maxVerticalEventAbsAmplitude[1] = Mathf.Max(
                    maxVerticalEventAbsAmplitude[1], absPitch);
                verticalEventAbsAmplitudeYSum += absY;
                verticalEventAbsAmplitudePitchSum += absPitch;
                maxVerticalEventAbsAmplitudeY = Mathf.Max(maxVerticalEventAbsAmplitudeY, absY);
                maxVerticalEventAbsAmplitudePitch = Mathf.Max(maxVerticalEventAbsAmplitudePitch, absPitch);
            }
            if (horizontalController.CompletedThisStep)
            {
                completedHorizontalEventCount++;
                float[] target = horizontalController.TargetPose;
                for (int axis = 0; axis < runtime.Data.horizontalStateIndices.Length; axis++)
                {
                    int feature = runtime.Data.horizontalStateIndices[axis];
                    maxHorizontalEventEndpointResidual = Mathf.Max(
                        maxHorizontalEventEndpointResidual,
                        Mathf.Abs(composed[feature] - target[axis]));
                }
            }
            if (verticalController.CompletedThisStep)
            {
                completedVerticalEventCount++;
                float[] target = verticalController.TargetPose;
                for (int axis = 0; axis < runtime.Data.verticalStateIndices.Length; axis++)
                {
                    int feature = runtime.Data.verticalStateIndices[axis];
                    maxVerticalEventEndpointResidual = Mathf.Max(
                        maxVerticalEventEndpointResidual,
                        Mathf.Abs(composed[feature] - target[axis]));
                }
                // Retain legacy field names in reports, but these are endpoint
                // residuals in v4 (events hold their target rather than return).
                maxVerticalEventAnchorClosureResidualY = maxVerticalEventEndpointResidual;
                maxVerticalEventAnchorClosureResidualPitch = maxVerticalEventEndpointResidual;
            }
            float[] rawRootInnovation = rootInnovation.Next(1f);
            float[] generatedRootInnovation = new float[3];
            for (int index = 0; index < 3; index++)
                generatedRootInnovation[index] = rawRootInnovation[index] * noiseScale;
            bool baseClipped = false;
            bool rootBaseClipped = false;
            bool stabilizedBaseClipped = false;
            bool appliedClipped = false;
            bool unexpectedClip = false;
            bool comfortEnvelopeActivated =
                horizontalController.ComfortEnvelopeActivatedThisStep ||
                verticalController.ComfortEnvelopeActivatedThisStep;
            if (comfortEnvelopeActivated) comfortEnvelopeActivationValueCount++;
            bool[] safetyClipped;
            float[] applied = runtime.ApplyInnovationAndSafety(
                composed, generatedRootInnovation, out safetyClipped);
            for (int feature = 0; feature < 9; feature++)
            {
                bool composedOutside = composed[feature] < runtime.Data.stateClipLow[feature] ||
                    composed[feature] > runtime.Data.stateClipHigh[feature];
                if (composedOutside && feature != 8)
                {
                    baseClipped = true;
                    baseClippedValueCount++;
                    stabilizedBaseClipped = true;
                    stabilizedBaseClippedValueCount++;
                    if (feature < 3)
                    {
                        rootBaseClipped = true;
                        rootBaseClippedValueCount++;
                    }
                }
                float innovationValue = feature < 3
                    ? generatedRootInnovation[feature]
                    : 0f;
                lastRaw[feature] = composed[feature];
                lastBaseFeedback[feature] = composed[feature];
                lastStabilizedBase[feature] = composed[feature];
                lastDriftCorrection[feature] = applied[feature] -
                    composed[feature] - innovationValue;
                lastRawInnovation[feature] = feature < 3
                    ? rawRootInnovation[feature]
                    : 0f;
                lastAppliedInnovation[feature] = feature < 3
                    ? applied[feature] - composed[feature]
                    : 0f;
                if (feature < 3 && !safetyClipped[feature])
                {
                    float expectedCandidate = (float)(composed[feature] + innovationValue);
                    maxRootInnovationPassthroughError = Mathf.Max(
                        maxRootInnovationPassthroughError,
                        Mathf.Abs(applied[feature] - expectedCandidate));
                }
                else
                {
                    maxAbsoluteHeadInnovation = Mathf.Max(
                        maxAbsoluteHeadInnovation,
                        Mathf.Abs(lastRawInnovation[feature]));
                }
                if (!float.IsFinite(applied[feature]))
                {
                    nonFiniteCount++;
                    throw new InvalidOperationException(
                        "Non-finite generated state at step " + stepIndex + ", feature " + feature + ".");
                }
                if (safetyClipped[feature])
                {
                    appliedClipped = true;
                    if (feature == 8)
                    {
                        comfortEnvelopeActivated = true;
                        comfortEnvelopeActivationValueCount++;
                    }
                    else
                    {
                        unexpectedClip = true;
                        clippedValueCount++;
                    }
                }
                lastApplied[feature] = applied[feature];
            }
            if (baseClipped)
            {
                baseClippedStepCount++;
            }
            if (rootBaseClipped)
            {
                rootBaseClippedStepCount++;
            }
            if (stabilizedBaseClipped)
            {
                stabilizedBaseClippedStepCount++;
            }
            if (unexpectedClip)
            {
                clippedStepCount++;
            }
            if (comfortEnvelopeActivated)
            {
                comfortEnvelopeActivationStepCount++;
            }

            float dt = runtime.Data.fixedDtSeconds;
            previousBodyPosition = bodyPosition;
            previousBodyYawDegrees = bodyYawDegrees;
            previousPlatformVerticalOffset = platformVerticalOffset;
            previousPlatformPitchDegrees = platformPitchDegrees;
            previousPlatformRollDegrees = platformRollDegrees;
            previousHeadLocalPosition = currentHeadLocalPosition;
            previousHeadLocalRotation = currentHeadLocalRotation;
            float learnedRight = lastApplied[0];
            float learnedForward = lastApplied[1];
            float learnedYawRate = lastApplied[2];
            float executedRight = learnedRight;
            float executedForward = learnedForward;
            float executedYawRate = learnedYawRate;
            if (motionDriver != null)
            {
                motionDriver.GuideRootMotion(
                    stepIndex * dt,
                    dt,
                    bodyPosition,
                    bodyYawDegrees,
                    learnedRight,
                    learnedForward,
                    learnedYawRate,
                    out executedRight,
                    out executedForward,
                    out executedYawRate);
                if (!float.IsFinite(executedRight) ||
                    !float.IsFinite(executedForward) ||
                    !float.IsFinite(executedYawRate))
                {
                    throw new InvalidOperationException(
                        "Root motion driver produced a non-finite command at step " + stepIndex + ".");
                }
            }
            float requestedPlatformVertical = 0f;
            float requestedPlatformPitch = 0f;
            float requestedPlatformRoll = 0f;
            bool hasPlatformPose = motionDriver != null &&
                motionDriver.TryGetPlatformPose(
                    stepIndex * dt,
                    out requestedPlatformVertical,
                    out requestedPlatformPitch,
                    out requestedPlatformRoll);
            if (hasPlatformPose)
            {
                if (!float.IsFinite(requestedPlatformVertical) ||
                    !float.IsFinite(requestedPlatformPitch) ||
                    !float.IsFinite(requestedPlatformRoll))
                {
                    throw new InvalidOperationException(
                        "Root motion driver produced a non-finite platform pose at step " + stepIndex + ".");
                }
                platformVerticalOffset = Mathf.Clamp(requestedPlatformVertical, -3f, 6f);
                platformPitchDegrees = Mathf.Clamp(requestedPlatformPitch, -35f, 35f);
                platformRollDegrees = Mathf.Clamp(requestedPlatformRoll, -9f, 9f);
            }
            else
            {
                platformVerticalOffset = 0f;
                platformPitchDegrees = 0f;
                platformRollDegrees = 0f;
            }
            float yawDeltaDegrees = -executedYawRate * Mathf.Rad2Deg * dt;
            Quaternion midpointRotation = Quaternion.Euler(0f, bodyYawDegrees + yawDeltaDegrees * 0.5f, 0f);
            Vector3 localVelocity = new Vector3(executedRight, 0f, executedForward);
            Vector3 movement = midpointRotation * localVelocity * dt;
            bodyPosition += movement;
            bodyYawDegrees = Mathf.Repeat(bodyYawDegrees + yawDeltaDegrees + 180f, 360f) - 180f;
            distanceMetres += movement.magnitude;
            SetCurrentHeadState(lastApplied);
            // v4 deployment contract: the final safe state is the next input
            // input. This includes joint BodyRoot innovation, while every one
            // of the six HeadLocal channels remains free of per-frame noise.
            runtime.AdvanceExecuted(lastApplied);
            stepIndex++;
            ValidateSimulatedCameraPose();

            // BodyRoot is constructed from yaw only, so this checks the exact
            // simulation pose rather than the interpolated presentation pose.
            Vector3 simulatedUp = Quaternion.Euler(0f, bodyYawDegrees, 0f) * Vector3.up;
            maxBodyTiltDegrees = Mathf.Max(maxBodyTiltDegrees, Vector3.Angle(simulatedUp, Vector3.up));
            AccumulateDynamics(lastApplied, stepIndex * dt);
            Array.Copy(lastApplied, previousApplied, 9);
            AppendTrail();
            WriteTelemetry(unexpectedClip, comfortEnvelopeActivated, appliedClipped);
            if (motionDriver != null &&
                motionDriverEpisodeBegan &&
                !motionDriverEpisodeCompleted)
            {
                Vector3 simulatedCameraPosition;
                Quaternion simulatedCameraRotation;
                ComputeSimulatedCameraPose(
                    out simulatedCameraPosition,
                    out simulatedCameraRotation);
                motionDriver.ObserveSimulationStep(new SceneAwareStepSample
                {
                    stepIndex = stepIndex,
                    simulatedTime = stepIndex * dt,
                    deltaTime = dt,
                    bodyPosition = bodyPosition,
                    bodyYawDegrees = bodyYawDegrees,
                    cameraPosition = simulatedCameraPosition,
                    cameraRotation = simulatedCameraRotation,
                    headLocalPosition = currentHeadLocalPosition,
                    headLocalRotation = currentHeadLocalRotation,
                    learnedRight = learnedRight,
                    learnedForward = learnedForward,
                    learnedYawRate = learnedYawRate,
                    executedRight = executedRight,
                    executedForward = executedForward,
                    executedYawRate = executedYawRate,
                });
            }
        }

        private void SetCurrentHeadState(float[] state)
        {
            currentHeadLocalPosition = new Vector3(state[3], state[4], state[5]);
            Vector3 unityRotationVector = new Vector3(-state[6], -state[7], state[8]);
            float angleRadians = unityRotationVector.magnitude;
            Quaternion localRotation = angleRadians > 0.0000001f
                ? Quaternion.AngleAxis(angleRadians * Mathf.Rad2Deg, unityRotationVector / angleRadians)
                : Quaternion.identity;
            float norm = Mathf.Sqrt(
                localRotation.x * localRotation.x + localRotation.y * localRotation.y +
                localRotation.z * localRotation.z + localRotation.w * localRotation.w);
            maxQuaternionNormError = Mathf.Max(maxQuaternionNormError, Mathf.Abs(norm - 1f));
            currentHeadLocalRotation = localRotation.normalized;
        }

        private void ApplyVisualPose(float interpolation)
        {
            // Presentation-only interpolation bridges the 37 Hz model clock to
            // the display refresh. None of these transforms are fed back to the
            // model or used for fixed-step telemetry.
            Vector3 visualBodyPosition = Vector3.Lerp(
                previousBodyPosition, bodyPosition, interpolation);
            float visualBodyYaw = Mathf.LerpAngle(
                previousBodyYawDegrees, bodyYawDegrees, interpolation);
            float visualPlatformVertical = Mathf.Lerp(
                previousPlatformVerticalOffset, platformVerticalOffset, interpolation);
            float visualPlatformPitch = Mathf.LerpAngle(
                previousPlatformPitchDegrees, platformPitchDegrees, interpolation);
            float visualPlatformRoll = Mathf.LerpAngle(
                previousPlatformRollDegrees, platformRollDegrees, interpolation);
            Quaternion visualRootRotation = Quaternion.Euler(
                visualPlatformPitch, visualBodyYaw, visualPlatformRoll);
            transform.SetPositionAndRotation(
                visualBodyPosition + Vector3.up * visualPlatformVertical,
                visualRootRotation);
            trackingOrigin.localRotation = Quaternion.identity;
            Vector3 visualHead = Vector3.Lerp(
                previousHeadLocalPosition, currentHeadLocalPosition, interpolation);
            // The event controller has already generated a quintic
            // minimum-jerk trajectory. Applying SmoothStep again inside every
            // 27 ms model interval would force the displayed vertical speed
            // toward zero at every tick and create a visible stop/start bob.
            // The same linear presentation interpolation is therefore used on
            // all three axes; it preserves the sampled minimum-jerk curve.
            headLocal.localPosition = visualHead;
            headLocal.localRotation = Quaternion.Slerp(
                previousHeadLocalRotation, currentHeadLocalRotation, interpolation);
        }

        private void ComputeSimulatedCameraPose(out Vector3 position, out Quaternion rotation)
        {
            Quaternion rootRotation = Quaternion.Euler(
                platformPitchDegrees, bodyYawDegrees, platformRollDegrees);
            position = bodyPosition + Vector3.up * platformVerticalOffset + rootRotation *
                (Vector3.up * runtime.Data.nominalEyeHeightM + currentHeadLocalPosition);
            rotation = (rootRotation * currentHeadLocalRotation).normalized;
        }

        private void ValidateSimulatedCameraPose()
        {
            Vector3 position;
            Quaternion rotation;
            ComputeSimulatedCameraPose(out position, out rotation);

            Matrix4x4 rootToWorld = Matrix4x4.TRS(
                bodyPosition + Vector3.up * platformVerticalOffset,
                Quaternion.Euler(platformPitchDegrees, bodyYawDegrees, platformRollDegrees),
                Vector3.one);
            Vector3 hierarchyLocalPosition = new Vector3(
                currentHeadLocalPosition.x,
                runtime.Data.nominalEyeHeightM + currentHeadLocalPosition.y,
                currentHeadLocalPosition.z);
            Vector3 hierarchyPosition = rootToWorld.MultiplyPoint3x4(hierarchyLocalPosition);
            Quaternion hierarchyRotation = (
                Quaternion.Euler(platformPitchDegrees, bodyYawDegrees, platformRollDegrees) *
                currentHeadLocalRotation).normalized;
            maxCameraHierarchyPositionErrorMetres = Mathf.Max(
                maxCameraHierarchyPositionErrorMetres,
                Vector3.Distance(position, hierarchyPosition));
            maxCameraHierarchyRotationErrorDegrees = Mathf.Max(
                maxCameraHierarchyRotationErrorDegrees,
                Quaternion.Angle(rotation, hierarchyRotation));
            // Project world-up onto the camera screen basis.  atan2(right.y,
            // up.y) measures visual-horizon roll even when pitch is non-zero,
            // without Euler-angle wraparound.
            Vector3 cameraRight = rotation * Vector3.right;
            Vector3 cameraUp = rotation * Vector3.up;
            float cameraRollDegrees = Mathf.Abs(
                Mathf.Atan2(cameraRight.y, cameraUp.y) * Mathf.Rad2Deg);
            maxAbsoluteCameraRollDegrees = Mathf.Max(
                maxAbsoluteCameraRollDegrees, cameraRollDegrees);

            if (Vector3.Distance(position, previousSimulatedCameraPosition) > 0.000001f ||
                Quaternion.Angle(rotation, previousSimulatedCameraRotation) > 0.0001f)
            {
                cameraPoseChangedStepCount++;
            }
            previousSimulatedCameraPosition = position;
            previousSimulatedCameraRotation = rotation;
        }

        private void AccumulateDynamics(float[] state, float elapsed)
        {
            float planarSpeedEnergy = state[0] * state[0] + state[1] * state[1];
            if (elapsed <= 10f)
            {
                firstDeltaCount++;
                firstPlanarSpeedEnergyCount++;
                firstPlanarSpeedEnergySum += planarSpeedEnergy;
                for (int index = 0; index < 9; index++)
                {
                    float delta = state[index] - previousApplied[index];
                    firstDeltaSquareSum[index] += delta * delta;
                }
            }
            if (elapsed >= EpisodeSeconds - 10f)
            {
                lastDeltaCount++;
                lastPlanarSpeedEnergyCount++;
                lastPlanarSpeedEnergySum += planarSpeedEnergy;
                for (int index = 0; index < 9; index++)
                {
                    float delta = state[index] - previousApplied[index];
                    lastDeltaSquareSum[index] += delta * delta;
                }
            }

            float[] currentHeadDelta = new float[6];
            for (int index = 0; index < currentHeadDelta.Length; index++)
            {
                currentHeadDelta[index] = state[3 + index] - previousApplied[3 + index];
            }
            // Telemetry starts at generated step 1, so its first observable
            // delta is step 1 -> step 2. Exclude anchor -> step 1 here as well
            // so the built-in gate and the external CSV validator are exact.
            if (stepIndex <= 1)
            {
                return;
            }
            if (hasPreviousHeadDelta)
            {
                headDeltaPairCount++;
                for (int index = 0; index < currentHeadDelta.Length; index++)
                {
                    double previous = previousHeadDelta[index];
                    double current = currentHeadDelta[index];
                    headDeltaPreviousSum[index] += previous;
                    headDeltaCurrentSum[index] += current;
                    headDeltaPreviousSquareSum[index] += previous * previous;
                    headDeltaCurrentSquareSum[index] += current * current;
                    headDeltaCrossSum[index] += previous * current;
                    if (previous * current < 0.0)
                    {
                        headDirectionReversalCount[index]++;
                    }
                }
            }
            Array.Copy(currentHeadDelta, previousHeadDelta, currentHeadDelta.Length);
            hasPreviousHeadDelta = true;
        }

        private float[] FinishHeadDeltaLag1Correlation()
        {
            float[] result = new float[6];
            if (headDeltaPairCount < 2) return result;
            double count = headDeltaPairCount;
            for (int index = 0; index < result.Length; index++)
            {
                double covariance = count * headDeltaCrossSum[index] -
                    headDeltaPreviousSum[index] * headDeltaCurrentSum[index];
                double previousVariance = count * headDeltaPreviousSquareSum[index] -
                    headDeltaPreviousSum[index] * headDeltaPreviousSum[index];
                double currentVariance = count * headDeltaCurrentSquareSum[index] -
                    headDeltaCurrentSum[index] * headDeltaCurrentSum[index];
                double denominator = Math.Sqrt(Math.Max(0.0, previousVariance * currentVariance));
                result[index] = denominator > 1e-20
                    ? (float)(covariance / denominator)
                    : 0f;
            }
            return result;
        }

        private float[] FinishHeadDirectionReversalsPerSecond()
        {
            float[] result = new float[6];
            // N lag pairs arise from N+1 observable telemetry deltas.
            float duration = (headDeltaPairCount + 1) * runtime.Data.fixedDtSeconds;
            if (!(duration > 0f)) return result;
            for (int index = 0; index < result.Length; index++)
            {
                result[index] = headDirectionReversalCount[index] / duration;
            }
            return result;
        }

        private void AppendTrail()
        {
            // Decimate to about 10 Hz so the LineRenderer remains lightweight.
            int stride = Mathf.Max(1, Mathf.RoundToInt(0.1f / runtime.Data.fixedDtSeconds));
            if (stepIndex % stride != 0 && stepIndex != targetSteps)
            {
                return;
            }
            int index = trail.positionCount;
            trail.positionCount = index + 1;
            trail.SetPosition(index, bodyPosition + Vector3.up * 0.04f);
        }

        private void SetCameraMode(bool useFirstPerson)
        {
            firstPerson = useFirstPerson;
            if (participantCamera != null) participantCamera.enabled = useFirstPerson;
            if (observerCamera != null) observerCamera.enabled = !useFirstPerson;
            if (participantAudio != null) participantAudio.enabled = useFirstPerson;
            if (observerAudio == null && observerCamera != null)
            {
                observerAudio = observerCamera.GetComponent<AudioListener>();
            }
            if (observerAudio != null) observerAudio.enabled = !useFirstPerson;
            if (headRenderer != null) headRenderer.enabled = !useFirstPerson;
            if (visorRenderer != null) visorRenderer.enabled = !useFirstPerson;
            if (bodyRenderer != null) bodyRenderer.enabled = !useFirstPerson;
        }

        private void OpenLog(bool replay)
        {
            episodeStartedUtc = DateTime.UtcNow;
            string requestedRoot = ReadArgumentValue("--log-root=");
            string root = string.IsNullOrEmpty(requestedRoot)
                ? Path.Combine(Application.persistentDataPath, "NormalHmdAgentLogs")
                : Path.GetFullPath(requestedRoot);
            string folder = episodeStartedUtc.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture) +
                "_seed_" + episodeSeed.ToString(CultureInfo.InvariantCulture);
            logDirectory = Path.Combine(root, folder);
            Directory.CreateDirectory(logDirectory);
            telemetryPath = Path.Combine(logDirectory, "telemetry.csv");
            telemetry = new StreamWriter(telemetryPath, false, new UTF8Encoding(false));

            StringBuilder header = new StringBuilder(
                "step,time_s,body_x_m,body_y_m,body_z_m,body_yaw_deg," +
                "head_local_x_m,head_local_y_offset_m,head_local_z_m," +
                "camera_x_m,camera_y_m,camera_z_m,camera_qx,camera_qy,camera_qz,camera_qw," +
                "camera_forward_x,camera_forward_y,camera_forward_z," +
                "unexpected_clip,comfort_envelope_activation,any_safety_clip," +
                "horizontal_probability,horizontal_eligible,horizontal_trigger," +
                "horizontal_active,horizontal_completed,horizontal_event_id,horizontal_phase," +
                "horizontal_cooldown_steps,horizontal_duration_steps," +
                "horizontal_target_x_m,horizontal_target_z_m,horizontal_target_yaw_rad," +
                "horizontal_target_roll_rad,horizontal_amplitude_x_m,horizontal_amplitude_z_m," +
                "horizontal_amplitude_yaw_rad,horizontal_amplitude_roll_rad," +
                "vertical_probability,vertical_eligible,vertical_trigger,vertical_active," +
                "vertical_completed,vertical_event_id,vertical_phase,vertical_cooldown_steps," +
                "vertical_duration_steps,vertical_target_y_m,vertical_target_pitch_rad," +
                "vertical_amplitude_y_m,vertical_amplitude_pitch_rad");
            string[] headNames =
            {
                "root_x_norm", "root_z_norm", "root_yaw_norm",
                "h_pose_x_norm", "h_pose_z_norm", "h_pose_yaw_norm", "h_pose_roll_norm",
                "h_hazard_logit", "h_param_mean_x", "h_param_mean_z", "h_param_mean_yaw",
                "h_param_mean_roll", "h_param_mean_logduration", "h_param_logstd_x",
                "h_param_logstd_z", "h_param_logstd_yaw", "h_param_logstd_roll",
                "h_param_logstd_logduration", "v_hazard_logit", "v_param_mean_y",
                "v_param_mean_pitch", "v_param_mean_logduration", "v_param_logstd_y",
                "v_param_logstd_pitch", "v_param_logstd_logduration"
            };
            foreach (string name in headNames) header.Append(",head_").Append(name);
            for (int i = 0; i < 9; i++) header.Append(",base_raw_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",base_feedback_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",stabilized_base_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",drift_correction_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",innovation_raw_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",innovation_applied_").Append(runtime.Data.stateFeatureNames[i]);
            for (int i = 0; i < 9; i++) header.Append(",applied_").Append(runtime.Data.stateFeatureNames[i]);
            telemetry.WriteLine(header.ToString());

            RunManifest manifest = new RunManifest
            {
                product = ProductLabel,
                schemaVersion = runtime.Data.schemaVersion,
                runSeed = episodeSeed,
                replay = replay,
                seedTrajectory = activeSeed.trajectoryId,
                seedSplit = activeSeed.split,
                seedStartFrame = activeSeed.startFrame,
                historyFrames = runtime.Data.historyLength,
                targetSeconds = episodeSafetyLimitSeconds,
                targetSteps = targetSteps,
                fixedDtSeconds = runtime.Data.fixedDtSeconds,
                bodyDefinition = "EyeNavGS-derived causal proxy: world XZ + yaw only; not measured torso",
                headDefinition = "HeadLocal position + rotation vector relative to BodyRoot proxy",
                recurrentFeedbackPolicy = runtime.Data.runtimeFeedbackContract,
                deploymentStabilizer = "event targets use learned pose bounds intersected with the deployment safety envelope; display interpolation does not overshoot",
                rootYawDcRemovalTauSeconds = 0f,
                headPoseDcRemovalTauSeconds = 0f,
                headPoseAnchor = "independent H/V absolute endpoints and durations are sampled once at onset and held",
                headPoseAnchorMarginSigma = 0f,
                rawHeadPoseAnchor = new[] { lastApplied[3], lastApplied[4], lastApplied[5], lastApplied[6], lastApplied[7], lastApplied[8] },
                clippedHeadPoseAnchor = new[] { lastApplied[3], lastApplied[4], lastApplied[5], lastApplied[6], lastApplied[7], lastApplied[8] },
                stochasticInnovation = "joint seeded BodyRoot[0:3] AR(1), safety-clipped and recurrently fed back; HeadLocal[3:9] exactly zero stochastic innovation",
                headInnovationPolicy = runtime.Data.headInnovationPolicy,
                headInnovationSmoothingTauSeconds = 0f,
                headInnovationSmoothingGain = 0f,
                verticalEventController = "independent eligible Bernoulli hazards; quintic minimum-jerk absolute target transition; endpoint hold; cooldown=4",
                verticalEventThreshold = -1f,
                verticalDurationMinSteps = runtime.Data.durationMinSteps,
                verticalDurationMaxSteps = runtime.Data.durationMaxSteps,
                rootInnovationRho = runtime.Data.rootInnovationRho,
                rootInnovationCholesky = runtime.Data.rootInnovationCholesky,
                rootInnovationFeedsBack = runtime.Data.rootInnovationFeedsBack,
                noiseScale = noiseScale,
                sourceModelSha256 = runtime.Data.sourceModelSha256,
                exportedModelSha256 = runtime.ExportSha256,
                datasetFingerprintSha256 = runtime.Data.datasetFingerprintSha256,
                unityModelGoldenMaxAbsError = runtime.GoldenMaxAbsError,
                unityEventGoldenMaxAbsError = runtime.EventGoldenMaxAbsError,
                unityInnovationGoldenMaxAbsError = runtime.InnovationGoldenMaxAbsError,
                hasCombinedStochasticGolden = runtime.HasCombinedStochasticGolden,
                unityCombinedStochasticGoldenMaxAbsError = runtime.CombinedStochasticGoldenMaxAbsError,
                startedAtUtc = episodeStartedUtc.ToString("O"),
            };
            File.WriteAllText(
                Path.Combine(logDirectory, "manifest.json"),
                JsonUtility.ToJson(manifest, true),
                new UTF8Encoding(false));
        }

        private void WriteTelemetry(bool unexpectedClip, bool comfortEnvelopeActivated,
            bool appliedClipped)
        {
            if (telemetry == null) return;
            Vector3 simulatedCameraPosition;
            Quaternion q;
            ComputeSimulatedCameraPose(out simulatedCameraPosition, out q);
            Vector3 forward = q * Vector3.forward;
            StringBuilder row = new StringBuilder();
            row.Append(stepIndex.ToString(CultureInfo.InvariantCulture));
            AppendCsv(row, stepIndex * runtime.Data.fixedDtSeconds);
            AppendCsv(row, bodyPosition.x); AppendCsv(row, bodyPosition.y); AppendCsv(row, bodyPosition.z);
            AppendCsv(row, bodyYawDegrees);
            AppendCsv(row, currentHeadLocalPosition.x); AppendCsv(row, currentHeadLocalPosition.y); AppendCsv(row, currentHeadLocalPosition.z);
            AppendCsv(row, simulatedCameraPosition.x); AppendCsv(row, simulatedCameraPosition.y); AppendCsv(row, simulatedCameraPosition.z);
            AppendCsv(row, q.x); AppendCsv(row, q.y); AppendCsv(row, q.z); AppendCsv(row, q.w);
            AppendCsv(row, forward.x); AppendCsv(row, forward.y); AppendCsv(row, forward.z);
            AppendCsv(row, unexpectedClip ? 1 : 0);
            AppendCsv(row, comfortEnvelopeActivated ? 1 : 0);
            AppendCsv(row, appliedClipped ? 1 : 0);
            float[] horizontalTarget = horizontalController.TargetPose;
            float[] horizontalAmplitude = horizontalController.Amplitude;
            float[] verticalTarget = verticalController.TargetPose;
            float[] verticalAmplitude = verticalController.Amplitude;
            AppendCsv(row, horizontalController.Probability);
            AppendCsv(row, horizontalController.EligibleThisStep ? 1 : 0);
            AppendCsv(row, horizontalController.TriggeredThisStep ? 1 : 0);
            AppendCsv(row, horizontalController.Active ? 1 : 0);
            AppendCsv(row, horizontalController.CompletedThisStep ? 1 : 0);
            AppendCsv(row, horizontalController.EventId);
            AppendCsv(row, horizontalController.Phase);
            AppendCsv(row, horizontalController.CooldownRemaining);
            AppendCsv(row, horizontalController.DurationSteps);
            for (int axis = 0; axis < 4; axis++) AppendCsv(row, horizontalTarget[axis]);
            for (int axis = 0; axis < 4; axis++) AppendCsv(row, horizontalAmplitude[axis]);
            AppendCsv(row, verticalController.Probability);
            AppendCsv(row, verticalController.EligibleThisStep ? 1 : 0);
            AppendCsv(row, verticalController.TriggeredThisStep ? 1 : 0);
            AppendCsv(row, verticalController.Active ? 1 : 0);
            AppendCsv(row, verticalController.CompletedThisStep ? 1 : 0);
            AppendCsv(row, verticalController.EventId);
            AppendCsv(row, verticalController.Phase);
            AppendCsv(row, verticalController.CooldownRemaining);
            AppendCsv(row, verticalController.DurationSteps);
            for (int axis = 0; axis < 2; axis++) AppendCsv(row, verticalTarget[axis]);
            for (int axis = 0; axis < 2; axis++) AppendCsv(row, verticalAmplitude[axis]);
            float[] rawHeads = lastStep.RawHeads();
            for (int index = 0; index < rawHeads.Length; index++) AppendCsv(row, rawHeads[index]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastRaw[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastBaseFeedback[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastStabilizedBase[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastDriftCorrection[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastRawInnovation[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastAppliedInnovation[i]);
            for (int i = 0; i < 9; i++) AppendCsv(row, lastApplied[i]);
            telemetry.WriteLine(row.ToString());
            if (stepIndex % 128 == 0) telemetry.Flush();
        }

        private void CompleteEpisode(string reason)
        {
            if (episodeComplete && telemetry == null)
            {
                CompleteMotionDriverEpisode(reason);
                return;
            }
            episodeComplete = true;
            try
            {
                telemetry?.Flush();
                telemetry?.Dispose();
                telemetry = null;
                if (logDirectory == "not started") return;
                RunSummary summary = BuildSummary(reason);
                if (File.Exists(telemetryPath)) summary.telemetrySha256 = Sha256File(telemetryPath);
                File.WriteAllText(
                    Path.Combine(logDirectory, "summary.json"),
                    JsonUtility.ToJson(summary, true),
                    new UTF8Encoding(false));
            }
            finally
            {
                CompleteMotionDriverEpisode(reason);
            }
        }

        private void CompleteMotionDriverEpisode(string reason)
        {
            if (motionDriver == null ||
                !motionDriverEpisodeBegan ||
                motionDriverEpisodeCompleted)
            {
                return;
            }

            // Mark before invoking external code so a re-entrant shutdown or a
            // driver exception can never result in a duplicate completion.
            motionDriverEpisodeCompleted = true;
            try
            {
                motionDriver.CompleteEpisode(reason);
            }
            catch (Exception exception)
            {
                Debug.LogError("Root motion driver completion failed: " + exception);
            }
        }

        private RunSummary BuildSummary(string reason)
        {
            float[] firstDeltaRms = FinishRms(firstDeltaSquareSum, firstDeltaCount);
            float[] lastDeltaRms = FinishRms(lastDeltaSquareSum, lastDeltaCount);
            float firstEnergy = firstPlanarSpeedEnergyCount > 0
                ? firstPlanarSpeedEnergySum / firstPlanarSpeedEnergyCount : 0f;
            float lastEnergy = lastPlanarSpeedEnergyCount > 0
                ? lastPlanarSpeedEnergySum / lastPlanarSpeedEnergyCount : 0f;
            float[] headDeltaLag1Correlation = FinishHeadDeltaLag1Correlation();
            float[] headDirectionReversalsPerSecond = FinishHeadDirectionReversalsPerSecond();
            return new RunSummary
            {
                reason = reason,
                steps = stepIndex,
                targetSteps = targetSteps,
                simulatedSeconds = stepIndex * (runtime == null ? 0f : runtime.Data.fixedDtSeconds),
                uninterrupted = episodeStartCount == 1,
                episodeStartCount = episodeStartCount,
                nonFiniteCount = nonFiniteCount,
                clippedStepCount = clippedStepCount,
                clippedValueCount = clippedValueCount,
                clipStepRate = stepIndex > 0 ? (float)clippedStepCount / stepIndex : 0f,
                comfortEnvelopeActivationStepCount = comfortEnvelopeActivationStepCount,
                comfortEnvelopeActivationValueCount = comfortEnvelopeActivationValueCount,
                comfortEnvelopeActivationRate = stepIndex > 0
                    ? (float)comfortEnvelopeActivationStepCount / stepIndex : 0f,
                baseClippedStepCount = baseClippedStepCount,
                baseClippedValueCount = baseClippedValueCount,
                baseClipStepRate = stepIndex > 0 ? (float)baseClippedStepCount / stepIndex : 0f,
                rootBaseClippedStepCount = rootBaseClippedStepCount,
                rootBaseClippedValueCount = rootBaseClippedValueCount,
                rootBaseClipStepRate = stepIndex > 0
                    ? (float)rootBaseClippedStepCount / stepIndex : 0f,
                stabilizedBaseClippedStepCount = stabilizedBaseClippedStepCount,
                stabilizedBaseClippedValueCount = stabilizedBaseClippedValueCount,
                stabilizedBaseClipStepRate = stepIndex > 0
                    ? (float)stabilizedBaseClippedStepCount / stepIndex : 0f,
                distanceMetres = distanceMetres,
                maxBodyTiltDegrees = maxBodyTiltDegrees,
                maxQuaternionNormError = maxQuaternionNormError,
                cameraPoseChangedStepCount = cameraPoseChangedStepCount,
                maxCameraHierarchyPositionErrorMetres = maxCameraHierarchyPositionErrorMetres,
                maxCameraHierarchyRotationErrorDegrees = maxCameraHierarchyRotationErrorDegrees,
                maxAbsoluteCameraRollDegrees = maxAbsoluteCameraRollDegrees,
                first10SecondsDeltaRms = firstDeltaRms,
                last10SecondsDeltaRms = lastDeltaRms,
                first10SecondsNormalizedDeltaRms = NormalizedCombinedRms(firstDeltaRms),
                last10SecondsNormalizedDeltaRms = NormalizedCombinedRms(lastDeltaRms),
                first10SecondsPlanarSpeedEnergy = firstEnergy,
                last10SecondsPlanarSpeedEnergy = lastEnergy,
                lastFirstPlanarSpeedEnergyRatio = firstEnergy > 0.000000001f
                    ? lastEnergy / firstEnergy : -1f,
                headDeltaLag1Correlation = headDeltaLag1Correlation,
                headDirectionReversalsPerSecond = headDirectionReversalsPerSecond,
                maxRootInnovationPassthroughError = maxRootInnovationPassthroughError,
                horizontalEventCount = horizontalEventCount,
                completedHorizontalEventCount = completedHorizontalEventCount,
                meanHorizontalEventAbsAmplitude = DivideArray(
                    horizontalEventAbsAmplitudeSum, horizontalEventCount),
                maxHorizontalEventAbsAmplitude =
                    (float[])maxHorizontalEventAbsAmplitude.Clone(),
                maxHorizontalEventEndpointResidual = maxHorizontalEventEndpointResidual,
                verticalEventCount = verticalEventCount,
                completedVerticalEventCount = completedVerticalEventCount,
                meanVerticalEventAbsAmplitude = DivideArray(
                    verticalEventAbsAmplitudeSum, verticalEventCount),
                maxVerticalEventAbsAmplitude =
                    (float[])maxVerticalEventAbsAmplitude.Clone(),
                maxVerticalEventEndpointResidual = maxVerticalEventEndpointResidual,
                meanVerticalEventAbsAmplitudeY = verticalEventCount > 0
                    ? verticalEventAbsAmplitudeYSum / verticalEventCount : 0f,
                meanVerticalEventAbsAmplitudePitch = verticalEventCount > 0
                    ? verticalEventAbsAmplitudePitchSum / verticalEventCount : 0f,
                maxVerticalEventAbsAmplitudeY = maxVerticalEventAbsAmplitudeY,
                maxVerticalEventAbsAmplitudePitch = maxVerticalEventAbsAmplitudePitch,
                maxVerticalEventAnchorClosureResidualY = maxVerticalEventAnchorClosureResidualY,
                maxVerticalEventAnchorClosureResidualPitch = maxVerticalEventAnchorClosureResidualPitch,
                maxAbsoluteHeadInnovation = maxAbsoluteHeadInnovation,
                headInnovationExactlyZero = maxAbsoluteHeadInnovation == 0f,
                telemetrySha256 = "",
                endedAtUtc = DateTime.UtcNow.ToString("O"),
            };
        }

        private void RunRolloutTest(int seed)
        {
            StartEpisode(seed, false);
            while (stepIndex < targetSteps)
            {
                StepModel();
            }
            RunSummary summary = BuildSummary("rollout_test");
            bool productionGateApplied = IsProductionModel();
            float maximumClipStepRate = productionGateApplied ? ProductionMaximumClipStepRate : 0.05f;
            float minimumNormalizedDeltaRms = productionGateApplied
                ? ProductionMinimumNormalizedDeltaRms : 0.001f;
            float minimumDistance = productionGateApplied
                ? ProductionMinimumDistanceMetres : 0.05f;
            float maximumDistance = productionGateApplied
                ? ProductionMaximumDistanceMetres : 100f;
            float maximumAbsoluteCameraRollDegrees = productionGateApplied
                ? ProductionMaximumAbsoluteCameraRollDegrees : 45f;
            bool deltaWindowsNonFrozen =
                summary.first10SecondsNormalizedDeltaRms >= minimumNormalizedDeltaRms &&
                summary.last10SecondsNormalizedDeltaRms >= minimumNormalizedDeltaRms;
            bool stabilizedBaseClipAcceptable =
                summary.stabilizedBaseClipStepRate <= maximumClipStepRate;
            bool rootBaseClipAcceptable = summary.rootBaseClipStepRate <= maximumClipStepRate;
            bool planarSpeedEnergyDiagnosticFinite =
                float.IsFinite(summary.lastFirstPlanarSpeedEnergyRatio);
            bool distanceReasonable =
                distanceMetres >= minimumDistance && distanceMetres <= maximumDistance;
            bool cameraPoseChanges = cameraPoseChangedStepCount >= targetSteps / 10;
            bool cameraHierarchyConsistent =
                maxCameraHierarchyPositionErrorMetres <= 0.00001f &&
                maxCameraHierarchyRotationErrorDegrees <= 0.0001f;
            bool horizonStable =
                maxAbsoluteCameraRollDegrees <= maximumAbsoluteCameraRollDegrees;
            float minimumHeadDeltaLag1Correlation = productionGateApplied
                ? ProductionMinimumHeadDeltaLag1Correlation : -1f;
            float maximumHeadDirectionReversalsPerSecond = productionGateApplied
                ? ProductionMaximumHeadDirectionReversalsPerSecond : 1000f;
            bool headTemporalSmoothnessAcceptable = true;
            for (int index = 0; index < 6; index++)
            {
                headTemporalSmoothnessAcceptable &=
                    summary.headDeltaLag1Correlation[index] >= minimumHeadDeltaLag1Correlation &&
                    summary.headDirectionReversalsPerSecond[index] <=
                        maximumHeadDirectionReversalsPerSecond;
            }
            bool rootInnovationPassthroughExact =
                summary.maxRootInnovationPassthroughError <=
                    ProductionMaximumRootInnovationPassthroughError;
            bool headInnovationExactlyZero = summary.headInnovationExactlyZero;
            bool horizontalEventControllerExercised =
                summary.horizontalEventCount >= 1 && summary.completedHorizontalEventCount >= 1;
            bool verticalEventControllerExercised =
                summary.verticalEventCount >= 1 && summary.completedVerticalEventCount >= 1;
            bool horizontalEventEndpointExact =
                summary.maxHorizontalEventEndpointResidual <= 0.000001f;
            bool verticalEventEndpointExact =
                summary.maxVerticalEventEndpointResidual <= 0.000001f;
            bool verticalEventParametersFinite =
                float.IsFinite(summary.meanVerticalEventAbsAmplitudeY) &&
                float.IsFinite(summary.meanVerticalEventAbsAmplitudePitch) &&
                float.IsFinite(summary.maxVerticalEventAbsAmplitudeY) &&
                float.IsFinite(summary.maxVerticalEventAbsAmplitudePitch);
            bool horizontalEventParametersFinite =
                AllFinite(summary.meanHorizontalEventAbsAmplitude) &&
                AllFinite(summary.maxHorizontalEventAbsAmplitude);
            string sameA = SequenceSignature(seed, 96);
            string sameB = SequenceSignature(seed, 96);
            string different = SequenceSignature(unchecked(seed + 1), 96);
            bool sameSeedDeterministic = sameA == sameB;
            bool differentSeedVaries = sameA != different;
            bool passed =
                stepIndex == targetSteps &&
                summary.simulatedSeconds >= EpisodeSeconds &&
                summary.uninterrupted &&
                nonFiniteCount == 0 &&
                maxBodyTiltDegrees <= 0.001f &&
                maxQuaternionNormError <= 0.0001f &&
                summary.clipStepRate <= maximumClipStepRate &&
                stabilizedBaseClipAcceptable &&
                rootBaseClipAcceptable &&
                deltaWindowsNonFrozen &&
                distanceReasonable &&
                cameraPoseChanges &&
                cameraHierarchyConsistent &&
                horizonStable &&
                headTemporalSmoothnessAcceptable &&
                rootInnovationPassthroughExact &&
                headInnovationExactlyZero &&
                horizontalEventControllerExercised &&
                verticalEventControllerExercised &&
                horizontalEventEndpointExact &&
                verticalEventEndpointExact &&
                horizontalEventParametersFinite &&
                verticalEventParametersFinite &&
                sameSeedDeterministic && differentSeedVaries;

            CompleteEpisode("rollout_test");
            RolloutTestReport report = new RolloutTestReport
            {
                status = passed ? "passed" : "failed",
                modelVerified = true,
                sourceModelSha256 = runtime.Data.sourceModelSha256,
                exportedModelSha256 = runtime.ExportSha256,
                modelGoldenMaxAbsError = runtime.GoldenMaxAbsError,
                eventGoldenMaxAbsError = runtime.EventGoldenMaxAbsError,
                innovationGoldenMaxAbsError = runtime.InnovationGoldenMaxAbsError,
                hasCombinedStochasticGolden = runtime.HasCombinedStochasticGolden,
                combinedStochasticGoldenMaxAbsError = runtime.CombinedStochasticGoldenMaxAbsError,
                seed = seed,
                seedTrajectory = activeSeed.trajectoryId,
                generatedSteps = stepIndex,
                generatedSeconds = stepIndex * runtime.Data.fixedDtSeconds,
                targetSeconds = EpisodeSeconds,
                episodeStartCount = episodeStartCount,
                noReset = episodeStartCount == 1,
                nonFiniteCount = nonFiniteCount,
                productionGateApplied = productionGateApplied,
                clipStepRate = summary.clipStepRate,
                maximumAllowedClipStepRate = maximumClipStepRate,
                comfortEnvelopeActivationStepCount =
                    summary.comfortEnvelopeActivationStepCount,
                comfortEnvelopeActivationRate = summary.comfortEnvelopeActivationRate,
                baseClipStepRate = summary.baseClipStepRate,
                rootBaseClipStepRate = summary.rootBaseClipStepRate,
                rootBaseClipAcceptable = rootBaseClipAcceptable,
                stabilizedBaseClipStepRate = summary.stabilizedBaseClipStepRate,
                stabilizedBaseClipAcceptable = stabilizedBaseClipAcceptable,
                deltaWindowsNonFrozen = deltaWindowsNonFrozen,
                minimumRequiredNormalizedDeltaRms = minimumNormalizedDeltaRms,
                first10SecondsNormalizedDeltaRms = summary.first10SecondsNormalizedDeltaRms,
                last10SecondsNormalizedDeltaRms = summary.last10SecondsNormalizedDeltaRms,
                first10SecondsDeltaRms = summary.first10SecondsDeltaRms,
                last10SecondsDeltaRms = summary.last10SecondsDeltaRms,
                planarSpeedEnergyRatioIsDiagnostic = true,
                planarSpeedEnergyDiagnosticFinite = planarSpeedEnergyDiagnosticFinite,
                first10SecondsPlanarSpeedEnergy = summary.first10SecondsPlanarSpeedEnergy,
                last10SecondsPlanarSpeedEnergy = summary.last10SecondsPlanarSpeedEnergy,
                lastFirstPlanarSpeedEnergyRatio = summary.lastFirstPlanarSpeedEnergyRatio,
                distanceReasonable = distanceReasonable,
                minimumAllowedDistanceMetres = minimumDistance,
                maximumAllowedDistanceMetres = maximumDistance,
                cameraPoseChanges = cameraPoseChanges,
                cameraPoseChangedStepCount = cameraPoseChangedStepCount,
                minimumRequiredCameraPoseChangedSteps = targetSteps / 10,
                cameraHierarchyConsistent = cameraHierarchyConsistent,
                maxCameraHierarchyPositionErrorMetres = maxCameraHierarchyPositionErrorMetres,
                maxCameraHierarchyRotationErrorDegrees = maxCameraHierarchyRotationErrorDegrees,
                horizonStable = horizonStable,
                maxAbsoluteCameraRollDegrees = maxAbsoluteCameraRollDegrees,
                maximumAllowedAbsoluteCameraRollDegrees = maximumAbsoluteCameraRollDegrees,
                headTemporalSmoothnessAcceptable = headTemporalSmoothnessAcceptable,
                minimumRequiredHeadDeltaLag1Correlation = minimumHeadDeltaLag1Correlation,
                maximumAllowedHeadDirectionReversalsPerSecond = maximumHeadDirectionReversalsPerSecond,
                headDeltaLag1Correlation = summary.headDeltaLag1Correlation,
                headDirectionReversalsPerSecond = summary.headDirectionReversalsPerSecond,
                rootInnovationPassthroughExact = rootInnovationPassthroughExact,
                maxRootInnovationPassthroughError = summary.maxRootInnovationPassthroughError,
                maximumAllowedRootInnovationPassthroughError =
                    ProductionMaximumRootInnovationPassthroughError,
                headInnovationExactlyZero = headInnovationExactlyZero,
                maxAbsoluteHeadInnovation = summary.maxAbsoluteHeadInnovation,
                horizontalEventControllerExercised = horizontalEventControllerExercised,
                horizontalEventEndpointExact = horizontalEventEndpointExact,
                horizontalEventParametersFinite = horizontalEventParametersFinite,
                horizontalEventCount = summary.horizontalEventCount,
                completedHorizontalEventCount = summary.completedHorizontalEventCount,
                meanHorizontalEventAbsAmplitude = summary.meanHorizontalEventAbsAmplitude,
                maxHorizontalEventAbsAmplitude = summary.maxHorizontalEventAbsAmplitude,
                maxHorizontalEventEndpointResidual = summary.maxHorizontalEventEndpointResidual,
                verticalEventControllerExercised = verticalEventControllerExercised,
                verticalEventClosureExact = verticalEventEndpointExact,
                verticalEventEndpointExact = verticalEventEndpointExact,
                verticalEventParametersFinite = verticalEventParametersFinite,
                verticalEventCount = summary.verticalEventCount,
                completedVerticalEventCount = summary.completedVerticalEventCount,
                meanVerticalEventAbsAmplitudeY = summary.meanVerticalEventAbsAmplitudeY,
                meanVerticalEventAbsAmplitudePitch = summary.meanVerticalEventAbsAmplitudePitch,
                maxVerticalEventAbsAmplitudeY = summary.maxVerticalEventAbsAmplitudeY,
                maxVerticalEventAbsAmplitudePitch = summary.maxVerticalEventAbsAmplitudePitch,
                maxVerticalEventAnchorClosureResidualY =
                    summary.maxVerticalEventAnchorClosureResidualY,
                maxVerticalEventAnchorClosureResidualPitch =
                    summary.maxVerticalEventAnchorClosureResidualPitch,
                maxBodyTiltDegrees = maxBodyTiltDegrees,
                maxQuaternionNormError = maxQuaternionNormError,
                distanceMetres = distanceMetres,
                sameSeedDeterministic = sameSeedDeterministic,
                differentSeedVaries = differentSeedVaries,
                sameSeedSignature = sameA,
                differentSeedSignature = different,
                logDirectory = logDirectory,
            };
            string output = ReadArgumentValue("--rollout-test-output=");
            if (string.IsNullOrEmpty(output))
            {
                output = Path.Combine(Application.persistentDataPath, "normal_hmd_rollout_test.json");
            }
            string fullOutput = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(fullOutput) ?? Application.persistentDataPath);
            File.WriteAllText(fullOutput, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            Debug.Log((passed ? "NORMAL_HMD_ROLLOUT_TEST_OK" : "NORMAL_HMD_ROLLOUT_TEST_FAILED") +
                " output=" + fullOutput);
            Application.Quit(passed ? 0 : 4);
        }

        private string SequenceSignature(int seed, int steps)
        {
            NormalHmdLstmRuntime verifier = NormalHmdLstmRuntime.LoadVerified();
            uint windowSeed = PortableHmdRandom.DeriveSeed(
                seed, verifier.Data.seedWindowXorConstant);
            PortableHmdRandom windowRandom = new PortableHmdRandom(windowSeed);
            NormalHmdSeedData selected = verifier.Data.seedWindows[
                (int)(windowRandom.NextUInt32() % (uint)verifier.Data.seedWindows.Length)];
            verifier.Warm(selected.historyRaw);
            uint rootSeed = PortableHmdRandom.DeriveSeed(
                seed, verifier.Data.rootSeedXorConstant);
            uint horizontalSeed = PortableHmdRandom.DeriveSeed(
                seed, verifier.Data.horizontalSeedXorConstant);
            uint verticalSeed = PortableHmdRandom.DeriveSeed(
                seed, verifier.Data.verticalSeedXorConstant);
            SeededRootInnovation noiseGenerator = new SeededRootInnovation(
                rootSeed,
                verifier.Data.rootInnovationRho,
                verifier.Data.rootInnovationCholesky);
            float[] previous = new float[9];
            int historyOffset = (verifier.Data.historyLength - 1) * 9;
            for (int feature = 0; feature < 9; feature++)
            {
                previous[feature] = Mathf.Clamp(
                    selected.historyRaw[historyOffset + feature],
                    verifier.Data.stateClipLow[feature],
                    verifier.Data.stateClipHigh[feature]);
            }
            NormalHmdEventController signatureHorizontal =
                verifier.CreateHorizontalController(previous, horizontalSeed);
            NormalHmdEventController signatureVertical =
                verifier.CreateVerticalController(previous, verticalSeed);
            using (SHA256 sha = SHA256.Create())
            using (MemoryStream stream = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                for (int step = 0; step < steps; step++)
                {
                    NormalHmdV4Step predicted = verifier.PredictHeads();
                    float[] composed = verifier.ComposeBeforeInnovation(previous, predicted,
                        signatureHorizontal, signatureVertical);
                    float[] rootNoise = noiseGenerator.Next(verifier.Data.defaultRootNoiseScale);
                    bool[] ignored;
                    float[] applied = verifier.ApplyInnovationAndSafety(
                        composed, rootNoise, out ignored);
                    for (int feature = 0; feature < 9; feature++)
                    {
                        writer.Write(applied[feature]);
                    }
                    verifier.AdvanceExecuted(applied);
                    previous = applied;
                }
                writer.Flush();
                return Hex(sha.ComputeHash(stream.ToArray()));
            }
        }

        private void OnGUI()
        {
            if (hudManagedExternally) return;
            BuildGuiStyles();
            float width = hudExpanded
                ? Mathf.Min(400f, Screen.width * 0.42f)
                : Mathf.Clamp(Screen.width * 0.25f, 220f, 310f);
            float height = hudExpanded
                ? Screen.height - 28f
                : Mathf.Min(132f, Screen.height - 28f);
            float left = Screen.width - width - 14f;
            Rect panelRect = new Rect(left, 14f, width, height);
            GUI.DrawTexture(panelRect, panelTexture, ScaleMode.StretchToFill, true);
            GUILayout.BeginArea(panelRect);
            GUILayout.Space(13f);
            GUILayout.BeginHorizontal();
            GUILayout.Space(15f);
            GUILayout.BeginVertical(GUILayout.Width(width - 30f));
            if (!hudExpanded)
            {
                GUILayout.Label("HEAD  ·  EyeNavGS Normal HMD v4", goodStyle);
                if (fatalError != null)
                {
                    GUILayout.Label("MODEL ERROR", errorStyle);
                }
                else
                {
                    float compactElapsed = runtime == null ? 0f : Mathf.Min(
                        episodeSafetyLimitSeconds,
                        stepIndex * runtime.Data.fixedDtSeconds);
                    string compactTime = adaptiveEpisodeDriver == null
                        ? compactElapsed.ToString("00.0") + " / 60.0 s"
                        : compactElapsed.ToString("0.0") + " s  ·  content-driven  ·  " +
                            episodeSafetyLimitSeconds.ToString("0") + " s safety cap";
                    GUILayout.Label(compactTime + "  ·  seed " + episodeSeed, smallStyle);
                    GUILayout.Label(
                        (episodeComplete ? "complete" : (paused ? "paused" : "online")) +
                        "  ·  " + (firstPerson ? "first person" : "observer"),
                        labelStyle);
                }
                if (GUILayout.Button("H  Show model details", GUILayout.Height(25f)))
                    hudExpanded = true;
                GUILayout.EndVertical(); GUILayout.EndHorizontal(); GUILayout.EndArea();
                GUI.DrawTexture(new Rect(left, 14f + height - 4f, width, 4f), accentTexture);
                return;
            }
            if (GUILayout.Button("H  Collapse details", GUILayout.Height(25f)))
                hudExpanded = false;
            GUILayout.Label(motionDriver == null
                ? "NORMAL HMD AGENT"
                : "NORMAL HMD + SCENE PLANNER", titleStyle);
            GUILayout.Label(motionDriver == null
                ? "EyeNavGS v4 three-branch 60-second first-person demo"
                : "Hybrid demo: trained human-motion prior + semantic skill planner", smallStyle);
            GUILayout.Space(10f);

            if (fatalError != null)
            {
                GUILayout.Label("MODEL/DEMO ERROR", errorStyle);
                GUILayout.Label(fatalError, smallStyle);
                GUILayout.EndVertical(); GUILayout.EndHorizontal(); GUILayout.EndArea();
                return;
            }
            if (runtime == null)
            {
                GUILayout.Label("Loading verified model…", labelStyle);
                GUILayout.EndVertical(); GUILayout.EndHorizontal(); GUILayout.EndArea();
                return;
            }

            float elapsed = Mathf.Min(
                episodeSafetyLimitSeconds,
                stepIndex * runtime.Data.fixedDtSeconds);
            GUILayout.Label("REAL TRAINED MODEL VERIFIED", goodStyle);
            GUILayout.Label("PyTorch ↔ Unity: " + runtime.GoldenMaxAbsError.ToString("G3") +
                "  | event RNG: " + runtime.EventGoldenMaxAbsError.ToString("G3") +
                "  | root RNG: " + runtime.InnovationGoldenMaxAbsError.ToString("G3"), smallStyle);
            GUILayout.Label("Combined stochastic loop: " +
                (runtime.HasCombinedStochasticGolden
                    ? runtime.CombinedStochasticGoldenMaxAbsError.ToString("G3")
                    : "n/a (smoke export)"), smallStyle);
            GUILayout.Label("v4: independent Root / Horizontal / Vertical LSTMs", smallStyle);
            GUILayout.Label("H and V: independent learned hazard  •  one endpoint/duration sample per onset  •  minimum jerk then hold",
                goodStyle);
            GUILayout.Label("HeadLocal random noise: OFF  •  only BodyRoot has joint learned innovation",
                goodStyle);
            if (motionDriver != null)
            {
                GUILayout.Label(
                    "TRAINED: HeadLocal motion and locomotion prior  •  RULE-BASED: goal/path steering of executed BodyRoot",
                    warningStyle);
                GUILayout.Label(
                    "Planner guidance does not replace the six learned HeadLocal channels or recurrent model state.",
                    smallStyle);
            }
            GUILayout.Space(8f);
            GUILayout.Label(adaptiveEpisodeDriver == null
                ? "TIME  " + elapsed.ToString("00.0") + " / 60.0 s"
                : "TIME  " + elapsed.ToString("0.0") +
                    " s  ·  until content complete  ·  safety cap " +
                    episodeSafetyLimitSeconds.ToString("0") + " s", labelStyle);
            GUILayout.HorizontalSlider(elapsed, 0f, episodeSafetyLimitSeconds);
            GUILayout.Label("Seed " + episodeSeed + "  •  " +
                (activeSeed == null ? "starting" : activeSeed.trajectoryId), smallStyle);
            GUILayout.Label("Camera: " + (firstPerson ? "AI participant (first person)" : "external observer"), labelStyle);
            GUILayout.Label("Distance: " + distanceMetres.ToString("0.00") + " m  •  unexpected clip: " +
                (stepIndex > 0 ? (100f * clippedStepCount / stepIndex).ToString("0.0") : "0.0") + "%", smallStyle);
            GUILayout.Label("Horizontal events: " + horizontalEventCount +
                "  •  p=" + (horizontalController == null ? "—" :
                    horizontalController.Probability.ToString("0.000")) +
                (horizontalController != null && horizontalController.Active
                    ? "  •  active " + horizontalController.Phase.ToString("0.00") : ""), smallStyle);
            GUILayout.Label("Vertical events: " + verticalEventCount +
                "  •  p=" + (verticalController == null ? "—" :
                    verticalController.Probability.ToString("0.000")) +
                (verticalController != null && verticalController.Active
                    ? "  •  active " + verticalController.Phase.ToString("0.00") : ""), smallStyle);
            GUILayout.Label(episodeComplete
                ? (adaptiveEpisodeDriver == null
                    ? "60 s CLOSED LOOP COMPLETE — no reset"
                    : "SCENE CONTENT RUN COMPLETE — no reset") :
                (paused ? "PAUSED" : "Generating online…"), episodeComplete ? goodStyle : labelStyle);

            GUILayout.Space(10f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Resume" : "Pause", GUILayout.Height(32f))) paused = !paused;
            if (GUILayout.Button("New random person", GUILayout.Height(32f)))
            {
                episodeStartCount = 0;
                StartEpisode(CreateRandomSeed(), false);
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Replay same seed", GUILayout.Height(30f)))
            {
                int replaySeed = episodeSeed;
                episodeStartCount = 0;
                StartEpisode(replaySeed, true);
            }
            if (GUILayout.Button(firstPerson ? "Observer camera" : "First person", GUILayout.Height(30f)))
            {
                SetCameraMode(!firstPerson);
            }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Speed", GUILayout.Width(44f));
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 0.5f), "0.5×", "Button")) simulationSpeed = 0.5f;
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 1f), "1×", "Button")) simulationSpeed = 1f;
            if (GUILayout.Toggle(Mathf.Approximately(simulationSpeed, 2f), "2×", "Button")) simulationSpeed = 2f;
            GUILayout.EndHorizontal();
            if (GUILayout.Button("Copy log folder", GUILayout.Height(27f))) GUIUtility.systemCopyBuffer = logDirectory;

            GUILayout.Space(10f);
            GUILayout.Label("HIERARCHY", smallStyle);
            GUILayout.Label(motionDriver == null
                ? "ParticipantRoot: learned ground XZ + yaw proxy\n  └ TrackingOrigin: always upright\n      └ HeadLocal: independently timed horizontal and vertical minimum-jerk events\n          └ ParticipantCamera: AI first-person view"
                : "ParticipantRoot: planner-guided execution of learned locomotion prior\n  └ TrackingOrigin: always upright\n      └ HeadLocal: trained H/V LSTMs + minimum-jerk event timing\n          └ ParticipantCamera: AI first-person view", smallStyle);
            GUILayout.Label("BodyRoot is a proxy inferred from HMD data. EyeNavGS does not contain measured torso/feet or sickness labels.", warningStyle);
            GUILayout.Label("Space: pause  •  C: switch camera  •  H: HUD", smallStyle);
            GUILayout.EndVertical();
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
            GUI.DrawTexture(new Rect(left, Screen.height - 18f, width, 4f), accentTexture);
        }

        private void BuildGuiStyles()
        {
            if (titleStyle != null) return;
            panelTexture = SolidTexture(new Color(0.025f, 0.045f, 0.065f, 0.94f));
            accentTexture = SolidTexture(new Color(0.14f, 0.91f, 0.79f, 0.95f));
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
            titleStyle.normal.textColor = new Color(0.74f, 1f, 0.94f);
            labelStyle = new GUIStyle(GUI.skin.label) { fontSize = 12, wordWrap = true };
            labelStyle.normal.textColor = new Color(0.89f, 0.94f, 0.97f);
            smallStyle = new GUIStyle(labelStyle) { fontSize = 10 };
            smallStyle.normal.textColor = new Color(0.63f, 0.75f, 0.80f);
            goodStyle = new GUIStyle(labelStyle) { fontStyle = FontStyle.Bold };
            goodStyle.normal.textColor = new Color(0.25f, 0.92f, 0.62f);
            warningStyle = new GUIStyle(smallStyle);
            warningStyle.normal.textColor = new Color(1f, 0.73f, 0.31f);
            errorStyle = new GUIStyle(labelStyle) { fontStyle = FontStyle.Bold };
            errorStyle.normal.textColor = new Color(1f, 0.3f, 0.3f);
        }

        private IEnumerator CaptureAndQuit(string output)
        {
            yield return new WaitForSecondsRealtime(3f);
            string fullPath = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Application.persistentDataPath);
            Camera captureCamera = firstPerson ? participantCamera : observerCamera;
            RenderTexture target = new RenderTexture(1280, 720, 24, RenderTextureFormat.ARGB32);
            RenderTexture previousTarget = captureCamera.targetTexture;
            RenderTexture previousActive = RenderTexture.active;
            Texture2D pixels = new Texture2D(1280, 720, TextureFormat.RGB24, false);
            try
            {
                captureCamera.targetTexture = target;
                captureCamera.Render();
                RenderTexture.active = target;
                pixels.ReadPixels(new Rect(0f, 0f, 1280f, 720f), 0, 0, false);
                pixels.Apply(false, false);
                File.WriteAllBytes(fullPath, pixels.EncodeToPNG());
            }
            finally
            {
                captureCamera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                Destroy(target);
                Destroy(pixels);
            }
            CompleteEpisode("capture_complete");
            Application.Quit(File.Exists(fullPath) ? 0 : 3);
        }

        private IEnumerator CaptureScreenAndQuit(string output)
        {
            // Unlike CaptureAndQuit, this evidence hook deliberately captures
            // the composed Player frame, including the compact HEAD/BRAIN HUD.
            // It is used by visual regression checks only when explicitly
            // requested on the command line.
            yield return new WaitForSecondsRealtime(3f);
            yield return new WaitForEndOfFrame();
            string fullPath = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Application.persistentDataPath);
            ScreenCapture.CaptureScreenshot(fullPath, 1);
            float deadline = Time.realtimeSinceStartup + 5f;
            while (!File.Exists(fullPath) && Time.realtimeSinceStartup < deadline)
                yield return null;
            yield return new WaitForSecondsRealtime(0.25f);
            CompleteEpisode("screen_capture_complete");
            Application.Quit(File.Exists(fullPath) ? 0 : 3);
        }

        private void OnApplicationQuit()
        {
            if (telemetry != null) CompleteEpisode("application_quit");
        }

        private void WriteFailedRolloutReportIfRequested(string message)
        {
            if (!HasArgument("--rollout-test")) return;
            string output = ReadArgumentValue("--rollout-test-output=");
            if (string.IsNullOrEmpty(output)) return;
            FailureReport report = new FailureReport { status = "failed", error = message };
            string fullPath = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Application.persistentDataPath);
            File.WriteAllText(fullPath, JsonUtility.ToJson(report, true), new UTF8Encoding(false));
            Application.Quit(5);
        }

        private static float[] FinishRms(float[] sum, int count)
        {
            float[] result = new float[sum.Length];
            if (count == 0) return result;
            for (int index = 0; index < sum.Length; index++) result[index] = Mathf.Sqrt(sum[index] / count);
            return result;
        }

        private static float[] DivideArray(float[] values, int count)
        {
            float[] result = new float[values.Length];
            if (count <= 0) return result;
            for (int index = 0; index < values.Length; index++)
                result[index] = values[index] / count;
            return result;
        }

        private static bool AllFinite(float[] values)
        {
            if (values == null) return false;
            foreach (float value in values)
                if (!float.IsFinite(value)) return false;
            return true;
        }

        private float NormalizedCombinedRms(float[] physicalDeltaRms)
        {
            float squareSum = 0f;
            for (int index = 0; index < physicalDeltaRms.Length; index++)
            {
                float normalized = physicalDeltaRms[index] / runtime.Data.stateStd[index];
                squareSum += normalized * normalized;
            }
            return Mathf.Sqrt(squareSum / physicalDeltaRms.Length);
        }

        private bool IsProductionModel()
        {
            // The production recipe is fixed at 64 history frames, a 64-step
            // training horizon and hidden size 128. The smaller smoke checkpoint
            // is useful only for compile/runtime wiring and receives wider gates.
            return runtime.Data.hiddenDim >= 96 &&
                runtime.Data.historyLength >= 64 &&
                runtime.Data.trainedRolloutSeconds >= 1.5f &&
                runtime.Data.schemaVersion == "normal-hmd-agent-unity-4.0";
        }

        private static void AppendCsv(StringBuilder builder, float value)
        {
            builder.Append(',').Append(value.ToString("G9", CultureInfo.InvariantCulture));
        }

        private static int CreateRandomSeed()
        {
            byte[] bytes = new byte[4];
            using (RandomNumberGenerator generator = RandomNumberGenerator.Create()) generator.GetBytes(bytes);
            return BitConverter.ToInt32(bytes, 0) & int.MaxValue;
        }

        private static bool HasArgument(string exact)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (string.Equals(argument, exact, StringComparison.OrdinalIgnoreCase)) return true;
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
            return int.TryParse(
                ReadArgumentValue("--seed="),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out seed) && seed >= 0;
        }

        private static string Sha256File(string path)
        {
            using (SHA256 sha = SHA256.Create())
            using (FileStream stream = File.OpenRead(path)) return Hex(sha.ComputeHash(stream));
        }

        private static string Hex(byte[] digest)
        {
            StringBuilder builder = new StringBuilder(digest.Length * 2);
            foreach (byte value in digest) builder.Append(value.ToString("x2"));
            return builder.ToString();
        }

        private static Texture2D SolidTexture(Color color)
        {
            Texture2D texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }

        [Serializable]
        private sealed class RunManifest
        {
            public string product;
            public string schemaVersion;
            public int runSeed;
            public bool replay;
            public string seedTrajectory;
            public string seedSplit;
            public int seedStartFrame;
            public int historyFrames;
            public float targetSeconds;
            public int targetSteps;
            public float fixedDtSeconds;
            public string bodyDefinition;
            public string headDefinition;
            public string recurrentFeedbackPolicy;
            public string deploymentStabilizer;
            public float rootYawDcRemovalTauSeconds;
            public float headPoseDcRemovalTauSeconds;
            public string headPoseAnchor;
            public float headPoseAnchorMarginSigma;
            public float[] rawHeadPoseAnchor;
            public float[] clippedHeadPoseAnchor;
            public string stochasticInnovation;
            public string headInnovationPolicy;
            public float headInnovationSmoothingTauSeconds;
            public float headInnovationSmoothingGain;
            public string verticalEventController;
            public float verticalEventThreshold;
            public float verticalDurationMinSteps;
            public float verticalDurationMaxSteps;
            public float[] rootInnovationRho;
            public float[] rootInnovationCholesky;
            public bool rootInnovationFeedsBack;
            public float noiseScale;
            public string sourceModelSha256;
            public string exportedModelSha256;
            public string datasetFingerprintSha256;
            public float unityModelGoldenMaxAbsError;
            public float unityEventGoldenMaxAbsError;
            public float unityInnovationGoldenMaxAbsError;
            public bool hasCombinedStochasticGolden;
            public float unityCombinedStochasticGoldenMaxAbsError;
            public string startedAtUtc;
        }

        [Serializable]
        private sealed class RunSummary
        {
            public string reason;
            public int steps;
            public int targetSteps;
            public float simulatedSeconds;
            public bool uninterrupted;
            public int episodeStartCount;
            public int nonFiniteCount;
            public int clippedStepCount;
            public int clippedValueCount;
            public float clipStepRate;
            public int comfortEnvelopeActivationStepCount;
            public int comfortEnvelopeActivationValueCount;
            public float comfortEnvelopeActivationRate;
            public int baseClippedStepCount;
            public int baseClippedValueCount;
            public float baseClipStepRate;
            public int rootBaseClippedStepCount;
            public int rootBaseClippedValueCount;
            public float rootBaseClipStepRate;
            public int stabilizedBaseClippedStepCount;
            public int stabilizedBaseClippedValueCount;
            public float stabilizedBaseClipStepRate;
            public float distanceMetres;
            public float maxBodyTiltDegrees;
            public float maxQuaternionNormError;
            public int cameraPoseChangedStepCount;
            public float maxCameraHierarchyPositionErrorMetres;
            public float maxCameraHierarchyRotationErrorDegrees;
            public float maxAbsoluteCameraRollDegrees;
            public float[] first10SecondsDeltaRms;
            public float[] last10SecondsDeltaRms;
            public float first10SecondsNormalizedDeltaRms;
            public float last10SecondsNormalizedDeltaRms;
            public float first10SecondsPlanarSpeedEnergy;
            public float last10SecondsPlanarSpeedEnergy;
            public float lastFirstPlanarSpeedEnergyRatio;
            public float[] headDeltaLag1Correlation;
            public float[] headDirectionReversalsPerSecond;
            public float maxRootInnovationPassthroughError;
            public int horizontalEventCount;
            public int completedHorizontalEventCount;
            public float[] meanHorizontalEventAbsAmplitude;
            public float[] maxHorizontalEventAbsAmplitude;
            public float maxHorizontalEventEndpointResidual;
            public int verticalEventCount;
            public int completedVerticalEventCount;
            public float[] meanVerticalEventAbsAmplitude;
            public float[] maxVerticalEventAbsAmplitude;
            public float maxVerticalEventEndpointResidual;
            public float meanVerticalEventAbsAmplitudeY;
            public float meanVerticalEventAbsAmplitudePitch;
            public float maxVerticalEventAbsAmplitudeY;
            public float maxVerticalEventAbsAmplitudePitch;
            public float maxVerticalEventAnchorClosureResidualY;
            public float maxVerticalEventAnchorClosureResidualPitch;
            public float maxAbsoluteHeadInnovation;
            public bool headInnovationExactlyZero;
            public string telemetrySha256;
            public string endedAtUtc;
        }

        [Serializable]
        private sealed class RolloutTestReport
        {
            public string status;
            public bool modelVerified;
            public string sourceModelSha256;
            public string exportedModelSha256;
            public float modelGoldenMaxAbsError;
            public float eventGoldenMaxAbsError;
            public float innovationGoldenMaxAbsError;
            public bool hasCombinedStochasticGolden;
            public float combinedStochasticGoldenMaxAbsError;
            public int seed;
            public string seedTrajectory;
            public int generatedSteps;
            public float generatedSeconds;
            public float targetSeconds;
            public int episodeStartCount;
            public bool noReset;
            public int nonFiniteCount;
            public bool productionGateApplied;
            public float clipStepRate;
            public float maximumAllowedClipStepRate;
            public int comfortEnvelopeActivationStepCount;
            public float comfortEnvelopeActivationRate;
            public float baseClipStepRate;
            public float rootBaseClipStepRate;
            public bool rootBaseClipAcceptable;
            public float stabilizedBaseClipStepRate;
            public bool stabilizedBaseClipAcceptable;
            public bool deltaWindowsNonFrozen;
            public float minimumRequiredNormalizedDeltaRms;
            public float first10SecondsNormalizedDeltaRms;
            public float last10SecondsNormalizedDeltaRms;
            public float[] first10SecondsDeltaRms;
            public float[] last10SecondsDeltaRms;
            public bool planarSpeedEnergyRatioIsDiagnostic;
            public bool planarSpeedEnergyDiagnosticFinite;
            public float first10SecondsPlanarSpeedEnergy;
            public float last10SecondsPlanarSpeedEnergy;
            public float lastFirstPlanarSpeedEnergyRatio;
            public bool distanceReasonable;
            public float minimumAllowedDistanceMetres;
            public float maximumAllowedDistanceMetres;
            public bool cameraPoseChanges;
            public int cameraPoseChangedStepCount;
            public int minimumRequiredCameraPoseChangedSteps;
            public bool cameraHierarchyConsistent;
            public float maxCameraHierarchyPositionErrorMetres;
            public float maxCameraHierarchyRotationErrorDegrees;
            public bool horizonStable;
            public float maxAbsoluteCameraRollDegrees;
            public float maximumAllowedAbsoluteCameraRollDegrees;
            public bool headTemporalSmoothnessAcceptable;
            public float minimumRequiredHeadDeltaLag1Correlation;
            public float maximumAllowedHeadDirectionReversalsPerSecond;
            public float[] headDeltaLag1Correlation;
            public float[] headDirectionReversalsPerSecond;
            public bool rootInnovationPassthroughExact;
            public float maxRootInnovationPassthroughError;
            public float maximumAllowedRootInnovationPassthroughError;
            public bool headInnovationExactlyZero;
            public float maxAbsoluteHeadInnovation;
            public bool horizontalEventControllerExercised;
            public bool horizontalEventEndpointExact;
            public bool horizontalEventParametersFinite;
            public int horizontalEventCount;
            public int completedHorizontalEventCount;
            public float[] meanHorizontalEventAbsAmplitude;
            public float[] maxHorizontalEventAbsAmplitude;
            public float maxHorizontalEventEndpointResidual;
            public bool verticalEventControllerExercised;
            public bool verticalEventClosureExact;
            public bool verticalEventEndpointExact;
            public bool verticalEventParametersFinite;
            public int verticalEventCount;
            public int completedVerticalEventCount;
            public float meanVerticalEventAbsAmplitudeY;
            public float meanVerticalEventAbsAmplitudePitch;
            public float maxVerticalEventAbsAmplitudeY;
            public float maxVerticalEventAbsAmplitudePitch;
            public float maxVerticalEventAnchorClosureResidualY;
            public float maxVerticalEventAnchorClosureResidualPitch;
            public float maxBodyTiltDegrees;
            public float maxQuaternionNormError;
            public float distanceMetres;
            public bool sameSeedDeterministic;
            public bool differentSeedVaries;
            public string sameSeedSignature;
            public string differentSeedSignature;
            public string logDirectory;
        }

        [Serializable]
        private sealed class FailureReport
        {
            public string status;
            public string error;
        }
    }
}
