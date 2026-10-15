using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Runtime counterpart of a SemanticAffordanceContract. It is deliberately
    /// explicit: an agent can query the supported verb and verify a state change
    /// instead of guessing an interaction from pixels.
    /// </summary>
    public sealed class SceneAwareAffordance : MonoBehaviour
    {
        public enum EffectKind
        {
            ConfirmOnly,
            OpenDoor,
            StartDynamicMotion,
            BoardVehicle,
            StartVehicle
        }

        private SemanticAffordanceContract contract;
        private EffectKind effectKind;
        private Transform effectTarget;
        private Renderer indicator;
        private Vector3 closedLocalPosition;
        private Vector3 openLocalPosition;
        private bool activated;
        private float activationProgress;

        public SemanticAffordanceContract Contract { get { return contract; } }
        public bool IsActivated { get { return activated; } }
        public bool SuccessSatisfied
        {
            get
            {
                if (!activated) return false;
                if (effectKind == EffectKind.OpenDoor) return activationProgress >= 0.999f;
                if (effectKind == EffectKind.StartDynamicMotion)
                {
                    SceneAwareDynamicMotionRig motion = effectTarget == null
                        ? null : effectTarget.GetComponent<SceneAwareDynamicMotionRig>();
                    return motion != null && motion.MotionEnabled;
                }
                if (effectKind == EffectKind.BoardVehicle || effectKind == EffectKind.StartVehicle)
                {
                    SceneAwareRollerCoasterRig vehicle = effectTarget == null
                        ? null : effectTarget.GetComponent<SceneAwareRollerCoasterRig>();
                    if (vehicle == null) return false;
                    return effectKind == EffectKind.BoardVehicle
                        ? vehicle.Boarded
                        : (vehicle.Running || vehicle.Complete);
                }
                return true;
            }
        }
        public string Evidence
        {
            get
            {
                if (!activated) return "state=idle";
                if (effectKind == EffectKind.OpenDoor)
                    return "door_open_fraction=" + activationProgress.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);
                if (effectKind == EffectKind.StartDynamicMotion)
                    return "dynamic_rig_running=true";
                if ((effectKind == EffectKind.BoardVehicle || effectKind == EffectKind.StartVehicle) &&
                    effectTarget != null)
                {
                    SceneAwareRollerCoasterRig vehicle = effectTarget.GetComponent<SceneAwareRollerCoasterRig>();
                    if (vehicle != null) return vehicle.Evidence;
                }
                return "activation_latched=true";
            }
        }

        public void Configure(
            SemanticAffordanceContract semanticContract,
            EffectKind kind,
            Transform target,
            Renderer statusIndicator)
        {
            contract = semanticContract;
            effectKind = kind;
            effectTarget = target;
            indicator = statusIndicator;
            if (effectTarget != null)
            {
                closedLocalPosition = effectTarget.localPosition;
                openLocalPosition = closedLocalPosition + Vector3.up * 2.8f;
            }
            ResetState();
        }

        public void ResetState()
        {
            activated = false;
            activationProgress = 0f;
            if (effectTarget != null)
            {
                effectTarget.localPosition = closedLocalPosition;
                SceneAwareDynamicMotionRig motion = effectTarget.GetComponent<SceneAwareDynamicMotionRig>();
                if (motion != null) motion.SetMotionEnabled(false);
                SceneAwareRollerCoasterRig vehicle = effectTarget.GetComponent<SceneAwareRollerCoasterRig>();
                if (vehicle != null) vehicle.ResetRide();
            }
            SetIndicatorColor(new Color(1.0f, 0.62f, 0.18f));
        }

        public bool Activate(out string evidence)
        {
            if (contract == null)
            {
                evidence = "missing_contract";
                return false;
            }

            bool succeeded = true;
            if (effectKind == EffectKind.StartDynamicMotion && effectTarget != null)
            {
                SceneAwareDynamicMotionRig motion = effectTarget.GetComponent<SceneAwareDynamicMotionRig>();
                if (motion != null) motion.SetMotionEnabled(true);
            }
            else if ((effectKind == EffectKind.BoardVehicle || effectKind == EffectKind.StartVehicle) &&
                effectTarget != null)
            {
                SceneAwareRollerCoasterRig vehicle = effectTarget.GetComponent<SceneAwareRollerCoasterRig>();
                if (vehicle == null)
                {
                    evidence = "vehicle_component_missing";
                    return false;
                }
                succeeded = effectKind == EffectKind.BoardVehicle
                    ? vehicle.Board(out evidence)
                    : vehicle.StartRide(out evidence);
                if (!succeeded) return false;
            }
            activated = succeeded;
            SetIndicatorColor(new Color(0.16f, 0.96f, 0.72f));
            evidence = Evidence;
            return succeeded;
        }

        /// <summary>
        /// Advances effects on the same fixed simulation clock as the HMD agent.
        /// Accelerated tests and the visible 1x demo therefore execute the same
        /// interaction timeline.
        /// </summary>
        public void AdvanceSimulation(float dt)
        {
            if (!activated || effectTarget == null) return;
            if (effectKind == EffectKind.OpenDoor)
            {
                activationProgress = Mathf.MoveTowards(activationProgress, 1f, dt * 1.4f);
                effectTarget.localPosition = Vector3.Lerp(
                    closedLocalPosition, openLocalPosition, activationProgress);
            }
            else if (effectKind == EffectKind.StartDynamicMotion)
            {
                SceneAwareDynamicMotionRig motion =
                    effectTarget.GetComponent<SceneAwareDynamicMotionRig>();
                if (motion != null) motion.AdvanceSimulation(dt);
            }
        }

        private void SetIndicatorColor(Color color)
        {
            if (indicator == null || indicator.material == null) return;
            if (indicator.material.HasProperty("_BaseColor")) indicator.material.SetColor("_BaseColor", color);
            indicator.material.color = color;
            if (indicator.material.HasProperty("_EmissionColor"))
            {
                indicator.material.EnableKeyword("_EMISSION");
                indicator.material.SetColor("_EmissionColor", color * 0.28f);
            }
        }
    }

    /// <summary>
    /// A controlled visual-motion stimulus. It stays stationary until the
    /// semantic Activate skill succeeds, which lets telemetry align the scene
    /// change with HMD/camera motion.
    /// </summary>
    public sealed class SceneAwareDynamicMotionRig : MonoBehaviour
    {
        [SerializeField] private bool motionEnabled;
        [SerializeField] private float rotationDegreesPerSecond = 38f;
        [SerializeField] private float verticalAmplitude = 0.9f;
        [SerializeField] private float verticalFrequencyHz = 0.28f;
        private Vector3 anchor;
        private float phase;

        public bool MotionEnabled { get { return motionEnabled; } }
        public float NormalizedIntensity { get { return motionEnabled ? 0.72f : 0f; } }

        private void Awake()
        {
            anchor = transform.localPosition;
        }

        public void SetMotionEnabled(bool value)
        {
            motionEnabled = value;
            if (!value)
            {
                phase = 0f;
                transform.localPosition = anchor;
                transform.localRotation = Quaternion.identity;
            }
        }

        public void AdvanceSimulation(float dt)
        {
            if (!motionEnabled) return;
            phase += dt;
            transform.Rotate(Vector3.up, rotationDegreesPerSecond * dt, Space.Self);
            float y = Mathf.Sin(phase * Mathf.PI * 2f * verticalFrequencyHz) * verticalAmplitude;
            transform.localPosition = anchor + Vector3.up * y;
        }
    }
}
