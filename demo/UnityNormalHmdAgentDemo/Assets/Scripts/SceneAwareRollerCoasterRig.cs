using System.Globalization;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Deterministic, queryable vehicle motion for the roller-coaster scenario.
    /// Time is advanced by the same simulated clock as the HMD model, so a
    /// speed-up test produces the exact same track motion as an interactive run.
    /// </summary>
    public sealed class SceneAwareRollerCoasterRig : MonoBehaviour
    {
        private Vector3 trackOrigin;
        private float rideDuration = 12f;
        private bool boarded;
        private bool startRequested;
        private bool running;
        private bool complete;
        private float rideStartedAt;
        private Vector3 currentWorldPosition;
        private Quaternion currentWorldRotation = Quaternion.identity;
        private Vector3 currentWorldVelocity;
        private float currentSpeed;
        private float currentAngularSpeed;
        private float currentPitch;
        private float currentRoll;
        private bool hasPose;

        public bool Boarded { get { return boarded; } }
        public bool Running { get { return running; } }
        public bool Complete { get { return complete; } }
        public Vector3 CurrentWorldPosition { get { return currentWorldPosition; } }
        public Quaternion CurrentWorldRotation { get { return currentWorldRotation; } }
        public Vector3 CurrentWorldVelocity { get { return currentWorldVelocity; } }
        public float CurrentSpeed { get { return currentSpeed; } }
        public float CurrentAngularSpeed { get { return currentAngularSpeed; } }
        public float CurrentVerticalOffset { get { return boarded ? currentWorldPosition.y : 0f; } }
        public float CurrentPitchDegrees { get { return boarded ? currentPitch : 0f; } }
        public float CurrentRollDegrees { get { return boarded ? currentRoll : 0f; } }
        public float NormalizedIntensity { get { return running ? 0.88f : (boarded ? 0.12f : 0f); } }
        public string StateLabel
        {
            get
            {
                if (complete) return "ride_complete";
                if (running) return "ride_running";
                if (startRequested) return "start_requested";
                if (boarded) return "boarded_stationary";
                return "waiting_for_boarding";
            }
        }

        public void Configure(Vector3 origin, float durationSeconds)
        {
            trackOrigin = origin;
            rideDuration = Mathf.Max(4f, durationSeconds);
            ResetRide();
        }

        public void ResetRide()
        {
            boarded = false;
            startRequested = false;
            running = false;
            complete = false;
            rideStartedAt = 0f;
            hasPose = false;
            currentWorldVelocity = Vector3.zero;
            currentSpeed = 0f;
            currentAngularSpeed = 0f;
            ApplyPose(0f, 0.027f);
        }

        public bool Board(out string evidence)
        {
            boarded = true;
            evidence = "vehicle_state=boarded_stationary;seat_latch=true";
            return true;
        }

        public bool StartRide(out string evidence)
        {
            if (!boarded)
            {
                evidence = "vehicle_state=waiting_for_boarding;start_rejected=true";
                return false;
            }
            startRequested = true;
            evidence = "vehicle_state=start_requested;safety_interlock=passed";
            return true;
        }

        public void Evaluate(float simulatedTime, float dt)
        {
            if (startRequested && !running && !complete)
            {
                running = true;
                rideStartedAt = simulatedTime;
            }
            float normalized = 0f;
            if (running)
            {
                normalized = Mathf.Clamp01((simulatedTime - rideStartedAt) / rideDuration);
                if (normalized >= 1f)
                {
                    normalized = 1f;
                    running = false;
                    complete = true;
                }
            }
            else if (complete)
            {
                normalized = 1f;
            }
            ApplyPose(normalized, dt);
        }

        public Vector3 SamplePosition(float normalized)
        {
            float angle = Mathf.Clamp01(normalized) * Mathf.PI * 2f;
            float x = Mathf.Sin(angle) * 2.85f;
            float z = (1f - Mathf.Cos(angle)) * 2.85f;
            float climb = Mathf.Pow(Mathf.Sin(angle * 0.5f), 2f) * 1.15f;
            float sine = Mathf.Sin(angle);
            float ripple = sine * sine * sine * 0.28f;
            return trackOrigin + new Vector3(x, climb + ripple, z);
        }

        public Vector3 SampleTangent(float normalized)
        {
            const float epsilon = 0.0005f;
            Vector3 before = SamplePosition(Mathf.Repeat(normalized - epsilon, 1f));
            Vector3 after = SamplePosition(Mathf.Repeat(normalized + epsilon, 1f));
            Vector3 tangent = after - before;
            return tangent.sqrMagnitude < 0.000001f ? Vector3.forward : tangent.normalized;
        }

        public string Evidence
        {
            get
            {
                return string.Format(
                    CultureInfo.InvariantCulture,
                    "vehicle_state={0};speed={1:F3};vertical_offset={2:F3}",
                    StateLabel,
                    currentSpeed,
                    CurrentVerticalOffset);
            }
        }

        private void ApplyPose(float normalized, float dt)
        {
            Vector3 priorPosition = currentWorldPosition;
            Quaternion priorRotation = currentWorldRotation;
            currentWorldPosition = SamplePosition(normalized);
            Vector3 tangent = SampleTangent(normalized);
            float pitch = -Mathf.Asin(Mathf.Clamp(tangent.y, -1f, 1f)) * Mathf.Rad2Deg;
            float angle = normalized * Mathf.PI * 2f;
            float roll = running ? Mathf.Sin(angle) * 3.5f : 0f;
            Vector3 flatTangent = new Vector3(tangent.x, 0f, tangent.z);
            if (flatTangent.sqrMagnitude < 0.000001f) flatTangent = Vector3.forward;
            float yaw = Mathf.Atan2(flatTangent.x, flatTangent.z) * Mathf.Rad2Deg;
            currentPitch = pitch;
            currentRoll = roll;
            currentWorldRotation = Quaternion.Euler(pitch, yaw, roll);
            transform.SetPositionAndRotation(currentWorldPosition, currentWorldRotation);

            if (hasPose && dt > 0.000001f)
            {
                currentWorldVelocity = (currentWorldPosition - priorPosition) / dt;
                currentSpeed = currentWorldVelocity.magnitude;
                currentAngularSpeed = Quaternion.Angle(priorRotation, currentWorldRotation) / dt;
            }
            else
            {
                currentWorldVelocity = Vector3.zero;
                currentSpeed = 0f;
                currentAngularSpeed = 0f;
            }
            hasPose = true;
        }
    }
}
