using System;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// One independent v4 event branch.  A Bernoulli draw occurs only while
    /// eligible; endpoint and duration are sampled exactly once on onset.  The
    /// absolute endpoint is reached through a quintic minimum-jerk transition
    /// and then held until the next event.
    /// </summary>
    internal sealed class NormalHmdEventController
    {
        private readonly int[] indices;
        private readonly float[] parameterCenter;
        private readonly float[] parameterScale;
        private readonly float[] parameterLow;
        private readonly float[] parameterHigh;
        private readonly float[] poseLow;
        private readonly float[] poseHigh;
        private readonly float[] deploymentPoseLow;
        private readonly float[] deploymentPoseHigh;
        private readonly float[] cholesky;
        private readonly float hazardBias;
        private readonly int durationMin;
        private readonly int durationMax;
        private readonly int cooldownSteps;
        private readonly float durationScale;
        private readonly PortableHmdRandom random;
        private readonly float[] start;
        private readonly float[] target;
        private readonly float[] amplitude;

        private bool active;
        private int cooldownRemaining;
        private int elapsedSteps;
        private int durationSteps;
        private int eventId;

        internal NormalHmdEventController(
            int[] stateIndices,
            float[] centers,
            float[] scales,
            float[] parameterMinimum,
            float[] parameterMaximum,
            float[] poseMinimum,
            float[] poseMaximum,
            float[] deploymentPoseMinimum,
            float[] deploymentPoseMaximum,
            float[] parameterCholesky,
            float calibratedHazardBias,
            int minimumDurationSteps,
            int maximumDurationSteps,
            int eventCooldownSteps,
            float eventDurationScale,
            uint derivedSeed,
            float[] previousPhysical)
        {
            indices = (int[])stateIndices.Clone();
            int poseDimension = indices.Length;
            int parameterDimension = poseDimension + 1;
            RequireLength(centers, parameterDimension, nameof(centers));
            RequireLength(scales, parameterDimension, nameof(scales));
            RequireLength(parameterMinimum, parameterDimension, nameof(parameterMinimum));
            RequireLength(parameterMaximum, parameterDimension, nameof(parameterMaximum));
            RequireLength(poseMinimum, poseDimension, nameof(poseMinimum));
            RequireLength(poseMaximum, poseDimension, nameof(poseMaximum));
            RequireLength(deploymentPoseMinimum, poseDimension,
                nameof(deploymentPoseMinimum));
            RequireLength(deploymentPoseMaximum, poseDimension,
                nameof(deploymentPoseMaximum));
            RequireLength(parameterCholesky, parameterDimension * parameterDimension,
                nameof(parameterCholesky));
            RequireLength(previousPhysical, 9, nameof(previousPhysical));
            parameterCenter = (float[])centers.Clone();
            parameterScale = (float[])scales.Clone();
            parameterLow = (float[])parameterMinimum.Clone();
            parameterHigh = (float[])parameterMaximum.Clone();
            poseLow = (float[])poseMinimum.Clone();
            poseHigh = (float[])poseMaximum.Clone();
            deploymentPoseLow = (float[])deploymentPoseMinimum.Clone();
            deploymentPoseHigh = (float[])deploymentPoseMaximum.Clone();
            cholesky = (float[])parameterCholesky.Clone();
            hazardBias = calibratedHazardBias;
            durationMin = minimumDurationSteps;
            durationMax = maximumDurationSteps;
            cooldownSteps = eventCooldownSteps;
            durationScale = eventDurationScale;
            random = new PortableHmdRandom(derivedSeed);
            start = new float[poseDimension];
            target = new float[poseDimension];
            amplitude = new float[poseDimension];
            for (int axis = 0; axis < poseDimension; axis++)
            {
                start[axis] = previousPhysical[indices[axis]];
                target[axis] = start[axis];
            }
            durationSteps = minimumDurationSteps;
        }

        internal bool Active => active;
        internal bool EligibleThisStep { get; private set; }
        internal bool TriggeredThisStep { get; private set; }
        internal bool CompletedThisStep { get; private set; }
        internal bool ComfortEnvelopeActivatedThisStep { get; private set; }
        internal int CooldownRemaining => cooldownRemaining;
        internal int ElapsedSteps => elapsedSteps;
        internal int DurationSteps => durationSteps;
        internal int EventId => eventId;
        internal float Probability { get; private set; }
        internal float Phase { get; private set; }
        internal float MinimumJerkWeight { get; private set; }
        internal float[] StartPose => (float[])start.Clone();
        internal float[] TargetPose => (float[])target.Clone();
        internal float[] Amplitude => (float[])amplitude.Clone();

        internal void Compose(float[] statePhysical, float hazardLogit,
            float[] parameterMeanNormalized, float[] parameterLogStd,
            bool? triggerOverride = null, bool sampleParameterNoise = true)
        {
            RequireLength(statePhysical, 9, nameof(statePhysical));
            int parameterDimension = indices.Length + 1;
            RequireLength(parameterMeanNormalized, parameterDimension,
                nameof(parameterMeanNormalized));
            RequireLength(parameterLogStd, parameterDimension, nameof(parameterLogStd));

            Probability = Sigmoid((float)(hazardLogit + hazardBias));
            EligibleThisStep = !active && cooldownRemaining <= 0;
            TriggeredThisStep = triggerOverride.HasValue
                ? EligibleThisStep && triggerOverride.Value
                : EligibleThisStep && random.NextUniformOpen() < Probability;
            CompletedThisStep = false;
            ComfortEnvelopeActivatedThisStep = false;

            if (TriggeredThisStep)
            {
                float[] sampled = SampleParameters(
                    parameterMeanNormalized, parameterLogStd, sampleParameterNoise);
                for (int axis = 0; axis < indices.Length; axis++)
                {
                    float current = statePhysical[indices[axis]];
                    float learnedEndpoint = Mathf.Clamp(
                        sampled[axis], poseLow[axis], poseHigh[axis]);
                    float endpoint = Mathf.Clamp(learnedEndpoint,
                        deploymentPoseLow[axis], deploymentPoseHigh[axis]);
                    ComfortEnvelopeActivatedThisStep |= endpoint != learnedEndpoint;
                    start[axis] = current;
                    target[axis] = endpoint;
                    amplitude[axis] = (float)(endpoint - current);
                }
                double durationPhysical =
                    Math.Exp(sampled[parameterDimension - 1]) * durationScale;
                int rounded = (int)Math.Round(durationPhysical, MidpointRounding.ToEven);
                durationSteps = Math.Max(durationMin, Math.Min(durationMax, rounded));
                elapsedSteps = 0;
                active = true;
                eventId++;
            }
            else if (active)
            {
                elapsedSteps++;
            }

            bool activeNow = active;
            if (activeNow)
            {
                float denominator = Mathf.Max(1f, durationSteps - 1f);
                Phase = Mathf.Clamp01(elapsedSteps / denominator);
                float u = Phase;
                MinimumJerkWeight = u * u * u * (10f + u * (-15f + 6f * u));
                for (int axis = 0; axis < indices.Length; axis++)
                {
                    statePhysical[indices[axis]] =
                        start[axis] + (target[axis] - start[axis]) * MinimumJerkWeight;
                }
                if (elapsedSteps >= durationSteps - 1)
                {
                    for (int axis = 0; axis < indices.Length; axis++)
                        statePhysical[indices[axis]] = target[axis];
                    active = false;
                    elapsedSteps = 0;
                    cooldownRemaining = cooldownSteps;
                    CompletedThisStep = true;
                    Phase = 1f;
                    MinimumJerkWeight = 1f;
                }
            }
            else
            {
                Phase = 1f;
                MinimumJerkWeight = 1f;
                for (int axis = 0; axis < indices.Length; axis++)
                    statePhysical[indices[axis]] = target[axis];
                if (cooldownRemaining > 0) cooldownRemaining--;
            }
        }

        // [active,cooldown,elapsed,duration,start...,target...,amplitude...]
        internal float[] StateVector()
        {
            float[] values = new float[4 + indices.Length * 3];
            values[0] = active ? 1f : 0f;
            values[1] = cooldownRemaining;
            values[2] = elapsedSteps;
            values[3] = durationSteps;
            Array.Copy(start, 0, values, 4, indices.Length);
            Array.Copy(target, 0, values, 4 + indices.Length, indices.Length);
            Array.Copy(amplitude, 0, values, 4 + indices.Length * 2, indices.Length);
            return values;
        }

        private float[] SampleParameters(float[] mean, float[] logStd, bool addNoise)
        {
            int dimension = mean.Length;
            float[] gaussian = new float[dimension];
            if (addNoise)
            {
                for (int index = 0; index < dimension; index++)
                    gaussian[index] = random.NextGaussianFloat();
            }
            float[] sampled = new float[dimension];
            for (int row = 0; row < dimension; row++)
            {
                float correlated = 0f;
                for (int column = 0; column <= row; column++)
                {
                    float product = (float)(cholesky[row * dimension + column] * gaussian[column]);
                    correlated = (float)(correlated + product);
                }
                float boundedLogStd = Mathf.Clamp(logStd[row], -5f, 2f);
                float normalized = (float)(mean[row] +
                    (float)((float)Math.Exp(boundedLogStd) * correlated));
                float physical = (float)(normalized * parameterScale[row] + parameterCenter[row]);
                float midpoint = 0.5f * (parameterLow[row] + parameterHigh[row]);
                float halfRange = Mathf.Max(0.000001f,
                    0.5f * (parameterHigh[row] - parameterLow[row]));
                sampled[row] = (float)(midpoint +
                    halfRange * (float)Math.Tanh((physical - midpoint) / halfRange));
            }
            return sampled;
        }

        private static float Sigmoid(float value)
        {
            if (value >= 0f)
            {
                float exponential = (float)Math.Exp(-value);
                return 1f / (1f + exponential);
            }
            float negative = (float)Math.Exp(value);
            return negative / (1f + negative);
        }

        private static void RequireLength(Array values, int expected, string name)
        {
            if (values == null || values.Length != expected)
                throw new ArgumentException(name + " length mismatch; expected " + expected + ".");
        }
    }
}
