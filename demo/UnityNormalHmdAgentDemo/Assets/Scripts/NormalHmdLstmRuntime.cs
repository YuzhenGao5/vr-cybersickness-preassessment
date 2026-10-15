using System;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    internal sealed class NormalHmdV4Step
    {
        internal readonly float[] RootNormalized;
        internal readonly float[] HorizontalPoseNormalized;
        internal readonly float HorizontalHazardLogit;
        internal readonly float[] HorizontalParameterMeanNormalized;
        internal readonly float[] HorizontalParameterLogStd;
        internal readonly float VerticalHazardLogit;
        internal readonly float[] VerticalParameterMeanNormalized;
        internal readonly float[] VerticalParameterLogStd;

        internal NormalHmdV4Step(float[] root, float[] horizontalPose,
            float horizontalHazard, float[] horizontalMean, float[] horizontalLogStd,
            float verticalHazard, float[] verticalMean, float[] verticalLogStd)
        {
            RootNormalized = root;
            HorizontalPoseNormalized = horizontalPose;
            HorizontalHazardLogit = horizontalHazard;
            HorizontalParameterMeanNormalized = horizontalMean;
            HorizontalParameterLogStd = horizontalLogStd;
            VerticalHazardLogit = verticalHazard;
            VerticalParameterMeanNormalized = verticalMean;
            VerticalParameterLogStd = verticalLogStd;
        }

        internal float[] RawHeads()
        {
            float[] values = new float[25];
            int offset = 0;
            Array.Copy(RootNormalized, 0, values, offset, 3); offset += 3;
            Array.Copy(HorizontalPoseNormalized, 0, values, offset, 4); offset += 4;
            values[offset++] = HorizontalHazardLogit;
            Array.Copy(HorizontalParameterMeanNormalized, 0, values, offset, 5); offset += 5;
            Array.Copy(HorizontalParameterLogStd, 0, values, offset, 5); offset += 5;
            values[offset++] = VerticalHazardLogit;
            Array.Copy(VerticalParameterMeanNormalized, 0, values, offset, 3); offset += 3;
            Array.Copy(VerticalParameterLogStd, 0, values, offset, 3);
            return values;
        }

        internal static NormalHmdV4Step FromRawHeads(float[] flat, int offset)
        {
            if (flat == null || offset < 0 || offset + 25 > flat.Length)
                throw new ArgumentException("A v4 raw-head row requires 25 values.");
            float[] root = Slice(flat, offset, 3); offset += 3;
            float[] pose = Slice(flat, offset, 4); offset += 4;
            float hHazard = flat[offset++];
            float[] hMean = Slice(flat, offset, 5); offset += 5;
            float[] hLogStd = Slice(flat, offset, 5); offset += 5;
            float vHazard = flat[offset++];
            float[] vMean = Slice(flat, offset, 3); offset += 3;
            float[] vLogStd = Slice(flat, offset, 3);
            return new NormalHmdV4Step(root, pose, hHazard, hMean, hLogStd,
                vHazard, vMean, vLogStd);
        }

        private static float[] Slice(float[] source, int offset, int count)
        {
            float[] values = new float[count];
            Array.Copy(source, offset, values, 0, count);
            return values;
        }
    }

    /// <summary>
    /// Dependency-free inference for the v4 three-branch model.  All branches
    /// receive the same final applied 9-D feedback but maintain independent
    /// recurrent state.  Unity startup runs Python-exported golden vectors for
    /// heads, event RNG/controllers, root RNG and the combined closed loop.
    /// </summary>
    public sealed class NormalHmdLstmRuntime
    {
        private sealed class Branch
        {
            internal readonly float[] Hidden;
            internal readonly float[] Cell;
            internal readonly float[] Gates;
            internal readonly float[] WeightInput;
            internal readonly float[] WeightHidden;
            internal readonly float[] BiasInput;
            internal readonly float[] BiasHidden;

            internal Branch(int hidden, float[] weightInput, float[] weightHidden,
                float[] biasInput, float[] biasHidden)
            {
                Hidden = new float[hidden];
                Cell = new float[hidden];
                Gates = new float[hidden * 4];
                WeightInput = weightInput;
                WeightHidden = weightHidden;
                BiasInput = biasInput;
                BiasHidden = biasHidden;
            }

            internal void Reset()
            {
                Array.Clear(Hidden, 0, Hidden.Length);
                Array.Clear(Cell, 0, Cell.Length);
            }
        }

        public NormalHmdModelData Data { get; }
        public string ExportSha256 { get; }
        public float GoldenMaxAbsError { get; }
        public float InnovationGoldenMaxAbsError { get; }
        public float EventGoldenMaxAbsError { get; }
        public float CombinedStochasticGoldenMaxAbsError { get; }
        public bool HasCombinedStochasticGolden => Data.combinedStochasticGoldenStepCount > 0;

        private readonly Branch root;
        private readonly Branch horizontal;
        private readonly Branch vertical;
        private readonly float[] normalizedInput;
        private bool warmed;

        private NormalHmdLstmRuntime(NormalHmdModelData data, string hash, bool runGolden)
        {
            Data = data;
            ExportSha256 = hash;
            root = new Branch(data.hiddenDim, data.rootWeightIhL0, data.rootWeightHhL0,
                data.rootBiasIhL0, data.rootBiasHhL0);
            horizontal = new Branch(data.hiddenDim, data.horizontalWeightIhL0,
                data.horizontalWeightHhL0, data.horizontalBiasIhL0,
                data.horizontalBiasHhL0);
            vertical = new Branch(data.hiddenDim, data.verticalWeightIhL0,
                data.verticalWeightHhL0, data.verticalBiasIhL0, data.verticalBiasHhL0);
            normalizedInput = new float[data.inputDim];
            GoldenMaxAbsError = runGolden ? RunDeterministicGoldenParity() : 0f;
            EventGoldenMaxAbsError = runGolden ? RunEventGoldenParity() : 0f;
            InnovationGoldenMaxAbsError = runGolden ? RunRootGoldenParity() : 0f;
            CombinedStochasticGoldenMaxAbsError = runGolden && HasCombinedStochasticGolden
                ? RunCombinedGoldenParity() : -1f;
        }

        public static NormalHmdLstmRuntime LoadVerified()
        {
            TextAsset asset = Resources.Load<TextAsset>("NormalHmdAgentModel");
            if (asset == null)
                throw new InvalidOperationException(
                    "NormalHmdAgentModel.json is missing. Export Normal HMD v4 first.");
            if (GeneratedNormalHmdIntegrity.ExportSha256 == "PENDING_EXPORT")
                throw new InvalidOperationException(
                    "GeneratedNormalHmdIntegrity.cs is a placeholder. Run the v4 exporter.");
            string actual = Sha256(asset.bytes);
            if (!string.Equals(actual, GeneratedNormalHmdIntegrity.ExportSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Embedded v4 model hash mismatch.");
            NormalHmdModelData data = JsonUtility.FromJson<NormalHmdModelData>(asset.text);
            Validate(data);
            if (!string.Equals(data.sourceModelSha256,
                    GeneratedNormalHmdIntegrity.SourceModelSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Normal HMD v4 checkpoint identity mismatch.");

            NormalHmdLstmRuntime runtime = new NormalHmdLstmRuntime(data, actual, true);
            float tolerance = data.goldenTolerance > 0f ? data.goldenTolerance : 0.0005f;
            RequireParity(runtime.GoldenMaxAbsError, tolerance, "deterministic heads/controllers");
            RequireParity(runtime.EventGoldenMaxAbsError, tolerance, "event RNG/controllers");
            RequireParity(runtime.InnovationGoldenMaxAbsError, tolerance, "root RNG");
            if (runtime.HasCombinedStochasticGolden)
                RequireParity(runtime.CombinedStochasticGoldenMaxAbsError, tolerance,
                    "combined closed loop");
            return runtime;
        }

        internal void Warm(float[] rawHistory)
        {
            RequireLength(rawHistory, Data.historyLength * 9, "seed history");
            root.Reset(); horizontal.Reset(); vertical.Reset();
            for (int step = 0; step < Data.historyLength; step++)
            {
                int offset = step * 9;
                for (int feature = 0; feature < 9; feature++)
                    normalizedInput[feature] =
                        (rawHistory[offset + feature] - Data.stateMean[feature]) /
                        Data.stateStd[feature];
                AdvanceNormalized(normalizedInput);
            }
            warmed = true;
        }

        internal NormalHmdV4Step PredictHeads()
        {
            if (!warmed) throw new InvalidOperationException("Warm must precede prediction.");
            float[] rootOutput = Dense(root.Hidden, Data.rootOutputWeight,
                Data.rootOutputBias, 3);
            float[] hPose = Dense(horizontal.Hidden, Data.horizontalPoseWeight,
                Data.horizontalPoseBias, 4);
            float hHazard = DenseScalar(horizontal.Hidden, Data.horizontalHazardWeight,
                Data.horizontalHazardBias[0]);
            float[] hMean = Dense(horizontal.Hidden, Data.horizontalParameterMeanWeight,
                Data.horizontalParameterMeanBias, 5);
            float[] hLogStd = Dense(horizontal.Hidden,
                Data.horizontalParameterLogStdWeight,
                Data.horizontalParameterLogStdBias, 5);
            ClampArray(hLogStd, -5f, 2f);
            float vHazard = DenseScalar(vertical.Hidden, Data.verticalHazardWeight,
                Data.verticalHazardBias[0]);
            float[] vMean = Dense(vertical.Hidden, Data.verticalParameterMeanWeight,
                Data.verticalParameterMeanBias, 3);
            float[] vLogStd = Dense(vertical.Hidden, Data.verticalParameterLogStdWeight,
                Data.verticalParameterLogStdBias, 3);
            ClampArray(vLogStd, -5f, 2f);
            return new NormalHmdV4Step(rootOutput, hPose, hHazard, hMean, hLogStd,
                vHazard, vMean, vLogStd);
        }

        internal void AdvanceExecuted(float[] finalAppliedPhysical)
        {
            if (!warmed) throw new InvalidOperationException("Warm must precede advance.");
            RequireLength(finalAppliedPhysical, 9, "final applied feedback");
            for (int feature = 0; feature < 9; feature++)
                normalizedInput[feature] =
                    (finalAppliedPhysical[feature] - Data.stateMean[feature]) /
                    Data.stateStd[feature];
            AdvanceNormalized(normalizedInput);
        }

        internal NormalHmdEventController CreateHorizontalController(
            float[] previousPhysical, uint derivedSeed)
        {
            float[] deploymentLow = IntersectPoseBounds(
                Data.horizontalStateIndices, Data.horizontalPoseLow, true);
            float[] deploymentHigh = IntersectPoseBounds(
                Data.horizontalStateIndices, Data.horizontalPoseHigh, false);
            return new NormalHmdEventController(Data.horizontalStateIndices,
                Data.horizontalParameterCenter, Data.horizontalParameterScale,
                Data.horizontalParameterLow, Data.horizontalParameterHigh,
                Data.horizontalPoseLow, Data.horizontalPoseHigh,
                deploymentLow, deploymentHigh,
                Data.horizontalParameterCholesky, Data.horizontalHazardCalibrationBias,
                Data.durationMinSteps, Data.durationMaxSteps, Data.cooldownSteps,
                Data.horizontalDurationScale, derivedSeed, previousPhysical);
        }

        internal NormalHmdEventController CreateVerticalController(
            float[] previousPhysical, uint derivedSeed)
        {
            float[] deploymentLow = IntersectPoseBounds(
                Data.verticalStateIndices, Data.verticalPoseLow, true);
            float[] deploymentHigh = IntersectPoseBounds(
                Data.verticalStateIndices, Data.verticalPoseHigh, false);
            return new NormalHmdEventController(Data.verticalStateIndices,
                Data.verticalParameterCenter, Data.verticalParameterScale,
                Data.verticalParameterLow, Data.verticalParameterHigh,
                Data.verticalPoseLow, Data.verticalPoseHigh,
                deploymentLow, deploymentHigh,
                Data.verticalParameterCholesky, Data.verticalHazardCalibrationBias,
                Data.durationMinSteps, Data.durationMaxSteps, Data.cooldownSteps,
                Data.verticalDurationScale, derivedSeed, previousPhysical);
        }

        private float[] IntersectPoseBounds(int[] indices, float[] poseBounds, bool lower)
        {
            float[] values = new float[indices.Length];
            for (int index = 0; index < indices.Length; index++)
            {
                float safety = lower
                    ? Data.stateClipLow[indices[index]]
                    : Data.stateClipHigh[indices[index]];
                values[index] = lower
                    ? Mathf.Max(poseBounds[index], safety)
                    : Mathf.Min(poseBounds[index], safety);
            }
            return values;
        }

        internal float[] ComposeBeforeInnovation(float[] previous, NormalHmdV4Step step,
            NormalHmdEventController hController, NormalHmdEventController vController,
            bool deterministicController = false)
        {
            float[] state = (float[])previous.Clone();
            for (int index = 0; index < 3; index++)
            {
                int stateIndex = Data.rootStateIndices[index];
                state[stateIndex] = step.RootNormalized[index] * Data.stateStd[stateIndex] +
                    Data.stateMean[stateIndex];
            }
            bool? hOverride = deterministicController
                ? step.HorizontalHazardLogit + Data.horizontalHazardCalibrationBias >= 0f : (bool?)null;
            bool? vOverride = deterministicController
                ? step.VerticalHazardLogit + Data.verticalHazardCalibrationBias >= 0f : (bool?)null;
            hController.Compose(state, step.HorizontalHazardLogit,
                step.HorizontalParameterMeanNormalized, step.HorizontalParameterLogStd,
                hOverride, !deterministicController);
            vController.Compose(state, step.VerticalHazardLogit,
                step.VerticalParameterMeanNormalized, step.VerticalParameterLogStd,
                vOverride, !deterministicController);
            return state;
        }

        internal float[] ApplyInnovationAndSafety(float[] composed,
            float[] rootInnovation, out bool[] clipped)
        {
            float[] applied = (float[])composed.Clone();
            if (rootInnovation != null)
            {
                for (int index = 0; index < 3; index++)
                    applied[Data.rootStateIndices[index]] += rootInnovation[index];
            }
            clipped = new bool[9];
            for (int feature = 0; feature < 9; feature++)
            {
                float safe = Mathf.Clamp(applied[feature], Data.stateClipLow[feature],
                    Data.stateClipHigh[feature]);
                clipped[feature] = safe != applied[feature];
                applied[feature] = safe;
            }
            return applied;
        }

        internal static float[] ControllerState(
            NormalHmdEventController horizontalController,
            NormalHmdEventController verticalController)
        {
            float[] h = horizontalController.StateVector();
            float[] v = verticalController.StateVector();
            float[] state = new float[h.Length + v.Length];
            Array.Copy(h, 0, state, 0, h.Length);
            Array.Copy(v, 0, state, h.Length, v.Length);
            return state;
        }

        internal static float[] Diagnostics(
            NormalHmdEventController horizontalController,
            NormalHmdEventController verticalController)
        {
            return new[]
            {
                horizontalController.TriggeredThisStep ? 1f : 0f,
                verticalController.TriggeredThisStep ? 1f : 0f,
                horizontalController.EligibleThisStep ? 1f : 0f,
                verticalController.EligibleThisStep ? 1f : 0f,
                horizontalController.Probability,
                verticalController.Probability,
            };
        }

        private float RunDeterministicGoldenParity()
        {
            Warm(Data.goldenHistoryRaw);
            float[] previous = LastHistoryState(Data.goldenHistoryRaw);
            uint hSeed = PortableHmdRandom.DeriveSeed(0, Data.horizontalSeedXorConstant);
            uint vSeed = PortableHmdRandom.DeriveSeed(0, Data.verticalSeedXorConstant);
            NormalHmdEventController hController = CreateHorizontalController(previous, hSeed);
            NormalHmdEventController vController = CreateVerticalController(previous, vSeed);
            float maximum = 0f;
            for (int stepIndex = 0; stepIndex < Data.goldenStepCount; stepIndex++)
            {
                NormalHmdV4Step step = PredictHeads();
                maximum = CompareRow(maximum, step.RawHeads(),
                    Data.goldenExpectedRawHeads, stepIndex, 25);
                float[] composed = ComposeBeforeInnovation(previous, step,
                    hController, vController, true);
                maximum = CompareRow(maximum, composed,
                    Data.goldenExpectedComposedBeforeInnovation, stepIndex, 9);
                bool[] ignored;
                float[] applied = ApplyInnovationAndSafety(composed, null, out ignored);
                maximum = CompareRow(maximum, applied,
                    Data.goldenExpectedApplied, stepIndex, 9);
                maximum = CompareRow(maximum, ControllerState(hController, vController),
                    Data.goldenExpectedControllerState, stepIndex, 26);
                maximum = CompareRow(maximum, Diagnostics(hController, vController),
                    Data.goldenExpectedDiagnostics, stepIndex, 6);
                previous = applied;
                AdvanceExecuted(applied);
            }
            return maximum;
        }

        private float RunEventGoldenParity()
        {
            float[] previous = (float[])Data.eventGoldenInitialPhysical.Clone();
            uint hSeed = PortableHmdRandom.DeriveSeed(Data.eventGoldenRunSeed,
                Data.horizontalSeedXorConstant);
            uint vSeed = PortableHmdRandom.DeriveSeed(Data.eventGoldenRunSeed,
                Data.verticalSeedXorConstant);
            NormalHmdEventController hController = CreateHorizontalController(previous, hSeed);
            NormalHmdEventController vController = CreateVerticalController(previous, vSeed);
            float maximum = 0f;
            for (int stepIndex = 0; stepIndex < Data.eventGoldenStepCount; stepIndex++)
            {
                NormalHmdV4Step step = NormalHmdV4Step.FromRawHeads(
                    Data.eventGoldenRawHeads, stepIndex * 25);
                float[] composed = ComposeBeforeInnovation(previous, step,
                    hController, vController, false);
                bool[] ignored;
                float[] applied = ApplyInnovationAndSafety(composed, null, out ignored);
                maximum = CompareRow(maximum, applied,
                    Data.eventGoldenExpectedApplied, stepIndex, 9);
                maximum = CompareRow(maximum, ControllerState(hController, vController),
                    Data.eventGoldenExpectedControllerState, stepIndex, 26);
                maximum = CompareRow(maximum, Diagnostics(hController, vController),
                    Data.eventGoldenExpectedDiagnostics, stepIndex, 6);
                previous = applied;
            }
            return maximum;
        }

        private float RunRootGoldenParity()
        {
            uint derived = PortableHmdRandom.DeriveSeed(Data.rootInnovationGoldenSeed,
                Data.rootSeedXorConstant);
            float maximum = Mathf.Abs((float)((long)derived -
                (long)Data.rootInnovationGoldenDerivedSeed));
            SeededRootInnovation innovation = new SeededRootInnovation(derived,
                Data.rootInnovationRho, Data.rootInnovationCholesky);
            maximum = CompareRow(maximum, innovation.InitialResidual,
                Data.rootInnovationGoldenInitialResidual, 0, 3);
            for (int step = 0; step < Data.rootInnovationGoldenStepCount; step++)
                maximum = CompareRow(maximum, innovation.Next(1f),
                    Data.rootInnovationGoldenExpected, step, 3);
            return maximum;
        }

        private float RunCombinedGoldenParity()
        {
            Warm(Data.goldenHistoryRaw);
            float[] previous = LastHistoryState(Data.goldenHistoryRaw);
            int runSeed = Data.combinedStochasticGoldenSeed;
            uint rootSeed = PortableHmdRandom.DeriveSeed(runSeed, Data.rootSeedXorConstant);
            uint hSeed = PortableHmdRandom.DeriveSeed(runSeed, Data.horizontalSeedXorConstant);
            uint vSeed = PortableHmdRandom.DeriveSeed(runSeed, Data.verticalSeedXorConstant);
            float maximum = 0f;
            maximum = CompareUInt(maximum, rootSeed,
                Data.combinedStochasticGoldenDerivedRootSeed);
            maximum = CompareUInt(maximum, hSeed,
                Data.combinedStochasticGoldenDerivedHorizontalSeed);
            maximum = CompareUInt(maximum, vSeed,
                Data.combinedStochasticGoldenDerivedVerticalSeed);
            SeededRootInnovation innovation = new SeededRootInnovation(rootSeed,
                Data.rootInnovationRho, Data.rootInnovationCholesky);
            NormalHmdEventController hController = CreateHorizontalController(previous, hSeed);
            NormalHmdEventController vController = CreateVerticalController(previous, vSeed);
            for (int stepIndex = 0;
                stepIndex < Data.combinedStochasticGoldenStepCount; stepIndex++)
            {
                NormalHmdV4Step step = PredictHeads();
                maximum = CompareRow(maximum, step.RawHeads(),
                    Data.combinedStochasticGoldenExpectedRawHeads, stepIndex, 25);
                float[] composed = ComposeBeforeInnovation(previous, step,
                    hController, vController, false);
                maximum = CompareRow(maximum, composed,
                    Data.combinedStochasticGoldenExpectedComposedBeforeInnovation,
                    stepIndex, 9);
                float[] rootResidual = innovation.Next(1f);
                maximum = CompareRow(maximum, rootResidual,
                    Data.combinedStochasticGoldenExpectedRootInnovation, stepIndex, 3);
                float[] rootNoise = new float[3];
                for (int index = 0; index < 3; index++)
                    rootNoise[index] = rootResidual[index] *
                        Data.combinedStochasticGoldenNoiseScale;
                bool[] ignored;
                float[] applied = ApplyInnovationAndSafety(composed, rootNoise, out ignored);
                maximum = CompareRow(maximum, applied,
                    Data.combinedStochasticGoldenExpectedApplied, stepIndex, 9);
                maximum = CompareRow(maximum, ControllerState(hController, vController),
                    Data.combinedStochasticGoldenExpectedControllerState, stepIndex, 26);
                maximum = CompareRow(maximum, Diagnostics(hController, vController),
                    Data.combinedStochasticGoldenExpectedDiagnostics, stepIndex, 6);
                previous = applied;
                AdvanceExecuted(applied);
            }
            return maximum;
        }

        private void AdvanceNormalized(float[] input)
        {
            LstmLayer(input, root);
            LstmLayer(input, horizontal);
            LstmLayer(input, vertical);
        }

        private void LstmLayer(float[] input, Branch branch)
        {
            int hidden = Data.hiddenDim;
            for (int row = 0; row < hidden * 4; row++)
            {
                float sum = branch.BiasInput[row] + branch.BiasHidden[row];
                int inputRow = row * Data.inputDim;
                for (int column = 0; column < Data.inputDim; column++)
                    sum += branch.WeightInput[inputRow + column] * input[column];
                int hiddenRow = row * hidden;
                for (int column = 0; column < hidden; column++)
                    sum += branch.WeightHidden[hiddenRow + column] * branch.Hidden[column];
                branch.Gates[row] = sum;
            }
            for (int index = 0; index < hidden; index++)
            {
                float inputGate = Sigmoid(branch.Gates[index]);
                float forgetGate = Sigmoid(branch.Gates[hidden + index]);
                float candidate = (float)Math.Tanh(branch.Gates[hidden * 2 + index]);
                float outputGate = Sigmoid(branch.Gates[hidden * 3 + index]);
                float nextCell = forgetGate * branch.Cell[index] + inputGate * candidate;
                branch.Cell[index] = nextCell;
                branch.Hidden[index] = outputGate * (float)Math.Tanh(nextCell);
            }
        }

        private static float[] Dense(float[] hidden, float[] weights,
            float[] bias, int outputCount)
        {
            float[] values = new float[outputCount];
            for (int output = 0; output < outputCount; output++)
            {
                float value = bias[output];
                int row = output * hidden.Length;
                for (int column = 0; column < hidden.Length; column++)
                    value += weights[row + column] * hidden[column];
                values[output] = value;
            }
            return values;
        }

        private static float DenseScalar(float[] hidden, float[] weights, float bias)
        {
            float value = bias;
            for (int column = 0; column < hidden.Length; column++)
                value += weights[column] * hidden[column];
            return value;
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

        private static void ClampArray(float[] values, float low, float high)
        {
            for (int index = 0; index < values.Length; index++)
                values[index] = Mathf.Clamp(values[index], low, high);
        }

        private float[] LastHistoryState(float[] history)
        {
            float[] state = new float[9];
            Array.Copy(history, history.Length - 9, state, 0, 9);
            for (int feature = 0; feature < 9; feature++)
                state[feature] = Mathf.Clamp(state[feature],
                    Data.stateClipLow[feature], Data.stateClipHigh[feature]);
            return state;
        }

        private static float CompareRow(float maximum, float[] actual,
            float[] expected, int row, int width)
        {
            int offset = row * width;
            for (int index = 0; index < width; index++)
                maximum = Mathf.Max(maximum,
                    Mathf.Abs(actual[index] - expected[offset + index]));
            return maximum;
        }

        private static float CompareUInt(float maximum, uint actual, uint expected)
        {
            return actual == expected ? maximum : float.PositiveInfinity;
        }

        private static void RequireParity(float error, float tolerance, string name)
        {
            if (!float.IsFinite(error) || error > tolerance)
                throw new InvalidOperationException("Unity/Python v4 " + name +
                    " parity failed: max_abs_error=" + error.ToString("G9") +
                    ", tolerance=" + tolerance.ToString("G9") + ".");
        }

        private static void Validate(NormalHmdModelData data)
        {
            if (data == null || data.schemaVersion != "normal-hmd-agent-unity-4.0")
                throw new InvalidOperationException("Unsupported Normal HMD schema; v4 required.");
            if (data.dataSource != "EyeNavGS" || data.syntheticTraining)
                throw new InvalidOperationException("Normal HMD v4 requires real EyeNavGS training.");
            if (data.architecture != "independent_root_horizontal_vertical_lstm_holding_events" ||
                data.lstmGateOrder != "ifgo" || data.tensorLayout != "row-major")
                throw new InvalidOperationException("Unexpected v4 architecture/layout.");
            if (data.inputDim != 9 || data.hiddenDim != 96 || data.numLayers != 1 ||
                data.historyLength < 2)
                throw new InvalidOperationException("Unexpected v4 dimensions.");
            RequireIndices(data.rootStateIndices, new[] { 0, 1, 2 }, "rootStateIndices");
            RequireIndices(data.horizontalStateIndices, new[] { 3, 5, 7, 8 },
                "horizontalStateIndices");
            RequireIndices(data.verticalStateIndices, new[] { 4, 6 }, "verticalStateIndices");
            RequireLength(data.stateFeatureNames, 9, "stateFeatureNames");
            RequireLength(data.stateMean, 9, "stateMean");
            RequireLength(data.stateStd, 9, "stateStd");
            RequireLength(data.stateClipLow, 9, "stateClipLow");
            RequireLength(data.stateClipHigh, 9, "stateClipHigh");
            for (int index = 0; index < 9; index++)
            {
                if (!(data.stateStd[index] > 0f) ||
                    !(data.stateClipHigh[index] > data.stateClipLow[index]))
                    throw new InvalidOperationException("Invalid state normalization/safety bounds.");
            }
            if (data.stateClipLow[8] < -0.120001f || data.stateClipHigh[8] > 0.120001f)
                throw new InvalidOperationException("v4 comfort roll envelope must remain ±0.12 rad.");
            RequireLength(data.horizontalParameterCenter, 5, "horizontalParameterCenter");
            RequireLength(data.horizontalParameterScale, 5, "horizontalParameterScale");
            RequireLength(data.horizontalParameterLow, 5, "horizontalParameterLow");
            RequireLength(data.horizontalParameterHigh, 5, "horizontalParameterHigh");
            RequireLength(data.verticalParameterCenter, 3, "verticalParameterCenter");
            RequireLength(data.verticalParameterScale, 3, "verticalParameterScale");
            RequireLength(data.verticalParameterLow, 3, "verticalParameterLow");
            RequireLength(data.verticalParameterHigh, 3, "verticalParameterHigh");
            RequireLength(data.horizontalPoseLow, 4, "horizontalPoseLow");
            RequireLength(data.horizontalPoseHigh, 4, "horizontalPoseHigh");
            RequireLength(data.verticalPoseLow, 2, "verticalPoseLow");
            RequireLength(data.verticalPoseHigh, 2, "verticalPoseHigh");
            RequireLength(data.horizontalParameterCholesky, 25,
                "horizontalParameterCholesky");
            RequireLength(data.verticalParameterCholesky, 9,
                "verticalParameterCholesky");
            if (data.durationMinSteps != 5 || data.durationMaxSteps != 148 ||
                data.cooldownSteps != 4 ||
                data.horizontalDurationScale != 1f || data.verticalDurationScale != 1f ||
                data.eventInterpolation != "10u^3-15u^4+6u^5")
                throw new InvalidOperationException("Unexpected v4 event-controller contract.");
            if (data.rootInnovationDim != 3 || !data.rootInnovationFeedsBack ||
                data.defaultRootNoiseScale < 0f)
                throw new InvalidOperationException("Invalid root innovation contract.");
            RequireLength(data.rootInnovationRho, 3, "rootInnovationRho");
            RequireLength(data.rootInnovationCholesky, 9, "rootInnovationCholesky");
            RequireLength(data.headInnovationStd, 6, "headInnovationStd");
            for (int index = 0; index < 6; index++)
                if (data.headInnovationStd[index] != 0f)
                    throw new InvalidOperationException("HeadLocal per-frame noise must be zero.");
            if (data.rootSeedXorConstant != 0xA341316Cu ||
                data.horizontalSeedXorConstant != 0xC8013EA4u ||
                data.verticalSeedXorConstant != 0xAD90777Du ||
                data.seedWindowXorConstant != 0x7E95761Eu)
                throw new InvalidOperationException("v4 independent RNG stream constants changed.");

            int h = data.hiddenDim;
            ValidateBranch(data.rootWeightIhL0, data.rootWeightHhL0,
                data.rootBiasIhL0, data.rootBiasHhL0, h, "root");
            ValidateBranch(data.horizontalWeightIhL0, data.horizontalWeightHhL0,
                data.horizontalBiasIhL0, data.horizontalBiasHhL0, h, "horizontal");
            ValidateBranch(data.verticalWeightIhL0, data.verticalWeightHhL0,
                data.verticalBiasIhL0, data.verticalBiasHhL0, h, "vertical");
            RequireLength(data.rootOutputWeight, 3 * h, "rootOutputWeight");
            RequireLength(data.rootOutputBias, 3, "rootOutputBias");
            RequireLength(data.horizontalPoseWeight, 4 * h, "horizontalPoseWeight");
            RequireLength(data.horizontalPoseBias, 4, "horizontalPoseBias");
            RequireLength(data.horizontalHazardWeight, h, "horizontalHazardWeight");
            RequireLength(data.horizontalHazardBias, 1, "horizontalHazardBias");
            RequireLength(data.horizontalParameterMeanWeight, 5 * h,
                "horizontalParameterMeanWeight");
            RequireLength(data.horizontalParameterMeanBias, 5,
                "horizontalParameterMeanBias");
            RequireLength(data.horizontalParameterLogStdWeight, 5 * h,
                "horizontalParameterLogStdWeight");
            RequireLength(data.horizontalParameterLogStdBias, 5,
                "horizontalParameterLogStdBias");
            RequireLength(data.verticalHazardWeight, h, "verticalHazardWeight");
            RequireLength(data.verticalHazardBias, 1, "verticalHazardBias");
            RequireLength(data.verticalParameterMeanWeight, 3 * h,
                "verticalParameterMeanWeight");
            RequireLength(data.verticalParameterMeanBias, 3,
                "verticalParameterMeanBias");
            RequireLength(data.verticalParameterLogStdWeight, 3 * h,
                "verticalParameterLogStdWeight");
            RequireLength(data.verticalParameterLogStdBias, 3,
                "verticalParameterLogStdBias");

            if (data.seedWindows == null || data.seedWindows.Length < 2 ||
                data.seedWindows.Length != GeneratedNormalHmdIntegrity.SeedWindowCount)
                throw new InvalidOperationException("Held-out validation seed bank is incomplete.");
            foreach (NormalHmdSeedData seed in data.seedWindows)
                if (seed == null || seed.split != "val" ||
                    seed.historyRaw == null || seed.historyRaw.Length != data.historyLength * 9)
                    throw new InvalidOperationException("Invalid held-out validation seed.");

            RequireLength(data.goldenHistoryRaw, data.historyLength * 9, "goldenHistoryRaw");
            if (data.goldenStepCount < 2 || data.eventGoldenStepCount < 2 ||
                data.rootInnovationGoldenStepCount < 1 ||
                data.combinedStochasticGoldenStepCount < 2)
                throw new InvalidOperationException("v4 golden coverage is incomplete.");
            RequireLength(data.goldenExpectedRawHeads, data.goldenStepCount * 25,
                "goldenExpectedRawHeads");
            RequireLength(data.goldenExpectedComposedBeforeInnovation,
                data.goldenStepCount * 9, "goldenExpectedComposedBeforeInnovation");
            RequireLength(data.goldenExpectedApplied, data.goldenStepCount * 9,
                "goldenExpectedApplied");
            RequireLength(data.goldenExpectedControllerState, data.goldenStepCount * 26,
                "goldenExpectedControllerState");
            RequireLength(data.goldenExpectedDiagnostics, data.goldenStepCount * 6,
                "goldenExpectedDiagnostics");
            RequireLength(data.eventGoldenRawHeads, data.eventGoldenStepCount * 25,
                "eventGoldenRawHeads");
            RequireLength(data.eventGoldenInitialPhysical, 9, "eventGoldenInitialPhysical");
            RequireLength(data.eventGoldenExpectedApplied, data.eventGoldenStepCount * 9,
                "eventGoldenExpectedApplied");
            RequireLength(data.eventGoldenExpectedControllerState,
                data.eventGoldenStepCount * 26, "eventGoldenExpectedControllerState");
            RequireLength(data.eventGoldenExpectedDiagnostics, data.eventGoldenStepCount * 6,
                "eventGoldenExpectedDiagnostics");
            RequireLength(data.rootInnovationGoldenInitialResidual, 3,
                "rootInnovationGoldenInitialResidual");
            RequireLength(data.rootInnovationGoldenExpected,
                data.rootInnovationGoldenStepCount * 3, "rootInnovationGoldenExpected");
            RequireLength(data.combinedStochasticGoldenExpectedRawHeads,
                data.combinedStochasticGoldenStepCount * 25,
                "combinedStochasticGoldenExpectedRawHeads");
            RequireLength(data.combinedStochasticGoldenExpectedComposedBeforeInnovation,
                data.combinedStochasticGoldenStepCount * 9,
                "combinedStochasticGoldenExpectedComposedBeforeInnovation");
            RequireLength(data.combinedStochasticGoldenExpectedControllerState,
                data.combinedStochasticGoldenStepCount * 26,
                "combinedStochasticGoldenExpectedControllerState");
            RequireLength(data.combinedStochasticGoldenExpectedDiagnostics,
                data.combinedStochasticGoldenStepCount * 6,
                "combinedStochasticGoldenExpectedDiagnostics");
            RequireLength(data.combinedStochasticGoldenExpectedApplied,
                data.combinedStochasticGoldenStepCount * 9,
                "combinedStochasticGoldenExpectedApplied");
            RequireLength(data.combinedStochasticGoldenExpectedRootInnovation,
                data.combinedStochasticGoldenStepCount * 3,
                "combinedStochasticGoldenExpectedRootInnovation");
        }

        private static void ValidateBranch(float[] wi, float[] wh, float[] bi,
            float[] bh, int hidden, string name)
        {
            RequireLength(wi, 4 * hidden * 9, name + "WeightIhL0");
            RequireLength(wh, 4 * hidden * hidden, name + "WeightHhL0");
            RequireLength(bi, 4 * hidden, name + "BiasIhL0");
            RequireLength(bh, 4 * hidden, name + "BiasHhL0");
        }

        private static void RequireIndices(int[] actual, int[] expected, string name)
        {
            RequireLength(actual, expected.Length, name);
            for (int index = 0; index < expected.Length; index++)
                if (actual[index] != expected[index])
                    throw new InvalidOperationException(name + " order mismatch.");
        }

        private static void RequireLength(Array values, int expected, string name)
        {
            if (values == null || values.Length != expected)
                throw new InvalidOperationException(name + " length mismatch; expected " +
                    expected + ".");
        }

        private static string Sha256(byte[] bytes)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(bytes);
                StringBuilder builder = new StringBuilder(digest.Length * 2);
                foreach (byte value in digest) builder.Append(value.ToString("x2"));
                return builder.ToString();
            }
        }
    }
}
