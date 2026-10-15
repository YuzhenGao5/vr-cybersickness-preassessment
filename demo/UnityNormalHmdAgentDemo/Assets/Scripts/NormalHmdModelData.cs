using System;

namespace NormalHmdAgentDemo
{
    [Serializable]
    public sealed class NormalHmdSeedData
    {
        public string trajectoryId;
        public string split;
        public int startFrame;
        public float[] historyRaw;
    }

    /// <summary>
    /// Flat, JsonUtility-compatible contract for Normal HMD Agent v4.
    /// Matrices are row-major and every LSTM uses PyTorch's i,f,g,o gate order.
    /// </summary>
    [Serializable]
    public sealed class NormalHmdModelData
    {
        public string schemaVersion;
        public string exportedAtUtc;
        public string dataSource;
        public bool syntheticTraining;
        public string sourceModelSha256;
        public string datasetFingerprintSha256;
        public string modelTask;
        public string architecture;
        public string lstmGateOrder;
        public string tensorLayout;
        public int parameterCount;

        public int historyLength;
        public int inputDim;
        public int hiddenDim;
        public int numLayers;
        public float fixedDtSeconds;
        public float trainedRolloutSeconds;
        public float nominalEyeHeightM;
        public string[] stateFeatureNames;
        public int[] rootStateIndices;
        public int[] horizontalStateIndices;
        public int[] verticalStateIndices;

        public float[] stateMean;
        public float[] stateStd;
        public float[] stateClipLow;
        public float[] stateClipHigh;

        public float[] horizontalParameterCenter;
        public float[] horizontalParameterScale;
        public float[] horizontalParameterLow;
        public float[] horizontalParameterHigh;
        public float[] verticalParameterCenter;
        public float[] verticalParameterScale;
        public float[] verticalParameterLow;
        public float[] verticalParameterHigh;
        public float[] horizontalPoseLow;
        public float[] horizontalPoseHigh;
        public float[] verticalPoseLow;
        public float[] verticalPoseHigh;
        public float[] horizontalParameterCholesky;
        public float[] verticalParameterCholesky;
        public float horizontalHazardCalibrationBias;
        public float verticalHazardCalibrationBias;

        public int durationMinSteps;
        public int durationMaxSteps;
        public int cooldownSteps;
        public float horizontalDurationScale;
        public float verticalDurationScale;
        public string eventInterpolation;
        public string runtimeFeedbackContract;
        public string headInnovationPolicy;

        public int rootInnovationDim;
        public float[] rootInnovationRho;
        public float[] rootInnovationCholesky;
        public float defaultRootNoiseScale;
        public float[] headInnovationStd;
        public bool rootInnovationFeedsBack;
        public string randomAlgorithm;
        public string seedDerivationAlgorithm;
        public uint rootSeedXorConstant;
        public uint horizontalSeedXorConstant;
        public uint verticalSeedXorConstant;
        public uint seedWindowXorConstant;

        public float[] rootWeightIhL0;
        public float[] rootWeightHhL0;
        public float[] rootBiasIhL0;
        public float[] rootBiasHhL0;
        public float[] horizontalWeightIhL0;
        public float[] horizontalWeightHhL0;
        public float[] horizontalBiasIhL0;
        public float[] horizontalBiasHhL0;
        public float[] verticalWeightIhL0;
        public float[] verticalWeightHhL0;
        public float[] verticalBiasIhL0;
        public float[] verticalBiasHhL0;

        public float[] rootOutputWeight;
        public float[] rootOutputBias;
        public float[] horizontalPoseWeight;
        public float[] horizontalPoseBias;
        public float[] horizontalHazardWeight;
        public float[] horizontalHazardBias;
        public float[] horizontalParameterMeanWeight;
        public float[] horizontalParameterMeanBias;
        public float[] horizontalParameterLogStdWeight;
        public float[] horizontalParameterLogStdBias;
        public float[] verticalHazardWeight;
        public float[] verticalHazardBias;
        public float[] verticalParameterMeanWeight;
        public float[] verticalParameterMeanBias;
        public float[] verticalParameterLogStdWeight;
        public float[] verticalParameterLogStdBias;

        public NormalHmdSeedData[] seedWindows;

        // Raw-head order is frozen as root[3], H pose[4], H hazard[1],
        // H mean[5], H logstd[5], V hazard[1], V mean[3], V logstd[3].
        public float[] goldenHistoryRaw;
        public int goldenSourceValRow;
        public int goldenStepCount;
        public float[] goldenExpectedRawHeads;
        public float[] goldenExpectedComposedBeforeInnovation;
        public float[] goldenExpectedApplied;
        public float[] goldenExpectedControllerState;
        public float[] goldenExpectedDiagnostics;

        public int eventGoldenRunSeed;
        public int eventGoldenStepCount;
        public float[] eventGoldenRawHeads;
        public float[] eventGoldenInitialPhysical;
        public float[] eventGoldenExpectedApplied;
        public float[] eventGoldenExpectedControllerState;
        public float[] eventGoldenExpectedDiagnostics;

        public int rootInnovationGoldenSeed;
        public uint rootInnovationGoldenDerivedSeed;
        public int rootInnovationGoldenStepCount;
        public float[] rootInnovationGoldenInitialResidual;
        public float[] rootInnovationGoldenExpected;

        public int combinedStochasticGoldenSeed;
        public uint combinedStochasticGoldenDerivedRootSeed;
        public uint combinedStochasticGoldenDerivedHorizontalSeed;
        public uint combinedStochasticGoldenDerivedVerticalSeed;
        public int combinedStochasticGoldenStepCount;
        public float combinedStochasticGoldenNoiseScale;
        public float[] combinedStochasticGoldenExpectedRawHeads;
        public float[] combinedStochasticGoldenExpectedComposedBeforeInnovation;
        public float[] combinedStochasticGoldenExpectedControllerState;
        public float[] combinedStochasticGoldenExpectedDiagnostics;
        public float[] combinedStochasticGoldenExpectedApplied;
        public float[] combinedStochasticGoldenExpectedRootInnovation;
        public float goldenTolerance;
    }
}
