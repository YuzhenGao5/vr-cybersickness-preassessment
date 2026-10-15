using System;

namespace RandomAgentDemo
{
    [Serializable]
    internal sealed class SeedWindowData
    {
        public string trajectoryId;
        public int windowIndex;
        public int seedRank;
        public float[] values;
    }

    [Serializable]
    internal sealed class EyeNavModelData
    {
        public string schemaVersion;
        public string exportedAtUtc;
        public string dataSource;
        public bool syntheticTraining;
        public string sourceModelSha256;
        public string datasetFingerprintSha256;
        public string modelTask;
        public int parameterCount;
        public string[] featureNames;
        public string lstmGateOrder;
        public string tensorLayout;
        public string sourceCoordinateSystem;
        public string unityCoordinateAdapter;
        public string testDataSha256;
        public int historyLength;
        public int inputDim;
        public int hiddenDim;
        public int outputDim;
        public int numLayers;
        public float medianDtSeconds;
        public float[] xMean;
        public float[] xStd;
        public float[] yMean;
        public float[] yStd;
        public float[] actionClipLow;
        public float[] actionClipHigh;
        public float[] weightIhL0;
        public float[] weightHhL0;
        public float[] biasIhL0;
        public float[] biasHhL0;
        public float[] weightIhL1;
        public float[] weightHhL1;
        public float[] biasIhL1;
        public float[] biasHhL1;
        public float[] fcWeight;
        public float[] fcBias;
        public SeedWindowData[] seedWindows;
        public float[] selfTestInput;
        public float[] selfTestExpectedOutput;
        public int testSampleCount;
        public int testTrajectoryCount;
        public float testLinearVelocityVectorMaeMps;
        public float testAngularVelocityVectorMaeDegps;
        public float persistenceLinearVelocityVectorMaeMps;
        public float persistenceAngularVelocityVectorMaeDegps;
    }
}
