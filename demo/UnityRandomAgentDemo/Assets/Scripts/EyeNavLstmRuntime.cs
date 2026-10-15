using System;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace RandomAgentDemo
{
    public sealed class EyeNavLstmRuntime
    {
        internal EyeNavModelData Data { get; }
        public string ExportSha256 { get; }
        public float SelfTestMaxAbsError { get; }

        private EyeNavLstmRuntime(EyeNavModelData data, string exportSha256, float selfTestError)
        {
            Data = data;
            ExportSha256 = exportSha256;
            SelfTestMaxAbsError = selfTestError;
        }

        public static EyeNavLstmRuntime LoadVerified()
        {
            TextAsset asset = Resources.Load<TextAsset>("EyeNavAgentModel");
            if (asset == null)
            {
                throw new InvalidOperationException("Embedded EyeNavAgentModel asset is missing.");
            }

            string actualHash;
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(asset.bytes);
                StringBuilder builder = new StringBuilder(digest.Length * 2);
                foreach (byte value in digest)
                {
                    builder.Append(value.ToString("x2"));
                }
                actualHash = builder.ToString();
            }

            if (!string.Equals(actualHash, GeneratedModelIntegrity.ExportSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    "Embedded model export hash mismatch. Expected " +
                    GeneratedModelIntegrity.ExportSha256 + ", got " + actualHash + ".");
            }

            EyeNavModelData data = JsonUtility.FromJson<EyeNavModelData>(asset.text);
            Validate(data);
            if (!string.Equals(
                    data.sourceModelSha256,
                    GeneratedModelIntegrity.SourceModelSha256,
                    StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("Source PyTorch checkpoint identity mismatch.");
            }

            float[] predicted = Predict(data, data.selfTestInput);
            float maxError = 0f;
            for (int index = 0; index < data.outputDim; index++)
            {
                maxError = Mathf.Max(maxError, Mathf.Abs(predicted[index] - data.selfTestExpectedOutput[index]));
            }
            if (!float.IsFinite(maxError) || maxError > 0.0002f)
            {
                throw new InvalidOperationException(
                    "Exported LSTM failed the PyTorch parity self-test. max_abs_error=" +
                    maxError.ToString("G9"));
            }

            return new EyeNavLstmRuntime(data, actualHash, maxError);
        }

        internal float[] Predict(float[] history)
        {
            return Predict(Data, history);
        }

        private static float[] Predict(EyeNavModelData data, float[] history)
        {
            int hidden = data.hiddenDim;
            float[] h0 = new float[hidden];
            float[] c0 = new float[hidden];
            float[] h1 = new float[hidden];
            float[] c1 = new float[hidden];
            float[] input = new float[data.inputDim];

            for (int step = 0; step < data.historyLength; step++)
            {
                int offset = step * data.inputDim;
                for (int feature = 0; feature < data.inputDim; feature++)
                {
                    input[feature] = (history[offset + feature] - data.xMean[feature]) / data.xStd[feature];
                }

                LstmLayer(
                    input,
                    data.inputDim,
                    h0,
                    c0,
                    data.weightIhL0,
                    data.weightHhL0,
                    data.biasIhL0,
                    data.biasHhL0);
                LstmLayer(
                    h0,
                    hidden,
                    h1,
                    c1,
                    data.weightIhL1,
                    data.weightHhL1,
                    data.biasIhL1,
                    data.biasHhL1);
            }

            float[] result = new float[data.outputDim];
            for (int output = 0; output < data.outputDim; output++)
            {
                float value = data.fcBias[output];
                int row = output * hidden;
                for (int column = 0; column < hidden; column++)
                {
                    value += data.fcWeight[row + column] * h1[column];
                }
                value = value * data.yStd[output] + data.yMean[output];
                result[output] = Mathf.Clamp(value, data.actionClipLow[output], data.actionClipHigh[output]);
            }
            return result;
        }

        private static void LstmLayer(
            float[] input,
            int inputSize,
            float[] hiddenState,
            float[] cellState,
            float[] weightInput,
            float[] weightHidden,
            float[] biasInput,
            float[] biasHidden)
        {
            int hiddenSize = hiddenState.Length;
            float[] gates = new float[hiddenSize * 4];
            for (int row = 0; row < gates.Length; row++)
            {
                float sum = biasInput[row] + biasHidden[row];
                int inputOffset = row * inputSize;
                for (int column = 0; column < inputSize; column++)
                {
                    sum += weightInput[inputOffset + column] * input[column];
                }
                int hiddenOffset = row * hiddenSize;
                for (int column = 0; column < hiddenSize; column++)
                {
                    sum += weightHidden[hiddenOffset + column] * hiddenState[column];
                }
                gates[row] = sum;
            }

            for (int index = 0; index < hiddenSize; index++)
            {
                float inputGate = Sigmoid(gates[index]);
                float forgetGate = Sigmoid(gates[hiddenSize + index]);
                float candidate = (float)Math.Tanh(gates[hiddenSize * 2 + index]);
                float outputGate = Sigmoid(gates[hiddenSize * 3 + index]);
                float cell = forgetGate * cellState[index] + inputGate * candidate;
                cellState[index] = cell;
                hiddenState[index] = outputGate * (float)Math.Tanh(cell);
            }
        }

        private static float Sigmoid(float value)
        {
            if (value >= 0f)
            {
                float exponential = (float)Math.Exp(-value);
                return 1f / (1f + exponential);
            }
            else
            {
                float exponential = (float)Math.Exp(value);
                return exponential / (1f + exponential);
            }
        }

        private static void Validate(EyeNavModelData data)
        {
            if (data == null || data.schemaVersion != "1.0.0")
            {
                throw new InvalidOperationException("Unsupported or unreadable model export.");
            }
            if (data.dataSource != "EyeNavGS" || data.syntheticTraining)
            {
                throw new InvalidOperationException("Demo requires a real-data EyeNavGS model.");
            }
            if (data.historyLength != 8 || data.inputDim != 6 || data.outputDim != 6 ||
                data.hiddenDim != 64 || data.numLayers != 2)
            {
                throw new InvalidOperationException("Unexpected trained model architecture.");
            }
            int h = data.hiddenDim;
            RequireLength(data.xMean, 6, "xMean");
            RequireLength(data.xStd, 6, "xStd");
            RequireLength(data.yMean, 6, "yMean");
            RequireLength(data.yStd, 6, "yStd");
            RequireLength(data.actionClipLow, 6, "actionClipLow");
            RequireLength(data.actionClipHigh, 6, "actionClipHigh");
            RequireLength(data.weightIhL0, 4 * h * 6, "weightIhL0");
            RequireLength(data.weightHhL0, 4 * h * h, "weightHhL0");
            RequireLength(data.biasIhL0, 4 * h, "biasIhL0");
            RequireLength(data.biasHhL0, 4 * h, "biasHhL0");
            RequireLength(data.weightIhL1, 4 * h * h, "weightIhL1");
            RequireLength(data.weightHhL1, 4 * h * h, "weightHhL1");
            RequireLength(data.biasIhL1, 4 * h, "biasIhL1");
            RequireLength(data.biasHhL1, 4 * h, "biasHhL1");
            RequireLength(data.fcWeight, 6 * h, "fcWeight");
            RequireLength(data.fcBias, 6, "fcBias");
            RequireLength(data.selfTestInput, 8 * 6, "selfTestInput");
            RequireLength(data.selfTestExpectedOutput, 6, "selfTestExpectedOutput");
            if (data.seedWindows == null || data.seedWindows.Length != GeneratedModelIntegrity.SeedWindowCount)
            {
                throw new InvalidOperationException("Held-out seed bank is missing or incomplete.");
            }
            foreach (SeedWindowData seed in data.seedWindows)
            {
                if (seed == null || string.IsNullOrEmpty(seed.trajectoryId) || seed.values == null || seed.values.Length != 48)
                {
                    throw new InvalidOperationException("Held-out seed bank contains an invalid entry.");
                }
            }
            for (int index = 0; index < 6; index++)
            {
                if (!(data.xStd[index] > 0f) || !(data.yStd[index] > 0f))
                {
                    throw new InvalidOperationException("Normalization scale must be positive.");
                }
            }
        }

        private static void RequireLength(float[] values, int expected, string name)
        {
            if (values == null || values.Length != expected)
            {
                throw new InvalidOperationException(
                    name + " length mismatch: expected " + expected + ".");
            }
        }
    }
}
