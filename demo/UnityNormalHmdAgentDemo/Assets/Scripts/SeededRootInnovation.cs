using System;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Portable xorshift32 stream with cached Box-Muller normals.  It is the
    /// only stochastic primitive used by the v4 deployment contract.
    /// </summary>
    internal sealed class PortableHmdRandom
    {
        private uint state;
        private bool hasSpare;
        private double spare;

        internal PortableHmdRandom(uint seed)
        {
            state = seed == 0u ? 0x6D2B79F5u : seed;
        }

        internal static uint DeriveSeed(int runSeed, uint xorConstant)
        {
            return SplitMix32(unchecked((uint)runSeed) ^ xorConstant);
        }

        // Frozen SplitMix32 finalizer. All operations deliberately wrap uint32.
        internal static uint SplitMix32(uint input)
        {
            unchecked
            {
                uint value = input + 0x9E3779B9u;
                value = (value ^ (value >> 16)) * 0x21F0AAADu;
                value = (value ^ (value >> 15)) * 0x735A2D97u;
                return value ^ (value >> 15);
            }
        }

        internal uint NextUInt32()
        {
            uint value = state;
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            state = value;
            return value;
        }

        internal double NextUniformOpen()
        {
            return (NextUInt32() + 0.5) / 4294967296.0;
        }

        internal float NextGaussianFloat()
        {
            if (hasSpare)
            {
                hasSpare = false;
                return (float)spare;
            }
            double u1 = NextUniformOpen();
            double u2 = NextUniformOpen();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = Math.PI * 2.0 * u2;
            spare = radius * Math.Sin(angle);
            hasSpare = true;
            return (float)(radius * Math.Cos(angle));
        }
    }

    /// <summary>
    /// Joint physical-unit BodyRoot AR(1) process.  Unlike v3, the residual is
    /// initialized from its stationary distribution before the first update,
    /// avoiding a deterministic low-motion warm-up at episode start.
    /// </summary>
    internal sealed class SeededRootInnovation
    {
        private readonly PortableHmdRandom random;
        private readonly float[] rho;
        private readonly float[] cholesky;
        private readonly float[] residual = new float[3];
        private readonly float[] initialResidual = new float[3];

        internal SeededRootInnovation(uint derivedSeed, float[] innovationRho,
            float[] innovationCholesky)
        {
            if (innovationRho == null || innovationRho.Length != 3 ||
                innovationCholesky == null || innovationCholesky.Length != 9)
            {
                throw new ArgumentException("Root innovation requires rho[3] and Cholesky[3,3].");
            }
            random = new PortableHmdRandom(derivedSeed);
            rho = (float[])innovationRho.Clone();
            cholesky = (float[])innovationCholesky.Clone();
            float[] initial = DrawCorrelated();
            Array.Copy(initial, residual, 3);
            Array.Copy(initial, initialResidual, 3);
        }

        internal float[] InitialResidual => (float[])initialResidual.Clone();

        internal float[] Next(float scale)
        {
            float[] epsilon = DrawCorrelated();
            float[] values = new float[3];
            for (int index = 0; index < 3; index++)
            {
                float rhoSquare = (float)(rho[index] * rho[index]);
                float gain = (float)Math.Sqrt(Math.Max(0.0, 1.0 - rhoSquare));
                float autoregressive = (float)(rho[index] * residual[index]);
                residual[index] = (float)(autoregressive + (float)(gain * epsilon[index]));
                values[index] = (float)(residual[index] * scale);
            }
            return values;
        }

        private float[] DrawCorrelated()
        {
            float[] gaussian = new float[3];
            for (int index = 0; index < 3; index++)
                gaussian[index] = random.NextGaussianFloat();
            float[] correlated = new float[3];
            for (int row = 0; row < 3; row++)
            {
                float value = 0f;
                for (int column = 0; column <= row; column++)
                {
                    float product = (float)(cholesky[row * 3 + column] * gaussian[column]);
                    value = (float)(value + product);
                }
                correlated[row] = value;
            }
            return correlated;
        }
    }
}
