using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using ILGPU.IR.Values;

public static class SimdBitUtils
{
    public static void ForwardPass(Layer2Bit layers, double[] input, double[] outputBuffer)
    {
        if (layers.Weights == null) throw new InvalidOperationException("Simd forward path requires packed weights.");

        var weights = layers.Weights;
        var activationFunction = layers._activationFunction;
        var weightLength = layers.WeightCount;
        int neuronCount = layers.NeuronCount;

        for (int i = 0; i < neuronCount; i++)
        {
            var output = PackedDotProduct(weights, i, input, weightLength);
            output += layers.Bias[i];
            outputBuffer[i] = activationFunction.Apply(output);
        }
    }
    public static void ForwardTrain(Layer2Bit layers, double[] input, double[] outputBuffer, out double[] preActivationValues)
    {
        SimdUtils.ForwardTrain(layers.GetNeuronWeights(), input, outputBuffer, layers.Bias, layers._activationFunction, out preActivationValues);
    }

    public static void ForwardPassParallel(Layer2Bit layers, double[] input, double[] outputBuffer)
    {
        if (layers.Weights == null) throw new InvalidOperationException("Simd parallel forward path requires packed weights.");

        var weights = layers.Weights;
        var activationFunction = layers._activationFunction;
        var weightLength = layers.WeightCount;
        int neuronCount = layers.NeuronCount;

        Parallel.For(0, neuronCount, i =>
        {
            outputBuffer[i] = activationFunction.Apply(PackedDotProduct(weights, i, input, weightLength) + layers.Bias[i]);
        });
    }

    public static void ForwardTrainParallel(Layer2Bit layers, double[] input, double[] outputBuffer, out double[] preActivationValues)
    {
        var weights = layers.LatentWeights ?? throw new InvalidOperationException("Simd parallel training requires latent weights.");
        var activationFunction = layers._activationFunction;
        var neuronCount = weights.GetLength(0);
        preActivationValues = new double[neuronCount];
        var pre = preActivationValues;
        // for each neuron in layer, calc output and build vector
        Parallel.For(0, neuronCount, i =>
        {
            var preAct = WeightInputDotProd(weights, i, input) + layers.Bias[i];
            pre[i] = preAct;
            outputBuffer[i] = activationFunction.Apply(preAct);
        });
    }


    public static unsafe double WeightInputDotProd(double[,] neurons, int neuronId, double[] input)
    {
        int len = neurons.GetLength(1);
        double result = 0;

        fixed (double* pN = neurons, pI = input)
        {
            double* row = pN + neuronId * len;

            if (Avx.IsSupported && Fma.IsSupported)
            {
                int i = 0;
                var acc = Vector256<double>.Zero;
                var zero = Vector256<double>.Zero;
                var one = Vector256.Create(1.0);
                var negOne = Vector256.Create(-1.0);

                for (; i <= len - 4; i += 4)
                {
                    var w = Avx.LoadVector256(row + i);
                    var inp = Avx.LoadVector256(pI + i);

                    // mask = w >= 0
                    var mask = Avx.CompareGreaterThanOrEqual(w, zero);

                    // sign = mask ? +1 : -1
                    var sign = Avx.BlendVariable(negOne, one, mask);

                    acc = Fma.MultiplyAdd(sign, inp, acc);
                }

                var low = acc.GetLower();
                var high = acc.GetUpper();
                var sum2 = Sse2.Add(low, high);
                var shuf = Sse2.UnpackHigh(sum2, sum2);
                result = Sse2.AddScalar(sum2, shuf).ToScalar();

                for (; i < len; i++)
                {
                    result += (row[i] >= 0 ? 1.0 : -1.0) * pI[i];
                }
            }
            else
            {
                for (int i = 0; i < len; i++)
                {
                    result += (row[i] >= 0 ? 1.0 : -1.0) * pI[i];
                }
            }
        }

        return result;
    }

    public static double PackedDotProduct(byte[,] layer, int neuronId, double[] values, int weightLength)
    {
        // values.length => layer.GetLength(0) * 4
        // this is because values uses a buffer so it may be larger than the actual input size, so we need to use weightLength to know how many values to actually use

        if (!Avx2.IsSupported)
        {
            return BitUtils.PackedDotProduct(layer, neuronId, values, weightLength);
        }

        // AVX2-enabled path: use block decoding with explicit 2-bit decode and avoid cross-byte vector shift mistakes.
        double sum = 0;
        int rowLen = (weightLength + 3) / 4;

        int b = 0;
        for (; b <= rowLen - 32; b += 32)
        {
            unsafe
            {
                fixed (byte* p = &layer[neuronId, b])
                {
                    for (int byteIdx = 0; byteIdx < 32; byteIdx++)
                    {
                        byte packed = p[byteIdx];
                        int baseI = (b + byteIdx) * 4;

                        if (baseI + 0 < weightLength)
                            sum += BitUtils.Decode((byte)(packed & 0x03)) * values[baseI + 0];
                        if (baseI + 1 < weightLength)
                            sum += BitUtils.Decode((byte)((packed >> 2) & 0x03)) * values[baseI + 1];
                        if (baseI + 2 < weightLength)
                            sum += BitUtils.Decode((byte)((packed >> 4) & 0x03)) * values[baseI + 2];
                        if (baseI + 3 < weightLength)
                            sum += BitUtils.Decode((byte)((packed >> 6) & 0x03)) * values[baseI + 3];
                    }
                }
            }
        }

        for (int i = b * 4; i < weightLength; i++)
        {
            sum += BitUtils.GetPair(layer[neuronId, i / 4], i % 4) * values[i];
        }

        return sum;
    }

    public static void ForwardTrain(double[,] neuronMatrix, double[] input, double[] outputBuffer, double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        var pre = preActivationValues;
        Parallel.For(0, neuronMatrix.GetLength(0), i =>
        {
            outputBuffer[i] = SimdUtils.CalcNeuronOutput(neuronMatrix, i, input, bias[i], activationFunction, out pre[i]);
        });
    }
}