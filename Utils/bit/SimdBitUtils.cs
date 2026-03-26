using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using ILGPU.IR.Values;

public static class SimdBitUtils
{
    public static void ForwardPass(Layer2Bit layers, double[] input, double[] outputBuffer)
    {
        var weights = layers.Weights;
        var activationFunction = layers._activationFunction;
        var weightLength = layers.Weights.GetLength(1);
        // for each neuron in layer, calc output and build vector
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            var output = PackedDotProduct(layers.Weights, i, input, weightLength);
            output += layers.Bias[i];
            output = activationFunction.Apply(output);
            outputBuffer[i] = output;
        }
    }
    public static void ForwardTrainParallel(Layer2Bit layers, double[] input, double[] outputBuffer, out double[] preActivationValues)
    {
        ParallelUtils.ForwardTrain(layers.GetNeuronWeights(), input, outputBuffer, layers.Bias, layers._activationFunction, out preActivationValues);
    }
    public static void ForwardTrain(Layer2Bit layers, double[] input, double[] outputBuffer, out double[] preActivationValues)
    {
        SimdUtils.ForwardTrain(layers.GetNeuronWeights(), input, outputBuffer, layers.Bias, layers._activationFunction, out preActivationValues);
    }

    public static void ForwardPassParallel(Layer2Bit layers, double[] input, double[] outputBuffer)
    {
        var weights = layers.Weights;
        var activationFunction = layers._activationFunction;
        var weightLength = layers.Weights.GetLength(1);
        // for each neuron in layer, calc output and build vector
        Parallel.For(0, weights.GetLength(0), i =>
        {
            outputBuffer[i] = activationFunction.Apply(PackedDotProduct(layers.Weights, i, input, weightLength) + layers.Bias[i]);
        });
    }

    public static double PackedDotProduct(byte[,] layer, int neuronId, double[] values, int weightLength)
    {
        // values.length => layer.GetLength(0) * 4
        // this is because values uses a buffer so it may be larger than the actual input size, so we need to use weightLength to know how many values to actually use
        double sum = 0;
        int rowLen = (weightLength + 3) / 4;

        if (!Avx2.IsSupported)
        {
            return BitUtils.PackedDotProduct(layer, neuronId, values, weightLength);
        }
        Vector256<byte> mask = Vector256.Create((byte)0x03);
        int b = 0;

        for (; b <= rowLen - 32; b += 32)
        {
            unsafe
            {
                fixed (byte* p = &layer[neuronId, b])
                {
                    Vector256<byte> raw = Avx2.LoadVector256(p);

                    Vector256<byte> p0 = Avx2.And(raw, mask);
                    Vector256<byte> p1 = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 2).AsByte(), mask);
                    Vector256<byte> p2 = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask);
                    Vector256<byte> p3 = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 6).AsByte(), mask);

                    for (int lane = 0; lane < 32; lane++)
                    {
                        int i0 = (b + lane) * 4 + 0; if (i0 < weightLength) sum += p0.GetElement(lane) * values[i0];
                        int i1 = (b + lane) * 4 + 1; if (i1 < weightLength) sum += p1.GetElement(lane) * values[i1];
                        int i2 = (b + lane) * 4 + 2; if (i2 < weightLength) sum += p2.GetElement(lane) * values[i2];
                        int i3 = (b + lane) * 4 + 3; if (i3 < weightLength) sum += p3.GetElement(lane) * values[i3];
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
}