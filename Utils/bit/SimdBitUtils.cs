using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class SimdBitUtils
{
    public static void ForwardPass(Layer2Bit layers, double[] input, double[] outputBuffer)
    {
        var weights = layers.Weights;
        var activationFunction = layers._activationFunction;
        // for each neuron in layer, calc output and build vector
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            outputBuffer[i] = activationFunction.Apply(PackedDotProduct(layers.Weights, i, input, input.Length) + layers.Bias[i]);
        }
    }

    public static double PackedDotProduct(byte[,] layer, int neuronId, double[] values, int length)
    {
        double sum = 0;
        int rowLen = (length + 3) / 4;

        if (!Avx2.IsSupported)
        {
            BitUtils.PackedDotProduct(layer, neuronId, values, length);
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
                        int i0 = (b + lane) * 4 + 0; if (i0 < length) sum += p0.GetElement(lane) * values[i0];
                        int i1 = (b + lane) * 4 + 1; if (i1 < length) sum += p1.GetElement(lane) * values[i1];
                        int i2 = (b + lane) * 4 + 2; if (i2 < length) sum += p2.GetElement(lane) * values[i2];
                        int i3 = (b + lane) * 4 + 3; if (i3 < length) sum += p3.GetElement(lane) * values[i3];
                    }
                }
            }
        }

        for (; b + 32 <= rowLen; b += 32)
            sum += BitUtils.GetPair(layer[neuronId, b / 4], b % 4) * values[b];

        return sum;
    }
}