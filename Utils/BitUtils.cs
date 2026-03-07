public static class BitUtils
{
    public static byte Encode(int value) => value switch
    {
        -1 => 0b00,
        0 => 0b01,
        1 => 0b10,
        _ => throw new ArgumentException($"Value must be -1, 0 or 1, got {value}")
    };

    public static int Decode(byte bits) => bits switch
    {
        0b00 => -1,
        0b01 => 0,
        0b10 => 1,
        _ => throw new ArgumentException($"Invalid 2-bit value: {bits}")
    };

    public static byte Pack(int v0, int v1, int v2, int v3)
    {
        return (byte)(
            (Encode(v0) << 0) |
            (Encode(v1) << 2) |
            (Encode(v2) << 4) |
            (Encode(v3) << 6)
        );
    }

    public static int GetPair(byte b, int n)
    {
        int shifted = (b >> (n * 2)) & 0b11;
        return Decode((byte)shifted);
    }

    public static byte[] PackArray(int[] values)
    {
        int byteCount = (values.Length + 3) / 4; 
        byte[] packed = new byte[byteCount];

        for (int i = 0; i < byteCount; i++)
        {
            int v0 = i * 4 + 0 < values.Length ? values[i * 4 + 0] : 0;
            int v1 = i * 4 + 1 < values.Length ? values[i * 4 + 1] : 0;
            int v2 = i * 4 + 2 < values.Length ? values[i * 4 + 2] : 0;
            int v3 = i * 4 + 3 < values.Length ? values[i * 4 + 3] : 0;
            packed[i] = Pack(v0, v1, v2, v3);
        }

        return packed;
    }

    public static int[] UnpackArray(byte[] packed, int originalLength)
    {
        int[] values = new int[originalLength];

        for (int i = 0; i < originalLength; i++)
        {
            int byteIndex = i / 4;
            int pairIndex = i % 4;
            values[i] = GetPair(packed[byteIndex], pairIndex);
        }

        return values;
    }

    public static int BitMultiply(int a, int b)
    {
        if (a == 0 || b == 0) return 0;
        if (a == b) return 1;
        return -1;
    }

    public static double PackedDotProduct(byte[] weight, double[] values, int lenght)
    {
        double sum = 0;

        for (int i = 0; i < lenght; i++)
        {
            int byteIndex = i / 4;
            int pairIndex = i % 4;

            int a = GetPair(weight[byteIndex], pairIndex);

            sum += a * values[i];
        }
        return sum;
    }
}