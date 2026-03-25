using ILGPU.IR.Values;

public struct NetworkConfig
{
    public AccelerationType AccelerationType;
    public NetworkConfig(AccelerationType accelerationType = AccelerationType.Simd)
    {
        AccelerationType = accelerationType;
    }
}