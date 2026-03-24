using ILGPU.IR.Values;

public struct NetworkConfig
{
    public bool UseGpu;
    public NetworkConfig(bool useGpu = false)
    {
        UseGpu = useGpu;
    }
}