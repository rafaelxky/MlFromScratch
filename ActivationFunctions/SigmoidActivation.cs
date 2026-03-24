// binary decisions 0 - 1
using ILGPU.Algorithms;

public struct SigmoidActivation : IActivationFunction
{
    // GPU-compatible Apply
    public double Apply(double value)
    {
        return 1.0 / (1.0 + XMath.Exp(-value));
    }

    // GPU-compatible Derivative
    public double Derivative(double value)
    {
        double sig = Apply(value);
        return sig * (1.0 - sig);
    }
}