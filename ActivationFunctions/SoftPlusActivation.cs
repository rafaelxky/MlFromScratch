using ILGPU.Algorithms;

public struct SoftplusActivation : IActivationFunction
{
    public double Apply(double value) => XMath.Log(1 + XMath.Exp(value));
    public double Derivative(double value) => 1 / (1 + XMath.Exp(-value));
}