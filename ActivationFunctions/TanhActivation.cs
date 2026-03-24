// -1 to 1
using ILGPU.Algorithms;

public struct TanhActivation : IActivationFunction
{
    public double Apply(double value) => XMath.Tanh(value);
    public double Derivative(double value) => 1 - XMath.Pow(XMath.Tanh(value), 2);
}