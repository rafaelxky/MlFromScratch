using ILGPU.Algorithms;

public struct SwishActivation : IActivationFunction
{
    private double Sigmoid(double value) => 1 / (1 + XMath.Exp(-value));
    public double Apply(double value) => value * Sigmoid(value);
    public double Derivative(double value) => Sigmoid(value) + Apply(value) * (1 - Sigmoid(value));
}