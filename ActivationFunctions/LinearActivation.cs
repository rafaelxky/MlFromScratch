public struct LinearActivation : IActivationFunction
{
    public double Apply(double value) => value;
    public double Derivative(double value) => 1;
}