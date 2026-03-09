public class SoftplusActivation : IActivationFunction
{
    public double Apply(double value) => Math.Log(1 + Math.Exp(value));
    public double Derivative(double value) => 1 / (1 + Math.Exp(-value));
}