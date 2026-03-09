// -1 to 1
public class TanhActivation : IActivationFunction
{
    public double Apply(double value) => Math.Tanh(value);
    public double Derivative(double value) => 1 - Math.Pow(Math.Tanh(value), 2);
}