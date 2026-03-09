public class HardTanhActivation : IActivationFunction
{
    public double Apply(double value) => Math.Clamp(value, -1, 1);
    public double Derivative(double value) => Math.Abs(value) <= 1 ? 1 : 0;
}