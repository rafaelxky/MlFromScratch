public class SigmoidActivation : IActivationFunction
{
    public double Apply(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }
}