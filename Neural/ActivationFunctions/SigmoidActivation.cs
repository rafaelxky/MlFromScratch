// binary decisions 0 - 1
public class SigmoidActivation : IActivationFunction
{
    public double Apply(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }

    public double Derivative(double value)
    {
        var result = Apply(value);
        return result * (1 - result);
    }
}