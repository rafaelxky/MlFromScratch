public struct LeakyReLUActivation : IActivationFunction
{
    public double Alpha;

    public LeakyReLUActivation(double alpha = 0.01)
    {
        Alpha = alpha;
    }

    public double Apply(double value)
    {
        return value > 0.0 ? value : Alpha * value;
    }

    public double Derivative(double value)
    {
        return value > 0.0 ? 1.0 : Alpha;
    }
}