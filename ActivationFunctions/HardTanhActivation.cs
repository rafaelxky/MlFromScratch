public struct HardTanhActivation : IActivationFunction
{
    public double Apply(double x)
    {
        // GPU-compatible clamp
        return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
    }
    public double Derivative(double value)
    {
        // GPU-compatible derivative
        return (value >= -1.0 && value <= 1.0) ? 1.0 : 0.0;
    }
}