public interface IActivationFunction
{
    public double Apply(double value);
    public double Derivative(double value);
    string Name => GetType().Name.ToLower();
}