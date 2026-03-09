public class LeakyReLUActivation : IActivationFunction
{
    private readonly double _alpha;
    public LeakyReLUActivation(double alpha = 0.01) => _alpha = alpha;
    public double Apply(double value) => value > 0 ? value : _alpha * value;
    public double Derivative(double value) => value > 0 ? 1 : _alpha;
}