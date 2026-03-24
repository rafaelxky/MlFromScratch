// 0 to infinity - general purpose
public struct ReLUActivation : IActivationFunction
{
    public double Apply(double value) => value > 0 ? value : 0;
    public double Derivative(double value) => value > 0 ? 1 : 0;

}