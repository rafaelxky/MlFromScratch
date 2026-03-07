public interface INeuron
{
    public double Calc(double[] values);
    public double SigmoidActivation(double value);
    public void Print();
}