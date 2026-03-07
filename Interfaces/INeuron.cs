public interface INeuron
{
    public double Calc(double[] values);
    public double SigmoidActivation(double value);
    public void Print();
    public object GetWeightsRaw();
    public void SetWeightsRaw(object data);
    public double GetBias();
    public void SetBias(double bias);
}
public interface INeuron<W>: INeuron
{
   
}