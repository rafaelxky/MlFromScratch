public interface INeuron
{
    public double Calc(double[] values, IActivationFunction activationFunction);
    public void Print();
    public object GetWeightsRaw();
    public void SetWeightsRaw(object data);
    public double GetBias();
    public void SetBias(double bias);
}
public interface INeuron<W>: INeuron
{
   
}