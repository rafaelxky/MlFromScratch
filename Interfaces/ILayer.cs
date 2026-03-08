public interface ILayer
{
    public double[] ForwardPass(double[] inputs, IActivationFunction activationFunction);
    public void Print();
    public INeuron[] GetNeurons();
}