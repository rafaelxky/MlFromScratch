public interface ITrainingLayer: ILayer
{
    public void BackPropagation(ITrainingLayer? nextLayer, double[]? targetOutputs, double learningRate, IActivationFunction activationFunction);
    public int GetLength();
    public ITrainingNeuron GetNeuron(int id);
    public ITrainingNeuron[] GetNeurons();
}