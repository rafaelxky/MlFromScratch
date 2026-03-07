public interface ITrainingLayer: ILayer
{
    public void BackPropagation(ITrainingLayer? nextLayer, double[]? targetOutputs, double learningRate);
    public int GetLength();
    public ITrainingNeuron GetNeuron(int id);
    public ITrainingNeuron[] GetNeurons();
}