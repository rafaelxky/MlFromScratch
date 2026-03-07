public interface ITrainingNeuronFactory
{
    public ITrainingNeuron NewNeuron(int inputSize, Random random);
    public ITrainingNeuron[] ArrFromLayerData(LayerData layerData);
}
public interface ITrainingNeuronFactory<T>: ITrainingNeuronFactory
{
    public ITrainingNeuron<T> NewNeuron(int inputSize, Random random);
}