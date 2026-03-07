public interface ITrainingNeuronFactory: INeuronFactory
{
    public ITrainingNeuron NewNeuron(int inputSize, Random random);
}
public interface ITrainingNeuronFactory<T>: ITrainingNeuronFactory
{
    public ITrainingNeuron<T> NewNeuron(int inputSize, Random random);
}