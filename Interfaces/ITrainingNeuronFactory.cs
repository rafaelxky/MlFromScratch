public interface ITrainingNeuronFactory: INeuronFactory
{
    public ITrainingNeuron NewNeuron(int inputSize, Random random);
}