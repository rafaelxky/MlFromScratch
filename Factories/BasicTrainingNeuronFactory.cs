
using MlNetworkTraining;

public class BasicTrainingNeuronFactory: ITrainingNeuronFactory
{
    public BasicTrainingNeuronFactory()
    {
        
    }

    public ITrainingNeuron NewNeuron(int inputSize, Random random)
    {
        return new TrainingNeuron(inputSize,random);
    }
}