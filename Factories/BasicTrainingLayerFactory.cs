
using MlNetworkTraining;

public class BasicTrainingLayerFactory: ITrainingLayerFactory
{
    ITrainingNeuronFactory _trainingNeuronFactory;
    public BasicTrainingLayerFactory(ITrainingNeuronFactory trainingNeuronFactory)
    {
        _trainingNeuronFactory = trainingNeuronFactory;
    }
    public ITrainingLayer NewLayer(int layerSize, Random random, int inputSize)
    {
        return new TrainingLayer(layerSize,random,inputSize,_trainingNeuronFactory);
    }
}