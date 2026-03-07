
using MlNetworkTraining;

public class TrainingLayerFactory: ITrainingLayerFactory
{
    ITrainingNeuronFactory _trainingNeuronFactory;
    public TrainingLayerFactory(ITrainingNeuronFactory trainingNeuronFactory)
    {
        _trainingNeuronFactory = trainingNeuronFactory;
    }

    public ITrainingLayer[] ArrFromNetworkData(NetworkData networkData)
    {
        var layer = new List<TrainingLayer>();
        foreach (var layerData in networkData.Layers) {
            layer.Add(new TrainingLayer(_trainingNeuronFactory.ArrFromLayerData(layerData)));
        }
        return layer.ToArray();
    }

    public ITrainingLayer NewLayer(int layerSize, Random random, int inputSize)
    {
        return new TrainingLayer(layerSize,random,inputSize,_trainingNeuronFactory);
    }
}