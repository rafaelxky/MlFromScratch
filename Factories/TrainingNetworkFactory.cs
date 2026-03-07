using MlNetworkTraining;

public class TrainingNetworkFactory : ITrainingNetworkFactory
{
    ITrainingLayerFactory _layerFactory;
    public TrainingNetworkFactory(ITrainingLayerFactory layerFactory)
    {
        _layerFactory = layerFactory;
    }
 

    public  ITrainingNetwork FromDto(NetworkData networkData)
    {
        var network = new TrainingNetwork();
        network.NeuralNetwork = _layerFactory.ArrFromNetworkData(networkData);
        return network;
    }
}