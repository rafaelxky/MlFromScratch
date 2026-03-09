using MlNetworkTraining;

public class TrainingNetworkFactory : ITrainingNetworkFactory
{
    ITrainingLayerFactory _layerFactory;
    IActivationFunction _activationFunction;
    public TrainingNetworkFactory(ITrainingLayerFactory layerFactory, IActivationFunction activationFunction)
    {
        _layerFactory = layerFactory;
        _activationFunction = activationFunction;
    }
 

    public  ITrainingNetwork FromDto(NetworkData networkData)
    {
        var network = new TrainingNetwork();
        network.ActivationFunction = _activationFunction;
        network.NeuralNetwork = _layerFactory.ArrFromNetworkData(networkData);
        return network;
    }
}