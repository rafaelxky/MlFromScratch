public class NetworkFactory : INetworkFactory
{
    ILayerFactory _layerFactory;
    IActivationFunction _activationFactory;
    public NetworkFactory(ILayerFactory layerFactory, IActivationFunction activationFunction)
    {
        _layerFactory = layerFactory;
        _activationFactory = activationFunction;
    }
    public INetwork FromDto(NetworkData networkData)
    {
        var network = new Network();
        network.NeuralNetwork = _layerFactory.ArrFromNetworkData(networkData);
        network.ActivationFunction = _activationFactory;
        return network;
    }
}