public class NetworkFactory : INetworkFactory
{
    ILayerFactory _layerFactory;
    public NetworkFactory(ILayerFactory layerFactory)
    {
        _layerFactory = layerFactory;
    }
    public INetwork FromDto(NetworkData networkData)
    {
        var network = new Network();
        network.NeuralNetwork = _layerFactory.ArrFromNetworkData(networkData);
        return network;
    }
}