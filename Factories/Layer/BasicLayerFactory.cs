
public class LayerFactory : ILayerFactory
{
    INeuronFactory _neuronFactory;
    public LayerFactory(INeuronFactory neuronFactory)
    {
        _neuronFactory = neuronFactory;
    }
    public ILayer[] ArrFromNetworkData(NetworkData networkData)
    {
        var layer = new List<Layer>();
        foreach (var layerData in networkData.Layers) {
            layer.Add(new Layer(_neuronFactory.ArrFromLayerData(layerData, networkData.NeuronType)));
        }
        return layer.ToArray();
    }
}