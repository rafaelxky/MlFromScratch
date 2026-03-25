public class MlNetworkTrainer
{
    private List<LayerCache>? _layerCaches;
    private INetwork _network;
    public MlNetworkTrainer(INetwork network)
    {
        _network = network;
    }
    public double[] ForwardTrain(double[] values)
    {
        var output = _network.ForwardTrain(values, out var layerCaches);
        _layerCaches = layerCaches;
        return output;
    }
    public void Train(double[] expected, double learningRate)
    {
        if (_layerCaches == null)
        {
            throw new Exception("Can't train if layer caches are null! Call ForwardTrain() before Train().");
        }
        _network.BackPropagation(expected, learningRate,_layerCaches);
    }
}