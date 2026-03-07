public interface ITrainingLayerFactory
{
    public ITrainingLayer NewLayer(int layerSize, Random random, int inputSize);
    public ITrainingLayer[] ArrFromNetworkData(NetworkData networkData);
}