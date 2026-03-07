public interface ITrainingLayerFactory
{
    public ITrainingLayer NewLayer(int layerSize, Random random, int inputSize);
}