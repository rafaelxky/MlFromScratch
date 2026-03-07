public interface ILayerFactory
{
    public ILayer NewLayer(int layerSize, Random random, int inputSize);
}