public interface INetwork
{
    public double[] ForwardTrain(double[] input, out List<LayerCache> layerCaches);
    public void BackPropagation(double[] expected, double learningRate, List<LayerCache> layerCaches);
}