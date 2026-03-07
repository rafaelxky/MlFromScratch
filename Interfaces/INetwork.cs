public interface INetwork
{
    public ILayer[] GetLayers();
    public double[] ForwardPass(double[] values);
}