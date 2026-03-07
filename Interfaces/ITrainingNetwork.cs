public interface ITrainingNetwork: INetwork
{
    public void BackPropagation(double[] expected, double learningRate);
}