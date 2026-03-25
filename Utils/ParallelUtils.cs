public class ParallelUtils
{
    public static double[] ForwardPass(Layer layer, double[] input)
    {
        return ForwardPass(layer.Neurons, input, layer.Bias, layer._activationFunction);
    }
    public static double[] ForwardPass(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        Parallel.For(0, neuronMatrix.GetLength(0), i =>
        {
            output[i] = SimdUtils.CalcNeuronOutput(neuronMatrix, i, input, bias[i], activationFunction, out var _);
        });
        return output;
    }
    public static double[] ForwardTrain(Layer layer, double[] input, out double[] preActivationValues)
    {
        var result = ForwardTrain(layer.Neurons, input, layer.Bias, layer._activationFunction, out var preActivationValuesInner);
        preActivationValues = preActivationValuesInner;
        return result;
    }
    public static double[] ForwardTrain(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        var pre = preActivationValues;

        Parallel.For(0, neuronMatrix.GetLength(0), i =>
        {
            output[i] = SimdUtils.CalcNeuronOutput(neuronMatrix, i, input, bias[i], activationFunction, out pre[i]);
        });
        return output;
    }
}