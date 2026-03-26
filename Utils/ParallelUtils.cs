public class ParallelUtils
{
    public static void ForwardPass(Layer layer, double[] input, double[] outputBuffer)
    {
        ForwardPass(layer.Neurons, input,outputBuffer, layer.Bias, layer._activationFunction);
    }
    public static void ForwardPass(double[,] neuronMatrix, double[] input, double[] outputBuffer,double[] bias, IActivationFunction activationFunction)
    {
        Parallel.For(0, neuronMatrix.GetLength(0), i =>
        {
            outputBuffer[i] = SimdUtils.CalcNeuronOutput(neuronMatrix, i, input, bias[i], activationFunction, out var _);
        });
    }
    public static void ForwardTrain(Layer layer, double[] input,double[]  outputBuffer,out double[] preActivationValues)
    {
        ForwardTrain(layer.GetNeuronWeights(), input, outputBuffer,layer.Bias, layer._activationFunction, out preActivationValues);
    }
    public static void ForwardTrain(double[,] neuronMatrix, double[] input,double[] outputBuffer ,double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        var pre = preActivationValues;
        Parallel.For(0, neuronMatrix.GetLength(0), i =>
        {
            outputBuffer[i] = SimdUtils.CalcNeuronOutput(neuronMatrix, i, input, bias[i], activationFunction, out pre[i]);
        });
    }
}