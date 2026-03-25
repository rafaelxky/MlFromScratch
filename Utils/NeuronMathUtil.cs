using Microsoft.VisualBasic;

public static class NeuronMathUtil
{
    // forward pass
    public static double Calc2BitNeuronOutput(double[,] layer, int neuronId, double[] input, double bias, IActivationFunction activationFunction, out double preActivation)
    {
        double output = 0;
        for (int i = 0; i < layer.GetLength(0); i++)
        {
            output += Math.Sign(layer[neuronId,i]) * input[i];
        }
        output += bias;
        preActivation = output;
        return activationFunction.Apply(output);
    }
    public static double[] ForwardPass(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        // foreach neuron in layer, calc output and build vector
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            output[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out var _);
        }

        return output;
    }
    public static double[] ForwardPass(Layer layer, double[] input)
    {
        return ForwardPass(layer.Neurons, input, layer.Bias, layer._activationFunction);
    }
    public static double[] ForwardTrain(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            output[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out preActivationValues[i]);
        }
        return output;
    }
    public static double[] ForwardTrain(Layer layer, double[] input, out double[] preActivationValues)
    {
        var result = ForwardTrain(layer.Neurons, input, layer.Bias, layer._activationFunction, out var preActivationValuesInner);
        preActivationValues = preActivationValuesInner;
        return result;
    }
    public static double CalcNeuronOutput(double[,] neurons,int neuronId, double[] input, double bias, IActivationFunction activationFunction, out double preActivation)
    {
        //double output = TensorUtils.WeightInputDotProd(neurons, neuronId, input);
        double output = TensorUtils.WeightInputDotProd(neurons,neuronId, input);
        output += bias;
        preActivation = output;
        return activationFunction.Apply(output);
    }

    // updates a single neuron
    public static void UpdateNeuronAtOutput(double[,] layer, int neuronId, double finalOutput, double targetOutput, double preActivation, IActivationFunction activationFunction, double learningRate, double[] neuronInputs, ref double neuronBias, out double delta)
    {
        var gradientStep = MathUtils.GetGradientStepAtOutput(learningRate, finalOutput, targetOutput, preActivation, activationFunction, out var deltaInner);
        delta = deltaInner;
        // for each weight
        for (int i = 0; i < layer.GetLength(0); i++)
        {
            layer[neuronId,i] -= gradientStep * neuronInputs[i];
        }
        neuronBias -= gradientStep;
    }
    public static void UpdateNeuron(
            double[,] layer,
            int neuronId,
            ILayer nextLayer,
            double learningRate,
            double preActivation,
            IActivationFunction activationFunction,
            double[] neuronInputs,
            double[] nextLayerDeltas,
            ref double neuronBias,
            out double delta
            )
    {
        double error = MathUtils.CalcError(nextLayer.GetNeuronWeights(), neuronId, nextLayerDeltas);
        var gradientStep = MathUtils.GetGradientStep(learningRate, error, preActivation, activationFunction, out var deltaInner);
        delta = deltaInner;
        for (int i = 0; i < layer.GetLength(1); i++)
        {
            layer[neuronId,i] -= gradientStep * neuronInputs[i];
        }
        neuronBias -= gradientStep;
    }
}