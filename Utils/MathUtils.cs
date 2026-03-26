public static class MathUtils
{
    public static double GetGradientStepAtOutput(double learningRate, double finalOutput, double targetOutput, double preActivation, IActivationFunction activationFunction, out double delta)
    {
        var error = CalcErrorAtOutput(finalOutput, targetOutput);
        delta = CalcDelta(preActivation, error, activationFunction);
        return learningRate * delta;
    }
    public static double GetGradientStep(double learningRate, double error, double preActivation, IActivationFunction activationFunction, out double delta)
    {
        delta = CalcDelta(preActivation, error, activationFunction);
        return learningRate * delta;
    }
    public static double CalcErrorAtOutput(double finalOutput, double targetOutput)
    {
        return finalOutput - targetOutput;
    }

    public static double CalcError(double[,] nextLayerNeurons, int neuronId, double[] nextLayerDeltas)
{
    // previous neuronId is next layers weight id
    double error = 0;
    int nextNeuronCount = nextLayerNeurons.GetLength(0);

    for (int nextNeuronId = 0; nextNeuronId < nextNeuronCount; nextNeuronId++)
    {
        error += nextLayerNeurons[nextNeuronId, neuronId] * nextLayerDeltas[nextNeuronId];
    }

    return error;
}

    public static double CalcDelta(double preActivation, double error, IActivationFunction activationFunction)
    {
        return error * activationFunction.Derivative(preActivation);
    }

    public static double CalcGradient(double delta, double inputValue)
    {
        return delta * inputValue;
    }
    public static double MeanSquareError(double finalOutput, double targetOutput)
    {
        return 0.5 * Math.Pow(finalOutput - targetOutput, 2);
    }
}