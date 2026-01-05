
using Microsoft.VisualBasic;

public class Neuron
{
    public double[] Weights;
    double[] Values;
    public double Bias;
    double Z;
    public double Output;
    public double Delta;
    public Neuron(int size, Random rand)
    {
        Weights = new double[size];
        for (int i = 0; i < size; i++)
        {
            // Random double between -0.5 and 0.5
            Weights[i] = rand.NextDouble() - 0.5;
        }
        Bias = rand.NextDouble() - 0.5;
    }
    public double Calc(double[] values)
    {
        Values = values;
        double value = 0;
        for (int i = 0; i < values.Length; i++)
        {
            value += values[i] * Weights[i];
        }
        value += Bias;
        Z = value;
        Output = SigmoidActivation(value);
        return Output;
    }
    public double MeanSquareError(double finalOutput, double targetOutput)
    {
        return 0.5 * Math.Pow(finalOutput - targetOutput, 2);
    }

    public double CalcErrorAtOutput(double finalOutput, double targetOutput)
    {
        return finalOutput - targetOutput;
    }
    public double CalcError(double nextWeight, double nextDelta)
    {
        // next weights corresponds to the weight of the output on the next neuron
        return nextWeight * nextDelta;
    }
    public double CalcGradient(double z, double inputValue, double error)
    {
        // finalOutput is the neuron output after activation function
        // target output is the final target output at the last neuron
        // z is the preactivation scalar
        var s = error * SigmoidDerivative(z);
        Delta = s;
        return s * inputValue;
    }
    public double CalcNewWeight(double currentWeight, double learningRate, double gradient)
    {
        return currentWeight - learningRate * gradient;
    }

    public double CalcNewBias(double gradient, double learningRate)
    {
        return Bias - learningRate * gradient;
    }
    public void SetWeights(double[] weights)
    {
        Weights = weights;
    }
    public void SetBias(double bias)
    {
        Bias = bias;
    }
    public double SigmoidActivation(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }
    public double SigmoidDerivative(double value)
    {
        var result = SigmoidActivation(value);
        return result * (1 - result);
    }

    public void RecalcWeights(double error, double learningRate)
    {
        // Compute delta
        Delta = error * SigmoidDerivative(Z);

        // Update weights
        for (int i = 0; i < Values.Length; i++)
        {
            Weights[i] -= learningRate * Delta * Values[i];
        }

        // Update bias
        Bias -= learningRate * Delta;
    }
}