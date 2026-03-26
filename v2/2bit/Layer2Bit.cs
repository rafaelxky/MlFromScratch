
using System.Text.Json;

public class Layer2Bit : ILayer
{

    public double[,] LatentWeights { get; set; }
    public byte[,] Weights { get; set; }
    public double[] Bias { get; set; }
    public string ActivationFunction { get; set; }
    public IActivationFunction _activationFunction;
    public int NeuronCount => LatentWeights.GetLength(0);
    public int WeightCount => LatentWeights.GetLength(1);

    // random instantiation
    public Layer2Bit(int neuronCount, int weightCount, Random rand, IActivationFunction activationFunction)
    {
        ActivationFunction = activationFunction.Name;
        _activationFunction = activationFunction;
        Bias = TensorUtils.NewRandomVector(rand, neuronCount);
        LatentWeights = TensorUtils.NewRandomMatrixContiguous(rand, neuronCount, weightCount);
    }
    public Layer2Bit(double[,] latentWeights, double[] bias, string activationFunction)
    {
        LatentWeights = latentWeights;
        Bias = bias;
        ActivationFunction = activationFunction;
        _activationFunction = ActivationFunctionRegistry.GetFunction(activationFunction);
    }
    public Layer2Bit(byte[,] weights, double[] bias, string activationFunction)
    {
        Weights = weights;
        Bias = bias;
        ActivationFunction = activationFunction;
        _activationFunction = ActivationFunctionRegistry.GetFunction(activationFunction);
    }

    public double[] ForwardPass(double[] input)
    {
        double[] output = new double[Weights.GetLength(0)];
        int weightLength = Weights.GetLength(1);
        // foreach neuron in layer, calc output and build vector
        for (int i = 0; i < Weights.GetLength(0); i++)
        {
            output[i] = _activationFunction.Apply(BitUtils.PackedDotProduct(Weights, i, input, weightLength) + Bias[i]);
        }
        return output;
    }
    public void Build()
    {
        int[,] bitmap = new int[LatentWeights.GetLength(0), LatentWeights.GetLength(1)];
        for (int i = 0; i < LatentWeights.GetLength(0); i++)
        {
            for (int j = 0; j < LatentWeights.GetLength(1); j++)
            {
                bitmap[i, j] = Math.Sign(LatentWeights[i, j]);
            }
        }
        Weights = BitUtils.PackMatrix(bitmap);
    }
    public double[] ForwardTrain(double[] input, out LayerCache trainingCache)
    {
        trainingCache = new LayerCache
        {
            Inputs = (double[])input.Clone()
        };

        double[] output = new double[NeuronCount];
        double[] preActivationValues = new double[NeuronCount];
        for (int i = 0; i < NeuronCount; i++)
        {
            output[i] = NeuronMathUtil.Calc2BitNeuronOutput(LatentWeights, i, input, Bias[i], _activationFunction, out var preActivationValue);
            preActivationValues[i] = preActivationValue;
        }

        trainingCache.PreActivationValues = (double[])preActivationValues.Clone();
        trainingCache.Outputs = (double[])output.Clone();
        return output;
    }

    public void BackPropagation(
        Layer2Bit? nextLayer,
        double[]? target,
        double learningRate,
        ref LayerCache layerCache,
        LayerCache? nextLayerCache
    )
    {
        // output layer
        if (target != null && nextLayer == null)
        {
            // for each neuron
            double[] deltas = new double[NeuronCount];
            for (int i = 0; i < NeuronCount; i++)
            {
                NeuronMathUtil.UpdateNeuronAtOutput(
                    LatentWeights,
                    i,
                    layerCache.Outputs[i],
                    target[i],
                    layerCache.PreActivationValues[i],
                    _activationFunction,
                    learningRate,
                    layerCache.Inputs,
                    ref Bias[i],
                    out var delta
                );
                deltas[i] = delta;
            }
            layerCache.Deltas = deltas;
        }
        else
        {
            // for each neuron
            double[] deltas = new double[NeuronCount];
            for (int i = 0; i < NeuronCount; i++)
            {
                NeuronMathUtil.UpdateNeuron(
                    LatentWeights,
                    i,
                    nextLayer!,
                    learningRate,
                    layerCache.PreActivationValues[i],
                    _activationFunction,
                    layerCache.Inputs,
                    nextLayerCache!.Value.Deltas,
                    ref Bias[i],
                    out var delta
                );
                deltas[i] = delta;
            }
            layerCache.Deltas = deltas;
        }
    }
    public void SetActivationFunction(IActivationFunction activationFunction)
    {
        _activationFunction = activationFunction;
    }

    public double[,] GetNeuronWeights()
    {
        return LatentWeights;
    }


}