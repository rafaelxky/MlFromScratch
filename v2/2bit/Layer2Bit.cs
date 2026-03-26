
using System.Text.Json;

public class Layer2Bit : ILayer
{

    public double[,]? LatentWeights { get; set; }
    public byte[,]? Weights { get; set; }
    public double[] Bias { get; set; } = Array.Empty<double>();
    public string ActivationFunction { get; set; } = string.Empty;
    public IActivationFunction _activationFunction = default!;

    public int NeuronCount => LatentWeights != null ? LatentWeights.GetLength(0) : Weights != null ? Weights.GetLength(0) : 0;
    public int WeightCount => LatentWeights != null ? LatentWeights.GetLength(1) : Weights != null ? Weights.GetLength(1) * 4 : 0;

    // random instantiation
    public Layer2Bit(int neuronCount, int weightCount, Random rand, IActivationFunction activationFunction)
    {
        ActivationFunction = activationFunction.Name;
        _activationFunction = activationFunction;
        Bias = TensorUtils.NewRandomVector(rand, neuronCount);
        LatentWeights = TensorUtils.NewRandomMatrixContiguousCentered(rand, neuronCount, weightCount);
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
        if (Weights != null)
        {
            int neuronCount = Weights.GetLength(0);
            int weightLength = WeightCount;
            double[] output = new double[neuronCount];

            for (int i = 0; i < neuronCount; i++)
            {
                output[i] = _activationFunction.Apply(BitUtils.PackedDotProduct(Weights, i, input, weightLength) + Bias[i]);
            }
            return output;
        }

        // If 2-bit packed weights are not available, use latent weights with sign during training.
        return NeuronMathUtil.ForwardTrain2Bit(GetNeuronWeights(), input, Bias, _activationFunction, out var _);
    }

    public void Build()
    {
        if (LatentWeights == null)
        {
            throw new InvalidOperationException("Cannot build 2-bit weights without latent weights.");
        }

        int neuronCount = LatentWeights.GetLength(0);
        int weightCount = LatentWeights.GetLength(1);
        int[,] bitmap = new int[neuronCount, weightCount];
        for (int i = 0; i < neuronCount; i++)
        {
            for (int j = 0; j < weightCount; j++)
            {
                bitmap[i, j] = Math.Sign(LatentWeights[i, j]);
            }
        }
        Weights = BitUtils.PackMatrix(bitmap);
    }
    public double[] ForwardTrain(double[] input, out double[] preActivationValues)
    {
        if (LatentWeights == null)
        {
            throw new InvalidOperationException("Latent weights required for training.");
        }
        return NeuronMathUtil.ForwardTrain(LatentWeights, input, Bias, _activationFunction, out preActivationValues);
    }

    public void BackPropagation(
        Layer2Bit? nextLayer,
        double[]? target,
        double learningRate,
        ref LayerCache layerCache,
        LayerCache? nextLayerCache
    )
    {
        if (LatentWeights == null && Weights != null)
        {
            LatentWeights = GetNeuronWeights();
        }

        if (LatentWeights == null)
        {
            throw new InvalidOperationException("No latent weights available for backpropagation.");
        }

        var latent = LatentWeights;

        // output layer
        if (target != null && nextLayer == null)
        {
            // for each neuron
            double[] deltas = new double[latent.GetLength(0)];
            for (int i = 0; i < latent.GetLength(0); i++)
            {
                Console.WriteLine($"i={i} | finalOutput={layerCache.Outputs[i]}, target={target[i]}, preActivation={layerCache.PreActivationValues[i]}, bias={Bias[i]}");
                NeuronMathUtil.UpdateNeuronAtOutput(
                    latent,
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
                Console.WriteLine($"i={i} delta={delta}, preActivation={layerCache.PreActivationValues[i]}, output={layerCache.Outputs[i]}, target={target[i]}, bias={Bias[i]}");
                deltas[i] = delta;
            }
            Console.WriteLine("Assigning deltas to layer cache: " + string.Join(", ", deltas));
            layerCache.Deltas = deltas;
        }
        else
        {
            // for each neuron
            double[] deltas = new double[latent.GetLength(0)];
            for (int i = 0; i < latent.GetLength(0); i++)
            {
                NeuronMathUtil.UpdateNeuron(
                    latent,
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

        // Note: Call Build() manually when ready to switch to packed forward pass.
    }
    public void SetActivationFunction(IActivationFunction activationFunction)
    {
        _activationFunction = activationFunction;
    }

    public double[,] GetNeuronWeights()
    {
        if (LatentWeights != null)
        {
            return LatentWeights;
        }

        if (Weights != null)
        {
            int neuronCount = Weights.GetLength(0);
            int weightCount = WeightCount;
            double[,] signs = new double[neuronCount, weightCount];

            for (int i = 0; i < neuronCount; i++)
            {
                for (int j = 0; j < weightCount; j++)
                {
                    signs[i, j] = BitUtils.GetPair(Weights[i, j / 4], j % 4);
                }
            }

            return signs;
        }

        throw new InvalidOperationException("Layer2Bit has no weights initialized");
    }
}