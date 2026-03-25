using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Serialization;

public class Layer: ILayer
{
    [JsonInclude]
    public double[,] Neurons { get; set; }
    [JsonInclude]
    public double[] Bias { get; set; }
    [JsonInclude]
    public string ActivationFunction { get; set; }
    [JsonIgnore]
    public IActivationFunction _activationFunction;
    [JsonIgnore]
    public int NeuronCount => Neurons.GetLength(0);
    [JsonIgnore]
    public int WeightCount => Neurons.GetLength(1);

    public Layer()
    {
        
    }

    // random instantiation
    [SetsRequiredMembers]
    public Layer(int neuronCount, int weightCount, Random rand, IActivationFunction activationFunction)
    {
        ActivationFunction = activationFunction.Name;
        _activationFunction = activationFunction;
        Bias = TensorUtils.NewRandomVector(rand, neuronCount);
        Neurons = TensorUtils.NewRandomMatrixContiguous(rand, neuronCount, weightCount);
    }

    public double[] ForwardPass(double[] input)
    {
        return NeuronMathUtil.ForwardPass(this, input);
        
    }
    public double[] ForwardTrain(double[] input, out double[] preActivationValues)
    {
        var output = NeuronMathUtil.ForwardTrain(Neurons,input,Bias,_activationFunction,out var layerCache);
        preActivationValues = layerCache;
        return output;
    }

    public void BackPropagation(
        Layer? nextLayer,
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
                    Neurons,
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
        } else
        {
            // for each neuron
            double[] deltas = new double[NeuronCount];
            for (int i = 0; i < NeuronCount; i++)
            {
                NeuronMathUtil.UpdateNeuron(
                    Neurons,
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
        return Neurons;
    }
}