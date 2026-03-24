using System.Text.Json;
using ILGPU.Backends;
using ILGPU.IR;

public class Network : INetwork
{
    public List<Layer> Layers { get; set; }
    public int Depth => Layers.Count;
    public Layer LastLayer => Layers[Layers.Count - 1];
    public Layer FirstLayer => Layers[0];
    private Random? _random;
    private GpuUtils gpu;
    public NetworkConfig Config;

    public Network()
    {
        gpu = new();
        gpu.CompileDoubleNetworkKernels();
        Config = new();
        _random = new Random();
    }

    public Network(int inputSize, int neuronCount, IActivationFunction activationFunction)
    {
        Layers = new();
        _random = new Random();
        Layers.Add(new Layer(neuronCount, inputSize, _random, activationFunction));
        gpu = new();
        gpu.CompileDoubleNetworkKernels();
        Config = new();
    }
    public void AddNewLayer(int neuronCount, IActivationFunction activationFunction)
    {
        var lastLayerNeuronCount = Layers[Layers.Count - 1].Neurons.GetLength(0);
        Layers.Add(new Layer(neuronCount, lastLayerNeuronCount, _random, activationFunction));
    }

    public double[] ForwardPass(double[] values)
    {
        if (Config.UseGpu)
        {
            return ForwardPassGpu(values);
        }
        else
        {
            return ForwardPassCpu(values);
        }
    }
    public double[] ForwardPassGpu(double[] values)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        return gpu.NetworkForwardPass(values, Layers);
    }
    public double[] ForwardPassCpu(double[] values)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = NeuronMathUtil.ForwardPass(layer, output);
        }
        return output;
    }
    public double[] ForwardTrain(double[] values, out List<LayerCache> layerCaches)
    {
        if (Config.UseGpu)
        {
            var result = ForwardTrainGpu(values, out var cache);
            layerCaches = cache;
            return result;
        }
        else
        {
            var result = ForwardTrainCpu(values, out var cache);
            layerCaches = cache;
            return result;
        }
    }
    public double[] ForwardTrainCpu(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        double[] output = values;

        foreach (var layer in Layers)
        {
            var cache = new LayerCache
            {
                Inputs = output
            };
            //output = gpu.ForwardTrain(layer,output,out var preActivationValues);
            output = NeuronMathUtil.ForwardTrain(layer, output, out var preActivationValues);
            cache.Outputs = output;
            cache.PreActivationValues = preActivationValues;
            layerCaches.Add(cache);
        }
        return output;
    }
    public double[] ForwardTrainGpu(double[] values, out List<LayerCache> layerCaches)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        var output = gpu.NetworkForwardTrain(values, Layers,out var preActivationValues);
        layerCaches = preActivationValues;
        return output;
    }

    public void BackPropagation(double[] expected, double learningRate, List<LayerCache> layerCaches)
    {
        if (expected.Length != LastLayer.NeuronCount)
        {
            throw new WrongInputSizeException($"Back propagation input must have length {LastLayer.WeightCount} for this network!");
        }
        for (int i = Depth - 1; i >= 0; i--)
        {
            Layer layer = Layers[i];
            // null if last layer
            // next means closer to output
            Layer? nextLayer = (i < Depth - 1) ? Layers[i + 1] : null;

            // null if first
            double[]? targetOutput = (i == Depth - 1) ? expected : null;

            LayerCache currentLayerCache = layerCaches[i];
            LayerCache? nextLayerCache = (layerCaches.Count > i + 1) ? layerCaches[i + 1] : null;

            layer.BackPropagation(
                nextLayer,
                targetOutput,
                learningRate,
                ref currentLayerCache,
                nextLayerCache
                );

            layerCaches[i] = currentLayerCache;
        }
    }

    public void Save(string path)
    {
        var options = new JsonSerializerOptions { WriteIndented = true };
        var jsonContent = JsonSerializer.Serialize(this, options);
        File.WriteAllText(path, jsonContent);
    }
    public static Network Load(string path)
    {
        var textContent = File.ReadAllText(path);
        var network = JsonSerializer.Deserialize<Network>(textContent);
        foreach (var layer in network.Layers)
        {
            layer.SetActivationFunction(ActivationFunctionRegistry.GetFunction(layer.ActivationFunction));
        }
        return network;
    }
}