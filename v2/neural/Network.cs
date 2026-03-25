using System.IO.Pipelines;
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
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        switch (Config.AccelerationType)
        {
            case AccelerationType.Simd:
                return ForwardPassSimd(values);
            case AccelerationType.Gpu:
                return ForwardPassGpu(values);
            default:
                return ForwardPassCpu(values);
        }
    }
    public double[] ForwardPassGpu(double[] values)
    {
        return gpu.NetworkForwardPass(values, Layers);
    }
    public double[] ForwardPassCpu(double[] values)
    {
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = NeuronMathUtil.ForwardPass(layer, output);
        }
        return output;
    }
    public double[] ForwardPassSimd(double[] values)
    {
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = SimdUtils.ForwardPass(layer, output);
        }
        return output;
    }
    public double[] ForwardTrain(double[] values, out List<LayerCache> layerCaches)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        double[] result;
        switch (Config.AccelerationType)
        {
            case AccelerationType.Simd:
                result = ForwardTrainSimd(values, out var cache1);
                layerCaches = cache1;
                break;
            case AccelerationType.Gpu:
                result = ForwardTrainGpu(values, out var cache2);
                layerCaches = cache2;
                break;
            default:
                result = ForwardTrainCpu(values, out var cache3);
                layerCaches = cache3;
                break;
        }
        return result;
    }
    public double[] ForwardTrainCpu(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();
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
    public double[] ForwardTrainSimd(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();
        double[] output = values;

        foreach (var layer in Layers)
        {
            var cache = new LayerCache
            {
                Inputs = output
            };
            //output = gpu.ForwardTrain(layer,output,out var preActivationValues);
            output = SimdUtils.ForwardTrain(layer, output, out var preActivationValues);
            cache.Outputs = output;
            cache.PreActivationValues = preActivationValues;
            layerCaches.Add(cache);
        }
        return output;
    }
    public double[] ForwardTrainGpu(double[] values, out List<LayerCache> layerCaches)
    {
        
        var output = gpu.NetworkForwardTrain(values, Layers, out var preActivationValues);
        layerCaches = preActivationValues;
        return output;
    }

    public void BackPropagation(double[] expected, double learningRate, List<LayerCache> layerCaches)
    {
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