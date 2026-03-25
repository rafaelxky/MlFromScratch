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
    private Random _random;
    private GpuUtils gpu;
    public NetworkConfig Config;
    public int MaxSize = 0;

    double[] bufferA;
    double[] bufferB;

    public Network(List<Layer> layers)
    {
        Layers = layers;
        gpu = new();
        gpu.CompileDoubleNetworkKernels();
        Config = new();
        _random = new Random();
        foreach (var layer in layers)
        {
            if (layer.NeuronCount > MaxSize)
            {
                MaxSize = layer.NeuronCount;
            }
        }
        bufferA = new double[MaxSize];
        bufferB = new double[MaxSize];
    }

    public Network(int inputSize, int neuronCount, IActivationFunction activationFunction)
    {
        MaxSize = inputSize;
        Layers = new();
        _random = new Random();
        Layers.Add(new Layer(neuronCount, inputSize, _random, activationFunction));
        gpu = new();
        gpu.CompileDoubleNetworkKernels();
        Config = new();
        bufferA = new double[MaxSize];
        bufferB = new double[MaxSize];
    }
    public void AddNewLayer(int neuronCount, IActivationFunction activationFunction)
    {
        if (neuronCount > MaxSize)
        {
            MaxSize = neuronCount;
            bufferA = new double[MaxSize];
            bufferB = new double[MaxSize];
        }
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
            case AccelerationType.SimdParallel:
                return ForwardPassSimdParallel(values);
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
        Console.WriteLine("Forward pass cpu");
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = NeuronMathUtil.ForwardPass(layer, output);
        }
        return output;
    }
    public double[] ForwardPassSimd(double[] values)
    {
        values.CopyTo(bufferA, 0);

        // layer input
        double[] current = bufferA;
        // layer output
        double[] next = bufferB;

        foreach (var layer in Layers)
        {
            SimdUtils.ForwardPass(layer, current, next);
            (current, next) = (next, current);
        }

        return current[..Layers[^1].NeuronCount];
    }
    public double[] ForwardPassSimdParallel(double[] values)
    {
        values.CopyTo(bufferA, 0);

        // layer input
        double[] current = bufferA;
        // layer output
        double[] next = bufferB;

        foreach (var layer in Layers)
        {
            // current (input) needs to be of correct lenght
            ParallelUtils.ForwardPass(layer, current, next);
            (current, next) = (next, current);
        }
        return current[..Layers[^1].NeuronCount];
    }
    public double[] ForwardTrain(double[] values, out List<LayerCache> layerCaches)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        switch (Config.AccelerationType)
        {
            case AccelerationType.Simd:
                return ForwardTrainSimd(values, out layerCaches);
            case AccelerationType.Gpu:
                return ForwardTrainGpu(values, out layerCaches);
            case AccelerationType.SimdParallel:
                return ForwardTrainSimdParallel(values, out layerCaches);
            default:
                return ForwardTrainCpu(values, out layerCaches);
        }
    }
    public double[] ForwardTrainCpu(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();
        double[] output = values;

        foreach (var layer in Layers)
        {
            var cache = new LayerCache
            {
                Inputs = (double[])output.Clone()
            };
            output = NeuronMathUtil.ForwardTrain(layer, output, out var preActivationValues);
            cache.Outputs = (double[])output.Clone();
            cache.PreActivationValues = preActivationValues;
            layerCaches.Add(cache);
        }
        return output;
    }
    public double[] ForwardTrainSimd(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();

        values.CopyTo(bufferA, 0);

        // layer input
        double[] current = bufferA;
        // layer output
        double[] next = bufferB;

        foreach (var layer in Layers)
        {
            int inputSize   = layer.Neurons.GetLength(1);
            int outputSize  = layer.NeuronCount;
            var cache = new LayerCache
            {
                Inputs = current[..inputSize].ToArray()
            };
            //output = gpu.ForwardTrain(layer,output,out var preActivationValues);
            SimdUtils.ForwardTrain(layer, current, next, out var preActivationValues);
            cache.Outputs = next[..outputSize].ToArray();
            cache.PreActivationValues = preActivationValues;
            layerCaches.Add(cache);
            (current, next) = (next, current);
        }
        return current[..Layers[^1].NeuronCount];
    }
    public double[] ForwardTrainSimdParallel(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();

        values.CopyTo(bufferA, 0);

        // layer input
        double[] current = bufferA;
        // layer output
        double[] next = bufferB;


        //Console.WriteLine($"inputs [{string.Join(", ", values)}]");
        foreach (var layer in Layers)
        {
            int inputSize   = layer.Neurons.GetLength(1);
            int outputSize  = layer.NeuronCount;

            var cache = new LayerCache
            {
                Inputs = current[..inputSize].ToArray()
            };
            //Console.WriteLine($"cacheInputs [{string.Join(", ", cache.Inputs)}]");
            //output = gpu.ForwardTrain(layer,output,out var preActivationValues);
            ParallelUtils.ForwardTrain(layer, current, next, out var preActivationValues);
            cache.Outputs = next[..outputSize].ToArray();
            //Console.WriteLine($"cacheOutputs [{string.Join(", ", cache.Inputs)}]");
            cache.PreActivationValues = preActivationValues;
            layerCaches.Add(cache);
            (current, next) = (next, current);
        }
        return current[..Layers[^1].NeuronCount];
    }
    public double[] ForwardTrainGpu(double[] values, out List<LayerCache> layerCaches)
    {

        var output = gpu.NetworkForwardTrain(values, Layers, out var preActivationValues);
        layerCaches = preActivationValues;
        return output;
    }

    public void BackPropagation(double[] expected, double learningRate, List<LayerCache> layerCaches)
    {
        // for each layer
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
        var options = new JsonSerializerOptions
        {
            Converters = { new TwoDimensionalArrayConverter() },
            WriteIndented = true
        };
        var jsonContent = JsonSerializer.Serialize(Layers, options);
        File.WriteAllText(path, jsonContent);
    }
    public static Network Load(string path)
    {
        var options = new JsonSerializerOptions
        {
            Converters = { new TwoDimensionalArrayConverter() },
            WriteIndented = true
        };

        try
        {
            var textContent = File.ReadAllText(path);
            var layers = JsonSerializer.Deserialize<List<Layer>>(textContent, options);
            foreach (var layer in layers!)
            {
                layer.SetActivationFunction(ActivationFunctionRegistry.GetFunction(layer.ActivationFunction));
            }
            return new Network(layers);
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"Failed to parse network: {ex.Message}");
            Environment.Exit(0);
            return null;
        }
    }
}