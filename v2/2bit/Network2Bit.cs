using System.Text.Json;

public class Network2Bit : INetwork
{
    public List<Layer2Bit> Layers { get; set; }
    public int Depth => Layers.Count;
    public Layer2Bit LastLayer => Layers[Layers.Count - 1];
    public Layer2Bit FirstLayer => Layers[0];
    private Random _random;
    public NetworkConfig Config;

    double[] bufferA;
    double[] bufferB;
    public int MaxSize = 0;

    public Network2Bit()
    {
        Config = new();
        Layers = new();
        _random = new();
    }

    public Network2Bit(int inputSize, int neuronCount, IActivationFunction activationFunction)
    {
        Config = new();
        Layers = new();
        _random = new Random();
        Layers.Add(new Layer2Bit(neuronCount, inputSize, _random, activationFunction));
        MaxSize = inputSize;
        bufferA = new double[MaxSize];
        bufferB = new double[MaxSize];
    }
    public void AddNewLayer(int neuronCount, IActivationFunction activationFunction)
    {
        var lastLayerNeuronCount = Layers[Layers.Count - 1].GetNeuronWeights().GetLength(0);
        Layers.Add(new Layer2Bit(neuronCount, lastLayerNeuronCount, _random, activationFunction));
        if (neuronCount > MaxSize)
        {
            MaxSize = neuronCount;
            bufferA = new double[MaxSize];
            bufferB = new double[MaxSize];
        }
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
            case AccelerationType.SimdParallel:
                return ForwardPassSimd(values);
            default:
                return ForwardPassCpu(values);
        }
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
            SimdBitUtils.ForwardPass(layer, current, next);
            (current, next) = (next, current);
        }
        return current[..Layers[^1].NeuronCount];
    }
    public double[] ForwardPassCpu(double[] values)
    {
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = layer.ForwardPass(output);
        }
        return output;
    }
    public double[] ForwardTrain(double[] values, out List<LayerCache> layerCaches)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }

        switch (Config.AccelerationType)
        {
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
            output = layer.ForwardTrain(output, out var layerCachesInner);
            layerCaches.Add(layerCachesInner);
        }
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
            Layer2Bit layer = Layers[i];
            // null if last layer
            // next means closer to output
            Layer2Bit? nextLayer = (i < Depth - 1) ? Layers[i + 1] : null;

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
        var layers = Layers;
        var layersSave = new Save2Bit[Depth];
        int i = 0;
        foreach (var layer in layers)
        {
            var save = new Save2Bit
            {
                Weights = layer.Weights,
                Bias = layer.Bias,
                ActivationFunction = layer.ActivationFunction
            };
            layersSave[i] = save;
            i++;
        }
        var options = new JsonSerializerOptions { WriteIndented = true };
        options.Converters.Add(new ByteArray2DConverter());
        var saveJson = JsonSerializer.Serialize(layersSave, options);
        File.WriteAllText(path, saveJson);
    }
    public void SaveLatent(string path)
    {
        var layers = Layers;
        var layersSave = new LatentWeightsSave[Depth];
        int i = 0;
        foreach (var layer in layers)
        {
            var save = new LatentWeightsSave
            {
                Weights = layer.LatentWeights,
                Bias = layer.Bias,
                ActivationFunction = layer.ActivationFunction
            };
            layersSave[i] = save;
            i++;
        }
        var options = new JsonSerializerOptions { WriteIndented = true };
        options.Converters.Add(new TwoDimensionalArrayConverter());
        var saveJson = JsonSerializer.Serialize(layersSave, options);
        File.WriteAllText(path, saveJson);
    }
    public static Network2Bit Load(string path)
    {
        Network2Bit network2Bit = new Network2Bit();
        string content = File.ReadAllText(path);
        Save2Bit[] weights = JsonSerializer.Deserialize<Save2Bit[]>(content);
        foreach (var layer in weights)
        {
            var newLayer = new Layer2Bit(layer.Weights, layer.Bias, layer.ActivationFunction);
            network2Bit.Layers.Add(newLayer);
        }
        return network2Bit;
    }
    public static Network2Bit LoadLatent(string path)
    {
        Network2Bit network2Bit = new Network2Bit();
        string content = File.ReadAllText(path);
        LatentWeightsSave[] latentWeights = JsonSerializer.Deserialize<LatentWeightsSave[]>(content);
        foreach (var layer in latentWeights)
        {
            var newLayer = new Layer2Bit(layer.Weights, layer.Bias, layer.ActivationFunction);
            network2Bit.Layers.Add(newLayer);
        }
        return network2Bit;
    }


    public void Build()
    {
        foreach (var layer in Layers)
        {
            layer.Build();
        }
    }

}