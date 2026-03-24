using System.Text.Json;

public class Network2Bit: INetwork
{
    public List<Layer2Bit> Layers { get; set; }
    public int Depth => Layers.Count;
    public Layer2Bit LastLayer => Layers[Layers.Count - 1];
    public Layer2Bit FirstLayer => Layers[0];
    private Random _random;

    public Network2Bit()
    {
        Layers = new();
        _random = new();
    }

    public Network2Bit(int inputSize, int neuronCount, IActivationFunction activationFunction)
    {
        Layers = new();
        _random = new Random();
        Layers.Add(new Layer2Bit(neuronCount, inputSize, _random, activationFunction));
    }
    public void AddNewLayer(int neuronCount, IActivationFunction activationFunction)
    {
        var lastLayerNeuronCount = Layers[Layers.Count - 1].GetNeuronWeights().GetLength(0);
        Layers.Add(new Layer2Bit(neuronCount, lastLayerNeuronCount, _random, activationFunction));
    }

    public double[] ForwardPass(double[] values)
    {
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        double[] output = values;
        foreach (var layer in Layers)
        {
            output = layer.ForwardPass(output);
        }
        return output;
    }
    public double[] ForwardTrain(double[] values, out List<LayerCache> layerCaches)
    {
        layerCaches = new();
        if (values.Length != FirstLayer.WeightCount)
        {
            throw new WrongInputSizeException($"Forward pass input must have lenght {FirstLayer.WeightCount} for this network!");
        }
        double[] output = values;

        foreach (var layer in Layers)
        {
            output = layer.ForwardTrain(output, out var trainingCache);
            layerCaches.Add(trainingCache);
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

    public void SaveLatent(string path)
    {
        var layers = Layers;
        var LatentWeightsSave = new LatentWeightsSave[Depth];
        int i = 0;
        foreach (var layer in layers)
        {
            var save = new LatentWeightsSave();
            save.Weights = layer.LatentWeights;
            save.Bias = layer.Bias;
            save.ActivationFunction = layer.ActivationFunction;
            LatentWeightsSave[i] = save;
            i++;
        }
        var options = new JsonSerializerOptions { WriteIndented = true };
        var saveJson = JsonSerializer.Serialize(LatentWeightsSave, options);
        File.WriteAllText(path, saveJson);
    }
    public void Save(string path)
    {
        var layers = Layers;
        var LatentWeightsSave = new Save2Bit[Depth];
        int i = 0;
        foreach (var layer in layers)
        {
            var save = new Save2Bit();
            save.Weights = layer.Weights;
            save.Bias = layer.Bias;
            save.ActivationFunction = layer.ActivationFunction;
            LatentWeightsSave[i] = save;
            i++;
        }
        var options = new JsonSerializerOptions { WriteIndented = true };
        var saveJson = JsonSerializer.Serialize(LatentWeightsSave, options);
        File.WriteAllText(path, saveJson);
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

    public void Build()
    {
        foreach (var layer in Layers)
        {
            layer.Build();
        }
    }

}