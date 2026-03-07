using System.Text.Json;

namespace Ml1_58
{
public class Network1_58
{
    public List<Layer1_58> NeuralNetwork { get; set; }
    public int Depth { get; set; }

    public Network1_58()
    {
        
    }
    public Network1_58(int inputSize, int[] hiddenLayerSizes, int outputSize)
    {
        NeuralNetwork = new();
        Random random = new();

        // Input -> first hidden
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayerSizes)
        {
            NeuralNetwork.Add(new Layer1_58(layerSize, random, previousSize));
            previousSize = layerSize;
        }

        // Output layer
        NeuralNetwork.Add(new Layer1_58(outputSize, random, previousSize));
        Depth = NeuralNetwork.Count;
    }
    public double[] ForwardPass(double[] values)
    {
        double[] last = values;
        foreach (var layer in NeuralNetwork)
        {
            last = layer.ForwardPass(last);
        }
        Console.WriteLine(string.Join(", ", last));
        return last;
    }

    public void BackPropagation(double[] expected, double learningRate)
    {
        for (int i = Depth - 1; i >= 0; i--) // start from output layer
        {
            Layer1_58 layer = NeuralNetwork[i];
            Layer1_58? nextLayer = (i < Depth - 1) ? NeuralNetwork[i + 1] : null;

            // Pass expected only to the output layer
            double[]? targets = (i == Depth - 1) ? expected : null;

            layer.BackPropagation(nextLayer, targets, learningRate);
        }
    }

    public void Print()
    {
        foreach (var layer in NeuralNetwork)
        {
            layer.Print();
        }
    }

    public void Save(string filePath)
    {
        var opts = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        var json = JsonSerializer.Serialize(this, opts);
        File.WriteAllText(filePath, json);
    }
    public static Network1_58 NewFromJson(string filePath)
    {
            var jsonText = File.ReadAllText(filePath);
            var network = JsonSerializer.Deserialize<Network1_58>(jsonText);
            return network!;
    }
    public static Network1_58 NewFromJsonOrDefault(string filePath, Network1_58 network)
    {
        if (File.Exists(filePath))
        {
            var jsonText = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<Network1_58>(jsonText)!;
        }
        return network;
    }
}
}