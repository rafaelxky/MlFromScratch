using System.Reflection;
using System.Text.Json;
using MlNetworkTraining;

public static class NeuralNetworkSerializer
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    public static void Save<L, N>(TrainingNetwork network, string path)
        where L : ILayer
        where N : INeuron
    {
        var data = new NetworkData(
            NeuronType: typeof(N).Name,
            Layers: network.NeuralNetwork.Select(layer => new LayerData(
                Neurons: layer.GetNeurons().Select(n => new NeuronData(
                    Weights: n.GetWeights().ToArray(),
                    Bias: n.GetBias()
                )).ToArray()
            )).ToArray()
        );

        File.WriteAllText(path, JsonSerializer.Serialize(data, Options));
    }

    // returns a non-generic base class so caller doesn't need to know N
    public static Network Load(string path)
    {
        string json = File.ReadAllText(path);
        NetworkData data = JsonSerializer.Deserialize<NetworkData>(json, Options)
            ?? throw new Exception("Failed to deserialize");

        // resolve neuron type from string at runtime
        Type neuronType = NeuronRegistry.Resolve(data.NeuronType);

        // use reflection to call the generic builder with the correct type
        var method = typeof(NeuralNetworkSerializer)
            .GetMethod(nameof(BuildNetwork), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(neuronType);

        return (Network)method.Invoke(null, new object[] { data })!;
    }

    private static Network BuildNetwork<N>(NetworkData data)
        where N : INeuron
    {
        var network = new Network
        {
            NeuralNetwork = data.Layers.Select(layerData =>
            {
                var layer = new Layer
                {
                    NeuronLayer = layerData.Neurons.Select(neuronData =>
                    {
                        N neuron = (N)Activator.CreateInstance(typeof(N), neuronData.Weights.Length)!;
                        neuron.SetWeights(neuronData.Weights);
                        neuron.SetBias(neuronData.Bias);
                        return neuron;
                    }).ToArray()
                };
                return layer;
            }).ToArray()
        };
        return network;
    }
}