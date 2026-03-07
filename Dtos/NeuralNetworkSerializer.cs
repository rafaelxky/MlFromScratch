using System.Text.Json;
using MlNetworkTraining;

public static class NetworkSerializer
{
    public static void Save(Network network, string path)
    {
        var data = new NetworkData(
            Layers: network.NeuralNetwork.Select(layer => new LayerData(
                Neurons: layer.GetNeurons().Select(n => new NeuronData(
                    Weights: n.GetWeightsRaw(),
                    Bias: n.GetBias()
                )).ToArray()
            )).ToArray(),
            NeuronType: network.NeuralNetwork[0].GetNeurons()[0].GetType().FullName!,
            LayerType: network.NeuralNetwork[0].GetType().FullName!
        );
        File.WriteAllText(path, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
    }
    public static void Save(TrainingNetwork network, string path)
    {
        var data = new NetworkData(
            Layers: network.NeuralNetwork.Select(layer => new LayerData(
                Neurons: layer.GetNeurons().Select(n => new NeuronData(
                    Weights: n.GetWeightsRaw(),
                    Bias: n.GetBias()
                )).ToArray()
            )).ToArray(),
            NeuronType: network.NeuralNetwork[0].GetNeurons()[0].GetType().FullName!,
            LayerType: network.NeuralNetwork[0].GetType().FullName!
        );
        File.WriteAllText(path, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static Network LoadNetwork(string path)
    {
        var data = JsonSerializer.Deserialize<NetworkData>(File.ReadAllText(path))!;

        // resolve types once, outside the loops
        Type neuronType = Type.GetType(data.NeuronType)
            ?? throw new Exception($"Could not resolve neuron type: {data.NeuronType}");

        Type layerType = Type.GetType(data.LayerType)
            ?? throw new Exception($"Could not resolve layer type: {data.LayerType}");

        // get W from INeuron<W> once
        Type weightType = neuronType.GetInterface(typeof(INeuron<>).Name)!
                                    .GetGenericArguments()[0];

        return new Network
        {
            NeuralNetwork = data.Layers.Select(layerData =>
            {
                INeuron[] neurons = layerData.Neurons.Select(neuronData =>
                {
                    JsonElement weightsJson = (JsonElement)neuronData.Weights;
                    object weights = JsonSerializer.Deserialize(weightsJson.GetRawText(), weightType)!;

                    INeuron neuron = (INeuron)Activator.CreateInstance(neuronType)!;
                    neuron.SetWeightsRaw(weights);
                    neuron.SetBias(neuronData.Bias);
                    return neuron;
                }).ToArray();

                // instantiate layer via reflection, same pattern as neurons
                return (ILayer)Activator.CreateInstance(layerType, new object[] { neurons })!;
            }).ToArray()
        };
    }
    public static TrainingNetwork LoadTrainingNetwork(string path)
    {
        var data = JsonSerializer.Deserialize<NetworkData>(File.ReadAllText(path))!;

        // resolve types once, outside the loops
        Type neuronType = Type.GetType(data.NeuronType)
            ?? throw new Exception($"Could not resolve neuron type: {data.NeuronType}");

        Type layerType = Type.GetType(data.LayerType)
            ?? throw new Exception($"Could not resolve layer type: {data.LayerType}");

        // get W from INeuron<W> once
        Type weightType = neuronType.GetInterface(typeof(ITrainingNeuron<>).Name)!
                                    .GetGenericArguments()[0];

        return new TrainingNetwork
        {
            NeuralNetwork = data.Layers.Select(layerData =>
            {
                ITrainingNeuron[] neurons = layerData.Neurons.Select(neuronData =>
                {
                    JsonElement weightsJson = (JsonElement)neuronData.Weights;
                    object weights = JsonSerializer.Deserialize(weightsJson.GetRawText(), weightType)!;

                    ITrainingNeuron neuron = (ITrainingNeuron)Activator.CreateInstance(neuronType)!;
                    neuron.SetWeightsRaw(weights);
                    neuron.SetBias(neuronData.Bias);
                    return neuron;
                }).ToArray();

                // instantiate layer via reflection, same pattern as neurons
                return (ITrainingLayer)Activator.CreateInstance(layerType, new object[] { neurons })!;
            }).ToArray()
        };
    }
}