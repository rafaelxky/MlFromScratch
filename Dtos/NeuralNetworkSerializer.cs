using System.Text.Json;
using MlNetworkTraining;

public static class NetworkSerializer
{
    public static void Save(INetwork network, string path)
    {
        var data = new NetworkData(
            Layers: network.GetLayers().Select(layer => new LayerData(
                Neurons: layer.GetNeurons().Select(n => new NeuronData(
                    Weights: n.GetWeightsRaw(),
                    Bias: n.GetBias()
                )).ToArray()
            )).ToArray(),
            NeuronType: network.GetLayers()[0].GetNeurons()[0].GetType().FullName!,
            LayerType: network.GetLayers()[0].GetType().FullName!
        );
        File.WriteAllText(path, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static INetwork LoadNetwork(string path, INetworkFactory networkFactory)
    {
        var data = JsonSerializer.Deserialize<NetworkData>(File.ReadAllText(path))!;
        return networkFactory.FromDto(data);
    }
    public static ITrainingNetwork LoadNetwork(string path, ITrainingNetworkFactory networkFactory)
    {
        var data = JsonSerializer.Deserialize<NetworkData>(File.ReadAllText(path))!;
        return networkFactory.FromDto(data);
    }
    public static ITrainingNetwork LoadNetworkDefault(string path)
    {
        var neuFac = new TrainingNeuronFactory();
        var layFac = new TrainingLayerFactory(neuFac);
        var netFac = new TrainingNetworkFactory(layFac);
        var data = JsonSerializer.Deserialize<NetworkData>(File.ReadAllText(path))!;
        return netFac.FromDto(data);
    }
}
    