using System.Text.Json;
using MlNetworkTraining;

public class NeuronFactory : INeuronFactory
{
    public INeuron[] ArrFromLayerData(LayerData layerData, string neuronType)
    {
        List<INeuron> neurons = new();
        foreach (var neuronData in layerData.Neurons)
        {
            var neuron = new Neuron();
            JsonElement element = (JsonElement)neuronData.Weights;
            double[] arr = element
    .EnumerateArray()
    .Select(x => x.GetDouble())
    .ToArray();
            neuron.Weights = arr;
            neuron.Bias = neuronData.Bias;
            neurons.Add(neuron);
        }
        return neurons.ToArray();
    }
}