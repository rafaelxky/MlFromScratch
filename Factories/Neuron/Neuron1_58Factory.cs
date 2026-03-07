using System.Text.Json;
using TrainingMl1_58;

public class Neuron1_58Factory : INeuronFactory
{
    public INeuron[] ArrFromLayerData(LayerData layerData)
    {
        List<INeuron> neurons = new();
        foreach (var neuronData in layerData.Neurons)
        {
            var neuron = new Neuron1_58();
            JsonElement element = (JsonElement)neuronData.Weights;
            byte[] arr = element
            .EnumerateArray()
            .Select(x => x.GetByte())
            .ToArray();
            neuron.Weights = arr;
            neuron.Bias = neuronData.Bias;
            neurons.Add(neuron);
        }
        return neurons.ToArray();
    }
}