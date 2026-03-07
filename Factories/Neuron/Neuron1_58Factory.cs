using System.Text.Json;
using TrainingMl1_58;

public class Neuron1_58Factory : INeuronFactory
{
    public INeuron[] ArrFromLayerData(LayerData layerData, string neuronType)
    {
        List<INeuron> neurons = new();
        if (neuronType.EndsWith(".Neuron1_58"))
        {
            foreach (var neuronData in layerData.Neurons)
            {
                JsonElement element = (JsonElement)neuronData.Weights;
                byte[] arr = element
                .EnumerateArray()
                .Select(x => x.GetByte())
                .ToArray();
                var neuron = new Neuron1_58(arr);
                neuron.Bias = neuronData.Bias;
                neurons.Add(neuron);
            }
        }
        else
        {
            foreach (var neuronData in layerData.Neurons)
            {
                JsonElement element = (JsonElement)neuronData.Weights;
                double[] arr = element
                .EnumerateArray()
                .Select(x => x.GetDouble())
                .ToArray();
                var neuron = new Neuron1_58(arr);
                neuron.Bias = neuronData.Bias;
                neurons.Add(neuron);
            }
        }
        return neurons.ToArray();
    }
}