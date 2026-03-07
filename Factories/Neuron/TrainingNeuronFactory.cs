
using System.Text.Json;
using MlNetworkTraining;

public class TrainingNeuronFactory: ITrainingNeuronFactory
{
    public TrainingNeuronFactory()
    {
        
    }

    public ITrainingNeuron[] ArrFromLayerData(LayerData layerData)
    {
          List<ITrainingNeuron> neurons = new();
        foreach (var neuronData in layerData.Neurons)
        {
            var neuron = new TrainingNeuron();
            var element = (JsonElement)neuronData.Weights;
            double[] arr = element
            .EnumerateArray()
            .Select(x => x.GetDouble())
            .ToArray();
            neuron.SetWeightsRaw(arr);
            neuron.Bias = neuronData.Bias;
            neurons.Add(neuron);
        }
        return neurons.ToArray();
    }

    public ITrainingNeuron NewNeuron(int inputSize, Random random)
    {
        return new TrainingNeuron(inputSize,random);
    }

}