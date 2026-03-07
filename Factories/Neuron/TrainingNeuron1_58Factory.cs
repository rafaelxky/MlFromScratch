
using System.Text.Json;
using MlNetworkTraining;
using TrainingMl1_58;

namespace TrainingMl1_58
{
    public class TrainingNeuron1_58Factory : ITrainingNeuronFactory
    {
        public ITrainingNeuron[] ArrFromLayerData(LayerData layerData)
        {
            List<ITrainingNeuron> neurons = new();
            foreach (var neuronData in layerData.Neurons)
            {
                var neuron = new TrainingNeuron1_58();
                neuron.Bias = neuronData.Bias;
                 JsonElement element = (JsonElement)neuronData.Weights;
                double[] arr = element
                .EnumerateArray()
                .Select(x => x.GetDouble())
                .ToArray();
                neuron.SetWeightsRaw(arr);
                neurons.Add(neuron);
            }
            return neurons.ToArray();
        }

        public ITrainingNeuron NewNeuron(int inputSize, Random random)
        {
            return new TrainingNeuron1_58(inputSize, random);
        }
    }
}