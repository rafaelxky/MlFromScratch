
using Microsoft.VisualBasic;
using MlNetwork;

namespace TrainingMl1_58
{
    public class Layer1_58
    {
        public List<Neuron1_58> NeuronLayer { get; set; }
        public int Length { get; set; }
        public Layer1_58()
        {
            
        }
        public Layer1_58(TrainingLayer1_58 layer)
        {
            List<Neuron1_58> packed1Neurons = new();
            foreach (var neuron in layer.NeuronLayer)
            {
                packed1Neurons.Add(new Neuron1_58(neuron));
            }   
            NeuronLayer = packed1Neurons;
            Length = layer.Length;
        }
        public double[] ForwardPass(double[] inputs)
        {
            double[] outputs = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                outputs[i] = NeuronLayer[i].Calc(inputs);
            }
            return outputs;
        }

        public void Print()
        {
            foreach (var neuron in NeuronLayer)
            {
                Console.WriteLine("Weights:" + string.Join(" ", neuron.Weights));
                Console.WriteLine("Bias:" + neuron.Bias);
            }
        }
    }
}