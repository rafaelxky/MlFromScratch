using Microsoft.VisualBasic;

namespace Ml1_58
{
    public class Layer1_58
    {
        public List<Neuron1_58> NeuronLayer { get; set; }
        public int Length { get; set; }
        double[]? Inputs;
        public Layer1_58()
        {

        }
        public Layer1_58(int numNeurons, Random random, int inputSize)
        {
            NeuronLayer = new();
            for (int i = 0; i < numNeurons; i++)
            {
                NeuronLayer.Add(new Neuron1_58(inputSize, random));
            }
            Length = NeuronLayer.Count;
        }

        public double[] ForwardPass(double[] inputs)
        {
            Inputs = inputs;
            double[] outputs = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                outputs[i] = NeuronLayer[i].Calc(inputs);
            }
            return outputs;
        }

        public void BackPropagation(Layer1_58? nextLayer, double[]? targetOutputs, double learningRate)
        {
            if (targetOutputs != null && nextLayer == null) // Output layer
            {
                for (int i = 0; i < Length; i++)
                {
                    Neuron1_58 neuron = NeuronLayer[i];
                    double error = neuron.CalcErrorAtOutput(neuron.Output, targetOutputs[i]);
                    neuron.RecalcWeights(error, learningRate);
                }
            }
            else // Hidden layer
            {
                for (int i = 0; i < Length; i++)
                {
                    Neuron1_58 neuron = NeuronLayer[i];

                    // Compute sum of weighted deltas from next layer
                    double error = 0;
                    for (int k = 0; k < nextLayer.Length; k++)
                    {
                        error += nextLayer.NeuronLayer[k].Weights[i] * nextLayer.NeuronLayer[k].Delta;
                    }

                    // Update neuron weights and bias
                    neuron.RecalcWeights(error, learningRate);
                }
            }
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