using Microsoft.VisualBasic;

namespace MlNetworkTraining
{
    public class TrainingLayer : ITrainingLayer
    {
        public ITrainingNeuron[] NeuronLayer { get; set; }
        public int Length => NeuronLayer.Length;
        public TrainingLayer()
        {

        }
        public TrainingLayer(ITrainingNeuron[] neurons)
        {
            NeuronLayer = neurons;
        }
        public TrainingLayer(int numNeurons, Random random, int inputSize, ITrainingNeuronFactory neuronFactory)
        {
            NeuronLayer = new ITrainingNeuron[numNeurons];
            for (int i = 0; i < numNeurons; i++)
            {
                NeuronLayer[i] = neuronFactory.NewNeuron(inputSize, random);
            }
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

        public void BackPropagation(ITrainingLayer? nextLayer, double[]? targetOutputs, double learningRate)
        {
            if (targetOutputs != null && nextLayer == null) // Output layer
            {
                for (int i = 0; i < Length; i++)
                {
                    var neuron = NeuronLayer[i];
                    double error = neuron.CalcErrorAtOutput(neuron.GetOutput(), targetOutputs[i]);
                    neuron.RecalcWeights(error, learningRate);
                }
            }
            else // Hidden layer
            {
                for (int i = 0; i < Length; i++)
                {
                    var neuron = NeuronLayer[i];

                    // Compute sum of weighted deltas from next layer
                    // switch to neuron instead
                    double error = 0;
                    for (int k = 0; k < nextLayer!.GetLength(); k++)
                    {
                        error += nextLayer.GetNeuron(k).GetWeight(i) * nextLayer.GetNeuron(k).GetDelta();
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
                neuron.Print();
            }
        }

        public int GetLength()
        {
            return this.Length;
        }

        public ITrainingNeuron GetNeuron(int id)
        {
            return this.NeuronLayer[id];
        }

        public ITrainingNeuron[] GetNeurons()
        {
            return NeuronLayer.ToArray();
        }

        INeuron[] ILayer.GetNeurons()
        {
            return GetNeurons();
        }
    }
}