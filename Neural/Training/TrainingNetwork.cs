using System.Text.Json;

namespace MlNetworkTraining
{
    public class TrainingNetwork : ITrainingNetwork
    {
        public ITrainingLayer[] NeuralNetwork { get; set; }
        public int Depth => NeuralNetwork.Length;
        public IActivationFunction ActivationFunction;

        public TrainingNetwork()
        {
            ActivationFunction = new SigmoidActivation();
        }

        public TrainingNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize, ITrainingLayerFactory layerFactory, IActivationFunction activationFunction)
        {
            ActivationFunction = activationFunction;
            NeuralNetwork = new ITrainingLayer[hiddenLayerSizes.Length + 1];
            Random random = new();

            // Input -> first hidden
            int previousSize = inputSize;

            int i = 0;
            foreach (var layerSize in hiddenLayerSizes)
            {
                //new TrainingLayer(layerSize, random, previousSize)
                NeuralNetwork[i] = layerFactory.NewLayer(layerSize, random, previousSize);
                previousSize = layerSize;
                i++;
            }

            // Output layer
            NeuralNetwork[i] = layerFactory.NewLayer(outputSize, random, previousSize);
        }
        public double[] ForwardPass(double[] values)
        {
            if (values.Length != NeuralNetwork[0].GetLength())
            {
                throw new WrongInputSizeException($"Forward pass input must have lenght {NeuralNetwork[0].GetLength()} for this network!");
            }
            double[] last = values;
            foreach (var layer in NeuralNetwork)
            {
                last = layer.ForwardPass(last, ActivationFunction);
            }
            return last;
        }

        public void BackPropagation(double[] expected, double learningRate)
        {
            if (expected.Length != NeuralNetwork[0].GetLength())
            {
                throw new WrongInputSizeException($"Back propagation input must have lenght {NeuralNetwork[0].GetLength()} for this network!");
            }
            for (int i = Depth - 1; i >= 0; i--) // start from output layer
            {
                ITrainingLayer layer = NeuralNetwork[i];
                ITrainingLayer? nextLayer = (i < Depth - 1) ? NeuralNetwork[i + 1] : null;

                // Pass expected only to the output layer
                double[]? targets = (i == Depth - 1) ? expected : null;

                layer.BackPropagation(nextLayer, targets, learningRate, ActivationFunction);
            }
        }

        public void Print()
        {
            foreach (var layer in NeuralNetwork)
            {
                layer.Print();
            }
        }

        public ILayer[] GetLayers()
        {
            return NeuralNetwork;
        }
    }
}