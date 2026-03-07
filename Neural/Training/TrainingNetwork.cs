using System.Text.Json;

namespace MlNetworkTraining
{
    public class TrainingNetwork
    {
        public ITrainingLayer[] NeuralNetwork { get; set; }
        public int Depth;

        public TrainingNetwork()
        {

        }

        public TrainingNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize, ITrainingLayerFactory layerFactory)
        {
            NeuralNetwork = new ITrainingLayer[hiddenLayerSizes.Length + 1];
            Random random = new();

            // Input -> first hidden
            int previousSize = inputSize;

            int i = 0;
            foreach (var layerSize in hiddenLayerSizes)
            {
                //new TrainingLayer(layerSize, random, previousSize)
                NeuralNetwork[i] = layerFactory.NewLayer(layerSize,random,previousSize);
                previousSize = layerSize;
                i++;
            }

            // Output layer
            NeuralNetwork[i] = layerFactory.NewLayer(outputSize,random,previousSize);
        }
        public double[] ForwardPass(double[] values)
        {
            double[] last = values;
            foreach (var layer in NeuralNetwork)
            {
                last = layer.ForwardPass(last);
            }
            return last;
        }

        public void BackPropagation(double[] expected, double learningRate)
        {
            for (int i = Depth - 1; i >= 0; i--) // start from output layer
            {
                ITrainingLayer layer = NeuralNetwork[i];
                ITrainingLayer? nextLayer = (i < Depth - 1) ? NeuralNetwork[i + 1] : null;

                // Pass expected only to the output layer
                double[]? targets = (i == Depth - 1) ? expected : null;

                layer.BackPropagation(nextLayer, targets, learningRate);
            }
        }

        public void Print()
        {
            foreach (var layer in NeuralNetwork)
            {
                layer.Print();
            }
        }

        public void Save(string filePath)
        {
            var opts = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            var json = JsonSerializer.Serialize(this, opts);
            File.WriteAllText(filePath, json);
        }
        public static TrainingNetwork NewFromJson(string filePath)
        {
            var jsonText = File.ReadAllText(filePath);
            var network = JsonSerializer.Deserialize<TrainingNetwork>(jsonText);
            return network!;
        }
        public static TrainingNetwork NewFromJsonOrDefault(string filePath, TrainingNetwork network)
        {
            if (File.Exists(filePath))
            {
                var jsonText = File.ReadAllText(filePath);
                return JsonSerializer.Deserialize<TrainingNetwork>(jsonText)!;
            }
            return network;
        }
    }
}