using System.Text.Json;

    public class Network
    {
        public List<ITrainingLayer> NeuralNetwork { get; set; }
        public int Depth { get; set; }

        public Network()
        {
            
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

        public void Print()
        {
            foreach (var layer in NeuralNetwork)
            {
                layer.Print();
            }
        }
        public static Network NewFromJson(string filePath,Type layerType,Type neuronType)
        {
            var jsonText = File.ReadAllText(filePath);
            var network = JsonSerializer.Deserialize<Network>(jsonText);
            return network!;
        }
    }