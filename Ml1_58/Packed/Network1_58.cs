
using System.Text.Json;

namespace TrainingMl1_58
{
    public class Network1_58
    {
        public List<Layer1_58> NeuralNetwork { get; set; }
        public int Depth { get; set; }

        public Network1_58()
        {
            
        }
        public Network1_58(TrainingNetwork1_58 network)
        {   
            List<Layer1_58> packed1Layers = new();
            foreach (var layer in network.NeuralNetwork)
            {
                packed1Layers.Add(new Layer1_58(layer));
            }
            NeuralNetwork = packed1Layers;
            Depth = network.Depth;
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

        public void Save(string filePath)
        {
            var opts = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            var json = JsonSerializer.Serialize(this, opts);
            File.WriteAllText(filePath, json);
        }
        public static Network1_58 NewFromJson(string filePath)
        {
            var jsonText = File.ReadAllText(filePath);
            var network = JsonSerializer.Deserialize<Network1_58>(jsonText);
            return network!;
        }
        public static Network1_58 NewFromJsonOrDefault(string filePath, Network1_58 network)
        {
            if (File.Exists(filePath))
            {
                var jsonText = File.ReadAllText(filePath);
                return JsonSerializer.Deserialize<Network1_58>(jsonText)!;
            }
            return network;
        }
    }
}