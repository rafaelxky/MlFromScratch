public class Network
{
    public List<Layer> NeuralNetwork;
    public int Depth;

    public Network(int inputSize, int[] hiddenLayerSizes, int outputSize)
    {
        NeuralNetwork = new();
        Random random = new();

        // Input -> first hidden
        int previousSize = inputSize;

        foreach (var layerSize in hiddenLayerSizes)
        {
            NeuralNetwork.Add(new Layer(layerSize, random, previousSize));
            previousSize = layerSize;
        }

        // Output layer
        NeuralNetwork.Add(new Layer(outputSize, random, previousSize));
        Depth = NeuralNetwork.Count;
    }
    public double[] ForwardPass(double[] values)
    {
        double[] last = values;
        foreach (var layer in NeuralNetwork)
        {
            last = layer.ForwardPass(last);
        }
        Console.WriteLine(string.Join(", ", last));
        return last;
    }

    public void BackPropagation(double[] expected, double learningRate)
    {
        for (int i = Depth - 1; i >= 0; i--) // start from output layer
        {
            Layer layer = NeuralNetwork[i];
            Layer? nextLayer = (i < Depth - 1) ? NeuralNetwork[i + 1] : null;

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
}