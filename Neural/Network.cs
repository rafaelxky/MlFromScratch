public class Network
{
    public List<Layer> NeuralNetwork;
    int Depth;
    int Length;
    public Network(int depth, int lenght)
    {
        NeuralNetwork = new();
        Depth = depth;
        Length = lenght;
        Random random = new();
        for (int i = 0; i < Depth; i++)
        {
            NeuralNetwork.Add(new(Length, random));
        }
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