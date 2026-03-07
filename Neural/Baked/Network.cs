using System.Text.Json;

public class Network
{
    public ILayer[] NeuralNetwork { get; set; }
    public int Depth => NeuralNetwork.Length;

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
}