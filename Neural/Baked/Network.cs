using System.Text.Json;

public class Network: INetwork
{
    public ILayer[] NeuralNetwork { get; set; }
    public int Depth => NeuralNetwork.Length;
    public IActivationFunction ActivationFunction {get;set;}

    public Network()
    {
        ActivationFunction = new SigmoidActivation();
    }

    public double[] ForwardPass(double[] values)
    {
        double[] last = values;
        foreach (var layer in NeuralNetwork)
        {
            last = layer.ForwardPass(last, ActivationFunction);
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

    public ILayer[] GetLayers()
    {
        return NeuralNetwork;
    }
}