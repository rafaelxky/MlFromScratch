public class Layer : ILayer
{
    public List<INeuron> NeuronLayer { get; set; }
    public int Length { get; set; }
    public Layer()
    {

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

    public void Print()
    {
        foreach (var neuron in NeuronLayer)
        {
            neuron.Print();
        }
    }
}