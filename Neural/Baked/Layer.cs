public class Layer : ILayer
{
    public INeuron[] NeuronLayer { get; set; }
    public int Length => NeuronLayer.Length;
    public Layer()
    {

    }
    public Layer(INeuron[] neurons)
    {
        this.NeuronLayer = neurons;
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

    public INeuron[] GetNeurons()
    {
        return NeuronLayer;
    }
}