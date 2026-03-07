public record NeuronData(object Weights, double Bias);
public record LayerData(NeuronData[] Neurons);
public record NetworkData(LayerData[] Layers, string NeuronType, string LayerType);