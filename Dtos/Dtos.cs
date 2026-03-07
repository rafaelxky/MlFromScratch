using System.Security.Cryptography.X509Certificates;

public record NeuronData(
    double[] Weights,
    double Bias
);

public record LayerData(
    NeuronData[] Neurons
);

public record NetworkData(
    string NeuronType,
    LayerData[] Layers
);