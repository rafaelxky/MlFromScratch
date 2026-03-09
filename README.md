# What is this?
This project is a machine learning algoritm using only standard c# libraries.

# How to use
- Create a new Network
To create a new network you can use the NetworkBuilder class
It provides defaults or fine tuned instantiation
ex:
NetworkBuilder.DefaultTrainingNetwork(3, [5], 3, new ReLUActivation());

The entry and exit layers must correspond with what you put in and expect out

You can call double[] Network.ForwardPass(double[] values)
to feed the algorithm values and it returns an output.
This will internally store the results for each involved neuron.

You can then call Network.BackPropagation(double[] expected, double learningRate) to update the weights acording to the output and expected (learning phase)

You can call the Network.Print() wich will print each neurons weights and biases

You can save the weights to json using NetworkSerializer.Save(string path)

You can load a model like so
NetworkSerializer.LoadTrainingNetworkDefault("newSerTest.json", new ReLUActivation());
Or you can provide your factories for deserialization

# Specifications
The weights are initiated with random values
It uses gradient descent

# Activation functions
- Linear
- ReLU (prefered)
- Sigmoid
- SoftPlus
- Tanh 

# Neuron Types
- Neuron (default)
- 1_58 (uses 2 bits per weight)

# 1bit llm
This project includes a 1 bit version.
This means that each weight is 2 bits
This presents a loss in precision but greatly reduces save space and ram 
Create a TrainingNetwork1_58 and train the same as the full double version
This will train the ml similarly to the base version
Its saves will be full double weights
This can then be saved and loaded with 1 bit weights

Notes: 
The weights can assume the values {-1, 0, 1}
So it would need 1.58 bits per weight
log₂(3) ≈ 1.58
Hence the name so technically not a 1 bit llm
The 1 bit version is much smaller but it has much less precision
The training speed is unchanged from the base double model
It knows ratios but cannot properly predict values 