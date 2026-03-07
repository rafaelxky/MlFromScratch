# What is this?
This project is a machine learning algoritm using only standard c# libraries.

# How to use
Create a new Network
You can define the size of the entry layer, the size of each hidden layer and the size of the output layer.
ex: 
var network = new Network(3,[20],[20],3);
// entry layer size 3, 2 hidden layers of size 20, exit layer of size 3

The entry and exit layers must correspond with what you put in and expect out

You can call double[] Network.ForwardPass(double[] values)
to feed the algorithm values and it returns an output.
This will internally store the results for each involved neuron.

You can then call Network.BackPropagation(double[] expected, double learningRate) to update the weights acording to the output and expected (learning phase)

You can call the Network.Print() wich will print each neurons weights and biases

You can save the weights to json using Network.Save(string path)
You can load them with Network.NewFromJson(string path) wich will return a new Network object

# Specifications
Currently the Ml algorithm uses sigmoid activation function so the outputs will be between 1 and 0
The weights are initiated with random values

# 




