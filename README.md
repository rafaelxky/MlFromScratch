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

# 1bit llm
This project includes a 1 bit version.
This means that each weight is 2 bits
This presents a loss in precision but greatly reduces save space and ram 
Create a TrainingNetwork1_58 and train the same as the full double version
This will train the ml similarly to the base version
Its saves will be full double weights
After you are done with training call .Bake(), this will return a Network1_58
This can then be saved and loaded with 1 bit weights

Note: 
The weights can assume the values {-1, 0, 1}
So it would need 1.58 bits per weight
log₂(3) ≈ 1.58
Hence the name so technically not a 1 bit llm

# Cases
Expected final outputs
[0.87, 0.65, 0.20];
[0.12,0.12,0.12];
[0.97,0.85,0.32];

A:
- full float llm
instantiated as...
var network = new Network(3, [250, 250, 250, 250, 250, 250, 250, 250], 3);
- 3.5 mb, 439,000 parameters (calculated)
- epochs = 1000
- learningRate = 0.01
- roughly 10 or 12 cycles per second
- saved file: 15mb
Values:
Final:0.8468605768490737 0.6418190351522041 0.21952344588236417
Final:0.1283683092636886 0.17754965572828021 0.1206645248286608
Final:0.9011495357964956 0.7496502615194393 0.3766156585357967



B:
- 1bit llm
instantiated as...
var network1_58 = new TrainingNetwork1_58(3, [250, 250, 250, 250, 250, 250, 250, 250], 3);
var network = network1_58.Bake();
from example A:
- 107KB, 439,000 parameters (calculated)
- epochs = 1000
- learningRate = 0.01
- roughly 10 or 12 cycles per second
- saved file: 310kb
Final:0.9552200664120242 0.9080152826557932 0.04460591208629829
Final:0.01269247112073543 0.005522931697733541 0.11479782858930695
Final:0.927270142872066 0.9268806751499197 0.2844031747723978

Notes: 
- The 1 bit version is much smaller but it has much less precision
- The training speed is unchanged from the base double model
- This may be diferent with other layer sizes and depths