
using System.Text.Json;
using MlNetworkTraining;
using TrainingMl1_58;

var savePath = Path.Join("1bSave.json");

Console.WriteLine("Started:");
//var network = new Network(3,[20],3);
//var network = new Network(3, [250, 250, 250, 250, 250, 250, 250, 250], 3);
//var network = Network.NewFromJson(Path.Join("DoubleNetworkSave.json"));

//var network = new TrainingNetwork1_58(3,[20],3);
//var network = new TrainingNetwork1_58(3, [250, 250, 250, 250, 250, 250, 250, 250], 3);
//var network = TrainingNetwork1_58.NewFromJson(Path.Join("NetworkSave.json"));

//var network = network1_58.Bake();
//var network = Network1_58.NewFromJson(Path.Join("1bSave.json"));

ITrainingNetwork trainingNetwork =
NetworkBuilder.TrainingNetwork1_58Default(3, [100,100,100,100,100,100,100,100,100,100], 3);
var network = trainingNetwork;

//var network = NetworkSerializer.Load1_58TrainingDefault("newSerTest.json");

double[][] inputs =
[
    [1.0, 0.5, 0.1],
    [0.1, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [0.9, 0.1, 0.2],
    [0.2, 0.8, 0.3],
    [0.3, 0.2, 0.9],
    [0.7, 0.6, 0.1],
    [0.4, 0.9, 0.4],
    [0.1, 0.3, 0.8],
    [0.6, 0.4, 0.6],
];
double[][] targets =
[
    [1.0, 1.0, 1.0],
    [0.1, 0.5, 0.1],
    [0.5, 0.5, 0.5],
    [0.9, 0.1, 0.2],
    [0.2, 0.8, 0.3],
    [0.3, 0.2, 0.9],
    [0.7, 0.6, 0.1],
    [0.4, 0.9, 0.4],
    [0.1, 0.3, 0.8],
    [0.6, 0.4, 0.6],
];

// linear relation
double[][] inputs2 =
[
    [0.9,0.9,0.9],
    [0.01,0.01,0.01],
    [0.5,0.5,0.5],
    [0.75,0.75,0.75],
    [0.25,0.25,0.25],
    [0.1, 0.5, 0.9],
    [0.9, 0.5, 0.1],
    [0.9, 0.2, 0.1],
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.3, 0.2, 0.1],
    [0.7, 0.8, 0.9],
    [0.87, 0.65, 0.20],
    [0.3892183, 0.9123901, 0.019139818457],
    [0.617312634, 0.45496840, 0.9091745859],
    [0.019139818457, 0.45496840, 0.617312634],
    [0.87, 0.65, 0.20],
    [0.20,0.40,0.60],
    [0.60,0.20,0.40],
    [0.40,0.60,0.20],
];

double[][] targets2 =
[
    [0.9,0.9,0.9],
    [0.01,0.01,0.01],
    [0.5,0.5,0.5],
    [0.75,0.75,0.75],
    [0.25,0.25,0.25],
    [0.1, 0.5, 0.9],
    [0.9, 0.5, 0.1],
    [0.9, 0.2, 0.1],
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.3, 0.2, 0.1],
    [0.7, 0.8, 0.9],
    [0.87, 0.65, 0.20],
    [0.3892183, 0.9123901, 0.019139818457],
    [0.617312634, 0.45496840, 0.9091745859],
    [0.019139818457, 0.45496840, 0.617312634],
    [0.87, 0.65, 0.20],
    [0.20,0.40,0.60],
    [0.60,0.20,0.40],
    [0.40,0.60,0.20],
];




// learning

var learningRate = 0.1;
var lenght = inputs2.Length;
for (int epoch = 0; epoch < 500; epoch++)
{
    for (int i = 0; i < inputs2.Length; i++)
    {
        var result = network.ForwardPass(inputs2[i]);
        network.BackPropagation(targets2[i], learningRate);
        Console.WriteLine(JsonSerializer.Serialize(result) +" - "+ JsonSerializer.Serialize(targets2[i]));
    }
}




// final expected values
double[] values1 = [0.87, 0.65, 0.20];
double[] values2 = [0.12,0.12,0.12];
double[] values3 = [0.97,0.85,0.32];

var result1 = network.ForwardPass(values1);
var result2 = network.ForwardPass(values2);
var result3 = network.ForwardPass(values3);

//network.Print();
Console.WriteLine("Final:" + string.Join(" ", result1));
Console.WriteLine("Final:" + string.Join(" ", result2));
Console.WriteLine("Final:" + string.Join(" ", result3));
//Console.WriteLine("For: depth - " + network.Depth + " - learningRate - " + learningRate);

// save
NetworkSerializer.Save(network, "newSerTest.json");
