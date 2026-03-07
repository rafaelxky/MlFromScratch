
using Ml1_58;
using MlNetwork;

Console.WriteLine("Started:");
//var network = new Network(3,[20],3);
//var network = Network.NewFromJson(Path.Join("NetworkSave.json"));
var network = new Network1_58(3,[20],3);
// 1 is more positive the closer to 1 it is
// 2 is more positive the closer to 0.5 it is
// 3 is more positive the closer to 0.1 it is

// positive example 
// negative example
// evaluation
// here 1 should be ~= 0.75, 2 should be 1-0.5, 3 should be ~= 0.25
//double[] values3 = [0.65, 0.65, 0.65];

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
    [0.9, 0.2, 0.1],
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.3, 0.2, 0.1],
    [0.7, 0.8, 0.9],
    [0.87, 0.65, 0.20],
    [0.3892183, 0.9123901, 0.019139818457],
    [0.617312634, 0.45496840, 0.9091745859],
];

double[][] targets2 =
[
    [0.9,0.9,0.9],
    [0.01,0.01,0.01],
    [0.5,0.5,0.5],
    [0.75,0.75,0.75],
    [0.25,0.25,0.25],
    [0.1, 0.5, 0.9],
    [0.9, 0.2, 0.1],
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.3, 0.2, 0.1],
    [0.7, 0.8, 0.9],
    [0.87, 0.65, 0.20],
    [0.3892183, 0.9123901, 0.019139818457],
    [0.617312634, 0.45496840, 0.9091745859],
];

double[] values1 = [0.87, 0.65, 0.20];
double[] values2 = [0.12,0.12,0.12];
double[] values3 = [0.97,0.85,0.32];

var learningRate = 0.01;
var lenght = inputs2.Length;
for (int epoch = 0; epoch < 30000; epoch++)
{
    for (int i = 0; i < inputs2.Length; i++)
    {
        var result = network.ForwardPass(inputs2[i]);
        network.BackPropagation(targets2[i], learningRate);
    }
}


var result1 = network.ForwardPass(values1);
var result2 = network.ForwardPass(values2);
var result3 = network.ForwardPass(values3);

network.Save(Path.Join("NetworkSave.json"));

network.Print();
Console.WriteLine("Final:" + string.Join(" ", result1));
Console.WriteLine("Final:" + string.Join(" ", result2));
Console.WriteLine("Final:" + string.Join(" ", result3));
Console.WriteLine("For: depth - " + network.Depth + " - learningRate - " + learningRate);

bool AreAlmostEqual(double[] a, double[] b, double tolerance = 1e-6)
{
    if (a.Length != b.Length) return false;

    for (int i = 0; i < a.Length; i++)
    {
        if (Math.Abs(a[i] - b[i]) > tolerance)
            return false;
    }

    return true;
}

void Train(double[] values, double[] expected, double learningRate,double round)
{
    bool running = true;
    var i = 0;
    while (running)
    {
        var result1 = network.ForwardPass(values);
        network.BackPropagation(expected, 0.01);
        running = !AreAlmostEqual(result1, expected, 0.01);
        i++;
        Console.WriteLine("i:" + i);
    }
}