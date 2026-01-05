var Network = new Network(5, 3);
bool running = true;
double[] expected = [0.5,0.5,0.5];
while (running)
{
    var result = Network.ForwardPass([1,0.5,0.75]);
    Network.BackPropagation(expected,0.1);
    running = !AreAlmostEqual(result, expected);
}

Network.Print();

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

//