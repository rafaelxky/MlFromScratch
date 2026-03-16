using System.Diagnostics;

public class DoubleMlUnitTest
{
    ITrainingNetwork trainingNetwork = NetworkBuilder.DefaultTrainingNetwork(3, [16, 16], 3, new ReLUActivation());

    public void TestAll()
    {
        
    }

    public void SuccessForwardPassTest()
    {
        var result = trainingNetwork.ForwardPass([1, 2, 3]);
        if (result.Length != 3)
        {
            Console.WriteLine("ERROR: Forward pass returned wrong size!");
            return;
        }
        if (result == null)
        {
            Console.WriteLine("ERROR: null output!");
            return;
        }
        Console.WriteLine("Forward pass test success.");
    }

    public void WrongSizeForwardPassTest()
    {
        try
        {
            var result = trainingNetwork.ForwardPass([1, 2, 3, 4]);
            Console.WriteLine("ERROR: wrong size forward pass didnt trow!");
            return;
        }
        catch (WrongInputSizeException)
        {
            Console.WriteLine("Forward pass wrong size input success.");
        }
    }

    public void WrongSizeBackPropagationTest()
    {
        try
        {
            trainingNetwork.BackPropagation([1, 2, 3, 4],0.1f);
            Console.WriteLine("ERROR: wrong size back propagation didnt trow!");
            return;
        }
        catch (WrongInputSizeException)
        {
            Console.WriteLine("Back propagation wrong size input success.");
        }
    }


    public void SingleForwardPassSpeedTest()
    {
         var sw = Stopwatch.StartNew();
            trainingNetwork.ForwardPass([0,0,0]);
        sw.Stop();

        Console.WriteLine("Single forward pass speed test finished in: " + sw.Elapsed.TotalMilliseconds + "ms");
    }
    public void SingleBackPropagationSpeedTest()
    {
         var sw = Stopwatch.StartNew();
            trainingNetwork.BackPropagation([0,0,0],0.1f);
        sw.Stop();

        Console.WriteLine("Single backpropagation speed test finished in: " + sw.Elapsed.TotalMilliseconds + "ms");
    }
}