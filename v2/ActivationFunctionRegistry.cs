public class ActivationFunctionRegistry
{
    public static Dictionary<string, IActivationFunction> Functions = new();
    static ActivationFunctionRegistry()
    {
        RegisterFunction(new HardTanhActivation());
        RegisterFunction(new LeakyReLUActivation());
        RegisterFunction(new LinearActivation());
        RegisterFunction(new ReLUActivation());
        RegisterFunction(new SigmoidActivation());
        RegisterFunction(new SoftplusActivation());
        RegisterFunction(new SwishActivation());
        RegisterFunction(new TanhActivation());
    }
    public static void RegisterFunction(IActivationFunction activationFunction)
    {
        Functions.Add(activationFunction.Name, activationFunction);
    }
    public static IActivationFunction GetFunction(string name)
    {
        return Functions[name];
    }
}