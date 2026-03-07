public interface ITrainingNeuron: INeuron
{
    public double MeanSquareError(double finalOutput, double targetOutput);
    public double CalcErrorAtOutput(double finalOutput, double targetOutput);
    public double CalcError(double nextWeight, double nextDelta);
    public double CalcGradient(double z, double inputValue, double error);
    public double CalcNewWeight(double currentWeight, double learningRate, double gradient);
    public double CalcNewBias(double gradient, double learningRate);
    public void SetWeights(double[] weights);
    public void SetBias(double bias);
    public double SigmoidDerivative(double value);
    public void RecalcWeights(double error, double learningRate);
    public double GetOutput();
    public double GetWeight(int id);
    public double[] GetWeights();
    public double GetDelta();
    public double GetBias();
}

public interface ITrainingNeuron<W>: ITrainingNeuron
{
   
}