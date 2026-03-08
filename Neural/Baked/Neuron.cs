
namespace MlNetworkTraining
{
    public class Neuron: INeuron<double[]>
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public Neuron()
        {
            
        }
        public double Calc(double[] values, IActivationFunction activationFunction)
        {
            double value = 0;
            for (int i = 0; i < values.Length; i++)
            {
                value += values[i] * Weights[i];
            }
            value += Bias;
            var output = activationFunction.Apply(value);
            return output;
        }        
        public void SetBias(double bias)
        {
            Bias = bias;
        }
        public void Print()
        {
            Console.WriteLine("Weights:" + string.Join(" ", Weights));
            Console.WriteLine("Bias:" + Bias);
        }

        public object GetWeightsRaw()
        {
            return this.Weights;
        }

        public void SetWeightsRaw(object data)
        {
            this.Weights = (double[])data;
        }

        public double GetBias()
        {
            return this.Bias;
        }
    }
}