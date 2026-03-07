

using Microsoft.VisualBasic;

namespace TrainingMl1_58
{
    public class Neuron1_58: INeuron<byte[]>
    {
        public byte[] Weights { get; set; }
        public double Bias { get; set; }

        public Neuron1_58()
        {
            
        }
        public double Calc(double[] values)
        {
            var value = BitUtils.PackedDotProduct(Weights, values, values.Length);
            value += Bias;
            var output = SigmoidActivation(value);
            return output;
        }
      
        public double SigmoidActivation(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public void Print()
        {
            Console.WriteLine("Weights:" + string.Join(" ", Weights));
            Console.WriteLine("Bias:" + Bias);
        }

        public object GetWeightsRaw()
        {
            return Weights;
        }

        public void SetWeightsRaw(object data)
        {
            this.Weights = (byte[])data;
        }

        public double GetBias()
        {
            return this.Bias;
        }

        public void SetBias(double bias)
        {
            this.Bias = bias;
        }
    }
}