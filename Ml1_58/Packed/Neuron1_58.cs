

using Microsoft.VisualBasic;

namespace TrainingMl1_58
{
    public class Neuron1_58: INeuron
    {
        public byte[] Weights { get; set; }
        public double[] Values;
        public double Bias { get; set; }
        public double Output;
        public double Delta;

        public Neuron1_58()
        {
            
        }
        public Neuron1_58(TrainingNeuron1_58 neuron)
        {
            List<int> bitmap = new();
            foreach (var weight in neuron.Weights)
            {
                bitmap.Add(Math.Sign(weight));
            }
            Weights = BitUtils.PackArray(bitmap.ToArray());
            Bias = neuron.Bias;
        }
        public double Calc(double[] values)
        {
            Values = values;
            var value = BitUtils.PackedDotProduct(Weights, values, values.Length);
            value += Bias;
            Output = SigmoidActivation(value);
            return Output;
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
    }
}