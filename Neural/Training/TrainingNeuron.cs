
using System.Text.Json;
using Microsoft.VisualBasic;

namespace MlNetworkTraining
{
    public class TrainingNeuron: ITrainingNeuron<double[]>
    {
        public double[] Weights { get; set; }
        public double[] Values;
        public double Bias { get; set; }
        public double Z;
        public double Output;
        public double Delta;

        public TrainingNeuron()
        {
            
        }
        public TrainingNeuron(int size, Random rand)
        {
            Weights = new double[size];
            for (int i = 0; i < size; i++)
            {
                // Random double between -0.5 and 0.5
                Weights[i] = rand.NextDouble() - 0.5;
            }
            Bias = rand.NextDouble() - 0.5;
        }
        public double Calc(double[] values, IActivationFunction activationFunction)
        {
            Values = values;
            double value = 0;
            for (int i = 0; i < values.Length; i++)
            {
                value += values[i] * Weights[i];
            }
            value += Bias;
            Z = value;
            Output = activationFunction.Apply(value);
            return Output;
        }
        public double MeanSquareError(double finalOutput, double targetOutput)
        {
            return 0.5 * Math.Pow(finalOutput - targetOutput, 2);
        }

        public double CalcErrorAtOutput(double finalOutput, double targetOutput)
        {
            return finalOutput - targetOutput;
        }
        public double CalcError(double nextWeight, double nextDelta)
        {
            // next weights corresponds to the weight of the output on the next neuron
            return nextWeight * nextDelta;
        }
        public double CalcGradient(double z, double inputValue, double error, IActivationFunction activationFunction)
        {
            // finalOutput is the neuron output after activation function
            // target output is the final target output at the last neuron
            // z is the preactivation scalar
            var derivative = activationFunction.Derivative(z);
            var s = error * derivative;
            Delta = s;
            return s * inputValue;
        }
        public double CalcNewWeight(double currentWeight, double learningRate, double gradient)
        {
            return currentWeight - learningRate * gradient;
        }

        public double CalcNewBias(double gradient, double learningRate)
        {
            return Bias - learningRate * gradient;
        }
        public void SetWeights(double[] weights)
        {
            Weights = weights;
        }
        public void SetBias(double bias)
        {
            Bias = bias;
        }

        public void RecalcWeights(double error, double learningRate, IActivationFunction activationFunction)
        {
            // Compute delta
            Delta = error * activationFunction.Derivative(Z);

            // Update weights
            for (int i = 0; i < Values.Length; i++)
            {
                Weights[i] -= learningRate * Delta * Values[i];
            }

            // Update bias
            Bias -= learningRate * Delta;
        }

        public void Print()
        {
            Console.WriteLine("Weights:" + string.Join(" ", Weights));
            Console.WriteLine("Bias:" + Bias);
        }

        public double GetOutput()
        {
            return this.Output;
        }

        public double GetWeight(int id)
        {
            return this.Weights[id];
        }

        public double GetDelta()
        {
            return this.Delta;
        }

        public double[] GetWeights()
        {
            return Weights;
        }

        public double GetBias()
        {
            return Bias;
        }

        public object GetWeightsRaw()
        {
            return Weights;
        }

        public void SetWeightsRaw(object data)
        {
            this.Weights = (double[])data;
        }
    }
}