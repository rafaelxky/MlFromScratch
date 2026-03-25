using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class SimdUtils
{
    public static double[] ForwardPass(Layer layer, double[] input)
    {
        return ForwardPass(layer.Neurons, input, layer.Bias, layer._activationFunction);
    }
    public static double[] ForwardPass(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        // foreach neuron in layer, calc output and build vector
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            output[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out var _);
        }

        return output;
    }

    public static double[] ForwardTrain(Layer layer, double[] input, out double[] preActivationValues)
    {
        var result = ForwardTrain(layer.Neurons, input, layer.Bias, layer._activationFunction, out var preActivationValuesInner);
        preActivationValues = preActivationValuesInner;
        return result;
    }

    public static double[] ForwardTrain(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        double[] output = new double[neuronMatrix.GetLength(0)];
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            output[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out preActivationValues[i]);
        }
        return output;
    }

    public static double CalcNeuronOutput(double[,] neurons,int neuronId, double[] input, double bias, IActivationFunction activationFunction, out double preActivation)
    {
        //double output = TensorUtils.WeightInputDotProd(neurons, neuronId, input);
        double output = WeightInputDotProd(neurons,neuronId, input);
        output += bias;
        preActivation = output;
        return activationFunction.Apply(output);
    }

    public static unsafe double WeightInputDotProd(double[,] neurons, int neuronId, double[] input)
    {
        int len = neurons.GetLength(1);
        double result = 0;

        fixed (double* pN = neurons, pI = input)
        {
            double* row = pN + neuronId * len; 

            if (Avx.IsSupported && Fma.IsSupported)
            {
                int i = 0;
                var acc = Vector256<double>.Zero; 

                for (; i <= len - 4; i += 4)
                {
                    var n = Avx.LoadVector256(row + i);
                    var inp = Avx.LoadVector256(pI + i);
                    acc = Fma.MultiplyAdd(n, inp, acc);
                }

                var low = acc.GetLower();
                var high = acc.GetUpper();                     
                var sum2 = Sse2.Add(low, high);                
                var shuf = Sse2.UnpackHigh(sum2, sum2);         
                result = Sse2.AddScalar(sum2, shuf).ToScalar(); 

                for (; i < len; i++)
                    result += row[i] * pI[i];
            }
            else
            {
                for (int i = 0; i < len; i++)
                    result += row[i] * pI[i];
            }
        }

        return result;
    }
}