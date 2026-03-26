using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static class SimdUtils
{
    public static void ForwardPass(Layer layer, double[] input, double[] outputBuffer)
    {
        ForwardPass(layer.Neurons, input, layer.Bias, layer._activationFunction, outputBuffer);
    }
    public static void ForwardPass(double[,] neuronMatrix, double[] input, double[] bias, IActivationFunction activationFunction, double[] outputBuffer)
    {
        //double[] output = new double[neuronMatrix.GetLength(0)];
        // foreach neuron in layer, calc output and build vector
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            outputBuffer[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out var _);
        }
    }

    public static void ForwardTrain(Layer layer, double[] input,double[] outputBuffer ,out double[] preActivationValues)
    {
        ForwardTrain(layer.GetNeuronWeights(), input,outputBuffer,layer.Bias, layer._activationFunction, out var preActivationValuesInner);
        preActivationValues = preActivationValuesInner;
    }

    public static void ForwardTrain(double[,] neuronMatrix, double[] input, double[] outputBuffer,double[] bias, IActivationFunction activationFunction, out double[] preActivationValues)
    {
        preActivationValues = new double[neuronMatrix.GetLength(0)];
        for (int i = 0; i < neuronMatrix.GetLength(0); i++)
        {
            outputBuffer[i] = CalcNeuronOutput(neuronMatrix,i, input, bias[i], activationFunction, out preActivationValues[i]);
        }
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