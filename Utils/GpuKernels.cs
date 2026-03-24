using ILGPU;
using ILGPU.Runtime;

public static class GpuKernels
{
    public static void VecMultKernel(
          Index1D i,
          ArrayView<double> a,
          ArrayView<double> b,
          ArrayView<double> result)
    {
        result[i] = a[i] * b[i];
    }
    public static void VecSumKernel(
             Index1D i,
             ArrayView<double> a,
             ArrayView<double> b,
             ArrayView<double> result)
    {
        result[i] = a[i] + b[i];
    }

    public static void MatMulKernel(
           Index2D idx,
           ArrayView2D<double, Stride2D.DenseX> matA,
           ArrayView2D<double, Stride2D.DenseX> matB,
           ArrayView2D<double, Stride2D.DenseX> result,
           int M, int N, int K)
    {
        int row = idx.X;
        int col = idx.Y;
        if (row >= M || col >= N) return;

        double sum = 0.0;
        for (int k = 0; k < K; k++)
            sum += matA[row, k] * matB[k, col];

        result[row, col] = sum;
    }

    public static void ForwardPassKernel<TActivation>(
           Index1D neuronIdx,
           ArrayView2D<double, Stride2D.DenseX> neuronMatrix,
           ArrayView1D<double, Stride1D.Dense> input,
           ArrayView1D<double, Stride1D.Dense> bias,
           ArrayView1D<double, Stride1D.Dense> output,
           ArrayView1D<double, Stride1D.Dense> preActivation
           )
           where TActivation : struct, IActivationFunction
    {
        int i = neuronIdx;
        //if (i >= neuronMatrix.Extent.X) return;

        double sum = 0.0;
        for (int j = 0; j < input.Length; j++)
            sum += input[j] * neuronMatrix[i, j];

        sum += bias[i];
        if (preActivation.Length > 0)
            preActivation[i] = sum;

        TActivation activation = default;
        output[i] = activation.Apply(sum);
    }
}