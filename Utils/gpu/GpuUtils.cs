using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

using ForwardKernelType = System.Action<
    ILGPU.Index1D,
    ILGPU.Runtime.ArrayView2D<double, ILGPU.Stride2D.DenseX>,
    ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>,
    ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>,
    ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>,
    ILGPU.Runtime.ArrayView1D<double, ILGPU.Stride1D.Dense>
>;

public class GpuUtils : IDisposable
{
    Context context;
    Accelerator accelerator;
    private Dictionary<string, ForwardKernelType>? forwardKernelMap;



    public GpuUtils()
    {
        context = Context.Create(builder => builder.EnableAlgorithms());
        accelerator = context.
            GetPreferredDevice(preferCPU: false).
            CreateAccelerator(context);
    }

    public void CompileDoubleNetworkKernels()
    {
        forwardKernelMap = new();
        RegisterActivationFunctionInGpu<HardTanhActivation>();
        RegisterActivationFunctionInGpu<LeakyReLUActivation>();
        RegisterActivationFunctionInGpu<LeakyReLUActivation>();
        RegisterActivationFunctionInGpu<LinearActivation>();
        RegisterActivationFunctionInGpu<ReLUActivation>();
        RegisterActivationFunctionInGpu<SigmoidActivation>();
        RegisterActivationFunctionInGpu<SoftplusActivation>();
        RegisterActivationFunctionInGpu<SwishActivation>();
        RegisterActivationFunctionInGpu<TanhActivation>();
    }

    public void RegisterActivationFunctionInGpu<TActivation>()
    where TActivation : struct, IActivationFunction
    {
        TActivation activation = default;
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView2D<double, Stride2D.DenseX>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>,
            ArrayView1D<double, Stride1D.Dense>
        >(GpuKernels.ForwardPassKernel<TActivation>);

        forwardKernelMap![activation.Name] = kernel;
    }
    public double[] VecMult(double[] a, double[] b)
    {
        var adata = accelerator.Allocate1D(a);
        var bdata = accelerator.Allocate1D(b);
        var output = accelerator.Allocate1D<double>(a.Length);
        var loadedKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(GpuKernels.VecMultKernel);
        loadedKernel(a.Length, adata.View, bdata.View, output.View);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }
    public double[] VecSum(double[] a, double[] b)
    {
        var adata = accelerator.Allocate1D(a);
        var bdata = accelerator.Allocate1D(b);
        var output = accelerator.Allocate1D<double>(a.Length);
        var loadedKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(GpuKernels.VecSumKernel);
        loadedKernel(a.Length, adata.View, bdata.View, output.View);
        accelerator.Synchronize();
        return output.GetAsArray1D();
    }
    public MemoryBuffer1D<double, Stride1D.Dense> ExecuteForwardPassKernel(
        double[,] neuronMatrix,
        int neuronCount,
        int weightCount,
        MemoryBuffer1D<double, Stride1D.Dense> inputs,
        double[] bias,
        string activationFunction,
        MemoryBuffer1D<double, Stride1D.Dense> preactivationBuffer,
        out MemoryBuffer1D<double, Stride1D.Dense> preActivationValues
    )
    {
        var extent = new Index2D(neuronCount, weightCount);
        using var neuronsMatrixBuffer = accelerator.Allocate2DDenseX<double>(extent);
        neuronsMatrixBuffer.CopyFromCPU(neuronMatrix);
        var inputsBuffer = inputs;
        var biasBuffer = accelerator.Allocate1D(bias);
        var outputBuffer = accelerator.Allocate1D<double>(neuronMatrix.GetLength(0));
        var loadedKernel = forwardKernelMap![activationFunction];
        loadedKernel(neuronCount, neuronsMatrixBuffer.View, inputsBuffer.View, biasBuffer.View, outputBuffer.View, preactivationBuffer);

        accelerator.Synchronize();
        preActivationValues = preactivationBuffer;

        return outputBuffer;
    }
    public double[] NetworkForwardPass(double[] input, List<Layer> layers)
    {
        var output = accelerator.Allocate1D(input);
        foreach (var layer in layers)
        {
            output = ForwardPass(layer, output);
        }
        return output.GetAsArray1D();
    }

    public double[] NetworkForwardTrain(double[] input, List<Layer> layers, out List<LayerCache> preActivationValues)
    {
        var output = accelerator.Allocate1D(input);
        preActivationValues = new();
        foreach (var layer in layers)
        {
            var cache = new LayerCache
            {
                Inputs = output.GetAsArray1D()
            };
            output = ForwardTrain(layer, output, out var preAct);
            cache.PreActivationValues = preAct.GetAsArray1D();
            cache.Outputs = output.GetAsArray1D();
            preActivationValues.Add(cache);
        }
        return output.GetAsArray1D();
    }

    public MemoryBuffer1D<double, Stride1D.Dense> ForwardTrain
    (
        Layer layer,
        MemoryBuffer1D<double, Stride1D.Dense> inputs,
        out MemoryBuffer1D<double, Stride1D.Dense> preActivationValues
    )
    {
        var output = ExecuteForwardPassKernel(
            layer.Neurons,
            layer.NeuronCount,
            layer.WeightCount,
            inputs,
            layer.Bias,
            layer.ActivationFunction,
            accelerator.Allocate1D<double>(layer.Neurons.GetLength(0)),
            out var preActivationValuesInner
        );
        preActivationValues = preActivationValuesInner;
        return output;
    }
    public MemoryBuffer1D<double, Stride1D.Dense> ForwardPass
           (
               Layer layer,
               MemoryBuffer1D<double, Stride1D.Dense> inputs
           )
    {
        var dummy = accelerator.Allocate1D<double>(0);
        var output = ExecuteForwardPassKernel(
            layer.Neurons,
            layer.NeuronCount,
            layer.WeightCount,
            inputs,
            layer.Bias,
            layer.ActivationFunction,
            dummy,
            out var _
        );
        return output;
    }

    public void Dispose()
    {
        accelerator.Dispose();
        context.Dispose();
    }
}
