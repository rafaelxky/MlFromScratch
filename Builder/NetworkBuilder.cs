using System.Dynamic;
using MlNetworkTraining;
using TrainingMl1_58;

public class NetworkBuilder
{
    private INeuronFactory? _neuronFactory;
    private ITrainingNeuronFactory? _trainingNeuronFactory;
    private ILayerFactory? _layerFactory;
    private ITrainingLayerFactory? _trainingLayerFactory;
    private int _inputSize = 3; 
    private int[] _hiddenLayerSizes = [10,10,10];
    private int _outputSize = 3;

    public TrainingNetwork BuildTrainingNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize)
    {
        _inputSize = inputSize;
        _hiddenLayerSizes = hiddenLayerSizes;
        _outputSize = outputSize;
        return BuildTrainingNetwork();
    }

    public TrainingNetwork BuildTrainingNetwork()
    {
        if (_trainingLayerFactory == null)
        {
            throw new Exception("Called BuildTrainingNetwork() but TrainingLayerFactory not set!");
        }  
        if (_trainingNeuronFactory == null)
        {
            throw new Exception("Called BuildTrainingNetwork() but TrainingNeuronFactory not set!");
        }
        return new TrainingNetwork(_inputSize, _hiddenLayerSizes,_outputSize, _trainingLayerFactory);
    }
    public static TrainingNetwork DefaultTrainingNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize)
    {
        var basicLayerFactory = new BasicTrainingLayerFactory(new BasicTrainingNeuronFactory());
        return new TrainingNetwork(inputSize, hiddenLayerSizes,outputSize, basicLayerFactory);
    }
    
    public NetworkBuilder BasicTrainingNeuron()
    {
        _trainingNeuronFactory = new BasicTrainingNeuronFactory();
        return this;
    }
    public NetworkBuilder TrainingNeuron1_58()
    {
        _trainingNeuronFactory = new TrainingNeuron1_58Factory();
        return this;
    }
    public NetworkBuilder BasicTrainingLayer()
    {
        if (_trainingNeuronFactory == null)
        {
            throw new Exception("Tried creating basic training layer factory before setting neuron factory!");
        }
        this._trainingLayerFactory = new BasicTrainingLayerFactory(_trainingNeuronFactory);
        return this;
    }
    public NetworkBuilder NeuronFactory(INeuronFactory neuronFactory)
    {
        _neuronFactory = neuronFactory;
        return this;
    }  
    public NetworkBuilder TrainingNeuronFactory(ITrainingNeuronFactory neuronFactory)
    {
        _trainingNeuronFactory = neuronFactory;
        return this;
    } public NetworkBuilder LayerFactory(ILayerFactory layerFactory)
    {
        _layerFactory = layerFactory;
        return this;
    } public NetworkBuilder NeuronFactory(ITrainingLayerFactory layerFactory)
    {
        _trainingLayerFactory = layerFactory;
        return this;
    }
    public NetworkBuilder InputSize(int size)
    {
        _inputSize = size;
        return this;
    }
    public NetworkBuilder HiddenLayers(int[] hiddenLayers)
    {
        _hiddenLayerSizes = hiddenLayers;
        return this;
    }
    public NetworkBuilder OutputSize(int outputSize)
    {
        _outputSize = outputSize;
        return this;
    }
}