
using TrainingMl1_58;

namespace TrainingMl1_58
{
    public class TrainingNeuron1_58Factory : ITrainingNeuronFactory
    {
        public ITrainingNeuron NewNeuron(int inputSize, Random random)
        {
            return new TrainingNeuron1_58(inputSize,random);
        }
    }
}