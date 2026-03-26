public struct LatentWeightsSave
{
    public double[,] Weights { get; set; }
    public double[] Bias { get; set; }
    public string ActivationFunction { get; set; }

    public LatentWeightsSave()
    {
        
    }
}