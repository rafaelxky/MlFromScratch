public struct Save2Bit
{
    public byte[,] Weights { get; set; }
    public double[] Bias { get; set; }
    public string ActivationFunction { get; set; }

    public Save2Bit()
    {
        
    }
}