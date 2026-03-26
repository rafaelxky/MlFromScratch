using System.Data;

public static class TensorUtils
{
    public static double[][] NewRandomMatrix(Random random, int rows, int cols)
    {
        double[][] matrix = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            matrix[i] = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                matrix[i][j] = random.NextDouble();
            }
        }
        return matrix;
    }
    public static double[,] NewRandomMatrixContiguous(Random random, int rows, int cols)
    {
        double[,] matrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble();
            }
        }

        return matrix;
    }
    
    public static double[,] NewRandomMatrixContiguousCentered(Random random, int rows, int cols)
    {
        double[,] matrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        return matrix;
    }

    public static double[] NewRandomVector(Random random, int size)
    {
        var vector = new double[size];
        for (int i = 0; i < size; i++)
        {
            vector[i] = random.NextDouble();
        }
        return vector;
    }

    // A and B must have the same size;
    public static double DotProd(double[] vecA, double[] vecB)
    {
        double output = 0;
        for (int i = 0; i < vecA.Length; i++)
        {
            output += vecA[i] * vecB[i];
        }
        return output;
    }

    public static double WeightInputDotProd(double[,] neurons, int neuronId, double[] input)
    {
        double output = 0;
        for (int i = 0; i < neurons.GetLength(1); i++)
        {
            output += neurons[neuronId, i] * input[i];
        }
        return output;
    }

    public static double WeightInputDotProdSigned(double[,] neurons, int neuronId, double[] input)
    {
        double output = 0;
        for (int i = 0; i < neurons.GetLength(1); i++)
        {
            output += Math.Sign(neurons[neuronId, i]) * input[i];
        }
        return output;
    }

    public static double[,] FlattenMatrix(double[][] matrix, int rows, int cols)
    {
        double[,] flatMatrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flatMatrix[i, j] = matrix[i][j];
        return flatMatrix;
    }


}