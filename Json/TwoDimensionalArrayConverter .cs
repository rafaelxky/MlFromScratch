using System.Text.Json;
using System.Text.Json.Serialization;

public class TwoDimensionalArrayConverter : JsonConverter<double[,]>
{
    public override double[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var jagged = JsonSerializer.Deserialize<double[][]>(ref reader, options)!;
        
        int rows = jagged.Length;
        int cols = jagged[0].Length;
        var result = new double[rows, cols];
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = jagged[i][j];
        
        return result;
    }

    public override void Write(Utf8JsonWriter writer, double[,] value, JsonSerializerOptions options)
    {
        int rows = value.GetLength(0);
        int cols = value.GetLength(1);
        
        var jagged = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            jagged[i] = new double[cols];
            for (int j = 0; j < cols; j++)
                jagged[i][j] = value[i, j];
        }
        
        JsonSerializer.Serialize(writer, jagged, options);
    }
}