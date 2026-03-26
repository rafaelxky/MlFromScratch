using System.Text.Json;
using System.Text.Json.Serialization;

public class ByteArray2DConverter : JsonConverter<byte[,]>
{
    public override byte[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var jagged = JsonSerializer.Deserialize<byte[][]>(ref reader, options)!;
        int rows = jagged.Length;
        int cols = rows == 0 ? 0 : jagged[0].Length;
        var result = new byte[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = jagged[i][j];
        return result;
    }

    public override void Write(Utf8JsonWriter writer, byte[,] value, JsonSerializerOptions options)
    {
        int rows = value.GetLength(0);
        int cols = value.GetLength(1);
        writer.WriteStartArray();
        for (int i = 0; i < rows; i++)
        {
            writer.WriteStartArray();
            for (int j = 0; j < cols; j++)
                writer.WriteNumberValue(value[i, j]);
            writer.WriteEndArray();
        }
        writer.WriteEndArray();
    }
}