public class WrongInputSizeException: Exception
{
    public WrongInputSizeException()
    {
        
    }
    public WrongInputSizeException(string message): base(message)
    {
        
    }

    public WrongInputSizeException(string message, Exception inner): base(message, inner)
    {
        
    }
}