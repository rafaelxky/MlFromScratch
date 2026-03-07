using System.ComponentModel;

public interface INetworkFactory
{
    public INetwork FromDto(NetworkData networkData);
}