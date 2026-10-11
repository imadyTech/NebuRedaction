using Xunit;

namespace imady.NebuRedaction.Tests;

public class NebuRedactionInfoTests
{
    [Fact]
    public void VersionIsAThreePartVersionString()
    {
        Assert.True(System.Version.TryParse(NebuRedactionInfo.Version, out var version));
        Assert.Equal(3, version!.ToString().Split('.').Length);
    }

    [Fact]
    public void HasEmbeddedModelIsFalseBeforeAModelIsTrained()
    {
        Assert.False(NebuRedactionInfo.HasEmbeddedModel);
    }
}
