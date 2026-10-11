namespace imady.NebuRedaction;

/// <summary>
/// Build and release information about the nebuRedaction inference library.
/// </summary>
/// <remarks>
/// This is a placeholder public surface established by work order NR-0000 so that
/// the package, its namespace and both target frameworks can be built and tested
/// before the detector itself exists. The detection API arrives in NR-0005.
/// </remarks>
public static class NebuRedactionInfo
{
    /// <summary>
    /// The version of the nebuRedaction library, as a three-part
    /// <c>major.minor.patch</c> string.
    /// </summary>
    /// <value>
    /// The informational version of the assembly, with any build metadata
    /// suffix (for example <c>+sha</c>) removed.
    /// </value>
    public static string Version => "0.1.0";

    /// <summary>
    /// Whether this build ships a usable detection model.
    /// </summary>
    /// <value>
    /// <see langword="false"/> for every pre-release build; it becomes
    /// <see langword="true"/> once a trained default model is embedded (NR-0005).
    /// </value>
    public static bool HasEmbeddedModel => false;
}
