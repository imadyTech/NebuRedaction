using System;

namespace imady.NebuRedaction;

/// <summary>Argument checks that compile on both target frameworks.</summary>
/// <remarks>
/// <c>ArgumentNullException.ThrowIfNull</c> is .NET 6 and later, and this
/// library also targets <c>netstandard2.1</c>, so the throw has to be written
/// out. Doing it here rather than at each call site keeps the analyzer
/// suppression to one place instead of a dozen, and keeps the two frameworks
/// throwing the identical exception with the identical parameter name.
/// </remarks>
internal static class Guard
{
#pragma warning disable CA1510 // ArgumentNullException.ThrowIfNull is not available on netstandard2.1.
    /// <summary>Throws when <paramref name="value"/> is null.</summary>
    /// <param name="value">The argument.</param>
    /// <param name="name">Its parameter name.</param>
    internal static void NotNull(object? value, string name)
    {
        if (value is null)
        {
            throw new ArgumentNullException(name);
        }
    }
#pragma warning restore CA1510
}
