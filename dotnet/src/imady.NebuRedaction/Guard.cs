using System;

namespace imady.NebuRedaction;

/// <summary>Argument checks that compile on both target frameworks.</summary>
/// <remarks>
/// <para>
/// <c>ArgumentNullException.ThrowIfNull</c> is .NET 6 and later, and this
/// library also targets <c>netstandard2.1</c>, where the throw has to be
/// written out. Doing it here rather than at each call site keeps both
/// frameworks throwing the identical exception with the identical parameter
/// name, from one place.
/// </para>
/// <para>
/// The two frameworks are split with <c>#if</c> rather than written once and
/// the resulting CA1510 suppressed: EXECUTION.md §2 item 3 forbids silencing
/// an analyzer unless the work order allows it, and this one is right — on
/// .NET 10 the helper really should call <c>ThrowIfNull</c>, so it does.
/// </para>
/// </remarks>
internal static class Guard
{
    /// <summary>Throws when <paramref name="value"/> is null.</summary>
    /// <param name="value">The argument.</param>
    /// <param name="name">Its parameter name.</param>
    internal static void NotNull(object? value, string name)
    {
#if NET
        ArgumentNullException.ThrowIfNull(value, name);
#else
        if (value is null)
        {
            throw new ArgumentNullException(name);
        }
#endif
    }
}
