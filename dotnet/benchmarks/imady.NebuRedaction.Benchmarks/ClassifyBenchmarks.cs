using System;
using System.IO;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace imady.NebuRedaction.Benchmarks;

/// <summary>Adds the percentile columns §2.5 asks for.</summary>
/// <remarks>
/// BenchmarkDotNet exposes percentiles up to P95 and no P99:
/// <see cref="StatisticColumn"/> has P0, P25, P50, P67, P80, P85, P90, P95 and
/// P100. A P99 taken from the iteration count a sub-microsecond method needs
/// would mostly estimate its own noise, so the report states P95 as measured
/// and says P99 is not available rather than relabelling one as the other.
/// </remarks>
public sealed class PercentileConfig : ManualConfig
{
    /// <summary>Adds P50, P90 and P95 to the summary.</summary>
    public PercentileConfig()
    {
        AddColumn(StatisticColumn.P50, StatisticColumn.P90, StatisticColumn.P95);
    }
}

/// <summary>
/// Measures <see cref="SecretDetector.Classify"/> (NR-0005 §2.5).
/// </summary>
/// <remarks>
/// <para>
/// §2.5 asks for p50 and p99 latency and bytes allocated per call, compared
/// against the plan's p99 ≤ 20 µs. This is a report, not a gate.
/// </para>
/// <para>
/// BenchmarkDotNet reports a mean and a distribution rather than percentiles by
/// default, so <see cref="PercentileConfig"/> adds P50, P90 and P95. There is
/// no P99 column to add; see that type for why, and why the report does not
/// invent one.
/// </para>
/// <para>
/// Three inputs, because they cost different amounts: the feature extractor
/// scans the value, so a long value is more work than a short one, and the
/// trees are walked once per class either way.
/// </para>
/// </remarks>
[MemoryDiagnoser]
[Config(typeof(PercentileConfig))]
public class ClassifyBenchmarks
{
    private SecretDetector _detector = null!;

    /// <summary>A short non-secret, the commonest case in a real log.</summary>
    private const string ShortValue = "29";

    /// <summary>A mid-length structural value.</summary>
    private const string HexValue = "9f86d081884c7d659a2feaa0c55ad015";

    /// <summary>A long value, where the value scanners do the most work.</summary>
    private static readonly string LongValue = new('x', 512);

    private const string Template = "order {OrderId} dispatched to {Region}";

    /// <summary>Loads the model once, outside the measurement.</summary>
    [GlobalSetup]
    public void Setup()
    {
        string path = RepositoryFile("models/neburedaction-v0.1.0.nrm");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "the benchmark needs the release model; export it with "
                    + "training/export_nrm.py first",
                path);
        }

        _detector = SecretDetector.Load(path);
    }

    /// <summary>Classifies a short non-secret attribute.</summary>
    /// <returns>Whether it would be redacted, so the call is not elided.</returns>
    [Benchmark(Baseline = true, Description = "short value (2 chars)")]
    public bool ShortNonSecret() =>
        _detector.Classify("OrderId", ShortValue, Template, "otel.attribute").ShouldRedact;

    /// <summary>Classifies a 32-character hex value.</summary>
    /// <returns>Whether it would be redacted.</returns>
    [Benchmark(Description = "hex value (32 chars)")]
    public bool HexDigest() =>
        _detector.Classify("digest", HexValue, Template, "otel.attribute").ShouldRedact;

    /// <summary>Classifies a 512-character value.</summary>
    /// <returns>Whether it would be redacted.</returns>
    [Benchmark(Description = "long value (512 chars)")]
    public bool LongAttribute() =>
        _detector.Classify("payload", LongValue, Template, "otel.attribute").ShouldRedact;

    /// <summary>Masks a value.</summary>
    /// <returns>The mask, so the call is not elided.</returns>
    [Benchmark(Description = "Mask (SHA-256 of the value)")]
#pragma warning disable CA1822 // BenchmarkDotNet refuses static benchmark methods.
    public string Mask() => SecretDetector.Mask(HexValue);
#pragma warning restore CA1822

    /// <summary>Walks up from the assembly to the repository root.</summary>
    /// <param name="relative">A repository-relative path.</param>
    /// <returns>The absolute path.</returns>
    internal static string RepositoryFile(string relative)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null
            && !File.Exists(Path.Combine(directory.FullName, "global.json")))
        {
            directory = directory.Parent;
        }

        if (directory is null)
        {
            throw new InvalidOperationException("could not locate the repository root");
        }

        return Path.Combine(directory.FullName, relative.Replace('/', Path.DirectorySeparatorChar));
    }
}

/// <summary>Entry point.</summary>
internal static class Program
{
    private static void Main(string[] arguments) =>
        BenchmarkSwitcher.FromAssembly(typeof(ClassifyBenchmarks).Assembly).Run(arguments);
}
