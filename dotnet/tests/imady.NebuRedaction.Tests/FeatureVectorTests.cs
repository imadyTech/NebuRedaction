using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Xunit;

namespace imady.NebuRedaction.Tests;

/// <summary>
/// Checks the .NET feature extractor against the vectors the normative Python
/// definition produced (NR-0005 §2.2 item 2).
/// </summary>
/// <remarks>
/// Plan §6.4's G3 makes agreement to within 1e-6 a hard condition. These
/// vectors are the first line of that: if a feature disagrees here, the
/// per-sample probability comparison would disagree too, and this says which
/// feature rather than only that something is wrong.
/// </remarks>
public class FeatureVectorTests
{
    private static readonly JsonDocument Vectors = LoadVectors();

    private static JsonDocument LoadVectors()
    {
        string path = RepositoryFile("tests/fixtures/feature-vectors-v1.json");
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    internal static string RepositoryFile(string relative)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "global.json")))
        {
            directory = directory.Parent;
        }

        if (directory is null)
        {
            throw new InvalidOperationException("could not locate the repository root");
        }

        return Path.Combine(directory.FullName, relative.Replace('/', Path.DirectorySeparatorChar));
    }

    [Fact]
    public void FeatureVersionMatches()
    {
        // A model is refused when these differ, so the two definitions must
        // agree on what version they are before anything else is compared.
        Assert.Equal(
            Features.Version,
            Vectors.RootElement.GetProperty("feature_version").GetString());
    }

    [Fact]
    public void DenseFeatureOrderMatches()
    {
        // The order is the order a model's coefficients are in. A mismatch
        // would not fail loudly anywhere else.
        var expected = new List<string>();
        foreach (JsonElement name in Vectors.RootElement.GetProperty("dense_feature_order").EnumerateArray())
        {
            expected.Add(name.GetString()!);
        }

        Assert.Equal(expected, Features.DenseFeatureNames);
    }

    [Fact]
    public void Fnv1aMatchesThePublishedVectors()
    {
        // Checked against FNV's own published values via the fixture, so the
        // hash is verified against its specification rather than against the
        // other implementation of itself.
        foreach (JsonProperty entry in Vectors.RootElement.GetProperty("fnv1a_32").EnumerateObject())
        {
            Assert.Equal(entry.Value.GetUInt32(), Features.Fnv1a(entry.Name));
        }
    }

    [Fact]
    public void EveryDenseFeatureMatchesForEveryCase()
    {
        int compared = 0;
        var failures = new List<string>();

        foreach (JsonElement testCase in Vectors.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = testCase.GetProperty("name").GetString()!;
            JsonElement input = testCase.GetProperty("input");
            double[] actual = Features.Dense(
                input.GetProperty("key").GetString()!,
                input.GetProperty("value").GetString()!,
                Text(input, "template"),
                Text(input, "scope"));

            foreach (JsonProperty expected in testCase.GetProperty("dense").EnumerateObject())
            {
                int index = Array.IndexOf(Features.DenseFeatureNames, expected.Name);
                Assert.True(index >= 0, $"unknown dense feature {expected.Name}");
                double want = expected.Value.GetDouble();
                double got = actual[index];
                compared++;
                if (Math.Abs(want - got) > 1e-9)
                {
                    failures.Add(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0}/{1}: expected {2:R}, got {3:R}",
                        name,
                        expected.Name,
                        want,
                        got));
                }
            }
        }

        Assert.True(
            failures.Count == 0,
            $"{failures.Count} of {compared} feature comparisons differ:{Environment.NewLine}"
                + string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void HashedFeatureSummariesMatchForEveryCase()
    {
        // The vectors record a summary rather than every bucket: a case can
        // activate a few hundred of 32,768, so the full maps would be most of a
        // megabyte of indices telling a reader nothing. The summary still fails
        // loudly if the hashing differs, and the FNV vectors above localise why.
        int buckets = Vectors.RootElement.GetProperty("buckets").GetInt32();
        var failures = new List<string>();

        foreach (JsonElement testCase in Vectors.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = testCase.GetProperty("name").GetString()!;
            JsonElement input = testCase.GetProperty("input");
            Dictionary<int, double> hashed = Features.Hashed(
                input.GetProperty("key").GetString()!,
                input.GetProperty("value").GetString()!,
                Text(input, "template"),
                Text(input, "scope"),
                buckets);

            JsonElement summary = testCase.GetProperty("hashed");
            int activeBuckets = summary.GetProperty("active_buckets").GetInt32();
            long bucketSum = summary.GetProperty("bucket_sum").GetInt64();

            long actualBucketSum = 0;
            foreach (int bucket in hashed.Keys)
            {
                actualBucketSum += bucket;
            }

            if (hashed.Count != activeBuckets)
            {
                failures.Add($"{name}: active buckets expected {activeBuckets}, got {hashed.Count}");
            }

            if (actualBucketSum != bucketSum)
            {
                failures.Add($"{name}: bucket sum expected {bucketSum}, got {actualBucketSum}");
            }
        }

        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void BucketCountsOutsideTheAllowedRangeAreRefused()
    {
        // Plan §4.2 allows 2^14 to 2^16, and a non-power-of-two would turn the
        // bucket mask into a modulo, biasing the low buckets.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Features.BucketOf("x", Features.ValueNamespace, 1000, out _));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Features.BucketOf("x", Features.ValueNamespace, 1 << 13, out _));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Features.BucketOf("x", Features.ValueNamespace, 1 << 17, out _));
    }

    [Fact]
    public void NamespacesSeparateTheFeatureFamilies()
    {
        // The token "token" as a key word and in a template are two features.
        int key = Features.BucketOf("token", Features.KeyNamespace, 1 << 15, out _);
        int context = Features.BucketOf("token", Features.ContextNamespace, 1 << 15, out _);
        int value = Features.BucketOf("token", Features.ValueNamespace, 1 << 15, out _);
        Assert.Equal(3, new HashSet<int> { key, context, value }.Count);
    }

    [Fact]
    public void TheAllocationFreePathAgreesWithTheReferencePath()
    {
        // `Project` accumulates only the buckets a model reads, instead of
        // building the whole map and discarding most of it. That is exact only
        // because the hashed features are not normalised, so it is worth
        // proving rather than asserting: for every committed case, the two
        // paths must produce the same value in every column.
        const int buckets = 1 << 15;
        var columns = new Dictionary<int, int>();
        for (int bucket = 0; bucket < buckets; bucket++)
        {
            columns[bucket] = bucket;
        }

        foreach (JsonElement testCase in Vectors.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = testCase.GetProperty("name").GetString()!;
            JsonElement input = testCase.GetProperty("input");
            string key = input.GetProperty("key").GetString()!;
            string value = input.GetProperty("value").GetString()!;
            string? template = Text(input, "template");
            string? scope = Text(input, "scope");

            Dictionary<int, double> reference =
                Features.Hashed(key, value, template, scope, buckets);
            var accumulated = new double[buckets];
            Features.AccumulateHashed(
                key, value, template, scope, buckets, columns, accumulated);

            foreach (KeyValuePair<int, double> entry in reference)
            {
                Assert.Equal(entry.Value, accumulated[entry.Key]);
            }

            double total = 0.0;
            foreach (double weight in accumulated)
            {
                total += Math.Abs(weight);
            }

            double referenceTotal = 0.0;
            foreach (double weight in reference.Values)
            {
                referenceTotal += Math.Abs(weight);
            }

            // Catches a bucket the fast path sets and the reference does not,
            // which the loop above cannot see.
            Assert.Equal(referenceTotal, total);
            Assert.True(total >= 0.0, name);
        }
    }

    [Fact]
    public void TheSpanHashMatchesTheStringHash()
    {
        // The namespace seed is only exact because FNV-1a is streaming.
        foreach (string token in new[] { "a", "ab", "abcd", "token", "", "键名" })
        {
            foreach (string ns in new[]
            {
                Features.ValueNamespace, Features.KeyNamespace, Features.ContextNamespace,
            })
            {
                int expected = Features.BucketOf(token, ns, 1 << 15, out double expectedSign);
                int actual = Features.BucketOfSpan(
                    Features.NamespaceSeed(ns), token.AsSpan(), (1 << 15) - 1, out double sign);
                Assert.Equal(expected, actual);
                Assert.Equal(expectedSign, sign);
            }
        }
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement property)
            && property.ValueKind != JsonValueKind.Null
            ? property.GetString()
            : null;
}
