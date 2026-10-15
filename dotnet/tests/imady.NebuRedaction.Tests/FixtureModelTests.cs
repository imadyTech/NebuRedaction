using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Xunit;

namespace imady.NebuRedaction.Tests;

/// <summary>
/// Reads the committed fixture model and compares against the Python
/// reference's recorded outputs (NR-0005 §2.6 item 1).
/// </summary>
/// <remarks>
/// <para>
/// This is what CI can actually run. The trained model is deliberately not in
/// the repository (§7, 2026-10-14) and neither is the corpus, so on a CI runner
/// every test needing either of them skips — which would leave the binary
/// reader, the tree walk and the Platt calibration, the parts most likely to
/// differ between Windows and Linux, verified on neither.
/// </para>
/// <para>
/// <b>This is not G3.</b> It proves the two implementations read the same bytes
/// and do the same arithmetic on four hand-built trees. Agreement on 2,400 real
/// samples is <see cref="ConsistencyTests"/>, which needs the model and stays a
/// local gate while the model stays out of git.
/// </para>
/// </remarks>
public class FixtureModelTests
{
    /// <summary>The same tolerance G3 fixes.</summary>
    private const double Tolerance = 1e-6;

    private static readonly string ModelPath =
        FeatureVectorTests.RepositoryFile("tests/fixtures/fixture-model-v1.nrm");

    private static readonly JsonDocument Expected = JsonDocument.Parse(
        File.ReadAllText(
            FeatureVectorTests.RepositoryFile("tests/fixtures/fixture-model-expected-v1.json")));

    [Fact]
    public void TheFixtureModelIsCommittedAndReadable()
    {
        // Committed on purpose, unlike every other model: it holds no
        // credential and nothing trained on one.
        Assert.True(File.Exists(ModelPath), $"{ModelPath} is missing");

        SecretDetector detector = SecretDetector.Load(ModelPath);
        var labels = new List<string>();
        foreach (JsonElement label in Expected.RootElement.GetProperty("labels").EnumerateArray())
        {
            labels.Add(label.GetString()!);
        }

        Assert.Equal(labels, detector.Labels);
        Assert.Equal("gbdt", detector.ModelMetadata["model_type"]);
        Assert.Equal(Features.Version, detector.ModelMetadata["feature_version"]);
    }

    [Fact]
    public void EveryProbabilityMatchesThePythonReference()
    {
        SecretDetector balanced = SecretDetector.Load(ModelPath);
        SecretDetector conservative = SecretDetector.Load(
            ModelPath,
            new SecretDetectorOptions { Profile = RedactionProfile.Conservative });

        JsonElement thresholds = Expected.RootElement.GetProperty("thresholds");
        Assert.Equal(thresholds.GetProperty("balanced").GetDouble(), balanced.Threshold, 12);
        Assert.Equal(
            thresholds.GetProperty("conservative").GetDouble(),
            conservative.Threshold,
            12);

        var failures = new List<string>();
        int cases = 0;
        bool sawRedaction = false;
        bool sawNoRedaction = false;
        bool sawProfilesDisagree = false;

        foreach (JsonElement testCase in Expected.RootElement.GetProperty("cases").EnumerateArray())
        {
            cases++;
            string key = testCase.GetProperty("key").GetString()!;
            string value = testCase.GetProperty("value").GetString()!;
            string? template = Text(testCase, "template");
            string? scope = Text(testCase, "scope");
            string label = $"{key}/{(value.Length > 16 ? value[..16] + "…" : value)}";

            DetectionResult result = balanced.Classify(key, value, template, scope);
            bool strict = conservative.Classify(key, value, template, scope).ShouldRedact;

            foreach (JsonProperty expected in testCase.GetProperty("probabilities").EnumerateObject())
            {
                double want = expected.Value.GetDouble();
                double got = result.Probabilities[expected.Name];
                if (Math.Abs(want - got) > Tolerance)
                {
                    failures.Add(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0}: class {1} expected {2:R}, got {3:R}",
                        label,
                        expected.Name,
                        want,
                        got));
                }
            }

            if (Math.Abs(testCase.GetProperty("score").GetDouble() - result.Score) > Tolerance)
            {
                failures.Add($"{label}: score differs");
            }

            if (result.Category != testCase.GetProperty("category").GetString())
            {
                failures.Add($"{label}: category expected "
                    + $"{testCase.GetProperty("category").GetString()}, got {result.Category}");
            }

            bool wantBalanced = testCase.GetProperty("should_redact_balanced").GetBoolean();
            bool wantStrict = testCase.GetProperty("should_redact_conservative").GetBoolean();
            if (result.ShouldRedact != wantBalanced || strict != wantStrict)
            {
                failures.Add($"{label}: decision differs (.NET {result.ShouldRedact}/{strict}, "
                    + $"Python {wantBalanced}/{wantStrict})");
            }

            sawRedaction |= wantBalanced;
            sawNoRedaction |= !wantBalanced;
            sawProfilesDisagree |= wantBalanced != wantStrict;
        }

        Assert.True(
            failures.Count == 0,
            $"{failures.Count} disagreement(s) over {cases} cases:{Environment.NewLine}"
                + string.Join(Environment.NewLine, failures));

        // A fixture where nothing is redacted, or where both profiles always
        // agree, would pass this test while proving much less than it looks
        // like it proves.
        Assert.True(sawRedaction, "no case is redacted; the fixture tests one branch");
        Assert.True(sawNoRedaction, "every case is redacted; the fixture tests one branch");
        Assert.True(sawProfilesDisagree, "the two profiles never differ in this fixture");
    }

    [Fact]
    public void AnUnknownFormatVersionIsRefusedRatherThanRead()
    {
        // The point of versioning the container. Bumping the version byte in a
        // copy of a real file is the cheapest way to test it.
        byte[] bytes = File.ReadAllBytes(ModelPath);
        bytes[4] = 99;

        using var stream = new MemoryStream(bytes);
        NrmException error = Assert.Throws<NrmException>(() => SecretDetector.Load(stream));
        Assert.Contains("99", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelFittedUnderAnotherFeatureVersionIsRefused()
    {
        // The check that matters in practice: different columns, not stale
        // numbers, so every prediction would be plausible nonsense.
        byte[] bytes = File.ReadAllBytes(ModelPath);
        string text = System.Text.Encoding.UTF8.GetString(bytes);
        int at = text.IndexOf("\"feature_version\": \"1.0\"", StringComparison.Ordinal);
        Assert.True(at > 0, "the fixture's metadata no longer spells feature_version that way");

        // Same length, so neither the metadata length nor any later offset moves.
        byte[] replacement = System.Text.Encoding.UTF8.GetBytes("\"feature_version\": \"9.9\"");
        Array.Copy(replacement, 0, bytes, at, replacement.Length);

        using var stream = new MemoryStream(bytes);
        NrmException error = Assert.Throws<NrmException>(() => SecretDetector.Load(stream));
        Assert.Contains("9.9", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATruncatedFileIsRefusedRatherThanReadAsZeroes()
    {
        byte[] bytes = File.ReadAllBytes(ModelPath);
        using var stream = new MemoryStream(bytes, 0, bytes.Length - 8);
        Assert.Throws<NrmException>(() => SecretDetector.Load(stream));
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement property)
            && property.ValueKind != JsonValueKind.Null
            ? property.GetString()
            : null;
}
