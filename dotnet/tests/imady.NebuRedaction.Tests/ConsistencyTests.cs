using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Xunit;

namespace imady.NebuRedaction.Tests;

/// <summary>
/// Compares the .NET library against the Python reference implementation,
/// sample by sample (NR-0005 §2.4, plan §6.4's gate G3).
/// </summary>
/// <remarks>
/// <para>
/// G3 is a hard gate: every class probability must agree to within 1e-6
/// absolute and every redaction decision must be identical. §2.4 item 3
/// forbids relaxing the tolerance in response to a failure — a disagreement
/// has to be traced to a feature or an operation and fixed — so
/// <see cref="Tolerance"/> is a constant and the failure message is built to
/// say which sample and which class, since that is what makes tracing possible.
/// </para>
/// <para>
/// The set is generated rather than committed: the synthetic section carries
/// format-correct vendor credentials, which must not enter git. When it is
/// absent these tests skip, and <see cref="TheSetIsPresentWhenRequired"/> turns
/// that skip into a failure wherever <c>NEBUREDACTION_REQUIRE_CONSISTENCY_SET</c>
/// is set, which CI does. Without that guard a CI job that forgot to generate
/// the set would report a clean pass over nothing.
/// </para>
/// </remarks>
public class ConsistencyTests
{
    /// <summary>The tolerance plan §6.4 fixes. Not to be relaxed.</summary>
    private const double Tolerance = 1e-6;

    private const string SetPath = "data/consistency/consistency-set-v1.json";
    private const string RequireVariable = "NEBUREDACTION_REQUIRE_CONSISTENCY_SET";

    /// <summary>How many disagreements to name before truncating the message.</summary>
    private const int ReportedFailures = 20;

    [Fact]
    public void TheSetIsPresentWhenRequired()
    {
        // Guards the skip above: in CI the set must exist, so a job that
        // skipped the comparison cannot pass as if it had run it.
        if (Environment.GetEnvironmentVariable(RequireVariable) is null)
        {
            Assert.Skip($"{RequireVariable} is not set; the consistency set is optional here");
        }

        Assert.True(
            File.Exists(FeatureVectorTests.RepositoryFile(SetPath)),
            $"{RequireVariable} is set but {SetPath} is missing; run "
                + "tools/generate_consistency_set.py before the .NET tests");
    }

    [Fact]
    public void EveryClassProbabilityMatchesPython()
    {
        JsonElement document = LoadSet();

        var labels = new List<string>();
        foreach (JsonElement label in document.GetProperty("labels").EnumerateArray())
        {
            labels.Add(label.GetString()!);
        }

        SecretDetector balanced = LoadDetector(RedactionProfile.Balanced);
        SecretDetector conservative = LoadDetector(RedactionProfile.Conservative);

        // The label order is what every probability comparison below is
        // indexed by; if it differs, each class would be compared against a
        // different class and the message would say nothing useful.
        Assert.Equal(labels, balanced.Labels);
        Assert.Equal(
            Features.Version,
            document.GetProperty("feature_version").GetString());

        JsonElement thresholds = document.GetProperty("thresholds");
        Assert.Equal(thresholds.GetProperty("balanced").GetDouble(), balanced.Threshold, 12);
        Assert.Equal(
            thresholds.GetProperty("conservative").GetDouble(),
            conservative.Threshold,
            12);

        int compared = 0;
        int samples = 0;
        double worst = 0.0;
        string worstSample = "none";
        var failures = new List<string>();

        foreach (JsonElement sample in document.GetProperty("samples").EnumerateArray())
        {
            samples++;
            string id = sample.GetProperty("id").GetString()!;
            string section = sample.GetProperty("section").GetString()!;
            DetectionResult result = balanced.Classify(
                sample.GetProperty("key").GetString()!,
                sample.GetProperty("value").GetString()!,
                Text(sample, "template"),
                Text(sample, "scope"));

            foreach (JsonProperty expected in sample.GetProperty("probabilities").EnumerateObject())
            {
                double want = expected.Value.GetDouble();
                double got = result.Probabilities[expected.Name];
                double difference = Math.Abs(want - got);
                compared++;
                if (difference > worst)
                {
                    worst = difference;
                    worstSample = $"{section}/{id}/{expected.Name}";
                }

                if (difference > Tolerance)
                {
                    Add(failures, string.Format(
                        CultureInfo.InvariantCulture,
                        "{0}/{1}: class {2} expected {3:R}, got {4:R} (|delta| {5:E3})",
                        section,
                        id,
                        expected.Name,
                        want,
                        got,
                        difference));
                }
            }

            if (result.ShouldRedact != sample.GetProperty("should_redact_balanced").GetBoolean())
            {
                Add(failures, $"{section}/{id}: balanced decision differs "
                    + $"(.NET {result.ShouldRedact}, score {SecretDetector.FormatScore(result.Score)})");
            }

            bool strict = conservative.Classify(
                sample.GetProperty("key").GetString()!,
                sample.GetProperty("value").GetString()!,
                Text(sample, "template"),
                Text(sample, "scope")).ShouldRedact;
            if (strict != sample.GetProperty("should_redact_conservative").GetBoolean())
            {
                Add(failures, $"{section}/{id}: conservative decision differs (.NET {strict})");
            }

            if (result.Category != sample.GetProperty("category").GetString())
            {
                Add(failures, $"{section}/{id}: category expected "
                    + $"{sample.GetProperty("category").GetString()}, got {result.Category}");
            }
        }

        Assert.True(samples > 0, "the consistency set carries no samples");
        Assert.True(
            failures.Count == 0,
            string.Format(
                CultureInfo.InvariantCulture,
                "{0} disagreement(s) over {1} samples and {2} probability comparisons; "
                    + "worst |delta| {3:E3} at {4}:{5}{6}",
                failures.Count,
                samples,
                compared,
                worst,
                worstSample,
                Environment.NewLine,
                string.Join(Environment.NewLine, failures)));

        // Stated as an assertion rather than only in the message so that a
        // tolerance quietly widened later still fails here.
        Assert.True(worst <= Tolerance, $"worst |delta| was {worst:E3}");
    }

    private static void Add(List<string> failures, string failure)
    {
        if (failures.Count < ReportedFailures)
        {
            failures.Add(failure);
        }
        else if (failures.Count == ReportedFailures)
        {
            failures.Add("... further disagreements not listed");
        }
    }

    private static SecretDetector LoadDetector(RedactionProfile profile) =>
        SecretDetector.Load(
            FeatureVectorTests.RepositoryFile("models/neburedaction-v0.1.0.nrm"),
            new SecretDetectorOptions { Profile = profile });

    private static JsonElement LoadSet()
    {
        string path = FeatureVectorTests.RepositoryFile(SetPath);
        if (!File.Exists(path))
        {
            Assert.Skip(
                $"{SetPath} has not been generated; run "
                    + "tools/generate_consistency_set.py (it needs the project's data/ corpus)");
        }

        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement property)
            && property.ValueKind != JsonValueKind.Null
            ? property.GetString()
            : null;
}
