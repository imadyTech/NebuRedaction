using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Xunit;

namespace imady.NebuRedaction.Tests;

/// <summary>
/// Checks the inference API's documented semantics (NR-0005 §2.3).
/// </summary>
/// <remarks>
/// These mirror <c>tests/neburedaction/test_detector.py</c> assertion for
/// assertion. They are the behaviours no probability comparison would catch —
/// the key-name comparison, the precedence between the override lists, the
/// mask's digest — so each one is asserted on both sides rather than inferred
/// from the other passing.
/// </remarks>
public class SecretDetectorTests
{
    private static readonly string ModelPath =
        FeatureVectorTests.RepositoryFile("models/neburedaction-v0.1.0.nrm");

    private static SecretDetector Detector(SecretDetectorOptions? options = null)
    {
        if (!File.Exists(ModelPath))
        {
            Assert.Skip("neburedaction-v0.1.0.nrm is not in this checkout");
        }

        return SecretDetector.Load(ModelPath, options);
    }

    [Fact]
    public void MaskIsTheFirstFourHexCharactersOfSha256()
    {
        // Stated as the construction rather than as a golden string, so the
        // test says what the mask is instead of what it currently produces.
        const string value = "whatever-this-is";
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        string expected = "***#" + Convert.ToHexString(digest)[..4].ToLowerInvariant();
        Assert.Equal(expected, SecretDetector.Mask(value));
    }

    [Fact]
    public void MaskIsStableAndDistinguishing()
    {
        Assert.Equal(SecretDetector.Mask("a"), SecretDetector.Mask("a"));
        Assert.NotEqual(SecretDetector.Mask("a"), SecretDetector.Mask("b"));
        Assert.StartsWith("***#", SecretDetector.Mask(string.Empty), StringComparison.Ordinal);
    }

    [Fact]
    public void ProbabilitiesSumToOneAndNameEveryClass()
    {
        SecretDetector detector = Detector();
        DetectionResult result = detector.Classify("note", "hello");

        Assert.Equal(
            detector.Labels.OrderBy(label => label, StringComparer.Ordinal),
            result.Probabilities.Keys.OrderBy(label => label, StringComparer.Ordinal));
        Assert.Equal(result.Probabilities.Values.Max(), result.Probabilities[result.Category]);
        Assert.Equal(1.0, result.Probabilities.Values.Sum(), 9);
    }

    [Fact]
    public void AlwaysRedactWinsOverNever()
    {
        // The documented precedence, which is otherwise arbitrary.
        DetectionResult result = Detector(new SecretDetectorOptions
        {
            AlwaysRedactKeys = ["ticket"],
            NeverRedactKeys = ["ticket"],
        }).Classify("ticket", "ABC-1234");

        Assert.True(result.ShouldRedact);
        Assert.Equal(RedactionReason.AlwaysRedactKey, result.Reason);
    }

    [Fact]
    public void KeyOverridesAreCaseInsensitive()
    {
        SecretDetector detector = Detector(new SecretDetectorOptions
        {
            AlwaysRedactKeys = ["API_KEY"],
        });

        Assert.Equal(RedactionReason.AlwaysRedactKey, detector.Classify("api_key", "x").Reason);
        Assert.Equal(RedactionReason.AlwaysRedactKey, detector.Classify("Api_Key", "x").Reason);
    }

    [Fact]
    public void AnOverrideLeavesTheProbabilitiesUntouched()
    {
        DetectionResult plain = Detector().Classify("comment", "ordinary text here");
        DetectionResult overridden = Detector(new SecretDetectorOptions
        {
            AlwaysRedactKeys = ["comment"],
        }).Classify("comment", "ordinary text here");

        Assert.Equal(plain.Probabilities, overridden.Probabilities);
        Assert.True(overridden.ShouldRedact);
        Assert.False(plain.ShouldRedact);
    }

    [Fact]
    public void NeverRedactSuppressesAModelDecision()
    {
        DetectionResult result = Detector(new SecretDetectorOptions
        {
            NeverRedactKeys = ["anykey"],
        }).Classify("anykey", "AKIAIOSFODNN7EXAMPLE");

        Assert.False(result.ShouldRedact);
        Assert.Equal(RedactionReason.NeverRedactKey, result.Reason);
    }

    [Fact]
    public void ConservativeIsTheStricterThreshold()
    {
        // Ten times stricter means a higher bar, so it can only redact less.
        double balanced = Detector().Threshold;
        double conservative = Detector(new SecretDetectorOptions
        {
            Profile = RedactionProfile.Conservative,
        }).Threshold;

        Assert.True(conservative >= balanced, $"{conservative} < {balanced}");
    }

    [Fact]
    public void TheModelCarriesTheMetadataAReportNeeds()
    {
        IReadOnlyDictionary<string, string> metadata = Detector().ModelMetadata;

        Assert.Equal("gbdt", metadata["model_type"]);
        Assert.Equal(Features.Version, metadata["feature_version"]);
    }

    [Fact]
    public void LoadingFromAStreamMatchesLoadingFromAPath()
    {
        if (!File.Exists(ModelPath))
        {
            Assert.Skip("neburedaction-v0.1.0.nrm is not in this checkout");
        }

        using FileStream stream = File.OpenRead(ModelPath);
        SecretDetector fromStream = SecretDetector.Load(stream);

        Assert.Equal(
            Detector().Classify("token", "notarealvalue1").Probabilities,
            fromStream.Classify("token", "notarealvalue1").Probabilities);
    }

    [Fact]
    public void ANonNrmFileIsRefused()
    {
        // The magic check exists so that a wrong path fails here rather than as
        // nonsense probabilities later.
        using var stream = new MemoryStream(Encoding.ASCII.GetBytes("not a model at all"));
        NrmException error = Assert.Throws<NrmException>(() => SecretDetector.Load(stream));
        // The same wording the Python reader uses, so a caller matching on it
        // is matching on one string rather than two.
        Assert.Contains("wrong magic", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ClassifyMessageDropsProseAndUsesTheMessageAsTemplate()
    {
        SecretDetector detector = Detector();

        Assert.Empty(detector.ClassifyMessage(
            "connection established to the downstream auckland region"));

        const string message = "session token=notarealvalue1 opened";
        IReadOnlyList<KeyValuePair<string, DetectionResult>> results =
            detector.ClassifyMessage(message);

        Assert.Equal(["notarealvalue1"], results.Select(entry => entry.Key));
        Assert.Equal(
            detector.Classify("token", "notarealvalue1", message, "message.kv").Probabilities,
            results[0].Value.Probabilities);
    }

    [Fact]
    public void NullArgumentsAreRefused()
    {
        SecretDetector detector = Detector();

        Assert.Throws<ArgumentNullException>(() => detector.Classify(null!, "x"));
        Assert.Throws<ArgumentNullException>(() => detector.Classify("x", null!));
        Assert.Throws<ArgumentNullException>(() => detector.ClassifyMessage(null!));
        Assert.Throws<ArgumentNullException>(() => SecretDetector.Mask(null!));
    }
}
