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
    /// <summary>The committed fixture model these tests run against.</summary>
    /// <remarks>
    /// Deliberately the 581-byte fixture rather than the trained model. What
    /// these tests assert is <b>semantics</b> — override precedence, the
    /// key-name comparison, the ordering of the two thresholds, the mask's
    /// construction — and none of that reads the weights. Pointing them at
    /// `models/neburedaction-v0.1.0.nrm` only meant that in any clean checkout,
    /// CI included, all of them skipped and §6 item 4's "the L1 options and
    /// Mask have tests" was true of nobody's machine but this one.
    /// </remarks>
    private static readonly string ModelPath =
        FeatureVectorTests.RepositoryFile("tests/fixtures/fixture-model-v1.nrm");

    private static SecretDetector Detector(SecretDetectorOptions? options = null) =>
        SecretDetector.Load(ModelPath, options);

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
        // Which way the model votes is not the point and must not be assumed:
        // the detector is asked first, then the override that contradicts it is
        // applied. That keeps the assertion about the override rather than
        // about the weights, so it holds for any model.
        const string key = "comment";
        const string value = "ordinary text here";

        DetectionResult plain = Detector().Classify(key, value);
        var contradicting = new SecretDetectorOptions();
        if (plain.ShouldRedact)
        {
            contradicting.NeverRedactKeys = [key];
        }
        else
        {
            contradicting.AlwaysRedactKeys = [key];
        }

        DetectionResult overridden = Detector(contradicting).Classify(key, value);

        Assert.Equal(plain.Probabilities, overridden.Probabilities);
        Assert.Equal(plain.Score, overridden.Score);
        Assert.NotEqual(plain.ShouldRedact, overridden.ShouldRedact);
        Assert.Equal(RedactionReason.Model, plain.Reason);
        Assert.NotEqual(RedactionReason.Model, overridden.Reason);
    }

    [Fact]
    public void NeverRedactSuppressesAModelDecision()
    {
        // A value the fixture model does redact, so the override is actually
        // suppressing something rather than agreeing with it.
        const string key = "payload";
        string value = new('x', 300);
        Assert.True(Detector().Classify(key, value).ShouldRedact, "the premise no longer holds");

        DetectionResult result = Detector(new SecretDetectorOptions
        {
            NeverRedactKeys = [key],
        }).Classify(key, value);

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
    public void TheEmbeddedDefaultModelLoadsAndAgreesWithTheFileOnDisk()
    {
        // NR-0005 §2.2 item 3 asked for `Default`; it could not be written until
        // the weights were embedded, which NR-0006 settled on 2026-10-22. What
        // matters is not only that it loads but that it is the *same* model the
        // consistency gate was measured on — an embedded resource that drifted
        // from the exported file would pass every other test here.
        string onDisk = FeatureVectorTests.RepositoryFile("models/neburedaction-v0.1.0.nrm");
        if (!File.Exists(onDisk))
        {
            Assert.Skip("the release model is not in this checkout, so Default has no resource");
        }

        SecretDetector embedded = SecretDetector.Default;
        SecretDetector fromFile = SecretDetector.Load(onDisk);

        Assert.Equal(fromFile.Labels, embedded.Labels);
        Assert.Equal(fromFile.Threshold, embedded.Threshold, 12);
        Assert.Equal(fromFile.ModelMetadata["feature_version"], embedded.ModelMetadata["feature_version"]);

        foreach ((string key, string value) in new[]
        {
            ("OrderId", "29"),
            ("digest", "9f86d081884c7d659a2feaa0c55ad015"),
            ("payload", new string('x', 300)),
        })
        {
            Assert.Equal(
                fromFile.Classify(key, value).Probabilities,
                embedded.Classify(key, value).Probabilities);
        }
    }

    [Fact]
    public void DefaultIsASingleSharedInstance()
    {
        // Parsing the model allocates the trees and the bucket map; re-parsing it
        // per call would be a surprising cost behind a property.
        if (!File.Exists(FeatureVectorTests.RepositoryFile("models/neburedaction-v0.1.0.nrm")))
        {
            Assert.Skip("the release model is not in this checkout");
        }

        Assert.Same(SecretDetector.Default, SecretDetector.Default);
    }

    [Fact]
    public void LoadDefaultHonoursOptionsWithoutMutatingDefault()
    {
        if (!File.Exists(FeatureVectorTests.RepositoryFile("models/neburedaction-v0.1.0.nrm")))
        {
            Assert.Skip("the release model is not in this checkout");
        }

        SecretDetector strict = SecretDetector.LoadDefault(
            new SecretDetectorOptions { Profile = RedactionProfile.Conservative });

        Assert.True(strict.Threshold >= SecretDetector.Default.Threshold);
        Assert.NotSame(SecretDetector.Default, strict);
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
