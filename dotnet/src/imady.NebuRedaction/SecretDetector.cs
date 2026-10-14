using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace imady.NebuRedaction;

/// <summary>Which threshold preset a detector operates at.</summary>
public enum RedactionProfile
{
    /// <summary>
    /// Plan §4.5's default: a 0.1% false-positive target on the validation
    /// split.
    /// </summary>
    Balanced,

    /// <summary>
    /// Ten times stricter, a 0.01% target. Redacts less and damages fewer
    /// logs, at lower recall. Worth preferring where the data may not look
    /// like what the model was trained on, since NR-0004 measured a
    /// false-positive rate several times its target on unseen format families.
    /// </summary>
    Conservative,
}

/// <summary>Which rule decided a unit's fate.</summary>
public enum RedactionReason
{
    /// <summary>The model's calibrated score met the profile's threshold.</summary>
    Model,

    /// <summary>The key name was listed in <see cref="SecretDetectorOptions.AlwaysRedactKeys"/>.</summary>
    AlwaysRedactKey,

    /// <summary>The key name was listed in <see cref="SecretDetectorOptions.NeverRedactKeys"/>.</summary>
    NeverRedactKey,
}

/// <summary>What the detector concluded about one log attribute.</summary>
public sealed class DetectionResult
{
    internal DetectionResult(
        IReadOnlyDictionary<string, double> probabilities,
        string category,
        bool shouldRedact,
        double score,
        RedactionReason reason)
    {
        Probabilities = probabilities;
        Category = category;
        ShouldRedact = shouldRedact;
        Score = score;
        Reason = reason;
    }

    /// <summary>One probability per class, in the model's label order.</summary>
    public IReadOnlyDictionary<string, double> Probabilities { get; }

    /// <summary>The most likely class.</summary>
    public string Category { get; }

    /// <summary>Whether the attribute should be masked.</summary>
    public bool ShouldRedact { get; }

    /// <summary>The calibrated probability that the attribute is a credential.</summary>
    public double Score { get; }

    /// <summary>Which rule decided.</summary>
    /// <remarks>
    /// An override does not change <see cref="Probabilities"/>, so without this
    /// a caller could not tell a model decision from a configured one.
    /// </remarks>
    public RedactionReason Reason { get; }
}

/// <summary>Runtime options (plan §7.4's L1).</summary>
public sealed class SecretDetectorOptions
{
    /// <summary>Which threshold preset to operate at.</summary>
    public RedactionProfile Profile { get; set; } = RedactionProfile.Balanced;

    /// <summary>Attribute names always treated as secret.</summary>
    public IReadOnlyCollection<string> AlwaysRedactKeys { get; set; } = Array.Empty<string>();

    /// <summary>Attribute names never treated as secret.</summary>
    public IReadOnlyCollection<string> NeverRedactKeys { get; set; } = Array.Empty<string>();
}

/// <summary>
/// Classifies structured-log attributes as secret credentials, and masks them.
/// </summary>
/// <remarks>
/// <para>
/// <b>Experimental — not recommended for production.</b> Plan §1.2 makes this
/// release about being correct and runnable rather than good: NR-0004 measured
/// gate G2 as not met, with the best candidate reaching 0.804 recall at a 0.1%
/// false-positive rate against a bar of 0.834, and a false-positive rate
/// several times its target on format families it was not trained on. Treat it
/// as defence in depth, never as the only thing keeping credentials out of
/// logs.
/// </para>
/// <para>
/// An instance is read-only after construction and safe to share between
/// threads. <see cref="Classify"/> allocates the feature vectors it needs and
/// nothing else per call.
/// </para>
/// </remarks>
public sealed class SecretDetector
{
    private static readonly char[] MaskHexDigits = "0123456789abcdef".ToCharArray();

    private readonly NrmModel _model;
    private readonly HashSet<string> _alwaysRedact;
    private readonly HashSet<string> _neverRedact;
    private readonly double _threshold;

    private SecretDetector(NrmModel model, SecretDetectorOptions options)
    {
        _model = model;
        string profile = options.Profile == RedactionProfile.Conservative
            ? "conservative"
            : "balanced";
        if (!model.Thresholds.TryGetValue(profile, out double threshold))
        {
            throw new NrmException(
                $"this model carries no '{profile}' threshold; it has "
                    + string.Join(", ", SortedKeys(model.Thresholds)));
        }

        _threshold = threshold;

        // Key names compare case-insensitively, and the comparison is ordinal
        // rather than culture-aware: a Turkish 'I' must not change whether an
        // attribute called "ID" is redacted.
        _alwaysRedact = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string key in options.AlwaysRedactKeys)
        {
            _alwaysRedact.Add(key);
        }

        _neverRedact = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string key in options.NeverRedactKeys)
        {
            _neverRedact.Add(key);
        }
    }

    private static List<string> SortedKeys(IReadOnlyDictionary<string, double> source)
    {
        var keys = new List<string>(source.Keys);
        keys.Sort(StringComparer.Ordinal);
        return keys;
    }

    /// <summary>The class order <see cref="DetectionResult.Probabilities"/> uses.</summary>
    public IReadOnlyList<string> Labels => _model.Labels;

    /// <summary>The calibrated-probability threshold the active profile uses.</summary>
    public double Threshold => _threshold;

    /// <summary>Selected metadata from the loaded model file.</summary>
    public IReadOnlyDictionary<string, string> ModelMetadata => _model.Metadata;

    /// <summary>Loads a detector from a <c>.nrm</c> file.</summary>
    /// <param name="path">Path to the file.</param>
    /// <param name="options">Runtime options; defaults to the balanced profile.</param>
    /// <returns>The detector.</returns>
    /// <exception cref="NrmException">If the file is not one this build can read.</exception>
    public static SecretDetector Load(string path, SecretDetectorOptions? options = null) =>
        new(NrmModel.Load(path), options ?? new SecretDetectorOptions());

    /// <summary>Loads a detector from a stream.</summary>
    /// <param name="stream">A readable stream positioned at the file's start.</param>
    /// <param name="options">Runtime options; defaults to the balanced profile.</param>
    /// <returns>The detector.</returns>
    /// <exception cref="NrmException">If the file is not one this build can read.</exception>
    public static SecretDetector Load(Stream stream, SecretDetectorOptions? options = null) =>
        new(NrmModel.Load(stream), options ?? new SecretDetectorOptions());

    /// <summary>Classifies one log attribute.</summary>
    /// <param name="key">The attribute name.</param>
    /// <param name="value">The attribute value.</param>
    /// <param name="template">The message template it appeared in, if any.</param>
    /// <param name="scope">The scope name, if any.</param>
    /// <returns>The result.</returns>
    /// <remarks>
    /// The override lists are consulted <i>after</i> scoring rather than
    /// instead of it, so an overridden attribute still carries real
    /// probabilities and a caller can see what the model thought. Skipping the
    /// model would make an override invisible in any diagnostic.
    /// </remarks>
    public DetectionResult Classify(
        string key,
        string value,
        string? template = null,
        string? scope = null)
    {
        Guard.NotNull(key, nameof(key));

        Guard.NotNull(value, nameof(value));

        double[] probabilities = _model.ClassProbabilities(key, value, template, scope);
        double score = _model.RedactionScore(probabilities);

        int best = 0;
        for (int index = 1; index < probabilities.Length; index++)
        {
            if (probabilities[index] > probabilities[best])
            {
                best = index;
            }
        }

        bool decision;
        RedactionReason reason;

        // `never` loses to `always`. A caller listing a name in both is
        // contradicting themselves, and the library has to pick; it picks the
        // safer reading — a name marked as a secret stays a secret — rather
        // than depending on which list happens to be consulted first.
        if (_alwaysRedact.Contains(key))
        {
            decision = true;
            reason = RedactionReason.AlwaysRedactKey;
        }
        else if (_neverRedact.Contains(key))
        {
            decision = false;
            reason = RedactionReason.NeverRedactKey;
        }
        else
        {
            decision = score >= _threshold;
            reason = RedactionReason.Model;
        }

        var named = new Dictionary<string, double>(probabilities.Length, StringComparer.Ordinal);
        for (int index = 0; index < probabilities.Length; index++)
        {
            named[_model.Labels[index]] = probabilities[index];
        }

        return new DetectionResult(named, _model.Labels[best], decision, score, reason);
    }

    /// <summary>Classifies the candidate fragments of a rendered message body.</summary>
    /// <param name="message">The rendered message.</param>
    /// <returns>Each candidate fragment and its result, in order of appearance.</returns>
    /// <remarks>
    /// Body mode is attribute mode applied to fragments: the tokenizer decides
    /// what counts as a candidate, and each is classified exactly as an
    /// attribute would be with the whole message as its template.
    /// </remarks>
    public IReadOnlyList<KeyValuePair<string, DetectionResult>> ClassifyMessage(string message)
    {
        Guard.NotNull(message, nameof(message));

        var results = new List<KeyValuePair<string, DetectionResult>>();
        foreach (BodyCandidate candidate in BodyTokenizer.Candidates(message))
        {
            results.Add(new KeyValuePair<string, DetectionResult>(
                candidate.Value,
                Classify(candidate.Key, candidate.Value, message, candidate.Scope)));
        }

        return results;
    }

    /// <summary>Masks a value as <c>***#&lt;fingerprint&gt;</c>.</summary>
    /// <param name="value">The value to mask.</param>
    /// <returns>
    /// <c>***#</c> followed by the first four hex characters of the value's
    /// SHA-256.
    /// </returns>
    /// <remarks>
    /// The fingerprint lets the same value be correlated across log lines
    /// without being exposed. Four hex characters is sixteen bits, which is
    /// deliberately weak: enough to see "this is the same token as the
    /// previous line", and far too little to confirm a guess at the value,
    /// since roughly one value in 65,536 shares any fingerprint.
    /// </remarks>
    public static string Mask(string value)
    {
        Guard.NotNull(value, nameof(value));

#if NET
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
#else
        byte[] digest;
        using (var sha256 = SHA256.Create())
        {
            digest = sha256.ComputeHash(Encoding.UTF8.GetBytes(value));
        }
#endif

        // Four hex characters is two bytes; formatting by hand avoids both a
        // ToString("x2") allocation per byte and any dependence on the current
        // culture's digits.
        var masked = new char[8];
        masked[0] = '*';
        masked[1] = '*';
        masked[2] = '*';
        masked[3] = '#';
        masked[4] = MaskHexDigits[digest[0] >> 4];
        masked[5] = MaskHexDigits[digest[0] & 0x0F];
        masked[6] = MaskHexDigits[digest[1] >> 4];
        masked[7] = MaskHexDigits[digest[1] & 0x0F];
        return new string(masked);
    }

    /// <summary>Formats a score for a log line, invariantly.</summary>
    /// <param name="score">A calibrated score.</param>
    /// <returns>The score with four decimal places.</returns>
    public static string FormatScore(double score) =>
        score.ToString("F4", CultureInfo.InvariantCulture);
}
