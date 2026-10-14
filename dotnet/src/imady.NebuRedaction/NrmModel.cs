using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace imady.NebuRedaction;

/// <summary>
/// Thrown when a <c>.nrm</c> file cannot be loaded.
/// </summary>
/// <remarks>
/// Deliberately not an <see cref="ArgumentException"/>: every case this covers
/// is a refusal to use a file, not a complaint about a parameter, and a caller
/// catching argument exceptions around a load should not swallow "this model
/// was fitted under a different feature definition".
/// </remarks>
public sealed class NrmException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What was wrong with the file.</param>
    public NrmException(string message)
        : base(message)
    {
    }
}

/// <summary>
/// A gradient-boosted tree from a <c>.nrm</c> file.
/// </summary>
internal sealed class NrmTree
{
    internal int ClassIndex;
    internal ushort[] Feature = Array.Empty<ushort>();
    internal float[] Threshold = Array.Empty<float>();
    internal short[] Left = Array.Empty<short>();
    internal short[] Right = Array.Empty<short>();
    internal float[] Leaf = Array.Empty<float>();

    /// <summary>Walks the tree for one projected feature vector.</summary>
    /// <param name="values">The projected feature vector.</param>
    /// <returns>The leaf value reached.</returns>
    /// <remarks>
    /// A split goes left on <c>value &lt;= threshold</c>, which is the
    /// convention the exporter preserved from the trainer. Getting the
    /// direction wrong is invisible on most units and wrong on exactly those
    /// sitting on a split.
    /// </remarks>
    internal double Evaluate(double[] values)
    {
        if (Feature.Length == 0)
        {
            return Leaf[0];
        }

        int node = 0;
        while (true)
        {
            short child = values[Feature[node]] <= Threshold[node] ? Left[node] : Right[node];
            if (child < 0)
            {
                return Leaf[~child];
            }

            node = child;
        }
    }
}

/// <summary>
/// A model loaded from a <c>.nrm</c> v1 file.
/// </summary>
/// <remarks>
/// <para>
/// The format is specified in <c>docs/spec/nrm-v1.md</c>. This reader and the
/// Python one are the two implementations plan §6.4's G3 requires to agree to
/// within 1e-6 per sample, so the places they could differ are pinned here:
/// scores accumulate in <see cref="double"/> even though leaf values are
/// stored as <see cref="float"/>, trees are summed in file order because
/// floating-point addition is not associative, and the logit clamps at the
/// same 1e-9.
/// </para>
/// <para>Experimental, like the rest of this pre-release library.</para>
/// </remarks>
public sealed class NrmModel
{
    /// <summary>The file's magic bytes.</summary>
    internal static readonly byte[] Magic = { (byte)'N', (byte)'R', (byte)'M', (byte)'1' };

    /// <summary>The format version this reader implements.</summary>
    public const int SupportedFormatVersion = 1;

    /// <summary>
    /// Clamp applied inside the logit so a saturated probability does not
    /// produce an infinity. The Python reader uses the same number; a
    /// different clamp is a different score for a saturated unit.
    /// </summary>
    public const double LogitEpsilon = 1e-9;

    private readonly NrmTree[] _trees;
    private readonly int[] _bucketIds;
    private readonly int _denseFeatureCount;
    private readonly int _noneIndex;

    private NrmModel(
        IReadOnlyDictionary<string, string> metadataStrings,
        string[] labels,
        NrmTree[] trees,
        int denseFeatureCount,
        int[] bucketIds,
        int bucketCount,
        double slope,
        double intercept,
        IReadOnlyDictionary<string, double> thresholds)
    {
        Metadata = metadataStrings;
        Labels = labels;
        _trees = trees;
        _denseFeatureCount = denseFeatureCount;
        _bucketIds = bucketIds;
        BucketCount = bucketCount;
        CalibrationSlope = slope;
        CalibrationIntercept = intercept;
        Thresholds = thresholds;
        _noneIndex = Array.IndexOf(labels, "none");
        if (_noneIndex < 0)
        {
            throw new NrmException("the model's label list has no 'none' class");
        }
    }

    /// <summary>Selected metadata fields, as strings.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>The class order every probability array is in.</summary>
    public string[] Labels { get; }

    /// <summary>The hashing space the model's buckets index.</summary>
    public int BucketCount { get; }

    /// <summary>The Platt calibration's slope.</summary>
    public double CalibrationSlope { get; }

    /// <summary>The Platt calibration's intercept.</summary>
    public double CalibrationIntercept { get; }

    /// <summary>Threshold presets, in calibrated probability space.</summary>
    public IReadOnlyDictionary<string, double> Thresholds { get; }

    /// <summary>How many trees the ensemble holds.</summary>
    public int TreeCount => _trees.Length;

    /// <summary>Loads a model from a file.</summary>
    /// <param name="path">The <c>.nrm</c> file.</param>
    /// <returns>The model.</returns>
    /// <exception cref="NrmException">If the file is not one this build can read.</exception>
    public static NrmModel Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Loads a model from a stream.</summary>
    /// <param name="stream">A readable stream positioned at the file's start.</param>
    /// <returns>The model.</returns>
    /// <exception cref="NrmException">If the file is not one this build can read.</exception>
    public static NrmModel Load(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        byte[] magic = ReadBytes(reader, 4, "magic");
        for (int index = 0; index < 4; index++)
        {
            if (magic[index] != Magic[index])
            {
                throw new NrmException("not a .nrm file: wrong magic");
            }
        }

        ushort formatVersion = BitConverter.ToUInt16(Read(reader, 2, "format version"), 0);
        if (formatVersion != SupportedFormatVersion)
        {
            throw new NrmException(
                $"file is format version {formatVersion}; this reader implements {SupportedFormatVersion}");
        }

        uint metadataLength = BitConverter.ToUInt32(Read(reader, 4, "metadata length"), 0);
        string metadataText = Encoding.UTF8.GetString(
            ReadBytes(reader, checked((int)metadataLength), "metadata"));

        JsonValue root = Json.Parse(metadataText);

        string modelType = root.RequiredString("model_type");
        if (modelType != "gbdt")
        {
            throw new NrmException(
                $"file has model_type '{modelType}'; this reader implements 'gbdt' only");
        }

        string featureVersion = root.RequiredString("feature_version");
        if (featureVersion != Features.Version)
        {
            throw new NrmException(
                $"file was fitted under feature version '{featureVersion}' but this build "
                    + $"produces '{Features.Version}'; the columns would not line up");
        }

        var labels = new List<string>();
        foreach (JsonValue label in root.RequiredArray("labels").Items)
        {
            labels.Add(label.AsString());
        }

        JsonValue calibration = root.RequiredObject("calibration");
        string method = calibration.RequiredString("method");
        if (method != "platt")
        {
            throw new NrmException(
                $"file's calibration is '{method}'; .nrm v1 stores Platt parameters");
        }

        var thresholds = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, JsonValue> entry in root.RequiredObject("thresholds").Members)
        {
            thresholds[entry.Key] = entry.Value.AsNumber();
        }

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in new[]
        {
            "model_type", "feature_version", "base_model_version", "license",
            "training_data_sha256",
        })
        {
            string? value = root.OptionalString(name);
            if (value is not null)
            {
                metadata[name] = value;
            }
        }

        uint classCount = BitConverter.ToUInt32(Read(reader, 4, "class count"), 0);
        uint treeCount = BitConverter.ToUInt32(Read(reader, 4, "tree count"), 0);
        uint denseCount = BitConverter.ToUInt32(Read(reader, 4, "dense feature count"), 0);
        uint hashedCount = BitConverter.ToUInt32(Read(reader, 4, "hashed column count"), 0);
        uint bucketCount = BitConverter.ToUInt32(Read(reader, 4, "bucket count"), 0);

        if (labels.Count != classCount)
        {
            throw new NrmException(
                $"file declares {labels.Count} labels but {classCount} classes of trees");
        }

        var bucketIds = new int[hashedCount];
        for (int index = 0; index < hashedCount; index++)
        {
            bucketIds[index] = BitConverter.ToInt32(Read(reader, 4, "bucket id"), 0);
        }

        var trees = new NrmTree[treeCount];
        for (int index = 0; index < treeCount; index++)
        {
            var tree = new NrmTree
            {
                ClassIndex = BitConverter.ToUInt16(Read(reader, 2, "tree class"), 0),
            };
            int internalCount = BitConverter.ToUInt16(Read(reader, 2, "internal count"), 0);
            int leafCount = BitConverter.ToUInt16(Read(reader, 2, "leaf count"), 0);

            tree.Feature = new ushort[internalCount];
            tree.Threshold = new float[internalCount];
            tree.Left = new short[internalCount];
            tree.Right = new short[internalCount];
            for (int node = 0; node < internalCount; node++)
            {
                tree.Feature[node] = BitConverter.ToUInt16(Read(reader, 2, "node feature"), 0);
                tree.Threshold[node] = BitConverter.ToSingle(Read(reader, 4, "node threshold"), 0);
                tree.Left[node] = BitConverter.ToInt16(Read(reader, 2, "node left"), 0);
                tree.Right[node] = BitConverter.ToInt16(Read(reader, 2, "node right"), 0);
            }

            tree.Leaf = new float[leafCount];
            for (int leaf = 0; leaf < leafCount; leaf++)
            {
                tree.Leaf[leaf] = BitConverter.ToSingle(Read(reader, 4, "leaf value"), 0);
            }

            trees[index] = tree;
        }

        return new NrmModel(
            metadata,
            labels.ToArray(),
            trees,
            checked((int)denseCount),
            bucketIds,
            checked((int)bucketCount),
            calibration.RequiredNumber("slope"),
            calibration.RequiredNumber("intercept"),
            thresholds);
    }

    /// <summary>Reads exactly <paramref name="count"/> bytes, unaltered.</summary>
    private static byte[] ReadBytes(BinaryReader reader, int count, string what)
    {
        byte[] buffer = reader.ReadBytes(count);
        if (buffer.Length != count)
        {
            throw new NrmException($"file truncated while reading {what}");
        }

        return buffer;
    }

    /// <summary>Reads one little-endian scalar's bytes, in host order.</summary>
    /// <remarks>
    /// Only for numbers. An earlier version of this method reversed on a
    /// big-endian host for *every* read, which would have reversed the four
    /// magic bytes and the whole metadata blob — corrupting both on exactly
    /// the hardware the reversal was meant to support. Text and numbers need
    /// different readers, so there are two.
    /// </remarks>
    private static byte[] Read(BinaryReader reader, int count, string what)
    {
        byte[] buffer = ReadBytes(reader, count, what);
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(buffer);
        }

        return buffer;
    }

    /// <summary>Builds the projected feature vector for one unit.</summary>
    /// <param name="key">Attribute name.</param>
    /// <param name="value">Attribute value.</param>
    /// <param name="template">Message template, if any.</param>
    /// <param name="scope">Scope name, if any.</param>
    /// <returns>The projected vector the trees index.</returns>
    public double[] Project(string key, string value, string? template, string? scope)
    {
        double[] dense = Features.Dense(key, value, template, scope);
        Dictionary<int, double> hashed = Features.Hashed(key, value, template, scope, BucketCount);

        var projected = new double[_denseFeatureCount + _bucketIds.Length];
        int copy = Math.Min(_denseFeatureCount, dense.Length);
        Array.Copy(dense, projected, copy);
        for (int index = 0; index < _bucketIds.Length; index++)
        {
            projected[_denseFeatureCount + index] =
                hashed.TryGetValue(_bucketIds[index], out double weight) ? weight : 0.0;
        }

        return projected;
    }

    /// <summary>Scores one unit, returning a probability per class.</summary>
    /// <param name="key">Attribute name.</param>
    /// <param name="value">Attribute value.</param>
    /// <param name="template">Message template, if any.</param>
    /// <param name="scope">Scope name, if any.</param>
    /// <returns>One probability per entry of <see cref="Labels"/>.</returns>
    public double[] ClassProbabilities(string key, string value, string? template, string? scope)
    {
        double[] projected = Project(key, value, template, scope);
        var scores = new double[Labels.Length];

        // File order, accumulating in double. Both are required by the
        // specification: float accumulation loses G3's 1e-6 agreement, and a
        // different summation order is a different answer.
        foreach (NrmTree tree in _trees)
        {
            scores[tree.ClassIndex] += tree.Evaluate(projected);
        }

        double largest = scores[0];
        for (int index = 1; index < scores.Length; index++)
        {
            if (scores[index] > largest)
            {
                largest = scores[index];
            }
        }

        double total = 0.0;
        for (int index = 0; index < scores.Length; index++)
        {
            scores[index] = Math.Exp(scores[index] - largest);
            total += scores[index];
        }

        for (int index = 0; index < scores.Length; index++)
        {
            scores[index] /= total;
        }

        return scores;
    }

    /// <summary>Calibrates the probability that a unit should be redacted.</summary>
    /// <param name="probabilities">Per-class probabilities.</param>
    /// <returns>The calibrated redaction probability.</returns>
    public double RedactionScore(double[] probabilities)
    {
        double raw = 1.0 - probabilities[_noneIndex];
        double clamped = Math.Min(Math.Max(raw, LogitEpsilon), 1.0 - LogitEpsilon);
        double logit = Math.Log(clamped / (1.0 - clamped));
        return 1.0 / (1.0 + Math.Exp(-((CalibrationSlope * logit) + CalibrationIntercept)));
    }
}
