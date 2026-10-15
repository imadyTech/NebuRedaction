using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace imady.NebuRedaction;

/// <summary>
/// The shared feature extractor, ported from the normative Python definition in
/// <c>python/src/neburedaction/features.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// Plan §6.4's G3 requires this implementation and the Python one to agree to
/// within 1e-6 per sample, and <c>tests/fixtures/feature-vectors-v1.json</c>
/// exists so the agreement is checked against recorded values rather than
/// argued from the code.
/// </para>
/// <para>
/// The hash is FNV-1a, not <see cref="string.GetHashCode()"/>. .NET randomises
/// string hashing per process by default, exactly as Python salts its own, so
/// using it would make a model unreadable by the next process and unportable
/// between the two languages. FNV-1a is specified in a dozen lines and is
/// byte-identical anywhere.
/// </para>
/// <para>Experimental, like the rest of this pre-release library.</para>
/// </remarks>
public static class Features
{
    /// <summary>The feature definition's version. A model fitted under another is refused.</summary>
    public const string Version = "1.0";

    private const uint FnvOffset = 0x811C9DC5;
    private const uint FnvPrime = 0x01000193;

    /// <summary>Smallest hashed-feature bucket count plan §4.2 allows.</summary>
    public const int MinBuckets = 1 << 14;

    /// <summary>Largest hashed-feature bucket count plan §4.2 allows.</summary>
    public const int MaxBuckets = 1 << 16;

    /// <summary>Namespace prefix for a value n-gram.</summary>
    public const string ValueNamespace = "v";

    /// <summary>Namespace prefix for a key word.</summary>
    public const string KeyNamespace = "k";

    /// <summary>Namespace prefix for a context word.</summary>
    public const string ContextNamespace = "c";

    /// <summary>
    /// Dense feature names, in the order a model's coefficients expect them.
    /// </summary>
    /// <remarks>
    /// Reordering this silently invalidates every trained model, which is why
    /// the feature definition carries a version and the loader checks it.
    /// </remarks>
    public static readonly string[] DenseFeatureNames =
    {
        "value_length_log", "value_entropy", "value_entropy_normalised",
        "ratio_lower", "ratio_upper", "ratio_digit", "ratio_symbol", "ratio_whitespace",
        "distinct_ratio", "longest_run_log", "class_transitions_ratio",
        "is_hex", "is_base64", "is_base64url", "is_uuid", "is_jwt_shaped",
        "has_pem_header", "has_url_credentials", "has_connection_string_password",
        "starts_with_eyj", "is_multiline", "has_dot_segments", "has_dash_segments",
        "has_underscore", "key_word_count", "key_has_sensitive_word",
        "key_has_neutralising_word", "key_is_slot", "template_has_sensitive_word",
        "scope_is_attribute", "has_template",
    };

    /// <summary>How many dense features there are.</summary>
    public static int DenseFeatureCount => DenseFeatureNames.Length;

    private static readonly HashSet<string> SensitiveKeyWords = new(StringComparer.Ordinal)
    {
        "apikey", "auth", "authorization", "credential", "credentials", "jwt", "key",
        "oauth", "passphrase", "passwd", "password", "pem", "pin", "pwd", "secret",
        "session", "sig", "signature", "token",
    };

    private static readonly HashSet<string> NeutralisingKeyWords = new(StringComparer.Ordinal)
    {
        "algorithm", "attempt", "attempts", "changed", "count", "enabled", "expired",
        "expires", "expiry", "failed", "format", "hash", "id", "kind", "last",
        "lastchanged", "length", "name", "policy", "prefix", "required", "rotation",
        "strength", "ttl", "type", "valid", "version",
    };

    /// <summary>Hashes a string with FNV-1a, 32-bit, over its UTF-8 bytes.</summary>
    /// <param name="text">The string to hash.</param>
    /// <returns>The 32-bit hash.</returns>
    /// <remarks>
    /// UTF-8 is the one detail a port has to match and the one a prose
    /// description usually omits.
    /// </remarks>
    public static uint Fnv1a(string text)
    {
        uint value = FnvOffset;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            value ^= b;
            value *= FnvPrime;
        }

        return value;
    }

    /// <summary>Maps a token to a bucket and a sign.</summary>
    /// <param name="token">The token.</param>
    /// <param name="ns">One of the namespace constants.</param>
    /// <param name="buckets">Bucket count; a power of two in the allowed range.</param>
    /// <param name="sign">Receives -1 or +1.</param>
    /// <returns>The bucket index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// If <paramref name="buckets"/> is outside plan §4.2's range or is not a
    /// power of two. A non-power-of-two would turn the mask into a modulo,
    /// which biases the low buckets.
    /// </exception>
    /// <remarks>
    /// The sign is the standard hashing-trick device: two colliding tokens
    /// cancel on average instead of always reinforcing, so a collision does not
    /// look like a strong signal. It is taken from a bit the bucket does not
    /// use, keeping the two independent.
    /// </remarks>
    public static int BucketOf(string token, string ns, int buckets, out double sign)
    {
        if (buckets < MinBuckets || buckets > MaxBuckets || (buckets & (buckets - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(buckets),
                buckets,
                $"buckets must be a power of two in [{MinBuckets}, {MaxBuckets}]");
        }

        uint digest = Fnv1a(ns + "\u001f" + token);
        sign = ((digest >> 31) & 1) == 1 ? 1.0 : -1.0;
        return (int)(digest & (uint)(buckets - 1));
    }

    /// <summary>Splits a key name into lower-case words.</summary>
    /// <param name="key">The attribute name.</param>
    /// <returns>The words, empties dropped.</returns>
    /// <remarks>
    /// Splits on camelCase, snake_case, kebab-case, dots and letter-to-digit
    /// boundaries, after stripping a <c>&lt;slot:...&gt;</c> marker so a
    /// body-mode fragment contributes the slot's name rather than punctuation.
    /// Written as a hand-rolled scanner rather than a regular expression,
    /// because .NET and Python disagree on enough regex details that matching
    /// behaviour is easier to guarantee by reading characters.
    /// </remarks>
    public static List<string> SplitKeyWords(string key)
    {
        string cleaned = StripSlot(key);
        var words = new List<string>();
        var current = new StringBuilder();

        for (int index = 0; index < cleaned.Length; index++)
        {
            char character = cleaned[index];
            if (!char.IsLetterOrDigit(character))
            {
                Flush(words, current);
                continue;
            }

            if (current.Length > 0)
            {
                char previous = cleaned[index - 1];
                bool camel = char.IsLower(previous) && char.IsUpper(character);
                bool digitAfterLetter = char.IsLetter(previous) && char.IsDigit(character);
                if (camel || digitAfterLetter)
                {
                    Flush(words, current);
                }
            }

            current.Append(char.ToLowerInvariant(character));
        }

        Flush(words, current);
        return words;
    }

    private static void Flush(List<string> words, StringBuilder current)
    {
        if (current.Length > 0)
        {
            words.Add(current.ToString());
            current.Clear();
        }
    }

    private static string StripSlot(string key)
    {
        int start = key.IndexOf("<slot:", StringComparison.Ordinal);
        if (start < 0)
        {
            return key;
        }

        var builder = new StringBuilder(key.Length);
        int position = 0;
        while (position < key.Length)
        {
            int open = key.IndexOf("<slot:", position, StringComparison.Ordinal);
            if (open < 0)
            {
                builder.Append(key, position, key.Length - position);
                break;
            }

            int close = key.IndexOf('>', open);
            if (close < 0)
            {
                builder.Append(key, position, key.Length - position);
                break;
            }

            builder.Append(key, position, open - position);
            builder.Append(key, open + 6, close - open - 6);
            position = close + 1;
        }

        return builder.ToString();
    }

    /// <summary>Splits a message template into lower-case context words.</summary>
    /// <param name="template">The template, or <see langword="null"/>.</param>
    /// <returns>The words, with placeholders removed first.</returns>
    /// <remarks>
    /// <c>&lt;*&gt;</c>, <c>{Name}</c>, <c>%s</c> and <c>$1</c> are removed
    /// before splitting: left in, they would be the corpus's most frequent
    /// context tokens while carrying no information.
    /// </remarks>
    public static List<string> TemplateWords(string? template)
    {
        if (string.IsNullOrEmpty(template))
        {
            return new List<string>();
        }

        return SplitKeyWords(RemovePlaceholders(template!));
    }

    private static string RemovePlaceholders(string template)
    {
        var builder = new StringBuilder(template.Length);
        int index = 0;
        while (index < template.Length)
        {
            char character = template[index];
            if (character == '<' && index + 2 < template.Length &&
                template[index + 1] == '*' && template[index + 2] == '>')
            {
                builder.Append(' ');
                index += 3;
                continue;
            }

            if (character == '{')
            {
                int close = template.IndexOf('}', index);
                if (close >= 0)
                {
                    builder.Append(' ');
                    index = close + 1;
                    continue;
                }
            }

            if (character == '%' && index + 1 < template.Length &&
                (template[index + 1] == 's' || template[index + 1] == 'd' || template[index + 1] == 'v'))
            {
                builder.Append(' ');
                index += 2;
                continue;
            }

            if (character == '$' && index + 1 < template.Length && char.IsDigit(template[index + 1]))
            {
                int end = index + 1;
                while (end < template.Length && char.IsDigit(template[end]))
                {
                    end++;
                }

                builder.Append(' ');
                index = end;
                continue;
            }

            builder.Append(character);
            index++;
        }

        return builder.ToString();
    }

    /// <summary>Shannon entropy of a string's characters, in bits.</summary>
    /// <param name="value">The string.</param>
    /// <returns>Entropy per character; 0 for an empty string.</returns>
    public static double ShannonEntropy(string value)
    {
        if (value.Length == 0)
        {
            return 0.0;
        }

        var counts = new Dictionary<char, int>();
        foreach (char character in value)
        {
            counts[character] = counts.TryGetValue(character, out int seen) ? seen + 1 : 1;
        }

        double total = value.Length;
        double entropy = 0.0;
        foreach (int count in counts.Values)
        {
            double share = count / total;
            entropy -= share * Log2(share);
        }

        return entropy;
    }

    private static char CharacterClass(char character)
    {
        if (char.IsLower(character))
        {
            return 'l';
        }

        if (char.IsUpper(character))
        {
            return 'u';
        }

        if (char.IsDigit(character))
        {
            return 'd';
        }

        return char.IsWhiteSpace(character) ? 'w' : 's';
    }

    /// <summary>Computes the morphological feature vector.</summary>
    /// <param name="key">The attribute name.</param>
    /// <param name="value">The attribute value.</param>
    /// <param name="template">The message template, if any.</param>
    /// <param name="scope">The scope name, if any.</param>
    /// <returns>Values in <see cref="DenseFeatureNames"/> order.</returns>
    /// <remarks>
    /// These describe shape, never content: none can be inverted to recover a
    /// value, which is what plan §7.5 item 4 asks the value side to be built
    /// from.
    /// </remarks>
    public static double[] Dense(string key, string value, string? template, string? scope)
    {
        int length = value.Length;
        string stripped = value.Trim();

        int lower = 0, upper = 0, digit = 0, symbol = 0, whitespace = 0;
        int transitions = 0, longestRun = 0, run = 0;
        char previous = '\0';
        bool first = true;
        var distinct = new HashSet<char>();

        foreach (char character in value)
        {
            char kind = CharacterClass(character);
            switch (kind)
            {
                case 'l': lower++; break;
                case 'u': upper++; break;
                case 'd': digit++; break;
                case 'w': whitespace++; break;
                default: symbol++; break;
            }

            if (!first && kind != previous)
            {
                transitions++;
            }

            run = !first && kind == previous ? run + 1 : 1;
            if (run > longestRun)
            {
                longestRun = run;
            }

            distinct.Add(character);
            previous = kind;
            first = false;
        }

        double entropy = ShannonEntropy(value);
        double ceiling = distinct.Count > 1 ? Log2(distinct.Count) : 1.0;

        List<string> words = SplitKeyWords(key);
        List<string> context = TemplateWords(template);

        double Ratio(int count) => length > 0 ? (double)count / length : 0.0;

        return new[]
        {
            Math.Log(1.0 + length),
            entropy,
            ceiling > 0 ? Math.Min(1.0, entropy / ceiling) : 0.0,
            Ratio(lower),
            Ratio(upper),
            Ratio(digit),
            Ratio(symbol),
            Ratio(whitespace),
            length > 0 ? (double)distinct.Count / length : 0.0,
            Math.Log(1.0 + longestRun),
            length > 1 ? (double)transitions / (length - 1) : 0.0,
            Flag(Shapes.IsHex(stripped) && length >= 8),
            Flag(Shapes.IsBase64(stripped) && length >= 8),
            Flag(Shapes.IsBase64Url(stripped) && length >= 8),
            Flag(Shapes.IsUuid(stripped)),
            Flag(Shapes.IsJwtShaped(stripped) && Count(stripped, '.') == 2),
            Flag(value.Contains("-----BEGIN", StringComparison.Ordinal)),
            Flag(Shapes.HasUrlCredentials(value)),
            Flag(ContainsConnectionStringPassword(value)),
            Flag(stripped.StartsWith("eyJ", StringComparison.Ordinal)),
            Flag(value.Contains('\n')),
            Flag(Count(stripped, '.') >= 2),
            Flag(Count(stripped, '-') >= 2),
            Flag(stripped.Contains('_')),
            words.Count,
            Flag(AnyIn(words, SensitiveKeyWords)),
            Flag(AnyIn(words, NeutralisingKeyWords)),
            Flag(key.StartsWith("<slot:", StringComparison.Ordinal)),
            Flag(AnyIn(context, SensitiveKeyWords)),
            Flag((scope ?? string.Empty).StartsWith("otel.attribute", StringComparison.Ordinal)),
            Flag(!string.IsNullOrEmpty(template)),
        };
    }

    private static bool ContainsConnectionStringPassword(string value)
    {
        string lowered = value.ToLowerInvariant();
        return lowered.Contains(";password=", StringComparison.Ordinal)
            || lowered.Contains(";pwd=", StringComparison.Ordinal);
    }

    private static bool AnyIn(List<string> words, HashSet<string> vocabulary)
    {
        foreach (string word in words)
        {
            if (vocabulary.Contains(word))
            {
                return true;
            }
        }

        return false;
    }

    private static int Count(string text, char character)
    {
        int count = 0;
        foreach (char candidate in text)
        {
            if (candidate == character)
            {
                count++;
            }
        }

        return count;
    }

    private static double Flag(bool condition) => condition ? 1.0 : 0.0;

    /// <summary>Computes the hashed feature map.</summary>
    /// <param name="key">The attribute name.</param>
    /// <param name="value">The attribute value.</param>
    /// <param name="template">The message template, if any.</param>
    /// <param name="scope">The scope name, if any.</param>
    /// <param name="buckets">Bucket count.</param>
    /// <returns>Bucket index to accumulated signed count; absent buckets are zero.</returns>
    /// <remarks>
    /// The three families are namespaced, so the token <c>token</c> appearing
    /// as a key word and in a template are different features rather than the
    /// same one counted twice.
    /// </remarks>
    public static Dictionary<int, double> Hashed(
        string key, string value, string? template, string? scope, int buckets)
    {
        var accumulated = new Dictionary<int, double>();

        void Add(string token, string ns)
        {
            int bucket = BucketOf(token, ns, buckets, out double sign);
            accumulated[bucket] = accumulated.TryGetValue(bucket, out double seen)
                ? seen + sign
                : sign;
        }

        for (int size = 1; size <= 4; size++)
        {
            for (int start = 0; start + size <= value.Length; start++)
            {
                Add(value.Substring(start, size), ValueNamespace);
            }
        }

        foreach (string word in SplitKeyWords(key))
        {
            Add(word, KeyNamespace);
        }

        foreach (string word in TemplateWords(template))
        {
            Add(word, ContextNamespace);
        }

        foreach (string word in SplitKeyWords(scope ?? string.Empty))
        {
            Add(word, ContextNamespace);
        }

        return accumulated;
    }

    /// <summary>FNV-1a state after consuming a namespace and its separator.</summary>
    /// <param name="ns">One of the namespace constants.</param>
    /// <returns>The intermediate hash state.</returns>
    /// <remarks>
    /// FNV-1a is a streaming hash, so hashing the namespace, a U+001F
    /// separator and the token as one string and
    /// hashing the namespace, the separator and then the token give the same
    /// result by construction. Precomputing the first part is therefore exact,
    /// not an approximation, and it is what lets the hot path hash an n-gram
    /// without building that concatenated string.
    /// </remarks>
    internal static uint NamespaceSeed(string ns)
    {
        uint value = FnvOffset;
        foreach (byte b in Encoding.UTF8.GetBytes(ns))
        {
            value ^= b;
            value *= FnvPrime;
        }

        value ^= 0x1F;
        value *= FnvPrime;
        return value;
    }

    /// <summary>Maps a token held in a span to a bucket and a sign.</summary>
    /// <param name="seed">A <see cref="NamespaceSeed"/> result.</param>
    /// <param name="token">The token's characters.</param>
    /// <param name="bucketMask"><c>buckets - 1</c>, validated by the caller.</param>
    /// <param name="sign">Receives -1 or +1.</param>
    /// <returns>The bucket index.</returns>
    /// <remarks>
    /// Equivalent to <see cref="BucketOf"/> with the same namespace and token,
    /// and allocation-free: the UTF-8 bytes go into a stack buffer sized for
    /// the longest token this extractor produces.
    /// </remarks>
    internal static int BucketOfSpan(
        uint seed, ReadOnlySpan<char> token, int bucketMask, out double sign)
    {
        // Four characters is the longest n-gram, and UTF-8 needs at most four
        // bytes per character, so 16 bytes covers every token including one
        // made entirely of surrogate pairs. Longer tokens (key and template
        // words) take the heap path below.
        uint value = seed;
        if (token.Length <= 4)
        {
            Span<byte> buffer = stackalloc byte[16];
            int written = Encoding.UTF8.GetBytes(token, buffer);
            for (int index = 0; index < written; index++)
            {
                value ^= buffer[index];
                value *= FnvPrime;
            }
        }
        else
        {
            foreach (byte b in Encoding.UTF8.GetBytes(token.ToString()))
            {
                value ^= b;
                value *= FnvPrime;
            }
        }

        sign = ((value >> 31) & 1) == 1 ? 1.0 : -1.0;
        return (int)(value & (uint)bucketMask);
    }

    /// <summary>
    /// Accumulates the hashed features for one unit directly into a projected
    /// vector, skipping the buckets the model does not use.
    /// </summary>
    /// <param name="key">The attribute name.</param>
    /// <param name="value">The attribute value.</param>
    /// <param name="template">The message template, if any.</param>
    /// <param name="scope">The scope name, if any.</param>
    /// <param name="buckets">Bucket count.</param>
    /// <param name="columnOfBucket">Bucket index to destination column.</param>
    /// <param name="destination">The vector to accumulate into.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// If <paramref name="buckets"/> is outside plan §4.2's range or is not a
    /// power of two.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This exists because <see cref="Hashed"/> allocates: a 512-character
    /// value has 2,042 n-grams, and building a substring, a namespaced
    /// concatenation and a UTF-8 byte array for each one cost about 200 KB a
    /// call. The measured model reads 256 buckets of 32,768, so the dictionary
    /// those allocations filled was discarded almost entirely.
    /// </para>
    /// <para>
    /// Dropping the unused buckets is exact rather than an approximation only
    /// because the hashed features are not normalised — each bucket's signed
    /// count is independent of the others, so a bucket no tree reads cannot
    /// affect one that is read. If a later feature version normalises them,
    /// this shortcut stops being valid and must go back through
    /// <see cref="Hashed"/>.
    /// </para>
    /// </remarks>
    internal static void AccumulateHashed(
        string key,
        string value,
        string? template,
        string? scope,
        int buckets,
        Dictionary<int, int> columnOfBucket,
        double[] destination)
    {
        if (buckets < MinBuckets || buckets > MaxBuckets || (buckets & (buckets - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(buckets),
                buckets,
                $"buckets must be a power of two in [{MinBuckets}, {MaxBuckets}]");
        }

        int mask = buckets - 1;
        uint valueSeed = NamespaceSeed(ValueNamespace);
        uint keySeed = NamespaceSeed(KeyNamespace);
        uint contextSeed = NamespaceSeed(ContextNamespace);

        void Add(uint seed, ReadOnlySpan<char> token)
        {
            int bucket = BucketOfSpan(seed, token, mask, out double sign);
            if (columnOfBucket.TryGetValue(bucket, out int column))
            {
                destination[column] += sign;
            }
        }

        ReadOnlySpan<char> characters = value.AsSpan();
        for (int size = 1; size <= 4; size++)
        {
            for (int start = 0; start + size <= characters.Length; start++)
            {
                Add(valueSeed, characters.Slice(start, size));
            }
        }

        foreach (string word in SplitKeyWords(key))
        {
            Add(keySeed, word.AsSpan());
        }

        foreach (string word in TemplateWords(template))
        {
            Add(contextSeed, word.AsSpan());
        }

        foreach (string word in SplitKeyWords(scope ?? string.Empty))
        {
            Add(contextSeed, word.AsSpan());
        }
    }

    /// <summary>Base-2 logarithm, available on both target frameworks.</summary>
    /// <param name="value">A positive number.</param>
    /// <returns>Its base-2 logarithm.</returns>
    /// <remarks>
    /// <c>Math.Log2</c> is .NET Core 3.0 and later, and this library also
    /// targets netstandard2.1. The fallback divides natural logarithms, which
    /// can differ from a dedicated <c>log2</c> in the last bit or two — a
    /// relative error around 1e-16, against G3's tolerance of 1e-6, so it is
    /// nine orders of magnitude from mattering. It is written as a division
    /// rather than a multiplication by the reciprocal of ln 2 because dividing
    /// keeps one rounding step instead of two.
    /// </remarks>
    private static double Log2(double value)
    {
#if NETSTANDARD2_1
        return Math.Log(value) / Math.Log(2.0);
#else
        return Math.Log2(value);
#endif
    }

    /// <summary>Formats a double the way the test vectors record it.</summary>
    /// <param name="value">The value.</param>
    /// <returns>A round-trippable decimal string.</returns>
    internal static string Format(double value) =>
        value.ToString("R", CultureInfo.InvariantCulture);
}
