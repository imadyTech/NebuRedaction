using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace imady.NebuRedaction;

/// <summary>One fragment of a message body.</summary>
public sealed class BodyCandidate
{
    internal BodyCandidate(string key, string value, string scope)
    {
        Key = key;
        Value = value;
        Scope = scope;
    }

    /// <summary>The adjacent key name, or <c>&lt;slot:value&gt;</c> when there is none.</summary>
    public string Key { get; }

    /// <summary>The fragment.</summary>
    public string Value { get; }

    /// <summary><c>message.kv</c> when a key carried it, <c>message.slot</c> otherwise.</summary>
    public string Scope { get; }
}

/// <summary>
/// Splits a rendered message into the fragments worth classifying (NR-0005
/// §2.4, from NR-0003 §2.0).
/// </summary>
/// <remarks>
/// <para>
/// A port of <c>neburedaction.body</c>, which is the normative definition.
/// NR-0001's export produced 1,545 body-mode units of which 1,510 were prose
/// words, so a fragment is kept only when something points at it being a value:
/// an assignment, enclosing quotes, a key-like name, or its own character
/// composition.
/// </para>
/// <para>
/// Unlike <see cref="Features"/>, this uses <c>System.Text.RegularExpressions</c>
/// rather than a hand-rolled scanner. The feature extractor runs on every unit
/// and its patterns are simple enough to hand-roll without risking divergence;
/// this one has an optional capture group feeding an alternation, where
/// re-implementing the backtracking by hand would be the more likely source of
/// a mismatch with Python. It is in-box on both target frameworks, so the
/// zero-dependency requirement is untouched.
/// </para>
/// </remarks>
public static class BodyTokenizer
{
    private static readonly Regex Candidate = new(
        @"(?:(?<key>[A-Za-z_][A-Za-z0-9_.\-]{0,40})\s*(?<sep>[=:])\s*)?"
            + @"(?:""(?<quoted>[^""]{8,})""|(?<value>[A-Za-z0-9+/_\-.:~%@]{8,}))",
        RegexOptions.CultureInvariant);

    /// <summary>Trailing sentence punctuation the pattern absorbs.</summary>
    /// <remarks>Not <c>=</c>, which is base64 padding.</remarks>
    private static readonly char[] TrailingPunctuation = ":.,;!?)".ToCharArray();

    private static readonly Regex ProseWord = new(@"^[A-Za-z]+$", RegexOptions.CultureInvariant);

    private static readonly Regex[] Unmistakable =
    [
        new(@"^[0-9a-fA-F]{16,}$", RegexOptions.CultureInvariant),
        new(
            @"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            RegexOptions.CultureInvariant),
        new(@"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$", RegexOptions.CultureInvariant),
        new(@"^[a-z][a-z0-9+.-]*://", RegexOptions.CultureInvariant),
        new(@"^-----BEGIN ", RegexOptions.CultureInvariant),
    ];

    /// <summary>
    /// Shortest pure-letter fragment that may still be a value, and only in
    /// mixed case.
    /// </summary>
    private const int PureLetterValueLength = 20;

    /// <summary>
    /// Key names that make a following fragment worth keeping whatever it looks
    /// like, so <c>password: hunter</c> survives.
    /// </summary>
    private static readonly HashSet<string> CredentialKeyWords = new(StringComparer.Ordinal)
    {
        "apikey", "auth", "authorization", "cookie", "credential", "credentials", "key",
        "passwd", "password", "pwd", "secret", "session", "signature", "token",
    };

    /// <summary>Decides whether a fragment looks like a value rather than prose.</summary>
    /// <param name="value">The fragment, already stripped of trailing punctuation.</param>
    /// <returns>
    /// True if its character composition is that of an identifier, digest or
    /// credential.
    /// </returns>
    /// <remarks>
    /// This will reject real secrets, and that is the trade NR-0003 §2.0 asks
    /// for. A password can be a single lowercase word, but with no key name and
    /// no delimiter there is no evidence to label on.
    /// </remarks>
    public static bool HasValueShape(string value)
    {
        Guard.NotNull(value, nameof(value));

        foreach (Regex pattern in Unmistakable)
        {
            if (pattern.IsMatch(value))
            {
                return true;
            }
        }

        bool mixedCase = !string.Equals(value, value.ToLowerInvariant(), StringComparison.Ordinal)
            && !string.Equals(value, value.ToUpperInvariant(), StringComparison.Ordinal);
        if (ProseWord.IsMatch(value))
        {
            return value.Length >= PureLetterValueLength && mixedCase;
        }

        bool hasDigit = false;
        bool hasLetter = false;
        bool separated = false;
        foreach (char character in value)
        {
            // char.IsDigit is Unicode category Nd, matching Python's
            // str.isdigit() for every character the candidate pattern admits.
            // A quoted fragment can hold anything, and there Python also
            // accepts a handful of non-Nd digits such as '²'; the two
            // definitions differ on those and nowhere else.
            hasDigit |= char.IsDigit(character);
            hasLetter |= char.IsLetter(character);
            separated |= IsSeparator(character);
        }

        if (value.Length >= 8 && hasDigit && hasLetter)
        {
            return true;
        }

        return value.Length >= 16 && (separated || mixedCase);
    }

    /// <summary>Whether a character is one of Python's <c>"+/_-.:~%@"</c> separators.</summary>
    /// <param name="character">The character.</param>
    /// <returns>True when it is a separator.</returns>
    private static bool IsSeparator(char character) =>
        character is '+' or '/' or '_' or '-' or '.' or ':' or '~' or '%' or '@';

    /// <summary>Decides whether a prefix reads as an attribute name rather than a word.</summary>
    /// <param name="key">The candidate key.</param>
    /// <returns>
    /// True for <c>access_token</c>, <c>db.host</c>, <c>retry2</c> and bare
    /// credential words; false for an ordinary word such as <c>received</c>.
    /// </returns>
    public static bool IsKeyLike(string key)
    {
        Guard.NotNull(key, nameof(key));

        if (CredentialKeyWords.Contains(key.ToLowerInvariant()))
        {
            return true;
        }

        foreach (char character in key)
        {
            if (character == '_' || character == '.' || character == '-' || char.IsDigit(character))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Decides whether a fragment becomes a detection unit.</summary>
    /// <param name="key">An adjacent key-like prefix, if the pattern found one.</param>
    /// <param name="separator">The delimiter between key and value, <c>=</c> or <c>:</c>.</param>
    /// <param name="quoted">Whether the fragment was enclosed in double quotes.</param>
    /// <param name="value">The fragment, stripped of trailing punctuation.</param>
    /// <returns>True if the fragment should be kept.</returns>
    /// <remarks>
    /// The two delimiters are not equally informative. <c>=</c> does not occur
    /// between two words of prose, so an assignment is evidence on its own.
    /// <c>:</c> occurs in prose constantly: <c>ack received: downstream
    /// authorised</c> parses as key <c>received</c>, value <c>downstream</c>,
    /// and admitting it reintroduces exactly the units this tokenizer exists to
    /// remove.
    /// </remarks>
    public static bool IsCandidate(string? key, string? separator, bool quoted, string value)
    {
        Guard.NotNull(value, nameof(value));

        if (value.Length < 8)
        {
            return false;
        }

        if (quoted)
        {
            return true;
        }

        if (key is not null && separator == "=")
        {
            return true;
        }

        if (key is not null && separator == ":" && IsKeyLike(key))
        {
            return true;
        }

        return HasValueShape(value);
    }

    /// <summary>Splits a message body into the fragments worth classifying.</summary>
    /// <param name="message">The rendered message.</param>
    /// <returns>The candidates, in order, without duplicates.</returns>
    public static IReadOnlyList<BodyCandidate> Candidates(string message)
    {
        Guard.NotNull(message, nameof(message));

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var candidates = new List<BodyCandidate>();
        foreach (Match match in Candidate.Matches(message))
        {
            bool quoted = match.Groups["quoted"].Success;
            string raw = quoted ? match.Groups["quoted"].Value : match.Groups["value"].Value;
            string value = raw.TrimEnd(TrailingPunctuation);
            Group keyGroup = match.Groups["key"];
            string? key = keyGroup.Success ? keyGroup.Value : null;
            string? separator = match.Groups["sep"].Success ? match.Groups["sep"].Value : null;
            if (!IsCandidate(key, separator, quoted, value) || !seen.Add(value))
            {
                continue;
            }

            // Python treats an empty key group as absent (`if key else`), which
            // the pattern cannot produce — its first character class is
            // non-optional — so a successful group is always a real key.
            candidates.Add(new BodyCandidate(
                key ?? "<slot:value>",
                value,
                key is null ? "message.slot" : "message.kv"));
        }

        return candidates;
    }
}
