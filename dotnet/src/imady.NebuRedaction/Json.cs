using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace imady.NebuRedaction;

/// <summary>
/// A minimal JSON reader for a <c>.nrm</c> file's metadata.
/// </summary>
/// <remarks>
/// <para>
/// <c>System.Text.Json</c> would be the obvious choice and is not available:
/// it is a NuGet package on <c>netstandard2.1</c>, and NR-0005 §2.2 item 1
/// requires the library to have **zero runtime dependencies**. That
/// requirement is worth the hundred lines here — a redaction library that
/// drags transitive packages into every logging pipeline it is added to is a
/// harder thing to adopt than one that does not.
/// </para>
/// <para>
/// Deliberately strict and deliberately small. It parses the subset the
/// exporter writes — objects, arrays, strings, numbers, booleans and null —
/// and refuses anything else rather than guessing. It is **not** a
/// general-purpose JSON parser and should not be used as one: no streaming, no
/// comments, no big-number handling, and the whole document is held in memory,
/// which is fine for a few kilobytes of metadata written by this project's own
/// exporter and wrong for anything arriving from outside.
/// </para>
/// </remarks>
internal static class Json
{
    /// <summary>Parses a JSON document.</summary>
    /// <param name="text">The document.</param>
    /// <returns>The root value.</returns>
    /// <exception cref="NrmException">If the text is not JSON this reader accepts.</exception>
    internal static JsonValue Parse(string text)
    {
        int position = 0;
        JsonValue value = ParseValue(text, ref position);
        SkipWhitespace(text, ref position);
        if (position != text.Length)
        {
            throw new NrmException($"trailing content in metadata at offset {position}");
        }

        return value;
    }

    private static JsonValue ParseValue(string text, ref int position)
    {
        SkipWhitespace(text, ref position);
        if (position >= text.Length)
        {
            throw new NrmException("metadata ended unexpectedly");
        }

        char character = text[position];
        switch (character)
        {
            case '{':
                return ParseObject(text, ref position);
            case '[':
                return ParseArray(text, ref position);
            case '"':
                return JsonValue.FromString(ParseString(text, ref position));
            case 't':
                Expect(text, ref position, "true");
                return JsonValue.FromBool(true);
            case 'f':
                Expect(text, ref position, "false");
                return JsonValue.FromBool(false);
            case 'n':
                Expect(text, ref position, "null");
                return JsonValue.Null;
            default:
                return JsonValue.FromNumber(ParseNumber(text, ref position));
        }
    }

    private static JsonValue ParseObject(string text, ref int position)
    {
        var members = new Dictionary<string, JsonValue>(StringComparer.Ordinal);
        position++;
        SkipWhitespace(text, ref position);
        if (position < text.Length && text[position] == '}')
        {
            position++;
            return JsonValue.FromObject(members);
        }

        while (true)
        {
            SkipWhitespace(text, ref position);
            if (position >= text.Length || text[position] != '"')
            {
                throw new NrmException($"expected a member name at offset {position}");
            }

            string name = ParseString(text, ref position);
            SkipWhitespace(text, ref position);
            if (position >= text.Length || text[position] != ':')
            {
                throw new NrmException($"expected ':' at offset {position}");
            }

            position++;
            members[name] = ParseValue(text, ref position);
            SkipWhitespace(text, ref position);
            if (position >= text.Length)
            {
                throw new NrmException("metadata ended inside an object");
            }

            if (text[position] == ',')
            {
                position++;
                continue;
            }

            if (text[position] == '}')
            {
                position++;
                return JsonValue.FromObject(members);
            }

            throw new NrmException($"expected ',' or '}}' at offset {position}");
        }
    }

    private static JsonValue ParseArray(string text, ref int position)
    {
        var items = new List<JsonValue>();
        position++;
        SkipWhitespace(text, ref position);
        if (position < text.Length && text[position] == ']')
        {
            position++;
            return JsonValue.FromArray(items);
        }

        while (true)
        {
            items.Add(ParseValue(text, ref position));
            SkipWhitespace(text, ref position);
            if (position >= text.Length)
            {
                throw new NrmException("metadata ended inside an array");
            }

            if (text[position] == ',')
            {
                position++;
                continue;
            }

            if (text[position] == ']')
            {
                position++;
                return JsonValue.FromArray(items);
            }

            throw new NrmException($"expected ',' or ']' at offset {position}");
        }
    }

    private static string ParseString(string text, ref int position)
    {
        position++;
        var builder = new StringBuilder();
        while (true)
        {
            if (position >= text.Length)
            {
                throw new NrmException("metadata ended inside a string");
            }

            char character = text[position++];
            if (character == '"')
            {
                return builder.ToString();
            }

            if (character != '\\')
            {
                builder.Append(character);
                continue;
            }

            if (position >= text.Length)
            {
                throw new NrmException("metadata ended inside an escape");
            }

            char escape = text[position++];
            switch (escape)
            {
                case '"': builder.Append('"'); break;
                case '\\': builder.Append('\\'); break;
                case '/': builder.Append('/'); break;
                case 'b': builder.Append('\b'); break;
                case 'f': builder.Append('\f'); break;
                case 'n': builder.Append('\n'); break;
                case 'r': builder.Append('\r'); break;
                case 't': builder.Append('\t'); break;
                case 'u':
                    if (position + 4 > text.Length)
                    {
                        throw new NrmException("metadata ended inside a \\u escape");
                    }

                    builder.Append((char)ushort.Parse(
                        text.AsSpan(position, 4),
                        NumberStyles.HexNumber,
                        CultureInfo.InvariantCulture));
                    position += 4;
                    break;
                default:
                    throw new NrmException($"unsupported escape '\\{escape}' in metadata");
            }
        }
    }

    private static double ParseNumber(string text, ref int position)
    {
        int start = position;
        if (position < text.Length && (text[position] == '-' || text[position] == '+'))
        {
            position++;
        }

        while (position < text.Length)
        {
            char character = text[position];
            bool numeric = (character >= '0' && character <= '9')
                || character == '.' || character == 'e' || character == 'E'
                || character == '+' || character == '-';
            if (!numeric)
            {
                break;
            }

            position++;
        }

        if (!double.TryParse(
                text.AsSpan(start, position - start),
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out double value))
        {
            throw new NrmException(
                $"'{text.Substring(start, position - start)}' is not a number this reader accepts");
        }

        return value;
    }

    private static void Expect(string text, ref int position, string literal)
    {
        if (position + literal.Length > text.Length
            || string.CompareOrdinal(text, position, literal, 0, literal.Length) != 0)
        {
            throw new NrmException($"expected '{literal}' at offset {position}");
        }

        position += literal.Length;
    }

    private static void SkipWhitespace(string text, ref int position)
    {
        while (position < text.Length)
        {
            char character = text[position];
            if (character != ' ' && character != '\t' && character != '\n' && character != '\r')
            {
                break;
            }

            position++;
        }
    }
}

/// <summary>One JSON value from a <c>.nrm</c> file's metadata.</summary>
internal sealed class JsonValue
{
    private readonly Dictionary<string, JsonValue>? _members;
    private readonly List<JsonValue>? _items;
    private readonly string? _text;
    private readonly double _number;
    private readonly bool _boolean;

    private JsonValue(
        NrmJsonKind kind,
        Dictionary<string, JsonValue>? members = null,
        List<JsonValue>? items = null,
        string? text = null,
        double number = 0.0,
        bool boolean = false)
    {
        Kind = kind;
        _members = members;
        _items = items;
        _text = text;
        _number = number;
        _boolean = boolean;
    }

    /// <summary>The null value.</summary>
    internal static JsonValue Null { get; } = new(NrmJsonKind.Null);

    /// <summary>What kind of value this is.</summary>
    internal NrmJsonKind Kind { get; }

    internal static JsonValue FromObject(Dictionary<string, JsonValue> members) =>
        new(NrmJsonKind.Object, members: members);

    internal static JsonValue FromArray(List<JsonValue> items) =>
        new(NrmJsonKind.Array, items: items);

    internal static JsonValue FromString(string text) => new(NrmJsonKind.String, text: text);

    internal static JsonValue FromNumber(double number) =>
        new(NrmJsonKind.Number, number: number);

    internal static JsonValue FromBool(bool boolean) =>
        new(NrmJsonKind.Bool, boolean: boolean);

    /// <summary>The array's items.</summary>
    internal IReadOnlyList<JsonValue> Items =>
        _items ?? throw new NrmException("expected an array in the metadata");

    /// <summary>The object's members.</summary>
    internal IReadOnlyDictionary<string, JsonValue> Members =>
        _members ?? throw new NrmException("expected an object in the metadata");

    /// <summary>Looks up a member, or null when absent.</summary>
    internal JsonValue? Member(string name) =>
        _members is not null && _members.TryGetValue(name, out JsonValue? value) ? value : null;

    /// <summary>Reads a member as a required string.</summary>
    internal string RequiredString(string name)
    {
        JsonValue? value = Member(name);
        if (value is null || value.Kind != NrmJsonKind.String)
        {
            throw new NrmException($"metadata is missing the string field '{name}'");
        }

        return value._text!;
    }

    /// <summary>Reads a member as a string, or null when absent.</summary>
    internal string? OptionalString(string name)
    {
        JsonValue? value = Member(name);
        return value is not null && value.Kind == NrmJsonKind.String ? value._text : null;
    }

    /// <summary>Reads a member as a required number.</summary>
    internal double RequiredNumber(string name)
    {
        JsonValue? value = Member(name);
        if (value is null || value.Kind != NrmJsonKind.Number)
        {
            throw new NrmException($"metadata is missing the numeric field '{name}'");
        }

        return value._number;
    }

    /// <summary>Reads a member as a required object.</summary>
    internal JsonValue RequiredObject(string name)
    {
        JsonValue? value = Member(name);
        if (value is null || value.Kind != NrmJsonKind.Object)
        {
            throw new NrmException($"metadata is missing the object field '{name}'");
        }

        return value;
    }

    /// <summary>Reads a member as a required array.</summary>
    internal JsonValue RequiredArray(string name)
    {
        JsonValue? value = Member(name);
        if (value is null || value.Kind != NrmJsonKind.Array)
        {
            throw new NrmException($"metadata is missing the array field '{name}'");
        }

        return value;
    }

    /// <summary>This value as a string.</summary>
    internal string AsString() =>
        Kind == NrmJsonKind.String
            ? _text!
            : throw new NrmException("expected a string in the metadata");

    /// <summary>This value as a number.</summary>
    internal double AsNumber() =>
        Kind == NrmJsonKind.Number
            ? _number
            : throw new NrmException("expected a number in the metadata");

    /// <summary>This value as a boolean.</summary>
    internal bool AsBool() =>
        Kind == NrmJsonKind.Bool
            ? _boolean
            : throw new NrmException("expected a boolean in the metadata");
}

/// <summary>The kinds of JSON value this reader produces.</summary>
/// <remarks>
/// Named <c>NrmJsonKind</c> rather than <c>JsonValueKind</c> so it does not
/// collide with <c>System.Text.Json.JsonValueKind</c> in any assembly that can
/// see this one's internals — the test assembly uses both.
/// </remarks>
internal enum NrmJsonKind
{
    /// <summary>A JSON object.</summary>
    Object,

    /// <summary>A JSON array.</summary>
    Array,

    /// <summary>A JSON string.</summary>
    String,

    /// <summary>A JSON number.</summary>
    Number,

    /// <summary>A JSON boolean.</summary>
    Bool,

    /// <summary>JSON null.</summary>
    Null,
}
