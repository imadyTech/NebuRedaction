using System;

namespace imady.NebuRedaction;

/// <summary>
/// The structural shape tests the dense features use.
/// </summary>
/// <remarks>
/// <para>
/// Written as hand-rolled scanners rather than regular expressions. The Python
/// definition uses anchored patterns, and .NET and Python agree on those only
/// if one is careful about what <c>$</c> matches, how character classes treat
/// non-ASCII and whether a pattern is culture-sensitive. Reading characters
/// removes the question: these are each a few lines, and a few lines that
/// obviously match are worth more than a regular expression that probably
/// does.
/// </para>
/// <para>
/// Every test here is ordinal and culture-invariant on purpose. A
/// culture-sensitive comparison would make the model's output depend on the
/// machine's locale, which is the kind of defect that only appears in
/// production in another country.
/// </para>
/// </remarks>
internal static class Shapes
{
    /// <summary>Whether every character is a hexadecimal digit, and there is at least one.</summary>
    internal static bool IsHex(string value)
    {
        if (value.Length == 0)
        {
            return false;
        }

        foreach (char character in value)
        {
            bool hex = (character >= '0' && character <= '9')
                || (character >= 'a' && character <= 'f')
                || (character >= 'A' && character <= 'F');
            if (!hex)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Whether the string is standard base64, allowing up to two padding characters.</summary>
    internal static bool IsBase64(string value) => IsBase64Like(value, urlSafe: false);

    /// <summary>Whether the string is URL-safe base64, allowing up to two padding characters.</summary>
    internal static bool IsBase64Url(string value) => IsBase64Like(value, urlSafe: true);

    private static bool IsBase64Like(string value, bool urlSafe)
    {
        if (value.Length == 0)
        {
            return false;
        }

        int padding = 0;
        while (padding < 2 && value.Length - padding - 1 >= 0 && value[value.Length - padding - 1] == '=')
        {
            padding++;
        }

        int body = value.Length - padding;
        if (body == 0)
        {
            return false;
        }

        for (int index = 0; index < body; index++)
        {
            char character = value[index];
            bool alphanumeric = (character >= 'A' && character <= 'Z')
                || (character >= 'a' && character <= 'z')
                || (character >= '0' && character <= '9');
            bool extra = urlSafe
                ? character == '_' || character == '-'
                : character == '+' || character == '/';
            if (!alphanumeric && !extra)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Whether the string is a canonical 8-4-4-4-12 hexadecimal UUID.</summary>
    internal static bool IsUuid(string value)
    {
        if (value.Length != 36)
        {
            return false;
        }

        for (int index = 0; index < 36; index++)
        {
            char character = value[index];
            bool shouldBeDash = index == 8 || index == 13 || index == 18 || index == 23;
            if (shouldBeDash)
            {
                if (character != '-')
                {
                    return false;
                }
            }
            else if (!IsHexDigit(character))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsHexDigit(char character) =>
        (character >= '0' && character <= '9')
        || (character >= 'a' && character <= 'f')
        || (character >= 'A' && character <= 'F');

    /// <summary>
    /// Whether the string is three dot-separated base64url segments, the last
    /// possibly empty.
    /// </summary>
    internal static bool IsJwtShaped(string value)
    {
        int firstDot = value.IndexOf('.');
        if (firstDot <= 0)
        {
            return false;
        }

        int secondDot = value.IndexOf('.', firstDot + 1);
        if (secondDot <= firstDot + 1)
        {
            return false;
        }

        if (value.IndexOf('.', secondDot + 1) >= 0)
        {
            return false;
        }

        return IsBase64UrlSegment(value, 0, firstDot)
            && IsBase64UrlSegment(value, firstDot + 1, secondDot)
            && IsBase64UrlSegment(value, secondDot + 1, value.Length, allowEmpty: true);
    }

    private static bool IsBase64UrlSegment(string value, int start, int end, bool allowEmpty = false)
    {
        if (end <= start)
        {
            return allowEmpty;
        }

        for (int index = start; index < end; index++)
        {
            char character = value[index];
            bool allowed = (character >= 'A' && character <= 'Z')
                || (character >= 'a' && character <= 'z')
                || (character >= '0' && character <= '9')
                || character == '_' || character == '-';
            if (!allowed)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Whether the string contains a URL with embedded credentials, as in
    /// <c>scheme://user:password@host</c>.
    /// </summary>
    internal static bool HasUrlCredentials(string value)
    {
        int from = 0;
        while (true)
        {
            int marker = value.IndexOf("://", from, StringComparison.Ordinal);
            if (marker < 0)
            {
                return false;
            }

            if (HasScheme(value, marker) && HasUserInfo(value, marker + 3))
            {
                return true;
            }

            from = marker + 3;
        }
    }

    private static bool HasScheme(string value, int marker)
    {
        int start = marker;
        while (start > 0)
        {
            char character = value[start - 1];
            bool allowed = IsAsciiLetterOrDigit(character)
                || character == '+' || character == '.' || character == '-';
            if (!allowed)
            {
                break;
            }

            start--;
        }

        return start < marker && IsAsciiLetter(value[start]);
    }

    private static bool HasUserInfo(string value, int start)
    {
        // user:password@ — neither part may contain /, whitespace, : or @.
        int index = start;
        int userLength = 0;
        while (index < value.Length && IsUserInfoCharacter(value[index]))
        {
            index++;
            userLength++;
        }

        if (userLength == 0 || index >= value.Length || value[index] != ':')
        {
            return false;
        }

        index++;
        int passwordLength = 0;
        while (index < value.Length && IsUserInfoCharacter(value[index]))
        {
            index++;
            passwordLength++;
        }

        return passwordLength > 0 && index < value.Length && value[index] == '@';
    }

    private static bool IsUserInfoCharacter(char character) =>
        character != '/' && character != ':' && character != '@' && !char.IsWhiteSpace(character);

    // char.IsAsciiLetter and friends are .NET 7+; this library also targets
    // netstandard2.1, so the ASCII tests are spelled out.
    private static bool IsAsciiLetter(char character) =>
        (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z');

    private static bool IsAsciiLetterOrDigit(char character) =>
        IsAsciiLetter(character) || (character >= '0' && character <= '9');
}
