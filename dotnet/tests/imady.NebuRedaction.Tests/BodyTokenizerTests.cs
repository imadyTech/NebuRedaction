using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Xunit;

namespace imady.NebuRedaction.Tests;

/// <summary>
/// Checks the .NET body tokenizer against the committed shared cases
/// (NR-0005 §2.4).
/// </summary>
/// <remarks>
/// The fixture is produced by <c>tools/generate_body_cases.py</c> from the
/// normative Python tokenizer and drives
/// <c>tests/neburedaction/test_body.py</c> too, so the two are shown to agree
/// against something neither of them computes.
/// </remarks>
public class BodyTokenizerTests
{
    private static readonly JsonDocument Cases = JsonDocument.Parse(
        File.ReadAllText(FeatureVectorTests.RepositoryFile("tests/fixtures/body-cases-v1.json")));

    [Fact]
    public void EveryCaseMatchesThePythonTokenizer()
    {
        var failures = new List<string>();
        int cases = 0;

        foreach (JsonElement testCase in Cases.RootElement.GetProperty("cases").EnumerateArray())
        {
            cases++;
            string name = testCase.GetProperty("name").GetString()!;
            string message = testCase.GetProperty("message").GetString()!;

            var expected = new List<string>();
            foreach (JsonElement candidate in testCase.GetProperty("candidates").EnumerateArray())
            {
                expected.Add(string.Join('\u001f', [
                    candidate.GetProperty("key").GetString()!,
                    candidate.GetProperty("value").GetString()!,
                    candidate.GetProperty("scope").GetString()!,
                ]));
            }

            List<string> actual = BodyTokenizer.Candidates(message)
                .Select(candidate =>
                    string.Join('\u001f', [candidate.Key, candidate.Value, candidate.Scope]))
                .ToList();

            if (!expected.SequenceEqual(actual, StringComparer.Ordinal))
            {
                failures.Add(
                    $"{name}:{Environment.NewLine}  expected [{string.Join(" | ", expected)}]"
                        + $"{Environment.NewLine}  got      [{string.Join(" | ", actual)}]");
            }
        }

        Assert.True(cases > 0, "the fixture carries no cases");
        Assert.True(
            failures.Count == 0,
            $"{failures.Count} of {cases} cases differ:{Environment.NewLine}"
                + string.Join(Environment.NewLine, failures));
    }

    [Fact]
    public void ProseIsRejectedAndStructureIsNot()
    {
        Assert.False(BodyTokenizer.HasValueShape("established"));
        Assert.False(BodyTokenizer.HasValueShape("auckland"));
        Assert.True(BodyTokenizer.HasValueShape("9f86d081884c7d659a2feaa0c55ad015"));
        Assert.True(BodyTokenizer.HasValueShape("3f2504e0-4f89-41d3-9a0c-0305e82c3301"));
        Assert.True(BodyTokenizer.HasValueShape("a1b2c3d4e5"));
    }

    [Fact]
    public void PureLettersNeedLengthAndMixedCase()
    {
        // Both conditions, not either.
        Assert.False(BodyTokenizer.HasValueShape("averylongallloweralpha"));
        Assert.False(BodyTokenizer.HasValueShape("ShortMixed"));
        Assert.True(BodyTokenizer.HasValueShape("VeryLongMixedCaseWordHere"));
    }

    [Fact]
    public void KeyLikeness()
    {
        Assert.True(BodyTokenizer.IsKeyLike("password"));
        Assert.True(BodyTokenizer.IsKeyLike("PASSWORD"));
        Assert.True(BodyTokenizer.IsKeyLike("access_token"));
        Assert.True(BodyTokenizer.IsKeyLike("db.host"));
        Assert.True(BodyTokenizer.IsKeyLike("retry2"));
        Assert.False(BodyTokenizer.IsKeyLike("received"));
    }

    [Fact]
    public void TheTwoDelimitersAreNotEquallyInformative()
    {
        // `=` is evidence on its own; `:` needs the key to be key-like.
        Assert.True(BodyTokenizer.IsCandidate("region", "=", false, "aucklandbay"));
        Assert.False(BodyTokenizer.IsCandidate("received", ":", false, "downstream"));
        Assert.True(BodyTokenizer.IsCandidate("access_token", ":", false, "shortval"));
    }

    [Fact]
    public void ShortFragmentsAndNullsAreHandled()
    {
        Assert.False(BodyTokenizer.IsCandidate("token", "=", true, "short"));
        Assert.Empty(BodyTokenizer.Candidates(string.Empty));
        Assert.Throws<ArgumentNullException>(() => BodyTokenizer.Candidates(null!));
        Assert.Throws<ArgumentNullException>(() => BodyTokenizer.HasValueShape(null!));
        Assert.Throws<ArgumentNullException>(() => BodyTokenizer.IsKeyLike(null!));
    }
}
