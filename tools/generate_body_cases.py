"""Generate the body-mode test cases the .NET tokenizer is checked against.

Usage:
    uv run python tools/generate_body_cases.py

NR-0005 §2.4 requires the Python and .NET body tokenizers to agree on shared
cases. The output is committed, so no message here carries a vendor-formatted
credential — the same rule `tools/generate_feature_vectors.py` follows, for the
same reason: a committed vendor-shaped string is indistinguishable from a leak
to any scanner matching vendor patterns. The values below are either obviously
synthetic or structural (hex, a UUID, a URL), which is all the tokenizer reads;
it never looks at a vendor prefix.

The messages are chosen to pin the decisions that NR-0003 §2.0 argued about and
that a port is most likely to get wrong: that `=` is evidence on its own while
`:` is not, that trailing sentence punctuation is stripped but `=` padding is
not, that a quoted fragment is always kept, and that prose words are dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "src"))

from neburedaction.body import body_candidates  # noqa: E402

OUT = REPOSITORY_ROOT / "tests" / "fixtures" / "body-cases-v1.json"

#: `(name, message, why)`. The reason is recorded in the fixture because a
#: future reader deleting a case should have to argue with the reason rather
#: than guess what it was for.
MESSAGES: tuple[tuple[str, str, str], ...] = (
    (
        "prose_only",
        "connection established to the downstream auckland region",
        "Every fragment is a prose word; none survives. This is the defect "
        "NR-0003 §2.0 exists to fix.",
    ),
    (
        "colon_prose",
        "ack received: downstream authorised",
        "A colon between two words is not evidence, and `received` is not "
        "key-like, so nothing is kept.",
    ),
    (
        "colon_key_like",
        "auth failed for access_token: shortval and for retry2: alsoshort",
        "After a colon the key must be key-like; an underscore and a digit "
        "both qualify, so both short values are kept.",
    ),
    (
        "colon_credential_word",
        "login rejected, password: hunterhunter",
        "A bare credential word is key-like even with no underscore or digit.",
    ),
    (
        "assignment",
        "starting worker region=aucklandbay retries=seventeen",
        "`=` is evidence on its own, so both values are kept although neither has value shape.",
    ),
    (
        "quoted_fragment",
        'rejected header "lowercaseonlyword" from the upstream',
        "Enclosing quotes keep a fragment that would otherwise be dropped as prose.",
    ),
    (
        "trailing_punctuation",
        "digest was 9f86d081884c7d659a2feaa0c55ad015, verified.",
        "The trailing comma is stripped; the fragment is unmistakable hex.",
    ),
    (
        "unquoted_base64_padding_is_not_captured",
        "payload cGFzc3dvcmRsZXNzc2FtcGxl== decoded",
        "The padding is absent from the candidate, and not because it was "
        "stripped: `=` is not in the unquoted value's character class, so the "
        "match ends before it. `_TRAILING_PUNCTUATION` excluding `=` therefore "
        "only matters for a quoted fragment, which the next case covers.",
    ),
    (
        "quoted_base64_padding_kept",
        'header "cGFzc3dvcmRsZXNzc2FtcGxl==" decoded',
        'Inside quotes the fragment is `[^"]{8,}`, so the padding is '
        "captured, and it survives because `=` is deliberately not trailing "
        "punctuation.",
    ),
    (
        "uuid_and_url",
        "session 3f2504e0-4f89-41d3-9a0c-0305e82c3301 opened via "
        "https://internal.example.test/v2/resource",
        "Two unmistakable shapes with no key name; both are kept as slots.",
    ),
    (
        "jwt_shape",
        "bearer " + "eyJ" + "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl" + " presented",
        "The JWT shape is unmistakable. Assembled from parts so the committed "
        "fixture does not contain a literal token-shaped string for a scanner "
        "to match.",
    ),
    (
        "public_key_header",
        "-----BEGIN PUBLIC KEY----- MIIBIjANBgkqhkiG9w0 -----END PUBLIC KEY-----",
        "The `-----BEGIN ` rule does **not** fire here: the space is not in "
        "the unquoted value class, so no candidate ever starts with it, and "
        "what survives is the base64 body instead. The rule is reachable for "
        "an attribute value and for a quoted fragment, which the next case "
        "covers. A public key, because a private-key header in a committed "
        "fixture is what gitleaks blocks.",
    ),
    (
        "quoted_public_key_header",
        'pem "-----BEGIN PUBLIC KEY----- MIIBIjANBgkqhkiG9w0" loaded',
        "Quoted, the fragment keeps its `-----BEGIN ` prefix, which is where "
        "the unmistakable rule applies.",
    ),
    (
        "duplicate_value",
        "token=samevalue1234 retry with token=samevalue1234",
        "The same value twice yields one candidate.",
    ),
    (
        "mixed_case_long_word",
        "label VeryLongMixedCaseWordHere and label shortMixed",
        "A pure-letter fragment needs 20 characters and mixed case; the second is too short.",
    ),
    (
        "digits_and_letters",
        "order a1b2c3d4e5 queued",
        "Eight or more characters with both a digit and a letter has value "
        "shape even with no key name.",
    ),
    (
        "empty",
        "",
        "No candidates, and no exception.",
    ),
)


def main() -> int:
    """Write the fixture.

    Returns:
        0 on success.
    """
    cases = []
    for name, message, why in MESSAGES:
        candidates = body_candidates(message)
        cases.append(
            {
                "name": name,
                "why": why,
                "message": message,
                "candidates": [
                    {"key": candidate.key, "value": candidate.value, "scope": candidate.scope}
                    for candidate in candidates
                ],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"cases": cases}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    kept = sum(len(case["candidates"]) for case in cases)
    print(f"wrote {len(cases)} cases, {kept} candidates, to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
