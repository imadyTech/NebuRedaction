"""Generate the feature test vectors NR-0005's .NET port is checked against.

Usage:
    python tools/generate_feature_vectors.py

NR-0004 §2 item 1 requires per-feature test vectors so that NR-0005's .NET
implementation has something to be verified against rather than a prose
description to interpret, and plan §6.4's G3 makes agreement to within 1e-6 a
hard condition. The output is committed.

**No vector carries a vendor-formatted credential.** The generator emits
format-correct vendor credentials and committing one would be indistinguishable
from a leak to any scanner matching vendor patterns — the conclusion NR-0002
reached about test fixtures, and the reason the synthetic corpus is generated
rather than committed. So the high-entropy vectors here use values with no
vendor prefix, and the credential-suggesting *key names* are exercised on short
non-secret values instead. Decoupling the two costs nothing: the key-name
features do not read the value and the value features do not read the key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "src"))

from neburedaction.features import (  # noqa: E402
    DEFAULT_BUCKETS,
    DENSE_FEATURES,
    FEATURE_VERSION,
    MAX_BUCKETS,
    MIN_BUCKETS,
    extract,
    fnv1a,
)

OUT = REPOSITORY_ROOT / "tests" / "fixtures" / "feature-vectors-v1.json"

#: Published FNV-1a 32-bit test vectors. A port that gets these right has the
#: hash right, which is the single most likely thing to differ between two
#: implementations — and the easiest to get silently wrong, because a wrong
#: hash still produces plausible-looking buckets.
HASH_VECTORS: tuple[str, ...] = ("", "a", "foobar", "nebuRedaction", "键名")

#: Inputs chosen to exercise each dense feature at least once, and the
#: structural flags individually. `(name, key, value, template, scope)`.
CASES: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    ("empty_value", "field", "", None, None),
    ("short_digits", "OrderId", "29", "Order 29 received for auckland", "otel.attribute"),
    ("plain_word", "Region", "auckland", None, "otel.attribute"),
    ("hex_32", "digest", "9f86d081884c7d659a2feaa0c55ad015", None, "otel.attribute"),
    (
        "hex_64",
        "checksum",
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        None,
        None,
    ),
    ("uuid", "correlation_id", "055d2131-3bae-4115-b42a-9f4c3bca8c7c", None, "otel.attribute"),
    # Base64 shape with no vendor prefix, so no scanner pattern matches it.
    ("base64_40", "field", "Zm9vYmFyYmF6cXV1eGNvcmdlZ3JhdWx0Z2FycGx5", None, None),
    ("base64url", "field", "Zm9vYmFyLWJhel9xdXV4LWNvcmdlZ3JhdWx0", None, None),
    # `is_jwt_shaped` and `starts_with_eyj` are exercised by *separate* cases,
    # because a value doing both is a JWT-shaped string, and gitleaks' `jwt`
    # rule matches that shape regardless of what the segments decode to —
    # correctly, since the shape is the whole signal. The first version of this
    # fixture tripped it. Three base64url segments not beginning with `eyJ`
    # move the first feature; a single `eyJ` segment with no dots moves the
    # second. Neither is a token.
    ("jwt_shaped", "field", "QUJDREVGRw.SElKS0xNTg.T1BRUlNUVVY", None, None),
    ("eyj_prefix", "field", "eyJhbGciOiJub25lIn0", None, None),
    (
        # A *public* key header. `has_pem_header` only tests for "-----BEGIN",
        # so this exercises the feature identically while being neither a
        # private key nor something detect-private-key must flag. The
        # private-key wording would make the committed fixture trip the
        # repository's own guard, and allowlisting it would mean switching off
        # a control to store a string chosen for its punctuation.
        "pem_header",
        "field",
        "-----BEGIN PUBLIC KEY-----\nQUJDREVG\n-----END PUBLIC KEY-----",
        None,
        None,
    ),
    ("url_credentials", "dsn", "postgres://appuser:hunter2@db.internal:5432/orders", None, None),
    (
        "connection_string",
        "dsn",
        "Server=tcp:db;Database=orders;User ID=app;Password=hunter2;",
        None,
        None,
    ),
    ("windows_path", "path", "C:\\Users\\frank\\AppData\\Local\\Temp\\report.txt", None, None),
    ("email", "subject", "s.chen@example.com", None, "otel.attribute"),
    ("ipv4", "peer", "198.51.100.7", None, "otel.attribute"),
    ("multiline", "field", "first line\nsecond line\nthird line", None, None),
    # Credential-suggesting key names, on values that are plainly not secrets.
    ("sensitive_key_plain_value", "password", "changeme", None, "otel.attribute"),
    ("sensitive_key_api", "ApiKey", "disabled", None, "otel.attribute"),
    ("neutralised_key", "PasswordPolicy", "strict", None, "otel.attribute"),
    ("neutralised_key_pair", "PasswordLastChanged", "2026-10-14", None, "otel.attribute"),
    ("partition_key", "PartitionKey", "users-2026", None, "otel.attribute"),
    # Template-hinted negative: NR-0000's failure and NR-0002's largest false
    # positive source, so the port must agree on the context features too.
    (
        "template_hint",
        "<slot:user>",
        "bright-copper-kettle",
        "Failed password for invalid user <*> from <*> port <*> ssh2",
        "message.slot",
    ),
    ("mixed_case_symbols", "field", "aB3-cD4_eF5.gH6+iJ7/kL8=", None, None),
    ("long_run", "field", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None, None),
    ("whitespace", "note", "some text with spaces", None, None),
    ("unicode", "city", "奥克兰", None, "otel.attribute"),
)


def build() -> dict[str, Any]:
    """Build the vector document.

    Returns:
        The document to serialise: the feature definition's version and shape,
        the hash vectors, and the dense features of every case.

    Hashed features are summarised by a checksum rather than written out in
    full. A case can activate a few hundred of 32,768 buckets, so the full maps
    would be most of a megabyte of indices that tell a reader nothing; the
    checksum still fails loudly if a port's hashing differs, and the hash
    vectors above localise *why*.
    """
    cases = []
    for name, key, value, template, scope in CASES:
        features = extract(key, value, template, scope)
        buckets = sorted(features.hashed.items())
        cases.append(
            {
                "name": name,
                "input": {"key": key, "value": value, "template": template, "scope": scope},
                "dense": {
                    feature: round(number, 12)
                    for feature, number in zip(DENSE_FEATURES, features.dense, strict=True)
                },
                "hashed": {
                    "active_buckets": len(buckets),
                    "bucket_sum": sum(bucket for bucket, _ in buckets),
                    "weight_sum": round(sum(weight for _, weight in buckets), 12),
                    # Order-independent on purpose: a port is free to accumulate
                    # in any order, and floating-point addition is not
                    # associative, so an order-dependent checksum would fail for
                    # a reason that is not a defect.
                    "checksum": sum((bucket + 1) * int(weight) for bucket, weight in buckets),
                },
            }
        )

    return {
        "note": (
            "Feature test vectors for NR-0005's .NET port (NR-0004 §2 item 1). The Python "
            "module neburedaction.features is the normative definition; this file is "
            "generated from it by tools/generate_feature_vectors.py. Plan §6.4's G3 "
            "requires the two implementations to agree to within 1e-6 per sample. No "
            "vector carries a vendor-formatted credential."
        ),
        "feature_version": FEATURE_VERSION,
        "buckets": DEFAULT_BUCKETS,
        "bucket_range": {"min": MIN_BUCKETS, "max": MAX_BUCKETS},
        "dense_feature_order": list(DENSE_FEATURES),
        "fnv1a_32": {text: fnv1a(text) for text in HASH_VECTORS},
        "cases": cases,
    }


def main() -> int:
    """Write the vectors.

    Returns:
        ``0`` on success.
    """
    document = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(document['cases'])} cases, {len(DENSE_FEATURES)} dense features")
    print(f"feature version {FEATURE_VERSION}, {DEFAULT_BUCKETS} buckets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
