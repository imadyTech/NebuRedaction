"""Build a tiny `.nrm` fixture model and its expected outputs.

Usage:
    uv run python tools/generate_fixture_model.py

NR-0005 §2.6 item 1 asks CI to run the .NET consistency tests on Windows and
Linux. The real consistency set cannot: §7 decided not to commit the trained
model, `.gitignore` excludes `*.nrm` and `data/`, so a CI runner has neither a
model nor a corpus and every model-dependent test would skip. CI would then
report a clean pass having exercised the binary reader, the tree walk and the
calibration — the parts most likely to differ between platforms — not at all.

This fills that gap. The model here is hand-specified, four trees over two
classes, small enough to read in full and large enough to reach every branch of
the reader: a multi-level tree, a single-leaf tree, a dense feature and a
hashed bucket, both children of a node, and a leaf reached through `~child`.
Its predictions mean nothing — it is not trained on anything — and the file says
so in its metadata.

Both outputs are committed. Neither contains a credential: the inputs are
ordinary log attributes and the weights are numbers chosen by hand.

**This is not a substitute for G3.** It proves the two implementations read the
same bytes and do the same arithmetic. Agreement on 2,400 real samples is
`tools/generate_consistency_set.py`, which stays a local gate for as long as
the model stays out of the repository.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "src"))

from neburedaction.detector import SecretDetector, SecretDetectorOptions  # noqa: E402
from neburedaction.features import DENSE_FEATURES, FEATURE_VERSION  # noqa: E402

MODEL_OUT = REPOSITORY_ROOT / "tests" / "fixtures" / "fixture-model-v1.nrm"
EXPECTED_OUT = REPOSITORY_ROOT / "tests" / "fixtures" / "fixture-model-expected-v1.json"

#: The hashing space, and the two buckets this model reads. Any two ids in
#: range will do; these are arbitrary and stored in the file, as the spec
#: requires, so nothing has to agree with them out of band.
BUCKET_COUNT = 1 << 15
BUCKET_IDS = (1234, 30000)

#: Two classes, so a softmax has something to divide.
LABELS = ("none", "token")

#: Column indices looked up by name rather than hard-coded: a tree indexing
#: the wrong column would still produce plausible numbers.
VALUE_LENGTH_LOG = DENSE_FEATURES.index("value_length_log")
RATIO_DIGIT = DENSE_FEATURES.index("ratio_digit")

#: `(class_index, internals, leaves)`. An internal node is
#: `(feature, threshold, left, right)` where a child below zero names leaf
#: `~child`.
TREES: tuple[tuple[int, tuple[tuple[int, float, int, int], ...], tuple[float, ...]], ...] = (
    # Two levels, so a reader that stops after the root is caught: short values
    # go left and are split again by how many digits they carry.
    (
        0,
        (
            (VALUE_LENGTH_LOG, 2.5, 1, -2),
            (RATIO_DIGIT, 0.1, -1, -3),
        ),
        (0.25, -0.75, 1.5),
    ),
    # A single leaf: internal_count == 0, value is leaf[0].
    (0, (), (0.125,)),
    # Reads a hashed bucket, which is column dense_feature_count + 0.
    (
        1,
        ((len(DENSE_FEATURES) + 0, 0.0, -1, -2),),
        (-0.5, 2.0),
    ),
    # Reads the second bucket, and inverts the branch direction so a reader
    # that swapped left and right fails here even if it passed above.
    (
        1,
        ((len(DENSE_FEATURES) + 1, -0.5, -2, -1),),
        (0.75, -1.25),
    ),
)

#: Platt parameters, chosen so neither threshold sits on a tie.
CALIBRATION = {"method": "platt", "slope": 1.5, "intercept": -0.25}
THRESHOLDS = {"balanced": 0.5, "conservative": 0.9}

#: Inputs that reach both sides of every branch above. `(key, value, template,
#: scope)`.
CASES: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("field", "", None, None),
    ("field", "short", None, None),
    ("field", "sh0rt", None, None),
    ("OrderId", "29", "Order 29 received", "otel.attribute"),
    ("digest", "9f86d081884c7d659a2feaa0c55ad015", None, "otel.attribute"),
    ("access_token", "notarealvalue1", "auth for {access_token}", "otel.attribute"),
    ("Region", "auckland", None, "otel.attribute"),
    ("payload", "x" * 300, None, None),
)


def build_weight_section() -> bytes:
    """Serialise the weight section the spec describes.

    Returns:
        The bytes following the metadata.
    """
    parts = [
        struct.pack(
            "<IIIII",
            len(LABELS),
            len(TREES),
            len(DENSE_FEATURES),
            len(BUCKET_IDS),
            BUCKET_COUNT,
        ),
        b"".join(struct.pack("<i", bucket) for bucket in BUCKET_IDS),
    ]

    for class_index, internals, leaves in TREES:
        parts.append(struct.pack("<HHH", class_index, len(internals), len(leaves)))
        for feature, threshold, left, right in internals:
            parts.append(struct.pack("<Hfhh", feature, threshold, left, right))
        for leaf in leaves:
            parts.append(struct.pack("<f", leaf))

    return b"".join(parts)


def main() -> int:
    """Write the fixture model and the expected outputs.

    Returns:
        0 on success.
    """
    metadata = {
        "format_version": 1,
        "model_type": "gbdt",
        "feature_version": FEATURE_VERSION,
        "base_model_version": "0.0.0-fixture",
        "labels": list(LABELS),
        "calibration": CALIBRATION,
        "thresholds": THRESHOLDS,
        "license": "MIT",
        "experimental": True,
        "note": (
            "Hand-built test fixture. Not trained on anything; its predictions "
            "are arithmetic, not judgements. See tools/generate_fixture_model.py."
        ),
    }
    encoded = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_bytes(
        b"NRM1"
        + struct.pack("<H", 1)
        + struct.pack("<I", len(encoded))
        + encoded
        + build_weight_section()
    )

    balanced = SecretDetector.load(MODEL_OUT, SecretDetectorOptions(profile="balanced"))
    conservative = SecretDetector.load(MODEL_OUT, SecretDetectorOptions(profile="conservative"))

    cases = []
    for key, value, template, scope in CASES:
        result = balanced.classify(key, value, template, scope)
        cases.append(
            {
                "key": key,
                "value": value,
                "template": template,
                "scope": scope,
                "probabilities": result.probabilities,
                "category": result.category,
                "score": result.score,
                "should_redact_balanced": result.should_redact,
                "should_redact_conservative": conservative.classify(
                    key, value, template, scope
                ).should_redact,
            }
        )

    EXPECTED_OUT.write_text(
        json.dumps(
            {
                "model": MODEL_OUT.name,
                "feature_version": FEATURE_VERSION,
                "labels": list(LABELS),
                "thresholds": THRESHOLDS,
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {MODEL_OUT.stat().st_size} bytes to {MODEL_OUT}")
    print(f"wrote {len(cases)} expected cases to {EXPECTED_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
