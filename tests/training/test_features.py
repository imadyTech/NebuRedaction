"""Tests for the shared feature extractor (NR-0004 §2 item 1).

This module is the normative definition that NR-0005 ports to .NET, and plan
§6.4's G3 makes agreement to within 1e-6 a hard condition. So the tests here
are mostly about the properties a port depends on — determinism, a specified
hash, a fixed feature order — rather than about particular numbers, with the
committed vectors covering the numbers.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from neburedaction.features import (
    DEFAULT_BUCKETS,
    DENSE_FEATURES,
    FEATURE_VERSION,
    MAX_BUCKETS,
    MIN_BUCKETS,
    NS_CONTEXT_WORD,
    NS_KEY_WORD,
    NS_VALUE_NGRAM,
    Features,
    apply_mask,
    bucket_of,
    dense_features,
    extract,
    fnv1a,
    hashed_features,
    rare_bucket_mask,
    shannon_entropy,
    split_key_words,
    template_words,
    to_dense_array,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORS = REPOSITORY_ROOT / "tests" / "fixtures" / "feature-vectors-v1.json"


# --- the hash ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0x811C9DC5),
        ("a", 0xE40C292C),
        ("foobar", 0xBF9CF968),
    ],
)
def test_fnv1a_matches_the_published_vectors(text: str, expected: int) -> None:
    """Checked against FNV's own published test vectors, not against itself.

    A hash that is merely self-consistent would let a .NET port be
    self-consistently different. These three values are external.
    """
    assert fnv1a(text) == expected


def test_the_hash_is_not_pythons_salted_one() -> None:
    """The hash must not be Python's salted built-in.

    Python's hash() of a string is salted per process, so a model trained in
    one process would read different features in the next and no .NET port
    could ever match it.

    Asserted by running a fresh interpreter with a different PYTHONHASHSEED and
    requiring the same answer.
    """
    environment = {**os.environ, "PYTHONHASHSEED": "12345"}
    script = (
        "import sys; sys.path.insert(0, r'"
        + str(REPOSITORY_ROOT / "python" / "src")
        + "'); from neburedaction.features import fnv1a; print(fnv1a('nebuRedaction'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=environment, check=True
    )
    assert int(result.stdout.strip()) == fnv1a("nebuRedaction")


def test_the_hash_covers_non_ascii() -> None:
    """UTF-8 is the one detail a port has to match and the one prose omits."""
    assert fnv1a("键名") == fnv1a("键名")
    assert fnv1a("键名") != fnv1a("jianming")


# --- bucketing ---------------------------------------------------------------


def test_namespaces_separate_the_feature_families() -> None:
    """The token `token` as a key word and in a template are two features."""
    key_bucket = bucket_of("token", NS_KEY_WORD, DEFAULT_BUCKETS)
    context_bucket = bucket_of("token", NS_CONTEXT_WORD, DEFAULT_BUCKETS)
    value_bucket = bucket_of("token", NS_VALUE_NGRAM, DEFAULT_BUCKETS)
    assert len({key_bucket, context_bucket, value_bucket}) == 3


def test_buckets_are_in_range_and_signs_are_unit() -> None:
    """Every bucket must index the vector, and the sign must be usable as one."""
    for index in range(500):
        bucket, sign = bucket_of(f"token-{index}", NS_VALUE_NGRAM, DEFAULT_BUCKETS)
        assert 0 <= bucket < DEFAULT_BUCKETS
        assert sign in (-1.0, 1.0)


def test_both_signs_occur() -> None:
    """The sign exists so colliding tokens cancel rather than reinforce.

    If it were constant the hashing trick would lose that property silently,
    and nothing else in the pipeline would notice.
    """
    signs = {bucket_of(f"t{index}", NS_VALUE_NGRAM, DEFAULT_BUCKETS)[1] for index in range(200)}
    assert signs == {-1.0, 1.0}


@pytest.mark.parametrize("buckets", [1 << 13, 1 << 17, 1000, 0])
def test_an_unsupported_bucket_count_is_refused(buckets: int) -> None:
    """Only the bucket counts plan §4.2 allows are accepted.

    A non-power-of-two would turn the bucket mask into a modulo, which biases
    the low buckets.
    """
    with pytest.raises(ValueError, match="power of two"):
        bucket_of("x", NS_VALUE_NGRAM, buckets)


@pytest.mark.parametrize("buckets", [MIN_BUCKETS, 1 << 15, MAX_BUCKETS])
def test_the_allowed_bucket_counts_work(buckets: int) -> None:
    """The ends of the range plan §4.2 permits."""
    bucket, _ = bucket_of("x", NS_VALUE_NGRAM, buckets)
    assert 0 <= bucket < buckets


# --- tokenisation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("AWS_SECRET_ACCESS_KEY", ["aws", "secret", "access", "key"]),
        ("apiKeyValue", ["api", "key", "value"]),
        ("api-key", ["api", "key"]),
        ("db.host", ["db", "host"]),
        ("PasswordLastChanged", ["password", "last", "changed"]),
        ("retry2", ["retry", "2"]),
        ("<slot:user>", ["user"]),
        ("", []),
    ],
)
def test_key_names_split_on_every_convention(key: str, expected: list[str]) -> None:
    """camelCase, snake_case, kebab-case, dots and digit boundaries."""
    assert split_key_words(key) == expected


def test_template_placeholders_are_dropped() -> None:
    """Template placeholders are dropped.

    `<*>` and `{Name}` would otherwise be the corpus's commonest context words,
    carrying no information while crowding the hashed space.
    """
    words = template_words("Failed password for user <*> from {Host} port %d")
    assert "password" in words
    assert not any(word in {"host"} for word in words) or "host" not in words
    assert all(word.isalnum() for word in words)


def test_an_absent_template_contributes_nothing() -> None:
    """Rather than a word meaning "absent", which the model would learn."""
    assert template_words(None) == []
    assert template_words("") == []


# --- dense features ----------------------------------------------------------


def test_the_dense_vector_matches_the_declared_order() -> None:
    """The dense vector matches the declared feature order.

    DENSE_FEATURES is the order a model's coefficients are in, so a mismatch
    would silently invalidate every trained model.
    """
    vector = dense_features("ApiKey", "abc123", None, "otel.attribute")
    assert len(vector) == len(DENSE_FEATURES)
    assert len(set(DENSE_FEATURES)) == len(DENSE_FEATURES)


def test_every_dense_feature_is_finite() -> None:
    """A NaN or an infinity propagates into the model weights and is very hard to trace back.

    The empty value is the case that would produce one.
    """
    for key, value in (("", ""), ("k", ""), ("", "v"), ("k", "\n"), ("k", " ")):
        for number in dense_features(key, value):
            assert math.isfinite(number)


def test_ratios_are_in_the_unit_interval() -> None:
    """They are ratios; a value outside [0, 1] means a counting bug."""
    names = [
        "ratio_lower",
        "ratio_upper",
        "ratio_digit",
        "ratio_symbol",
        "ratio_whitespace",
        "distinct_ratio",
        "class_transitions_ratio",
        "value_entropy_normalised",
    ]
    features = dict(
        zip(DENSE_FEATURES, dense_features("Key", "aB3-cD4_eF5 /=", None, None), strict=True)
    )
    for name in names:
        assert 0.0 <= features[name] <= 1.0, name


def test_the_character_class_ratios_sum_to_one() -> None:
    """Every character is in exactly one class, so they must."""
    features = dict(
        zip(DENSE_FEATURES, dense_features("k", "aB3-cD4_eF5 /=", None, None), strict=True)
    )
    total = sum(
        features[name]
        for name in (
            "ratio_lower",
            "ratio_upper",
            "ratio_digit",
            "ratio_symbol",
            "ratio_whitespace",
        )
    )
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("flag", "key", "value"),
    [
        ("is_hex", "d", "9f86d081884c7d65"),
        ("is_uuid", "d", "055d2131-3bae-4115-b42a-9f4c3bca8c7c"),
        ("has_pem_header", "d", "-----BEGIN PUBLIC KEY-----"),
        ("has_url_credentials", "d", "postgres://u:p@h:1/db"),
        ("has_connection_string_password", "d", "Server=db;Password=hunter2;"),
        ("starts_with_eyj", "d", "eyJhbGciOiJub25lIn0"),
        ("is_multiline", "d", "a\nb"),
        ("key_has_sensitive_word", "password", "x"),
        ("key_has_neutralising_word", "PasswordPolicy", "x"),
        ("key_is_slot", "<slot:user>", "x"),
    ],
)
def test_each_structural_flag_fires_on_its_own_case(flag: str, key: str, value: str) -> None:
    """One case per flag, so a flag that never fires cannot pass unnoticed."""
    features = dict(zip(DENSE_FEATURES, dense_features(key, value), strict=True))
    assert features[flag] == 1.0


def test_the_template_hint_feature_sees_the_template_not_the_value() -> None:
    """NR-0000's failure and NR-0002's largest false-positive source.

    The feature exists so a model can *learn* how much to discount a template
    hint — NR-0002 measured 416 false positives from acting on it directly —
    which it can only do if the hint is an input rather than a decision.
    """
    features = dict(
        zip(
            DENSE_FEATURES,
            dense_features(
                "<slot:user>",
                "bright-copper-kettle",
                "Failed password for invalid user <*>",
                "message.slot",
            ),
            strict=True,
        )
    )
    assert features["template_has_sensitive_word"] == 1.0
    assert features["key_has_sensitive_word"] == 0.0


def test_entropy_is_zero_for_one_repeated_character() -> None:
    """The textbook case, and a guard on the log."""
    assert shannon_entropy("aaaa") == pytest.approx(0.0)
    assert shannon_entropy("") == pytest.approx(0.0)
    assert shannon_entropy("ab") == pytest.approx(1.0)


# --- hashed features ---------------------------------------------------------


def test_hashed_features_are_sparse() -> None:
    """Hashed features are returned sparsely.

    At 2^15 buckets a dense row would be 99.9% zeros, and 200,000 rows of
    32,799 floats is 26 GB.
    """
    row = hashed_features("ApiKey", "abc123", "calling <*>", "otel.attribute")
    assert 0 < len(row) < 200


def test_extraction_is_deterministic() -> None:
    """The property the whole pipeline rests on."""
    first = extract("ApiKey", "abc123", "calling <*>", "otel.attribute")
    second = extract("ApiKey", "abc123", "calling <*>", "otel.attribute")
    assert first.dense == second.dense
    assert first.hashed == second.hashed
    assert first.version == FEATURE_VERSION


def test_an_empty_value_still_extracts() -> None:
    """Own-log and CredData units can have an empty value; it must not raise."""
    features = extract("k", "")
    assert len(features.dense) == len(DENSE_FEATURES)
    assert features.hashed  # the key name still contributes


# --- the rare-n-gram mask (plan §7.5 item 4) ---------------------------------


def test_a_bucket_seen_once_is_masked() -> None:
    """A bucket seen in only one row is masked (plan §7.5 item 4).

    A 4-gram occurring in exactly one training row can carry a weight that is,
    in effect, a lookup for that row's value.

    That is the memorisation mechanism, and suppressing it costs nothing because
    such a bucket generalises to nothing by definition.
    """
    rows = [{1: 1.0, 2: 1.0}, {1: 1.0, 3: 1.0}, {1: 1.0}]
    mask = rare_bucket_mask(rows, minimum_count=2)
    assert mask == {2, 3}
    assert 1 not in mask


def test_the_mask_is_applied_by_dropping_buckets() -> None:
    """So the learned weight is zero and inference needs no mask of its own."""
    assert apply_mask({1: 1.0, 2: -1.0}, {2}) == {1: 1.0}


def test_a_minimum_count_below_one_is_refused() -> None:
    """Zero would mask nothing while reading as though it masked everything."""
    with pytest.raises(ValueError, match="at least 1"):
        rare_bucket_mask([], minimum_count=0)


# --- the committed vectors ---------------------------------------------------


def test_the_committed_vectors_still_match_the_extractor() -> None:
    """The contract NR-0005's .NET port is checked against.

    If this fails, either the extractor changed — in which case FEATURE_VERSION
    must be bumped and every trained model retrained — or the file drifted. It
    is not a test to update casually.
    """
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert document["feature_version"] == FEATURE_VERSION
    assert document["dense_feature_order"] == list(DENSE_FEATURES)
    assert document["buckets"] == DEFAULT_BUCKETS

    for text, expected in document["fnv1a_32"].items():
        assert fnv1a(text) == expected, text

    for case in document["cases"]:
        inputs = case["input"]
        features = extract(inputs["key"], inputs["value"], inputs["template"], inputs["scope"])
        actual = features.as_dict()
        for name, expected_value in case["dense"].items():
            assert actual[name] == pytest.approx(expected_value, abs=1e-9), f"{case['name']}/{name}"
        summary = case["hashed"]
        buckets = sorted(features.hashed.items())
        assert summary["active_buckets"] == len(buckets), case["name"]
        assert summary["bucket_sum"] == sum(b for b, _ in buckets), case["name"]


def test_the_vectors_carry_no_vendor_credential() -> None:
    """The file is committed, so it is held to the same rule as every fixture.

    NR-0002 concluded that committing a format-correct vendor credential is
    indistinguishable from a leak to any scanner matching vendor patterns, and
    the repository's own pre-commit hooks enforce it. A test states the intent
    so a future case cannot quietly reintroduce one.
    """
    text = VECTORS.read_text(encoding="utf-8")
    for marker in ("AKIA", "ASIA", "ghp_", "glpat-", "xox", "sk_live_", "AIza", "npm_"):
        assert marker not in text, marker
    assert "PRIVATE KEY" not in text

    # A JWT-shaped string is caught by shape, not by prefix: gitleaks' `jwt`
    # rule matches three dot-separated base64url segments beginning with `eyJ`
    # regardless of what they decode to, and it is right to, because the shape
    # is the whole signal. The first version of this fixture tripped it, so the
    # guard checks the shape rather than a list of vendor prefixes.
    document = json.loads(text)
    jwt_shaped = re.compile(r"eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")
    for case in document["cases"]:
        value = case["input"]["value"]
        assert not jwt_shaped.search(value), case["name"]


def test_every_dense_feature_is_exercised_by_some_vector() -> None:
    """Every dense feature is exercised by some vector.

    A feature no vector moves is one the .NET port can get wrong without the
    comparison noticing.
    """
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    moved = {
        name
        for case in document["cases"]
        for name, value in case["dense"].items()
        if value not in (0.0, 0)
    }
    assert set(DENSE_FEATURES) - moved == set(), sorted(set(DENSE_FEATURES) - moved)


# --- the flattened form ------------------------------------------------------


def test_the_flat_vector_places_dense_first_then_buckets() -> None:
    """The layout a model's coefficient vector assumes."""
    features = Features(dense=tuple(float(i) for i in range(len(DENSE_FEATURES))), hashed={5: -1.0})
    row = to_dense_array(features, buckets=MIN_BUCKETS)
    assert len(row) == len(DENSE_FEATURES) + MIN_BUCKETS
    assert row[: len(DENSE_FEATURES)] == list(features.dense)
    assert row[len(DENSE_FEATURES) + 5] == -1.0


def test_a_masked_bucket_is_zero_in_the_flat_vector() -> None:
    """Masking must be equivalent to the bucket never having fired."""
    features = Features(dense=(0.0,) * len(DENSE_FEATURES), hashed={5: -1.0, 6: 1.0})
    row = to_dense_array(features, mask={5}, buckets=MIN_BUCKETS)
    assert row[len(DENSE_FEATURES) + 5] == 0.0
    assert row[len(DENSE_FEATURES) + 6] == 1.0
