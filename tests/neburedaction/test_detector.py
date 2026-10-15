"""Checks the inference API's documented semantics (NR-0005 §2.3).

These are the behaviours a reimplementation could get subtly wrong and no
probability comparison would catch: the key-name comparison, the precedence
between the override lists, and the mask's digest. The .NET suite asserts the
same things, so a divergence shows up as a test failure on one side rather
than as different redactions in production.

The model file is a release artifact and `.gitignore` excludes `*.nrm`, so the
tests that need it skip when it is absent rather than failing a fresh checkout.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datagen.families import REGISTRY

from neburedaction.detector import (
    BALANCED,
    CONSERVATIVE,
    SecretDetector,
    SecretDetectorOptions,
    mask,
)

MODEL = Path(__file__).resolve().parents[2] / "models" / "neburedaction-v0.1.0.nrm"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"{MODEL.name} is not in this checkout")


def _detector(**kwargs: object) -> SecretDetector:
    return SecretDetector.load(MODEL, SecretDetectorOptions(**kwargs))  # type: ignore[arg-type]


def test_mask_is_the_first_four_hex_characters_of_sha256() -> None:
    """Stated as the construction rather than as a golden string."""
    value = "whatever-this-is"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:4]
    assert mask(value) == f"***#{digest}"


def test_mask_is_stable_and_distinguishing() -> None:
    """The same value masks the same way; a different one usually differs."""
    assert mask("a") == mask("a")
    assert mask("a") != mask("b")
    assert mask("").startswith("***#")


def test_probabilities_sum_to_one_and_name_every_class() -> None:
    """A result reports a distribution, not scores."""
    detector = _detector()
    result = detector.classify("note", "hello")
    assert set(result.probabilities) == set(detector.labels)
    assert result.probabilities[result.category] == max(result.probabilities.values())
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


def test_always_redact_wins_over_never() -> None:
    """The documented precedence, which is otherwise arbitrary."""
    detector = _detector(
        always_redact_keys=frozenset({"ticket"}),
        never_redact_keys=frozenset({"ticket"}),
    )
    result = detector.classify("ticket", "ABC-1234")
    assert result.should_redact
    assert result.reason == "always"


def test_key_overrides_are_case_insensitive() -> None:
    """`API_KEY` and `api_key` are the same attribute name."""
    detector = _detector(always_redact_keys=frozenset({"API_KEY"}))
    assert detector.classify("api_key", "x").reason == "always"
    assert detector.classify("Api_Key", "x").reason == "always"


def test_an_override_leaves_the_probabilities_untouched() -> None:
    """The reason exists so an override is visible; the scores stay real."""
    plain = _detector().classify("comment", "ordinary text here")
    overridden = _detector(always_redact_keys=frozenset({"comment"})).classify(
        "comment", "ordinary text here"
    )
    assert overridden.probabilities == plain.probabilities
    assert overridden.should_redact and not plain.should_redact


def test_never_redact_suppresses_a_model_decision() -> None:
    """And records that it did.

    The value is produced by the generator rather than written out: this test
    previously carried a vendor's own published example access key id, which
    EXECUTION.md §5 item 3 forbids (NR-0008 §2.0 finding F4). The premise is
    asserted first, so the test cannot quietly decay into an override that
    merely agrees with the model.
    """
    value = REGISTRY["aws.access_key_id"].generate(random.Random(20261016))
    assert _detector().classify("anykey", value).should_redact, "premise no longer holds"

    detector = _detector(never_redact_keys=frozenset({"anykey"}))
    result = detector.classify("anykey", value)
    assert not result.should_redact
    assert result.reason == "never"


def test_conservative_is_the_stricter_threshold() -> None:
    """Ten times stricter means a higher bar, so it can only redact less."""
    balanced = _detector(profile=BALANCED)
    conservative = _detector(profile=CONSERVATIVE)
    assert conservative.threshold >= balanced.threshold


def test_an_unknown_profile_is_refused() -> None:
    """Rather than silently falling back to a threshold nobody asked for."""
    with pytest.raises(ValueError, match="profile"):
        _detector(profile="paranoid")


def test_classify_message_drops_prose_and_uses_the_message_as_template() -> None:
    """Body mode is attribute mode over the tokenizer's candidates."""
    detector = _detector()
    message = "connection established to the downstream auckland region"
    assert detector.classify_message(message) == []

    message = "session token=notarealvalue1 opened"
    results = detector.classify_message(message)
    assert [fragment for fragment, _ in results] == ["notarealvalue1"]
    assert (
        results[0][1].probabilities
        == detector.classify(
            "token", "notarealvalue1", template=message, scope="message.kv"
        ).probabilities
    )
