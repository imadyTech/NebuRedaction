"""Tests for the adversarial spot-check set (NR-0003 §2.2 item 6)."""

from __future__ import annotations

import pytest
from datagen.loader import load_for_training
from datagen.unit import DetectionUnit
from labeling.teacher.adversarial import (
    CARRIERS,
    FROZEN,
    MINIMUM_UNITS,
    carrier_of,
    digest_value,
    is_adversarial,
    load_frozen,
    measure,
    resolve,
    select,
)

REPOSITORY_ROOT = FROZEN.resolve().parent.parent.parent
TRAIN = REPOSITORY_ROOT / "data" / "synth" / "v0" / "train.jsonl"


def negative(unit_id: str, **meta: object) -> DetectionUnit:
    """Build a negative unit with the given meta."""
    return DetectionUnit(
        id=unit_id, key="ApiKey", value="not-a-secret", source="synth", label="none", meta=meta
    )


def test_the_template_carrier_is_read_from_the_metadata() -> None:
    """The one carrier NR-0001 records explicitly."""
    unit = negative("a", adversarial=True, adversarial_carrier="template", context="misleading")
    assert carrier_of(unit) == "template"


def test_the_key_carriers_are_derived_from_the_context() -> None:
    """NR-0001 does not record these, so they are derived.

    A metadata gap in the generator rather than a choice here: of the 27,840
    adversarial negatives, only the 8,352 template ones carry an explicit
    carrier. Reported as a defect for NR-0001 to close.
    """
    assert carrier_of(negative("b", adversarial=True, context="hinting")) == "sensitive_key"
    assert carrier_of(negative("c", adversarial=True, context="misleading")) == "misleading_key"


def test_a_plain_negative_is_not_adversarial() -> None:
    """111,306 of the training negatives are plain, and must not be included."""
    unit = negative("d", context="neutral")
    assert carrier_of(unit) is None
    assert not is_adversarial(unit)


def test_a_positive_is_never_adversarial() -> None:
    """The set measures false positives, so it must contain only negatives."""
    unit = DetectionUnit(
        id="e",
        key="ApiKey",
        value="x",
        source="synth",
        label="api_key",
        meta={"adversarial": True, "context": "hinting"},
    )
    assert not is_adversarial(unit)


def test_a_hinting_context_without_the_adversarial_flag_is_excluded() -> None:
    """42,941 units have a hinting context but are not adversarial negatives.

    The flag is what NR-0001 uses to mark the ones it constructed as
    adversarial, so keying on the context alone would roughly quadruple the
    population and change what the measurement means.
    """
    assert carrier_of(negative("f", context="hinting")) is None


def test_selection_is_balanced_across_carriers() -> None:
    """Selection is balanced across the three carriers.

    An unbalanced set would make the rate depend on the corpus's carrier mix
    rather than on the teacher, and the three carriers are not equally hard.
    """
    units = (
        [negative(f"s{index}", adversarial=True, context="hinting") for index in range(50)]
        + [negative(f"m{index}", adversarial=True, context="misleading") for index in range(50)]
        + [
            negative(
                f"t{index}", adversarial=True, adversarial_carrier="template", context="misleading"
            )
            for index in range(50)
        ]
    )
    chosen = select(units, 30)
    counts = dict.fromkeys(CARRIERS, 0)
    for unit in chosen:
        carrier = carrier_of(unit)
        assert carrier is not None
        counts[carrier] += 1
    assert counts == {"sensitive_key": 10, "misleading_key": 10, "template": 10}


def test_selection_is_deterministic() -> None:
    """Selection is repeatable without a seed.

    The set is frozen immediately and never resampled, so a seed would add a
    parameter without adding anything — but it must still be repeatable.
    """
    units = [negative(f"s{index}", adversarial=True, context="hinting") for index in range(30)]
    assert [unit.id for unit in select(units, 9)] == [unit.id for unit in select(units, 9)]


def test_the_frozen_set_meets_the_work_orders_floor() -> None:
    """§2.2 item 6 requires at least 200 units."""
    frozen = load_frozen()
    assert len(frozen) >= MINIMUM_UNITS


def test_the_frozen_set_stores_no_values() -> None:
    """The frozen set stores digests, never values.

    It is committed, unlike everything under data/, so it must not carry
    credential-shaped strings — and these units' values are non-credentials
    under credential-suggesting names, which is still worth not publishing.
    """
    text = FROZEN.read_text(encoding="utf-8")
    assert "value" not in load_frozen()
    for digest in load_frozen().values():
        assert len(digest) == 16
        assert int(digest, 16) >= 0
    assert "not-a-secret" not in text


@pytest.mark.skipif(not TRAIN.is_file(), reason="the synthetic corpus is generated, not committed")
def test_the_frozen_set_still_resolves_against_the_corpus() -> None:
    """The frozen set still resolves against the corpus.

    A regenerated corpus that reassigned ids would silently change what the
    measurement is measuring, so the digests are checked rather than trusted.
    """
    corpus = list(load_for_training([TRAIN]))
    resolved, problems = resolve(load_frozen(), corpus)
    assert problems == []
    assert len(resolved) == len(load_frozen())
    assert all(is_adversarial(unit) for unit in resolved)


def test_a_reassigned_id_is_detected() -> None:
    """Which is the whole reason the digests are stored alongside the ids."""
    units = [negative("x", adversarial=True, context="hinting")]
    frozen = {"x": digest_value("something-else")}
    resolved, problems = resolve(frozen, units)
    assert resolved == []
    assert problems == ["x: value digest changed"]


def test_a_missing_id_is_detected() -> None:
    """A shrunken corpus is also a comparability break."""
    resolved, problems = resolve({"gone": digest_value("v")}, [])
    assert resolved == []
    assert problems == ["gone: not in the corpus"]


def test_measurement_counts_any_non_none_choice_as_a_false_positive() -> None:
    """Every unit in the set is a negative, so any secret class is an error."""
    units = [
        negative("a", adversarial=True, context="hinting"),
        negative("b", adversarial=True, context="hinting"),
    ]
    teacher = {
        "a": {
            "none": 1.0,
            "password": 0.0,
            "token": 0.0,
            "api_key": 0.0,
            "private_key": 0.0,
            "connection_string": 0.0,
            "cookie": 0.0,
        },
        "b": {
            "none": 0.05,
            "password": 0.95,
            "token": 0.0,
            "api_key": 0.0,
            "private_key": 0.0,
            "connection_string": 0.0,
            "cookie": 0.0,
        },
    }
    result = measure(units, teacher)
    assert result["units"] == 2
    assert result["false_positives"] == 1
    assert result["false_positive_rate"] == 0.5
    assert result["called_instead"] == {"password": 1}


def test_the_confident_subset_is_reported_separately() -> None:
    """It is the number §2.3's auto-adopt threshold actually depends on.

    An 80% error rate where none of the errors is confident would be survivable
    under a 0.9 threshold; one where half are confident is not.
    """
    units = [negative(name, adversarial=True, context="hinting") for name in ("a", "b")]
    base = dict.fromkeys(
        ("none", "password", "token", "api_key", "private_key", "connection_string", "cookie"), 0.0
    )
    teacher = {
        "a": {**base, "password": 0.95, "none": 0.05},
        "b": {**base, "password": 0.55, "none": 0.45},
    }
    result = measure(units, teacher)
    assert result["false_positives"] == 2
    assert result["confident_false_positives_at_0.9"] == 1
