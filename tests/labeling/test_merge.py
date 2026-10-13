"""Tests for the label merge (NR-0003 §2.3, §2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datagen.unit import LABELS, DetectionUnit
from labeling.merge import SOURCES, merge_unit, statistics, write_merged
from labeling.review import Decision, decide
from labeling.rules import LOW, RuleLabel
from tests.labeling.test_review import distribution, queue_item


def rule(unit_id: str) -> RuleLabel:
    """Build a rule label for a test case."""
    return RuleLabel(unit_id, "none", LOW, "none", 0.0)


def synthetic(label: str = "api_key") -> DetectionUnit:
    """Build a labelled synthetic unit."""
    return DetectionUnit(
        id="s-1", key="ApiKey", value="value", source="synth", label=label, family="aws.x"
    )


def own_log() -> DetectionUnit:
    """Build an unlabelled own-log unit."""
    return DetectionUnit(id="own-1", key="OrderId", value="29", source="own", label=None)


# --- provenance --------------------------------------------------------------


def test_a_synthetic_unit_the_teacher_agreed_with_is_the_generators() -> None:
    """§2.3 row 1: agreement means the generator's label, plus a soft label."""
    merged = merge_unit(synthetic(), rule("s-1"), distribution("api_key", 0.8), None, None)
    assert merged is not None
    assert merged.label == "api_key"
    assert merged.label_source == "generator"
    assert merged.teacher_agreed is True


def test_a_synthetic_disagreement_still_keeps_the_generator_label() -> None:
    """§2.3 row 2: the generator wins even when the teacher disagrees.

    It is right by construction, so a disagreement is evidence about the
    teacher rather than a reason to overwrite a label.
    """
    merged = merge_unit(synthetic(), rule("s-1"), distribution("none", 0.95), None, None)
    assert merged is not None
    assert merged.label == "api_key"
    assert merged.label_source == "generator"
    assert merged.teacher_agreed is False


def test_a_synthetic_unit_outside_the_teachers_scope_is_still_included() -> None:
    """Units with no soft label are kept, with their construction-guaranteed one.

    Roughly 45,000 training units were never shown to the teacher because the
    scope samples the high-confidence pool. Dropping them would discard a
    quarter of the corpus to tidy a schema.
    """
    merged = merge_unit(synthetic(), rule("s-1"), None, None, None)
    assert merged is not None
    assert merged.label_source == "generator_unreviewed"
    assert merged.teacher is None
    assert merged.to_dict()["teacher"] is None


def test_an_own_log_unit_with_agreement_is_auto_adopted() -> None:
    """§2.3 row 3: two systems agree and the teacher is confident."""
    merged = merge_unit(own_log(), rule("own-1"), distribution("none", 0.95), False, None)
    assert merged is not None
    assert merged.label == "none"
    assert merged.label_source == "teacher_auto"


def test_a_reviewers_decision_beats_everything() -> None:
    """Which is the point of having a reviewer."""
    decision = decide(queue_item("api_key"), "frank", "none")
    merged = merge_unit(synthetic(), rule("s-1"), distribution("api_key", 0.9), None, decision)
    assert merged is not None
    assert merged.label == "none"
    assert merged.label_source == "human"
    assert merged.review_verdict == "corrected"


def test_a_confirmed_generator_label_is_marked_as_reviewed() -> None:
    """`generator` and `generator_after_review` are different evidence."""
    decision = decide(queue_item("api_key"), "frank", "api_key")
    merged = merge_unit(synthetic(), rule("s-1"), distribution("none", 0.9), None, decision)
    assert merged is not None
    assert merged.label_source == "generator_after_review"
    assert merged.review_verdict == "confirmed"


# --- the omission case -------------------------------------------------------


def test_an_undecided_own_log_unit_is_omitted_rather_than_defaulted() -> None:
    """The case that matters most, and the one with a tempting wrong answer.

    §2.3 routed this unit to a human because neither system can be trusted with
    it alone, and no human decided it — it fell outside the 500-unit cap. Both
    fallbacks are worse than omission: writing the teacher's choice puts into a
    training file the label the merge strategy just declined to auto-adopt, and
    writing `none` manufactures a negative out of an absence of evidence. The
    second would usually be *right* on this corpus, which is exactly what makes
    it dangerous — these are the units where the teacher saw a credential, so
    defaulting them would teach the student to ignore the cases a human was
    asked to look at.
    """
    assert merge_unit(own_log(), rule("own-1"), distribution("token", 0.6), False, None) is None


def test_a_skipped_own_log_unit_is_omitted_too() -> None:
    """The reviewer looked and declined to say, which is not a label."""
    decision = decide(queue_item("none"), "frank", None)
    assert merge_unit(own_log(), rule("own-1"), distribution("token", 0.6), False, decision) is None


def test_a_missing_second_opinion_omits_rather_than_adopts() -> None:
    """Measured for real during this work order.

    With no `credsweeper-own.json` present, all 2,692 own-log units were
    omitted rather than labelled on the teacher's word alone.
    """
    assert merge_unit(own_log(), rule("own-1"), distribution("none", 0.999), None, None) is None


def test_an_own_log_unit_with_no_teacher_label_is_an_error() -> None:
    """Every own-log unit was in scope, so a missing one is a corpus mismatch."""
    with pytest.raises(ValueError, match="different corpus"):
        merge_unit(own_log(), rule("own-1"), None, None, None)


# --- statistics and writing --------------------------------------------------


def test_agreement_is_reported_per_class() -> None:
    """§2.5 item 2, and the reason it is per class rather than an average.

    The measured agreement on `none` was 0.341 while every positive class was
    above 0.67. An aggregate would hide the one number that matters behind the
    positives.
    """
    merged = [
        merge_unit(synthetic("api_key"), rule("s-1"), distribution("api_key", 0.9), None, None),
        merge_unit(synthetic("api_key"), rule("s-1"), distribution("none", 0.9), None, None),
        merge_unit(synthetic("none"), rule("s-1"), distribution("token", 0.9), None, None),
    ]
    stats = statistics([item for item in merged if item is not None], [])
    agreement = stats["teacher_generator_agreement"]
    assert agreement["api_key"] == {"asked": 2, "agreed": 1, "rate": 0.5}
    assert agreement["none"] == {"asked": 1, "agreed": 0, "rate": 0.0}


def test_units_without_a_teacher_opinion_do_not_count_towards_agreement() -> None:
    """Otherwise the rate would be diluted by units never asked about."""
    merged = [merge_unit(synthetic(), rule("s-1"), None, None, None)]
    stats = statistics([item for item in merged if item is not None], [])
    assert stats["teacher_generator_agreement"] == {}
    assert stats["with_teacher_soft_label"] == 0


def test_the_review_verdicts_are_summarised() -> None:
    """§2.5 item 2 wants the review's conclusion distribution."""
    decisions = [Decision("a", "frank", "t", "api_key", "none", "corrected")]
    stats = statistics([], decisions)
    assert stats["review"]["by_verdict"] == {"corrected": 1}


def test_written_records_carry_a_provenance(tmp_path: Path) -> None:
    """A label with no provenance is a bug, not a default.

    NR-0004 trains on these files, and distillation and self-training are not
    the same thing.
    """
    merged = merge_unit(synthetic(), rule("s-1"), distribution("api_key", 0.9), None, None)
    assert merged is not None
    path = tmp_path / "train_v0.jsonl"
    assert write_merged([merged], path) == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["meta"]["label_source"] in SOURCES
    assert record["label"] in LABELS
    assert set(record["teacher"]) == set(LABELS)


def test_an_unknown_provenance_is_refused(tmp_path: Path) -> None:
    """Checked at the writer, because that is the last point before NR-0004."""
    merged = merge_unit(synthetic(), rule("s-1"), distribution("api_key", 0.9), None, None)
    assert merged is not None
    broken = type(merged)(
        unit=merged.unit,
        label=merged.label,
        label_source="invented",
        teacher=merged.teacher,
        teacher_agreed=merged.teacher_agreed,
        review_verdict=None,
    )
    with pytest.raises(ValueError, match="label_source"):
        write_merged([broken], tmp_path / "out.jsonl")


def test_an_unknown_label_is_refused(tmp_path: Path) -> None:
    """A reviewer typo or a drifted class set must not reach a training file."""
    merged = merge_unit(synthetic(), rule("s-1"), distribution("api_key", 0.9), None, None)
    assert merged is not None
    broken = type(merged)(
        unit=merged.unit,
        label="secret",
        label_source="generator",
        teacher=merged.teacher,
        teacher_agreed=None,
        review_verdict=None,
    )
    with pytest.raises(ValueError, match="is not one of"):
        write_merged([broken], tmp_path / "out.jsonl")
