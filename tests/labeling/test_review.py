"""Tests for the review queue and its records (NR-0003 §2.3, §2.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datagen.unit import LABELS, DetectionUnit
from labeling.review import (
    AUTO_ADOPT_THRESHOLD,
    Decision,
    QueueItem,
    append_decision,
    build_queue,
    classify_for_queue,
    decide,
    entropy,
    latest_decisions,
    read_decisions,
    statistics,
)
from labeling.rules import LOW, RuleLabel


def distribution(top: str, confidence: float) -> dict[str, float]:
    """Build a teacher distribution with a given top class and confidence."""
    rest = [label for label in LABELS if label != top]
    return {top: confidence, **{label: (1.0 - confidence) / len(rest) for label in rest}}


def unit(unit_id: str, *, source: str = "synth", label: str | None = "api_key") -> DetectionUnit:
    """Build a unit for a test case."""
    return DetectionUnit(id=unit_id, key="ApiKey", value="value", source=source, label=label)


def rule(unit_id: str) -> RuleLabel:
    """Build a rule label for a test case."""
    return RuleLabel(unit_id, "none", LOW, "none", 0.0)


def test_entropy_is_zero_when_certain_and_maximal_when_uniform() -> None:
    """The queue's secondary sort key must mean what the ordering assumes."""
    certain = dict.fromkeys(LABELS, 0.0)
    certain["token"] = 1.0
    assert entropy(certain) == pytest.approx(0.0)
    uniform = {label: 1.0 / len(LABELS) for label in LABELS}
    assert entropy(uniform) == pytest.approx(2.8074, abs=1e-4)


# --- §2.3's merge table, row by row -----------------------------------------


def test_synthetic_agreement_takes_the_generator_label() -> None:
    """Row 1: nothing to decide, so it does not reach the queue."""
    needs_review, reason, disposition = classify_for_queue(
        unit("s1", label="api_key"), rule("s1"), distribution("api_key", 0.8), None
    )
    assert needs_review is False
    assert disposition == "generator"
    assert reason == "synthetic:agreed"


def test_synthetic_disagreement_goes_to_review() -> None:
    """Row 2: the generator still wins, but the disagreement is evidence."""
    needs_review, reason, _ = classify_for_queue(
        unit("s2", label="api_key"), rule("s2"), distribution("none", 0.9), None
    )
    assert needs_review is True
    assert reason == "synthetic:teacher_generator_disagree"


def test_a_class_confusion_is_not_a_disagreement() -> None:
    """Agreement is about redaction, not class.

    A secret the teacher calls a token and the generator calls an api_key gets
    masked either way; only the mask's class fingerprint differs. Sending these
    to review would fill the 500-unit queue with units nobody needs to look at.
    """
    needs_review, reason, disposition = classify_for_queue(
        unit("s3", label="api_key"), rule("s3"), distribution("token", 0.8), None
    )
    assert needs_review is False
    assert disposition == "generator"
    assert reason == "synthetic:agreed"


def test_own_log_agreement_above_the_threshold_is_auto_adopted() -> None:
    """Row 3: two independent systems agree and the teacher is confident."""
    needs_review, reason, disposition = classify_for_queue(
        unit("o1", source="own", label=None),
        rule("o1"),
        distribution("none", 0.95),
        credsweeper=False,
    )
    assert needs_review is False
    assert disposition == "teacher_auto"
    assert reason == "own:auto_adopted"


def test_own_log_disagreement_goes_to_review() -> None:
    """Row 4: CredSweeper says secret, Laya says not. A human decides."""
    needs_review, reason, _ = classify_for_queue(
        unit("o2", source="own", label=None),
        rule("o2"),
        distribution("none", 0.99),
        credsweeper=True,
    )
    assert needs_review is True
    assert reason == "own:laya_credsweeper_disagree"


def test_own_log_below_the_threshold_goes_to_review() -> None:
    """Row 4 again: agreement is not enough if the teacher is unsure."""
    needs_review, reason, _ = classify_for_queue(
        unit("o3", source="own", label=None),
        rule("o3"),
        distribution("none", AUTO_ADOPT_THRESHOLD - 0.01),
        credsweeper=False,
    )
    assert needs_review is True
    assert reason == "own:below_threshold"


def test_an_own_log_unit_with_no_second_opinion_is_never_auto_adopted() -> None:
    """§2.3 requires two systems. A missing CredSweeper result is not consent.

    This is the one that would matter if the CredSweeper pass were skipped or
    failed: defaulting to adoption would auto-label the own logs on the
    uncalibrated word of a model that calls `OrderId = 29` a token at 0.979.
    """
    needs_review, reason, _ = classify_for_queue(
        unit("o4", source="own", label=None),
        rule("o4"),
        distribution("none", 0.999),
        credsweeper=None,
    )
    assert needs_review is True
    assert reason == "own:no_second_opinion"


def test_the_teachers_confident_false_positive_on_own_logs_is_caught() -> None:
    """The concrete case from this run, end to end.

    `OrderId = 29` scored `token` at 0.979. CredSweeper will not call a
    two-digit integer a credential, so the disagreement rule routes it to a
    human instead of adopting it.
    """
    order_id = DetectionUnit(
        id="own-000024-attr-00",
        key="OrderId",
        value="29",
        source="own",
        label=None,
        template="Order 29 received: cold brew to auckland",
    )
    needs_review, reason, _ = classify_for_queue(
        order_id, rule(order_id.id), distribution("token", 0.979), credsweeper=False
    )
    assert needs_review is True
    assert reason == "own:laya_credsweeper_disagree"


# --- ordering and the cap ----------------------------------------------------


def test_disagreements_sort_before_everything_else() -> None:
    """§2.4 item 2's primary key."""
    units = [
        unit("agree-low-entropy", label="api_key"),
        unit("disagree", label="api_key"),
    ]
    rules = {u.id: rule(u.id) for u in units}
    teacher = {
        # Not a disagreement, but maximum entropy.
        "agree-low-entropy": {label: 1.0 / len(LABELS) for label in LABELS},
        # A disagreement, but the teacher is certain.
        "disagree": distribution("none", 0.99),
    }
    # The agreeing unit must be forced into the queue to test the ordering, so
    # it is made unlabelled, which is its own review reason.
    units[0] = DetectionUnit(id="agree-low-entropy", key="k", value="v", source="synth", label=None)
    queue = build_queue(units, rules, teacher)
    assert next(item.unit.id for item in queue) == "disagree"


def test_within_a_group_higher_entropy_sorts_first() -> None:
    """§2.4 item 2's secondary key: the most confused units first."""
    units = [unit(f"d{index}", label="api_key") for index in range(3)]
    rules = {u.id: rule(u.id) for u in units}
    teacher = {
        "d0": distribution("none", 0.95),
        "d1": distribution("none", 0.40),
        "d2": distribution("none", 0.70),
    }
    queue = build_queue(units, rules, teacher)
    assert [item.unit.id for item in queue] == ["d1", "d2", "d0"]


def test_the_cap_is_applied_after_sorting() -> None:
    """The cap keeps the top of the ordering.

    Capping first would keep whatever came earliest in the corpus, which is the
    opposite of the point of ordering.
    """
    units = [unit(f"d{index:03d}", label="api_key") for index in range(50)]
    rules = {u.id: rule(u.id) for u in units}
    # Entropy rises with the index, so the last units are the most confused.
    teacher = {u.id: distribution("none", 0.95 - 0.01 * index) for index, u in enumerate(units)}
    queue = build_queue(units, rules, teacher, cap=5)
    assert len(queue) == 5
    assert [item.unit.id for item in queue] == ["d049", "d048", "d047", "d046", "d045"]


def test_the_queue_is_deterministic() -> None:
    """A partly-reviewed queue must not reshuffle under the reviewer."""
    units = [unit(f"d{index}", label="api_key") for index in range(10)]
    rules = {u.id: rule(u.id) for u in units}
    teacher = {u.id: distribution("none", 0.5) for u in units}
    first = [item.unit.id for item in build_queue(units, rules, teacher)]
    second = [item.unit.id for item in build_queue(units, rules, teacher)]
    assert first == second


def test_units_without_a_teacher_label_are_skipped_not_guessed() -> None:
    """Units outside the teacher's scope are not reviewed blind.

    The scope samples the high-confidence pool, so most units have no teacher
    opinion at all. Those are out of scope for review, not guessed at.
    """
    units = [unit("labelled", label="api_key"), unit("unlabelled", label="api_key")]
    rules = {u.id: rule(u.id) for u in units}
    queue = build_queue(units, rules, {"labelled": distribution("none", 0.9)})
    assert [item.unit.id for item in queue] == ["labelled"]


def test_a_teacher_label_without_a_rule_label_is_an_error() -> None:
    """It would mean the two passes ran over different corpora."""
    with pytest.raises(ValueError, match="no rule label"):
        build_queue([unit("x", label="api_key")], {}, {"x": distribution("none", 0.9)})


# --- decisions ---------------------------------------------------------------


def queue_item(current: str = "api_key") -> QueueItem:
    """Build a queue item whose current label is known."""
    return QueueItem(
        unit=unit("q1", label=current),
        rule=rule("q1"),
        teacher=distribution("none", 0.9),
        teacher_choice="none",
        credsweeper=None,
        reason="synthetic:teacher_generator_disagree",
        disagreement=True,
        entropy=1.0,
    )


def test_confirming_the_current_label_is_recorded_as_confirmed() -> None:
    """The common keystroke must not look like a correction in the statistics."""
    decision = decide(queue_item(), "frank", "api_key")
    assert decision.verdict == "confirmed"
    assert decision.original_label == "api_key"


def test_changing_the_label_is_recorded_as_corrected() -> None:
    """And the original is kept, so the change is auditable."""
    decision = decide(queue_item(), "frank", "none")
    assert decision.verdict == "corrected"
    assert decision.original_label == "api_key"
    assert decision.new_label == "none"


def test_skipping_is_recorded_rather_than_dropped() -> None:
    """A skipped unit is a fact about the data, not an absence of a decision."""
    decision = decide(queue_item(), "frank", None)
    assert decision.verdict == "skipped"
    assert decision.new_label is None


def test_an_invalid_label_is_refused() -> None:
    """A typo must not enter the training labels."""
    with pytest.raises(ValueError, match="is not one of"):
        decide(queue_item(), "frank", "secret")


def test_an_own_log_items_current_label_is_the_teachers_choice() -> None:
    """An own-log item's current label is the teacher's choice.

    There is no generator label to confirm, so Enter means "the teacher was
    right" — a decision the reviewer is making knowingly.
    """
    item = QueueItem(
        unit=unit("o9", source="own", label=None),
        rule=rule("o9"),
        teacher=distribution("none", 0.8),
        teacher_choice="none",
        credsweeper=False,
        reason="own:below_threshold",
        disagreement=False,
        entropy=1.0,
    )
    assert item.current_label == "none"


def test_decisions_are_flushed_per_decision(tmp_path: Path) -> None:
    """§2.4 item 1: interruptible. Three hundred units must not be lost."""
    path = tmp_path / "review_v0.jsonl"
    append_decision(decide(queue_item(), "frank", "none"), path)
    assert path.is_file()
    recovered = read_decisions(path)
    assert len(recovered) == 1
    assert recovered[0].reviewer == "frank"
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {
        "unit_id",
        "reviewer",
        "reviewed_utc",
        "original_label",
        "new_label",
        "verdict",
        "note",
    }


def test_an_undo_appends_rather_than_rewriting(tmp_path: Path) -> None:
    """An undo appends a further record.

    The log stays append-only, so a change of mind is auditable, and the last
    entry is the one that counts.
    """
    path = tmp_path / "review_v0.jsonl"
    item = queue_item()
    append_decision(decide(item, "frank", "none"), path)
    append_decision(decide(item, "frank", "token"), path)
    assert len(read_decisions(path)) == 2
    resolved = latest_decisions(read_decisions(path))
    assert len(resolved) == 1
    assert resolved["q1"].new_label == "token"


def test_statistics_report_verdicts_labels_and_notes() -> None:
    """§2.5 item 2 wants the review's conclusion distribution."""
    decisions = [
        Decision("a", "frank", "t", "api_key", "api_key", "confirmed"),
        Decision("b", "frank", "t", "api_key", "none", "corrected", note="generator wrong"),
        Decision("c", "frank", "t", "none", None, "skipped"),
    ]
    stats = statistics(decisions)
    assert stats["reviewed"] == 3
    assert stats["by_verdict"] == {"confirmed": 1, "corrected": 1, "skipped": 1}
    assert stats["by_label"] == {"api_key": 1, "none": 1}
    assert stats["notes"] == [{"unit_id": "b", "note": "generator wrong"}]


# --- the Enter default on own-log units --------------------------------------


def own_item(credsweeper: bool | None, teacher_choice: str = "token") -> QueueItem:
    """Build an own-log queue item with a given CredSweeper verdict."""
    return QueueItem(
        unit=unit("own-1", source="own", label=None),
        rule=rule("own-1"),
        teacher=distribution(teacher_choice, 0.18),
        teacher_choice=teacher_choice,
        credsweeper=credsweeper,
        reason="own:below_threshold",
        disagreement=False,
        entropy=2.79,
    )


def test_enter_means_none_when_credsweeper_says_not_a_secret() -> None:
    """The easiest keystroke must not be the wrong answer.

    After recalibration the teacher's top probability on own-log units is about
    0.18 against a uniform 0.14, so it is barely distinguishable from having no
    opinion. CredSweeper flagged none of the 2,692 own-log units and the rule
    layer fired on none either, so two independent systems say `none` and only
    the near-uninformative teacher disagrees. Defaulting Enter to the teacher's
    argmax would commit `token` for `OrderId = 27`, hundreds of times over.
    """
    assert own_item(credsweeper=False).current_label == "none"


def test_enter_falls_back_to_the_teacher_when_credsweeper_agrees_it_is_a_secret() -> None:
    """Enter falls back to the teacher when CredSweeper also sees a secret.

    Then the two systems agree that something is there, and the teacher's class
    is the only opinion available about which.
    """
    assert own_item(credsweeper=True).current_label == "token"


def test_enter_falls_back_to_the_teacher_with_no_second_opinion() -> None:
    """With nothing to corroborate, the default is not silently flipped."""
    assert own_item(credsweeper=None).current_label == "token"


def test_a_synthetic_units_default_is_always_the_generators_label() -> None:
    """§2.3: the generator wins on synthetic data, whatever anyone else says."""
    item = queue_item("api_key")
    assert item.current_label == "api_key"


# --- the source split --------------------------------------------------------


def test_own_log_units_get_their_reserved_share_of_the_queue() -> None:
    """§2.4 item 2's ordering applied literally admits none of them.

    Measured on this corpus: 86,460 units qualify for review and 83,768 of them
    — 96.9% — are synthetic disagreements, which sort first and fill the whole
    500-unit cap. But review has two different jobs (§2.3): on a synthetic unit
    it confirms a label that is already correct by construction, while on an
    own-log unit it creates the only label that will ever exist. Spending every
    slot on the first produces zero labels.
    """
    synthetic_units = [unit(f"s{index}", label="api_key") for index in range(100)]
    own_units = [unit(f"own-{index}", source="own", label=None) for index in range(100)]
    units = synthetic_units + own_units
    rules = {u.id: rule(u.id) for u in units}
    # Synthetic units disagree and so sort first; own units merely fall below
    # the threshold.
    teacher = {u.id: distribution("none", 0.15) for u in synthetic_units}
    teacher.update({u.id: distribution("token", 0.40) for u in own_units})
    credsweeper = {u.id: False for u in own_units}

    queue = build_queue(units, rules, teacher, credsweeper, cap=20, own_share=0.5)
    own_count = sum(1 for item in queue if item.unit.source == "own")
    assert len(queue) == 20
    assert own_count == 10


def test_the_literal_ordering_is_still_available() -> None:
    """The literal ordering remains available, and shows why it is a problem.

    own_share=None restores §2.4 item 2's wording, under which the own-log
    units are crowded out entirely.
    """
    synthetic_units = [unit(f"s{index}", label="api_key") for index in range(100)]
    own_units = [unit(f"own-{index}", source="own", label=None) for index in range(100)]
    units = synthetic_units + own_units
    rules = {u.id: rule(u.id) for u in units}
    teacher = {u.id: distribution("none", 0.15) for u in synthetic_units}
    teacher.update({u.id: distribution("token", 0.40) for u in own_units})
    credsweeper = {u.id: False for u in own_units}

    queue = build_queue(units, rules, teacher, credsweeper, cap=20, own_share=None)
    assert all(item.unit.source == "synth" for item in queue)


def test_a_reserved_slot_is_never_left_idle() -> None:
    """A reserved slot is never left idle.

    If one stratum cannot fill its share, the other takes the remainder.
    """
    synthetic_units = [unit(f"s{index}", label="api_key") for index in range(100)]
    own_units = [unit("own-0", source="own", label=None)]
    units = synthetic_units + own_units
    rules = {u.id: rule(u.id) for u in units}
    teacher = {u.id: distribution("none", 0.15) for u in synthetic_units}
    teacher["own-0"] = distribution("token", 0.40)

    queue = build_queue(units, rules, teacher, {"own-0": False}, cap=20, own_share=0.5)
    assert len(queue) == 20
    assert sum(1 for item in queue if item.unit.source == "own") == 1
