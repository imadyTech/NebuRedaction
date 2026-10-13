"""Tests for the teacher's call scope and the prompt (NR-0003 §2.2 items 1 and 3)."""

from __future__ import annotations

import pytest
from datagen.unit import LABELS, DetectionUnit
from labeling.rules import HIGH, LOW, RuleLabel
from labeling.teacher.prompt import PROMPTS_DIR, QUESTION_KEY, _parse, load_prompt
from labeling.teacher.scope import (
    BUDGET_HOURS,
    MEASURED_SAMPLES_PER_SECOND,
    plan_scope,
    projected_hours,
    ratios_within_budget,
)


def units_and_labels(
    per_class: int, high_fraction: float
) -> tuple[list[DetectionUnit], list[RuleLabel]]:
    """Build a synthetic corpus with a known class and confidence composition."""
    units: list[DetectionUnit] = []
    labels: list[RuleLabel] = []
    for label_name in LABELS:
        for index in range(per_class):
            unit_id = f"{label_name}-{index:04d}"
            units.append(
                DetectionUnit(id=unit_id, key="k", value="v", source="synth", label=label_name)
            )
            confidence = HIGH if index < per_class * high_fraction else LOW
            labels.append(RuleLabel(unit_id, label_name, confidence, "x", 0.0))
    return units, labels


def own(count: int) -> list[DetectionUnit]:
    """Build own-log units."""
    return [
        DetectionUnit(id=f"own-{index:04d}", key="OrderId", value="v", source="own", label=None)
        for index in range(count)
    ]


def test_the_work_orders_scope_includes_every_own_log_unit() -> None:
    """§2.2 item 3: own logs are all labelled, unconditionally."""
    units, labels = units_and_labels(100, 0.25)
    plan = plan_scope(
        units, labels, own(50), low_confidence_ratio=1.0, high_confidence_ratio=0.1, seed=1
    )
    assert plan.own == 50
    assert sum(1 for unit_id in plan.unit_ids if unit_id.startswith("own-")) == 50


def test_every_low_confidence_unit_is_in_scope_at_ratio_one() -> None:
    """The units no rule can settle are where the teacher's opinion is worth most."""
    units, labels = units_and_labels(100, 0.25)
    plan = plan_scope(
        units, labels, [], low_confidence_ratio=1.0, high_confidence_ratio=0.1, seed=1
    )
    assert plan.low_confidence_selected == plan.low_confidence_total


def test_the_high_confidence_sample_is_stratified_by_class() -> None:
    """§2.2 item 3 wants a per-class agreement rate, so every class must appear.

    An unstratified 10% of a corpus where one class is 3% of the units can miss
    that class entirely, and then it has no agreement rate at all.
    """
    units, labels = units_and_labels(100, 0.40)
    plan = plan_scope(
        units, labels, [], low_confidence_ratio=1.0, high_confidence_ratio=0.1, seed=1
    )
    sampled = plan.by_class["high_confidence"]
    assert set(sampled) == set(LABELS)
    assert all(count == 4 for count in sampled.values()), sampled


def test_a_rare_class_still_gets_at_least_one_unit() -> None:
    """Rounding to zero would silently drop a class from the measurement."""
    units = [
        DetectionUnit(id="a-1", key="k", value="v", source="synth", label="cookie"),
        *[
            DetectionUnit(id=f"b-{i}", key="k", value="v", source="synth", label="api_key")
            for i in range(500)
        ],
    ]
    labels = [RuleLabel(unit.id, "x", HIGH, "x", 0.0) for unit in units]
    plan = plan_scope(
        units, labels, [], low_confidence_ratio=1.0, high_confidence_ratio=0.01, seed=1
    )
    assert plan.by_class["high_confidence"]["cookie"] == 1


def test_the_plan_is_reproducible_from_its_seed() -> None:
    """§2.2 item 3 fixes the seed; two plans must be identical."""
    units, labels = units_and_labels(80, 0.5)
    first = plan_scope(
        units, labels, own(10), low_confidence_ratio=0.5, high_confidence_ratio=0.1, seed=7
    )
    second = plan_scope(
        units, labels, own(10), low_confidence_ratio=0.5, high_confidence_ratio=0.1, seed=7
    )
    assert first.unit_ids == second.unit_ids
    different = plan_scope(
        units, labels, own(10), low_confidence_ratio=0.5, high_confidence_ratio=0.1, seed=8
    )
    assert different.unit_ids != first.unit_ids


def test_the_scope_is_in_corpus_order_not_selection_order() -> None:
    """A resumed run must process units in the same sequence a fresh one would."""
    units, labels = units_and_labels(40, 0.5)
    plan = plan_scope(
        units, labels, own(5), low_confidence_ratio=1.0, high_confidence_ratio=1.0, seed=3
    )
    corpus_order = [unit.id for unit in units]
    selected = [unit_id for unit_id in plan.unit_ids if not unit_id.startswith("own-")]
    assert selected == [unit_id for unit_id in corpus_order if unit_id in set(selected)]


def test_a_unit_without_a_rule_label_is_an_error() -> None:
    """Silently dropping it would shrink both pools invisibly."""
    units, labels = units_and_labels(10, 0.5)
    with pytest.raises(ValueError, match="no rule label"):
        plan_scope(
            units, labels[:-1], [], low_confidence_ratio=1.0, high_confidence_ratio=0.1, seed=1
        )


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_ratios_outside_the_unit_interval_are_rejected(ratio: float) -> None:
    """A ratio above one would mean sampling with replacement, which is not meant."""
    units, labels = units_and_labels(10, 0.5)
    with pytest.raises(ValueError, match="must be in"):
        plan_scope(units, labels, [], low_confidence_ratio=ratio, high_confidence_ratio=0.1, seed=1)


def test_the_measured_corpus_fits_the_budget_without_subsampling() -> None:
    """The decision this module exists to justify, pinned as a test.

    With the §2.2A trial's measured rate the full scope as §2.2 item 3 writes it
    projects well inside the twelve-hour budget, so no ratio needs reducing.
    If a future change to the corpus or the rate breaks that, this fails and
    the subsampling question has to be reopened deliberately.
    """
    calls = 2692 + 149515 + 5049
    assert projected_hours(calls) < BUDGET_HOURS
    assert projected_hours(calls, MEASURED_SAMPLES_PER_SECOND) == pytest.approx(6.91, abs=0.05)


def test_the_budget_solver_keeps_the_high_confidence_sample() -> None:
    """Reducing the high ratio first, as §2.2 words it, cannot reach the budget.

    On the measured corpus the high-confidence pool is a quarter of the units
    and its 10% sample is 3% of the work, so following the letter means zeroing
    the only thing that estimates the per-class agreement rate and then cutting
    the low pool anyway. The solver cuts the low pool and keeps the sample.
    """
    low, high = ratios_within_budget(2692, 149515, 50485, budget_hours=4.0)
    assert high == pytest.approx(0.10)
    assert 0.0 < low < 1.0


def test_the_budget_solver_refuses_an_impossible_budget() -> None:
    """Own logs are included unconditionally; if they alone overrun, say so."""
    with pytest.raises(ValueError, match="cannot both be met"):
        ratios_within_budget(1_000_000, 10, 10, budget_hours=1.0)


def test_the_prompt_covers_exactly_the_seven_classes() -> None:
    """The prompt names all seven classes.

    A missing class is unreachable for the teacher while every output file still
    looks well-formed, so it is a hard failure rather than a default.
    """
    prompt = load_prompt()
    assert list(prompt.criteria) == list(LABELS)
    assert set(prompt.questions()[QUESTION_KEY]["criteria"]) == set(LABELS)


def test_the_prompt_digest_changes_with_the_file() -> None:
    """The digest is what ties a label to its wording."""
    first = load_prompt("v1")
    text = (PROMPTS_DIR / "v1.txt").read_text(encoding="utf-8")
    altered = _parse(text.replace("Answer 'none'", "Answer none"), "v1", "different-digest")
    assert altered.instructions != first.instructions
    assert altered.digest != first.digest


def test_a_prompt_missing_a_class_is_refused() -> None:
    """Parsing, not labelling, is where this must fail."""
    text = (PROMPTS_DIR / "v1.txt").read_text(encoding="utf-8")
    without_cookie = text.split("[criteria:cookie]")[0]
    with pytest.raises(ValueError, match="criteria missing"):
        _parse(without_cookie, "test", "d")


def test_a_duplicated_section_is_refused() -> None:
    """Two `[instructions]` blocks would silently use one of them."""
    text = (PROMPTS_DIR / "v1.txt").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate section"):
        _parse(text + "\n[instructions]\nsomething else\n", "test", "d")


def test_absent_optional_fields_render_as_a_constant() -> None:
    """Absent and empty optional fields render identically.

    So the prompt's shape never varies with the data.
    """
    prompt = load_prompt()
    missing = DetectionUnit(id="a", key="k", value="v", source="synth")
    empty = DetectionUnit(id="b", key="k", value="v", source="synth", template="", scope="")
    assert prompt.render(missing) == prompt.render(empty)
    assert "(none)" in prompt.render(missing)


def test_the_warning_section_is_part_of_the_instructions() -> None:
    """The warning reaches the model.

    It is kept separate in the file so the report can say when it was
    introduced, but the model has one instructions field and must see it.
    """
    prompt = load_prompt()
    instructions = prompt.questions()[QUESTION_KEY]["instructions"]
    assert prompt.warning in instructions
    assert "not evidence" in instructions
