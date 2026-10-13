"""Choosing which units the teacher is asked about (NR-0003 §2.2 item 3).

§2.2 item 3 sets the scope: all own-log units, all rule-low-confidence synthetic
training units, and a class-stratified 10% of the rule-high-confidence ones,
with a fixed seed. It then adds a budget rule — if NR-0000's measured throughput
projects more than 12 hours, reduce the synthetic sampling ratio and justify the
final figure in the execution report.

**Measured before building this, the budget rule as written does not work**, and
the arithmetic is worth stating because the module's shape follows from it. The
training set is 74.8% low confidence — 149,515 units of 200,000. The ratio §2.2
names governs only the high-confidence remainder, so taking it from 10% to 2%
removes 4,039 calls from a total of 157,255: 15.2 hours becomes 14.8. The clause
cannot reach its own target.

The budget therefore has to be met by subsampling the low-confidence pool as
well, which is a change to §2.2's text rather than an application of it. Both
ratios are parameters here, neither has a default that silently decides it, and
:func:`ratios_within_budget` solves for the pair that fits a given number of
hours so that the choice is arithmetic rather than taste. Which pair was used is
recorded in the plan and travels into the output.

What is lost by subsampling the low-confidence pool is worth being explicit
about: those are the units no rule can settle, so they are where the teacher's
opinion is worth most and where the distilled student has the most to learn. A
smaller sample there means a student trained on fewer of the hard cases, not
merely a coarser estimate of an agreement rate.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from datagen.unit import DetectionUnit

from labeling.rules import RuleLabel

#: Laya's measured throughput on this machine, in samples per second. Every
#: projection here is this one number times a count, so it is named once.
#:
#: Measured by the NR-0003 §2.2A trial: the `multilingual` checkpoint, 10 torch
#: threads, batch size 8, inputs length-bucketed within blocks, 128 samples per
#: configuration, two rounds.
#:
#: NR-0000 §9.5 reported 2.88 and the budget arithmetic in this module's
#: docstring was originally built on it. It understated throughput for three
#: reasons, each worth recording because each was a guess that turned out
#: wrong: it measured the `english` checkpoint rather than `multilingual` (the
#: larger model is the faster one here, 6.32 against 2.86), on ten samples, and
#: without length bucketing (+32%).
MEASURED_SAMPLES_PER_SECOND = 6.32

#: NR-0000's figure, kept so that a reader who finds it in that report can see
#: why the projections differ.
NR0000_SAMPLES_PER_SECOND = 2.88

#: The budget §2.2 item 3 sets, in hours.
BUDGET_HOURS = 12.0

#: The ratio §2.2 item 3 specifies for the high-confidence pool.
SPECIFIED_HIGH_CONFIDENCE_RATIO = 0.10


@dataclass(frozen=True, slots=True)
class ScopePlan:
    """What the teacher will be asked, and what it is projected to cost.

    Attributes:
        unit_ids: The selected unit ids, in corpus order. Order is corpus order
            rather than selection order so that a resumed run processes units in
            the same sequence whatever the sampling ratios were.
        own: How many own-log units were included.
        low_confidence_selected: Selected from the low-confidence pool.
        low_confidence_total: The pool's size.
        high_confidence_selected: Selected from the high-confidence pool.
        high_confidence_total: The pool's size.
        low_confidence_ratio: The ratio applied to the low-confidence pool.
        high_confidence_ratio: The ratio applied to the high-confidence pool.
        seed: The sampling seed.
        by_class: Selected counts per class, for the low and high pools.
    """

    unit_ids: tuple[str, ...]
    own: int
    low_confidence_selected: int
    low_confidence_total: int
    high_confidence_selected: int
    high_confidence_total: int
    low_confidence_ratio: float
    high_confidence_ratio: float
    seed: int
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        """Total teacher calls this plan requires."""
        return len(self.unit_ids)

    @property
    def projected_hours(self) -> float:
        """Projected wall-clock hours at NR-0000's measured throughput."""
        return projected_hours(self.calls)

    @property
    def within_budget(self) -> bool:
        """Whether the projection meets §2.2 item 3's 12-hour budget."""
        return self.projected_hours <= BUDGET_HOURS

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan for the run report, without the id list."""
        return {
            "calls": self.calls,
            "own": self.own,
            "low_confidence": {
                "selected": self.low_confidence_selected,
                "total": self.low_confidence_total,
                "ratio": self.low_confidence_ratio,
            },
            "high_confidence": {
                "selected": self.high_confidence_selected,
                "total": self.high_confidence_total,
                "ratio": self.high_confidence_ratio,
                "specified_by_work_order": SPECIFIED_HIGH_CONFIDENCE_RATIO,
            },
            "seed": self.seed,
            "by_class": self.by_class,
            "projection": {
                "samples_per_second": MEASURED_SAMPLES_PER_SECOND,
                "source": "NR-0003 §2.2A trial: multilingual, 10 threads, batch 8, length-bucketed",
                "hours": round(self.projected_hours, 2),
                "budget_hours": BUDGET_HOURS,
                "within_budget": self.within_budget,
            },
        }


def projected_hours(calls: int, rate: float = MEASURED_SAMPLES_PER_SECOND) -> float:
    """Project wall-clock hours for a number of teacher calls.

    Args:
        calls: How many units the teacher will be asked about.
        rate: Samples per second. Defaults to NR-0000's measurement.

    Returns:
        Projected hours.

    Raises:
        ValueError: If the rate is not positive.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    return calls / rate / 3600.0


def ratios_within_budget(
    own: int,
    low_confidence_total: int,
    high_confidence_total: int,
    budget_hours: float = BUDGET_HOURS,
    high_confidence_ratio: float = SPECIFIED_HIGH_CONFIDENCE_RATIO,
    rate: float = MEASURED_SAMPLES_PER_SECOND,
) -> tuple[float, float]:
    """Solve for sampling ratios that fit a time budget.

    Args:
        own: Own-log units, all of which are included and cannot be reduced.
        low_confidence_total: Size of the low-confidence pool.
        high_confidence_total: Size of the high-confidence pool.
        budget_hours: The budget to fit.
        high_confidence_ratio: The high-confidence ratio to try first.
        rate: Samples per second.

    Returns:
        ``(low_confidence_ratio, high_confidence_ratio)``, each in [0, 1].

    Raises:
        ValueError: If the own-log units alone exceed the budget. There is no
            ratio to return in that case, because §2.2 item 3 includes them
            unconditionally — the budget and the work order would be in direct
            conflict and that is a decision for the orchestrator, not a number
            to round.

    **The reduction order is the opposite of what §2.2's wording implies, and
    the measurement is the reason.** Reducing the high-confidence ratio first,
    as written, cannot reach the budget on this corpus — it governs 3% of the
    work — so following the letter means reducing it to zero and then cutting
    the low-confidence pool anyway. That is the worst of both: the
    high-confidence sample is the only thing that estimates the
    teacher/generator agreement rate per class, which is §2.2 item 3's own
    stated purpose for it, and at the specified 10% it costs 5,048 calls, under
    half an hour, 4% of a twelve-hour budget. Spending that to keep the estimate
    and taking the cut from the low-confidence pool loses less.

    So the high-confidence sample is preserved at its specified ratio and the
    low-confidence ratio absorbs the budget. Only if the own-log units plus the
    high-confidence sample alone overrun does the high-confidence ratio give
    way, and then the low pool is already at zero and the scope is in trouble
    for other reasons.
    """
    budget_calls = int(budget_hours * 3600.0 * rate)
    if own > budget_calls:
        raise ValueError(
            f"{own} own-log units alone project {projected_hours(own, rate):.1f} h, over the "
            f"{budget_hours} h budget; §2.2 item 3 includes them unconditionally, so the budget "
            "and the scope cannot both be met"
        )

    remaining = budget_calls - own
    wanted_high = int(high_confidence_total * high_confidence_ratio)

    if low_confidence_total + wanted_high <= remaining:
        return 1.0, high_confidence_ratio

    if wanted_high <= remaining:
        affordable_low = remaining - wanted_high
        ratio = affordable_low / low_confidence_total if low_confidence_total else 0.0
        return min(1.0, ratio), high_confidence_ratio

    # Even the high-confidence sample does not fit beside the own-log units.
    ratio = remaining / high_confidence_total if high_confidence_total else 0.0
    return 0.0, min(high_confidence_ratio, ratio)


def _stratified_sample(
    by_class: dict[str, list[str]], ratio: float, rng: random.Random
) -> dict[str, list[str]]:
    """Take a stratified sample from per-class id lists.

    Args:
        by_class: Unit ids grouped by class.
        ratio: Fraction to keep from each class, in [0, 1].
        rng: The seeded generator.

    Returns:
        The sampled ids, still grouped by class.

    Each class's count is computed as an exact integer with :func:`round` and
    then drawn, rather than keeping each unit with independent probability
    ``ratio``. The same choice as NR-0001's split composition, for the same
    reason: it makes the resulting proportion a guarantee instead of an
    expectation, so a rare class cannot vanish from the sample by luck.
    """
    if ratio >= 1.0:
        return {name: list(ids) for name, ids in by_class.items()}
    if ratio <= 0.0:
        return {name: [] for name in by_class}
    sampled: dict[str, list[str]] = {}
    for name in sorted(by_class):
        ids = by_class[name]
        # At least one unit from any non-empty class: a class the teacher is
        # never asked about has no agreement rate at all, which is worse than a
        # noisy one.
        count = max(1, round(len(ids) * ratio)) if ids else 0
        sampled[name] = rng.sample(sorted(ids), min(count, len(ids)))
    return sampled


def plan_scope(
    synthetic_units: Sequence[DetectionUnit],
    rule_labels: Sequence[RuleLabel],
    own_units: Sequence[DetectionUnit],
    *,
    low_confidence_ratio: float,
    high_confidence_ratio: float,
    seed: int,
) -> ScopePlan:
    """Choose the units the teacher will be asked about.

    Args:
        synthetic_units: The synthetic training units.
        rule_labels: Their rule labels, aligned by unit id.
        own_units: The own-log units; all are included.
        low_confidence_ratio: Fraction of the low-confidence pool to include.
            1.0 is §2.2 item 3 as written.
        high_confidence_ratio: Fraction of the high-confidence pool to include.
            0.10 is §2.2 item 3 as written.
        seed: Sampling seed, fixed so the scope is reproducible.

    Returns:
        The plan, with its projection.

    Raises:
        ValueError: If a synthetic unit has no rule label, or either ratio is
            outside [0, 1]. A missing label would silently drop units from both
            pools, so the two sequences must correspond exactly.

    Stratification is by the unit's own class, which is what makes §2.2 item 3's
    stated purpose — estimating the teacher/generator agreement rate *per class*
    — achievable. Using the true label to choose the sample is sound: it shapes
    which units are asked about, never what the teacher is told or how its
    answer is scored.
    """
    for name, ratio in (("low", low_confidence_ratio), ("high", high_confidence_ratio)):
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name}-confidence ratio must be in [0, 1], got {ratio}")

    labels_by_id = {label.unit_id: label for label in rule_labels}
    low_by_class: dict[str, list[str]] = {}
    high_by_class: dict[str, list[str]] = {}
    for unit in synthetic_units:
        label = labels_by_id.get(unit.id)
        if label is None:
            raise ValueError(f"unit {unit.id!r} has no rule label; run `python -m labeling.rules`")
        bucket = high_by_class if label.is_high_confidence else low_by_class
        bucket.setdefault(unit.label or "none", []).append(unit.id)

    rng = random.Random(seed)
    # Low first, then high, each from the same generator: the sequence of draws
    # therefore depends on both ratios, so a plan is reproducible from its
    # recorded ratios and seed but is not expected to be a subset of a plan made
    # with different ones. The run report records the ratios for that reason.
    low_sample = _stratified_sample(low_by_class, low_confidence_ratio, rng)
    high_sample = _stratified_sample(high_by_class, high_confidence_ratio, rng)

    selected = {unit_id for ids in low_sample.values() for unit_id in ids}
    selected |= {unit_id for ids in high_sample.values() for unit_id in ids}

    ordered = [unit.id for unit in own_units]
    ordered += [unit.id for unit in synthetic_units if unit.id in selected]

    return ScopePlan(
        unit_ids=tuple(ordered),
        own=len(own_units),
        low_confidence_selected=sum(len(ids) for ids in low_sample.values()),
        low_confidence_total=sum(len(ids) for ids in low_by_class.values()),
        high_confidence_selected=sum(len(ids) for ids in high_sample.values()),
        high_confidence_total=sum(len(ids) for ids in high_by_class.values()),
        low_confidence_ratio=low_confidence_ratio,
        high_confidence_ratio=high_confidence_ratio,
        seed=seed,
        by_class={
            "low_confidence": {name: len(ids) for name, ids in sorted(low_sample.items())},
            "high_confidence": {name: len(ids) for name, ids in sorted(high_sample.items())},
        },
    )
