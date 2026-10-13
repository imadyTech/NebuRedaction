"""Merging rule, teacher and review labels into the training files (NR-0003 §2.5).

Applies the §2.3 table — argued in ``labeling/README.md`` — and writes
``data/labels/train_v0.jsonl`` and ``val_v0.jsonl`` with a hard label, the
teacher's soft label where one exists, and the provenance of the hard label.

Three decisions about what goes in the files:

**Every synthetic unit is included, not only the ones the teacher saw.** The
teacher's scope samples the rule-high-confidence pool, so roughly 45,000
training units have no soft label. Their hard labels are still
construction-guaranteed and still worth training on, so they are written with
``teacher: null`` and NR-0004 filters if it wants only distillation targets.
Dropping them would throw away a quarter of the corpus to tidy a schema.

**The own logs go to validation, not training.** They are NebuLog's real
attribute distribution but contain no positives at all, so their training value
is a handful of negatives while their measurement value is the only
in-distribution read the project has on the false-positive rate — which is
NR-0002 finding 5's actual complaint. They carry ``source: own`` so the
evaluation can slice them out rather than blending them into a synthetic
average.

**A label with no provenance is a bug, not a default.** NR-0004 trains on these
files, and the difference between a construction-guaranteed label and a model's
guess is the difference between distillation and self-training. Every record
says which it is, and the writer refuses a record that does not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagen.unit import LABELS, DetectionUnit

from labeling.review import Decision, classify_for_queue, latest_decisions
from labeling.rules import RuleLabel

#: Accepted values for ``meta.label_source``; see labeling/README.md.
SOURCES = frozenset(
    {"generator", "teacher_auto", "human", "generator_after_review", "generator_unreviewed"}
)


@dataclass(frozen=True, slots=True)
class MergedLabel:
    """One merged label.

    Attributes:
        unit: The unit.
        label: The hard label.
        label_source: Where the hard label came from.
        teacher: The teacher's recalibrated distribution, or None if it was
            never asked about this unit.
        teacher_agreed: Whether the teacher agreed about redaction, or None
            when there is nothing to compare.
        review_verdict: The reviewer's verdict, when this unit was reviewed.
    """

    unit: DetectionUnit
    label: str
    label_source: str
    teacher: dict[str, float] | None
    teacher_agreed: bool | None
    review_verdict: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the training file."""
        meta = dict(self.unit.meta or {})
        meta["label_source"] = self.label_source
        if self.teacher_agreed is not None:
            meta["teacher_agreed"] = self.teacher_agreed
        if self.review_verdict is not None:
            meta["review_verdict"] = self.review_verdict
        record: dict[str, Any] = {
            "id": self.unit.id,
            "key": self.unit.key,
            "value": self.unit.value,
            "source": self.unit.source,
            "label": self.label,
            "meta": meta,
        }
        if self.unit.template is not None:
            record["template"] = self.unit.template
        if self.unit.scope is not None:
            record["scope"] = self.unit.scope
        if self.unit.family is not None:
            record["family"] = self.unit.family
        record["teacher"] = (
            {label: round(value, 6) for label, value in self.teacher.items()}
            if self.teacher is not None
            else None
        )
        return record


def merge_unit(
    unit: DetectionUnit,
    rule: RuleLabel | None,
    teacher: dict[str, float] | None,
    credsweeper: bool | None,
    decision: Decision | None,
) -> MergedLabel | None:
    """Decide one unit's final label.

    Args:
        unit: The unit.
        rule: Its rule label, if the rule pass covered it.
        teacher: The teacher's recalibrated distribution, or None.
        credsweeper: CredSweeper's binary opinion, for own-log units.
        decision: The reviewer's decision, if this unit was reviewed.

    Returns:
        The merged label, or ``None`` when the unit cannot be labelled and must
        be left out of the training files.

    Raises:
        ValueError: If an own-log unit has no teacher label either. Every
            own-log unit was in the teacher's scope, so a missing one means the
            merge is running against a different corpus than the run labelled.

    A reviewer's decision wins over everything, which is the point of the
    review.

    **The None case is the own-log unit that §2.3 sent to a human and no human
    decided** — because it fell outside §2.4's 500-unit cap, or because the
    reviewer skipped it. There is no honest label for it, and both candidate
    fallbacks are worse than omission. Writing the teacher's choice would put
    into a training file the very label the merge strategy declined to
    auto-adopt. Writing ``none`` would manufacture a negative out of an absence
    of evidence — and on this corpus it would usually be right, which is what
    makes it dangerous: these are the units where the teacher saw a credential,
    so defaulting them would systematically teach the student to ignore exactly
    the cases a human was asked to look at. Omitting them loses data and says
    so in the statistics.
    """
    choice = max(teacher, key=lambda label: teacher[label]) if teacher else None
    agreed: bool | None = None
    if teacher is not None and unit.label is not None:
        agreed = (choice != "none") == (unit.label != "none")

    if decision is not None and decision.new_label is not None:
        source = "human"
        if unit.label is not None and decision.new_label == unit.label:
            source = "generator_after_review"
        return MergedLabel(unit, decision.new_label, source, teacher, agreed, decision.verdict)

    verdict = decision.verdict if decision is not None else None

    if unit.label is not None:
        # Synthetic: the generator wins, reviewed or not (§2.3 row 1 and 2).
        if teacher is None:
            return MergedLabel(unit, unit.label, "generator_unreviewed", None, None, verdict)
        return MergedLabel(unit, unit.label, "generator", teacher, agreed, verdict)

    if teacher is None:
        raise ValueError(
            f"own-log unit {unit.id!r} has neither a generator label nor a teacher label; "
            "every own-log unit was in the teacher's scope, so this means the merge is "
            "running against a different corpus than the run labelled"
        )

    needs_review, _, disposition = classify_for_queue(
        unit, rule or RuleLabel(unit.id, "none", "low", "none", 0.0), teacher, credsweeper
    )
    if needs_review:
        # Queued but not decided: outside the cap, or skipped. The docstring
        # argues why this is an omission rather than a default.
        return None
    assert choice is not None
    return MergedLabel(unit, choice, disposition, teacher, None, verdict)


def _agreement_by_class(merged: Sequence[MergedLabel]) -> dict[str, dict[str, Any]]:
    """Teacher/generator agreement rate per class (NR-0003 §2.5 item 2).

    Args:
        merged: The merged labels.

    Returns:
        Per generator class: how many units the teacher was asked about, how
        many it agreed with about redaction, and the rate.

    Only units with both a generator label and a teacher opinion count. This is
    what §2.2 item 3's stratified high-confidence sample exists to make
    measurable for the classes the teacher would otherwise rarely be asked
    about.
    """
    counts: dict[str, dict[str, Any]] = {}
    for item in merged:
        if item.teacher_agreed is None or item.unit.label is None:
            continue
        bucket = counts.setdefault(item.unit.label, {"asked": 0, "agreed": 0})
        bucket["asked"] += 1
        bucket["agreed"] += int(item.teacher_agreed)
    for bucket in counts.values():
        bucket["rate"] = round(bucket["agreed"] / bucket["asked"], 4) if bucket["asked"] else None
    return dict(sorted(counts.items()))


def statistics(merged: Sequence[MergedLabel], decisions: Iterable[Decision]) -> dict[str, Any]:
    """Summarise the merge (NR-0003 §2.5 item 2).

    Args:
        merged: The merged labels.
        decisions: The resolved review decisions.

    Returns:
        Label counts, provenance counts, per-class teacher agreement and the
        review's verdict distribution.
    """
    from labeling.review import statistics as review_statistics

    by_label: dict[str, int] = {}
    by_source: dict[str, int] = {}
    with_teacher = 0
    for item in merged:
        by_label[item.label] = by_label.get(item.label, 0) + 1
        by_source[item.label_source] = by_source.get(item.label_source, 0) + 1
        with_teacher += int(item.teacher is not None)
    return {
        "units": len(merged),
        "with_teacher_soft_label": with_teacher,
        "by_label": dict(sorted(by_label.items())),
        "by_label_source": dict(sorted(by_source.items())),
        "teacher_generator_agreement": _agreement_by_class(merged),
        "review": review_statistics(decisions),
    }


def write_merged(merged: Iterable[MergedLabel], path: Path) -> int:
    """Write merged labels as JSON Lines.

    Args:
        merged: The labels to write.
        path: Output file.

    Returns:
        How many were written.

    Raises:
        ValueError: If a label or a provenance is not recognised. NR-0004 reads
            these files; an unknown value here becomes a silent training bug
            there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in merged:
            if item.label not in LABELS:
                raise ValueError(
                    f"{item.unit.id}: label {item.label!r} is not one of {list(LABELS)}"
                )
            if item.label_source not in SOURCES:
                raise ValueError(
                    f"{item.unit.id}: label_source {item.label_source!r} is not one of "
                    f"{sorted(SOURCES)}"
                )
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def resolve_decisions(path: Path) -> dict[str, Decision]:
    """Read and collapse the review log.

    Args:
        path: ``data/labels/review_v0.jsonl``.

    Returns:
        One decision per unit, the last recorded winning.
    """
    from labeling.review import read_decisions

    return latest_decisions(read_decisions(path))
