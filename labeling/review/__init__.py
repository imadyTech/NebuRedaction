"""The review queue and its records (NR-0003 §2.4).

The queue logic lives here and the terminal interface lives in ``__main__``,
because the ordering, the cap and the record format are what correctness depends
on and they should be testable without a keyboard.

**A command-line tool rather than a local web page**, which §2.4 item 1 leaves
to the executor with an explanation required. Four reasons, in the order they
mattered:

1. Reviewing up to 500 units in one to two hours is eight to fourteen seconds
   each. That is a single-keystroke task, which is what a terminal is for; a
   browser adds a mouse and a round trip to every decision.
2. A localhost server is an open port and a running process in a project whose
   entire subject is credential leakage. The values on screen during review are
   the most sensitive data the project handles — real CredData-adjacent material
   is excluded, but synthetic credentials and NebuLog's own logs are not
   nothing. A tool with no socket cannot be reached by anything.
3. Resumption is an append-only file read back at startup, with no session or
   server state to reconcile.
4. It works unchanged over SSH and in a remote session.

The one thing a page would do better is render a multi-line PEM block, so the
value is shown truncated with an explicit key to expand it, rather than letting
a 1,700-character private key scroll the context off the screen.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datagen.unit import LABELS, DetectionUnit

from labeling.rules import RuleLabel

#: §2.4 item 4 caps the queue. A cap is not a detail: an uncapped queue on this
#: corpus would be tens of thousands of units, which is not a review but an
#: abandonment, and the ordering below is what makes the first 500 the ones
#: worth a human's attention.
QUEUE_CAP = 500

#: The threshold §2.3 applies to own-log units, on *recalibrated* probabilities.
AUTO_ADOPT_THRESHOLD = 0.9


def entropy(probabilities: dict[str, float]) -> float:
    """Shannon entropy of a distribution, in bits.

    Args:
        probabilities: A distribution over the seven classes.

    Returns:
        The entropy. Zero when the teacher is certain, log2(7) ~ 2.807 when it
        has no idea.

    This is the queue's secondary sort key (§2.4 item 2). It measures how
    confused the teacher is, which is a different question from whether it
    disagrees with the generator: a unit where the teacher is torn between
    `token` and `api_key` is worth less review time than one where it spreads
    probability across all seven.
    """
    total = 0.0
    for value in probabilities.values():
        if value > 0.0:
            total -= value * math.log2(value)
    return total


@dataclass(frozen=True, slots=True)
class QueueItem:
    """One unit awaiting review.

    Attributes:
        unit: The detection unit.
        rule: The rule labeller's verdict.
        teacher: The teacher's recalibrated distribution.
        teacher_choice: The teacher's top class.
        credsweeper: CredSweeper's binary opinion for own-log units, or None
            when it was not consulted.
        reason: Why this unit is in the queue.
        disagreement: Whether this is a disagreement, which sorts first.
        entropy: The teacher's entropy, the secondary sort key.
    """

    unit: DetectionUnit
    rule: RuleLabel
    teacher: dict[str, float]
    teacher_choice: str
    credsweeper: bool | None
    reason: str
    disagreement: bool
    entropy: float

    @property
    def current_label(self) -> str:
        """The label that stands if the reviewer accepts it unchanged.

        For a synthetic unit that is the generator's label, which §2.3 says
        wins; for an own-log unit there is no prior label, so it is the
        teacher's choice.
        """
        return self.unit.label if self.unit.label is not None else self.teacher_choice

    def to_dict(self) -> dict[str, Any]:
        """Serialise the queue entry, values included.

        The queue file necessarily contains values — the reviewer has to see
        them — so it lives under `data/`, which `.gitignore` excludes in full.
        """
        return {
            "unit_id": self.unit.id,
            "key": self.unit.key,
            "value": self.unit.value,
            "template": self.unit.template,
            "scope": self.unit.scope,
            "source": self.unit.source,
            "family": self.unit.family,
            "generator_label": self.unit.label,
            "rule": {
                "label": self.rule.label,
                "confidence": self.rule.confidence,
                "evidence": self.rule.evidence,
            },
            "teacher": {
                "choice": self.teacher_choice,
                "probabilities": {label: round(v, 6) for label, v in self.teacher.items()},
                "entropy": round(self.entropy, 4),
            },
            "credsweeper_secret": self.credsweeper,
            "reason": self.reason,
            "disagreement": self.disagreement,
        }


def should_redact(label: str) -> bool:
    """Whether a class means "redact this".

    Args:
        label: One of the seven classes.

    Returns:
        True for every class but ``none``.
    """
    return label != "none"


def classify_for_queue(
    unit: DetectionUnit,
    rule: RuleLabel,
    teacher: dict[str, float],
    credsweeper: bool | None,
) -> tuple[bool, str, str]:
    """Apply §2.3's merge table to one unit.

    Args:
        unit: The unit.
        rule: Its rule label.
        teacher: The teacher's **recalibrated** distribution.
        credsweeper: CredSweeper's opinion, for own-log units.

    Returns:
        ``(needs_review, reason, disposition)``, where the disposition is the
        ``meta.label_source`` the merge will record if no review happens.

    The disagreement test is about redaction, not class. A secret the teacher
    calls a `token` and the generator calls an `api_key` is masked either way,
    so it is recorded as a class confusion in the statistics but is not worth a
    human's time; `labeling/README.md` argues this at length.
    """
    choice = max(teacher, key=lambda label: teacher[label])
    confidence = teacher[choice]

    if unit.source == "own":
        if credsweeper is None:
            return True, "own:no_second_opinion", "review"
        if should_redact(choice) != credsweeper:
            return True, "own:laya_credsweeper_disagree", "review"
        if confidence < AUTO_ADOPT_THRESHOLD:
            return True, "own:below_threshold", "review"
        return False, "own:auto_adopted", "teacher_auto"

    if unit.label is None:
        return True, "synthetic:unlabelled", "review"
    if should_redact(choice) != should_redact(unit.label):
        return True, "synthetic:teacher_generator_disagree", "review"
    return False, "synthetic:agreed", "generator"


def build_queue(
    units: Sequence[DetectionUnit],
    rules: dict[str, RuleLabel],
    teacher: dict[str, dict[str, float]],
    credsweeper: dict[str, bool] | None = None,
    cap: int = QUEUE_CAP,
) -> list[QueueItem]:
    """Build the review queue in review order.

    Args:
        units: Every unit that was teacher-labelled.
        rules: Rule labels by unit id.
        teacher: Recalibrated distributions by unit id.
        credsweeper: CredSweeper opinions by unit id, for own-log units.
        cap: Maximum queue length (§2.4 item 4).

    Returns:
        At most ``cap`` items, disagreements first and then by descending
        teacher entropy (§2.4 item 2).

    Raises:
        ValueError: If a unit has a teacher label but no rule label, which would
            mean the two passes ran over different corpora.

    The cap is applied **after** sorting, so what survives is the top of the
    ordering rather than whatever came first in the corpus. Ties are broken by
    unit id so that two runs over the same data produce the same queue and a
    partly-reviewed queue does not reshuffle underneath the reviewer.
    """
    opinions = credsweeper or {}
    items: list[QueueItem] = []
    for unit in units:
        distribution = teacher.get(unit.id)
        if distribution is None:
            continue
        rule = rules.get(unit.id)
        if rule is None:
            raise ValueError(f"unit {unit.id!r} has a teacher label but no rule label")
        needs_review, reason, _ = classify_for_queue(
            unit, rule, distribution, opinions.get(unit.id)
        )
        if not needs_review:
            continue
        items.append(
            QueueItem(
                unit=unit,
                rule=rule,
                teacher=distribution,
                teacher_choice=max(distribution, key=lambda label: distribution[label]),
                credsweeper=opinions.get(unit.id),
                reason=reason,
                disagreement="disagree" in reason,
                entropy=entropy(distribution),
            )
        )

    items.sort(key=lambda item: (not item.disagreement, -item.entropy, item.unit.id))
    return items[:cap]


@dataclass(frozen=True, slots=True)
class Decision:
    """One reviewer decision (NR-0003 §2.4 item 3).

    Attributes:
        unit_id: The unit reviewed.
        reviewer: Who reviewed it.
        reviewed_utc: When.
        original_label: The label that stood before review — the generator's for
            a synthetic unit, the teacher's choice for an own-log one.
        new_label: What the reviewer decided.
        verdict: ``confirmed`` when the original stands, ``corrected``
            otherwise, or ``skipped``.
        note: Optional free text, used to flag a suspected generator defect.
    """

    unit_id: str
    reviewer: str
    reviewed_utc: str
    original_label: str
    new_label: str | None
    verdict: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``data/labels/review_v0.jsonl``."""
        return {
            "unit_id": self.unit_id,
            "reviewer": self.reviewer,
            "reviewed_utc": self.reviewed_utc,
            "original_label": self.original_label,
            "new_label": self.new_label,
            "verdict": self.verdict,
            "note": self.note,
        }


def decide(item: QueueItem, reviewer: str, new_label: str | None, note: str = "") -> Decision:
    """Record a decision about a queue item.

    Args:
        item: The item reviewed.
        reviewer: Who is reviewing.
        new_label: The chosen class, or None to skip.
        note: Optional free text.

    Returns:
        The decision.

    Raises:
        ValueError: If the label is not one of the seven classes.
    """
    if new_label is not None and new_label not in LABELS:
        raise ValueError(f"{new_label!r} is not one of {list(LABELS)}")
    if new_label is None:
        verdict = "skipped"
    elif new_label == item.current_label:
        verdict = "confirmed"
    else:
        verdict = "corrected"
    return Decision(
        unit_id=item.unit.id,
        reviewer=reviewer,
        reviewed_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        original_label=item.current_label,
        new_label=new_label,
        verdict=verdict,
        note=note,
    )


def append_decision(decision: Decision, path: Path) -> None:
    """Append one decision, flushed immediately.

    Args:
        decision: The decision to record.
        path: ``data/labels/review_v0.jsonl``.

    Flushed per decision rather than per session: a review interrupted after
    three hundred units must not lose them, and §2.4 item 1 requires the tool to
    be interruptible and resumable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def read_decisions(path: Path) -> list[Decision]:
    """Read recorded decisions, latest last.

    Args:
        path: The review file; may not exist.

    Returns:
        The decisions in file order. A unit may appear more than once if it was
        revisited after an undo; the last entry is the one that counts, and
        :func:`latest_decisions` resolves that.
    """
    if not path.is_file():
        return []
    decisions: list[Decision] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            decisions.append(
                Decision(
                    unit_id=record["unit_id"],
                    reviewer=record["reviewer"],
                    reviewed_utc=record["reviewed_utc"],
                    original_label=record["original_label"],
                    new_label=record["new_label"],
                    verdict=record["verdict"],
                    note=record.get("note", ""),
                )
            )
    return decisions


def latest_decisions(decisions: Iterable[Decision]) -> dict[str, Decision]:
    """Resolve a decision log to one decision per unit.

    Args:
        decisions: Decisions in file order.

    Returns:
        The last decision for each unit.

    An undo is recorded as a further entry rather than by rewriting the file,
    so the log is append-only and the reviewer's changes of mind stay auditable.
    This is what collapses it for use.
    """
    resolved: dict[str, Decision] = {}
    for decision in decisions:
        resolved[decision.unit_id] = decision
    return resolved


def statistics(decisions: Iterable[Decision]) -> dict[str, Any]:
    """Summarise a review (NR-0003 §2.5 item 2).

    Args:
        decisions: The resolved decisions.

    Returns:
        Counts by verdict and by resulting label, and the notes, which are where
        a suspected generator defect gets flagged.
    """
    resolved = list(decisions)
    by_verdict: dict[str, int] = {}
    by_label: dict[str, int] = {}
    notes: list[dict[str, str]] = []
    for decision in resolved:
        by_verdict[decision.verdict] = by_verdict.get(decision.verdict, 0) + 1
        if decision.new_label is not None:
            by_label[decision.new_label] = by_label.get(decision.new_label, 0) + 1
        if decision.note:
            notes.append({"unit_id": decision.unit_id, "note": decision.note})
    return {
        "reviewed": len(resolved),
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_label": dict(sorted(by_label.items())),
        "notes": notes,
    }
