"""Rule labelling with a confidence tier (NR-0003 §2.1).

This reuses NR-0002's rule layer rather than reimplementing it. The layer
already decides a label and reports which rule fired; what §2.1 adds is a
*confidence tier*, and the tier is what the teacher's call scope is built on
(§2.2 item 3): low-confidence units all go to the teacher, high-confidence ones
are sampled.

**The tier is about the strength of the evidence, not the probability of being
right.** A matched vendor value pattern is structural evidence: a string shaped
like a GitHub token, checksum included, is one almost regardless of context. A
key name is only a *hint about a value that matched nothing*, and NR-0002
measured what acting on it costs — every one of the rule baseline's 1,296 false
positives came from a key name or a template, 416 from the template alone.
Entropy is weaker still. So:

- **high** — a value pattern matched.
- **low** — anything else: a key name, a template, raised entropy, or silence.

Silence sits in the low tier deliberately. A unit no rule fires on is labelled
``none``, but absence of evidence is not evidence of absence, and treating a
confident ``none`` as settled would hide exactly the secrets with no structural
marker that plan §3.3 and NR-0002's finding 3 identify as the project's reason
to exist.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagen.unit import DetectionUnit
from eval.baselines.rules import VALUE_RULES, classify_key
from eval.baselines.rules_entropy import normalised_entropy

VERSION = "0.1.0"

#: Confidence tiers. Two, not three: §2.2's call scope is a binary split, and a
#: middle tier would have to be given a meaning that no measurement supports.
HIGH = "high"
LOW = "low"

#: Normalised entropy above which a value is treated as a weak positive signal
#: in its own right. It only ever produces a *low*-confidence label, so the
#: exact figure moves which units the teacher sees, never a final label.
ENTROPY_SIGNAL = 0.90

#: Evidence string used when no rule fired at all.
NO_EVIDENCE = "none"


@dataclass(frozen=True, slots=True)
class RuleLabel:
    """One unit's rule label.

    Attributes:
        unit_id: The unit's id.
        label: The class the rules decided, or ``none`` when nothing fired.
        confidence: :data:`HIGH` or :data:`LOW`.
        evidence: What decided it — a value-rule name, ``key:<class>``,
            ``template:<class>``, ``entropy``, or ``none``.
        entropy: The value's normalised entropy, recorded whether or not it
            contributed, because the teacher's sampling and the later error
            analysis both want it.
    """

    unit_id: str
    label: str
    confidence: str
    evidence: str
    entropy: float

    @property
    def is_high_confidence(self) -> bool:
        """Whether this label rests on a matched value pattern."""
        return self.confidence == HIGH

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON Lines output."""
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "entropy": round(self.entropy, 6),
        }


def label_unit(unit: DetectionUnit) -> RuleLabel:
    """Assign a rule label and confidence tier to one unit.

    Args:
        unit: The unit to label. Its own ``label`` is never read, so this
            behaves identically on the labelled synthetic data and on the
            unlabelled own logs.

    Returns:
        The rule label.

    The order is value patterns, then key name, then template, then entropy —
    strongest evidence first, so the recorded evidence is the best available
    rather than whichever rule happened to be checked last. Only the first step
    yields high confidence.
    """
    entropy = normalised_entropy(unit.value)

    for rule in VALUE_RULES:
        if rule.pattern.search(unit.value):
            return RuleLabel(unit.id, rule.label, HIGH, rule.name, entropy)

    from_key = classify_key(unit.key)
    if from_key is not None:
        return RuleLabel(unit.id, from_key, LOW, f"key:{from_key}", entropy)

    # The template is consulted only for body-mode units, which have no real key
    # name. NR-0002 finding 2 measured the template as the single largest source
    # of false positives, so it is both the weakest signal here and the one whose
    # units most need a teacher opinion.
    if unit.template and unit.key.startswith("<slot:"):
        from_template = classify_key(unit.template)
        if from_template is not None:
            return RuleLabel(unit.id, from_template, LOW, f"template:{from_template}", entropy)

    if entropy >= ENTROPY_SIGNAL:
        # Entropy says "this looks random", not which class it is. There is no
        # honest class to report, so the label stays `none` and the evidence
        # records that something was seen — which is enough to route the unit to
        # the teacher, the only thing a low-confidence label is used for.
        return RuleLabel(unit.id, "none", LOW, "entropy", entropy)

    return RuleLabel(unit.id, "none", LOW, NO_EVIDENCE, entropy)


def label_units(units: Iterable[DetectionUnit]) -> Iterator[RuleLabel]:
    """Label a stream of units.

    Args:
        units: The units to label.

    Yields:
        One :class:`RuleLabel` per unit, in order.
    """
    for unit in units:
        yield label_unit(unit)


def statistics(labels: Sequence[RuleLabel]) -> dict[str, Any]:
    """Summarise a set of rule labels.

    Args:
        labels: The labels to summarise.

    Returns:
        Counts by confidence tier, by decided label and by evidence kind. The
        evidence kind collapses the many distinct value-rule names into
        ``value_rule`` while keeping ``key``, ``template``, ``entropy`` and
        ``none`` apart, which is the breakdown §2.5 item 2 reports on.
    """
    by_confidence: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_evidence: dict[str, int] = {}
    for label in labels:
        by_confidence[label.confidence] = by_confidence.get(label.confidence, 0) + 1
        by_label[label.label] = by_label.get(label.label, 0) + 1
        # The many distinct value-rule names collapse to one bucket; the weak
        # signals stay apart, because which weak signal fired is the interesting
        # part (NR-0002 finding 2 is entirely about `key` versus `template`).
        kind = "value_rule" if label.confidence == HIGH else label.evidence.split(":", 1)[0]
        by_evidence[kind] = by_evidence.get(kind, 0) + 1
    return {
        "units": len(labels),
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_evidence": dict(sorted(by_evidence.items())),
    }


def write_labels(labels: Iterable[RuleLabel], path: Path) -> int:
    """Write rule labels as JSON Lines.

    Args:
        labels: The labels to write.
        path: Output file; parent directories are created.

    Returns:
        How many labels were written.

    No value is ever written, only the unit id and the decision. The file is
    therefore safe to keep beside the data whichever corpus it describes, which
    matters because the same labeller runs over the own logs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_labels(path: Path) -> list[RuleLabel]:
    """Read rule labels back.

    Args:
        path: A file written by :func:`write_labels`.

    Returns:
        The labels, in file order.
    """
    labels: list[RuleLabel] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            labels.append(
                RuleLabel(
                    unit_id=record["unit_id"],
                    label=record["label"],
                    confidence=record["confidence"],
                    evidence=record["evidence"],
                    entropy=float(record["entropy"]),
                )
            )
    return labels
