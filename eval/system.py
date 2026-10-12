"""The interface every evaluated system implements.

NR-0002 §2.1 item 1: a system takes detection units and returns, for each unit,
a score per class. Everything downstream — thresholds, metrics, slices, the
report — works only against this interface, so the rule baselines, CredSweeper,
the Laya teacher and the student models are all measured by the same code on the
same inputs, which is what plan §6.3 requires.

Two kinds of system exist and they cannot be measured the same way:

* **continuous** — emits a score, so an operating point can be *chosen*. Its
  recall is reported at thresholds selected for a target false-positive rate.
* **binary** — a pure rule set either fires or it does not. It has exactly one
  operating point, and asking for "its recall at 0.1% FPR" is meaningless. Its
  own measured FPR and recall are reported instead (NR-0002 §2.1 item 2).

Conflating the two is the easiest way to produce a flattering number by
accident, so the kind is declared rather than inferred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from datagen.unit import LABELS, SECRET_LABELS, DetectionUnit

#: How a system's output may be interpreted.
SystemKind = Literal["continuous", "binary"]


@dataclass(frozen=True, slots=True)
class SystemOutput:
    """One system's predictions over a sequence of units.

    Attributes:
        name: Identifier used in reports and on the command line.
        kind: ``continuous`` or ``binary``; see the module docstring.
        scores: One mapping per unit, from class name to score. For a binary
            system the values are 0.0 or 1.0.
        version: The evaluated system's version, recorded so a report can be
            tied to what produced it.
        notes: Anything worth carrying into the report, such as how a vendored
            tool was invoked.
    """

    name: str
    kind: SystemKind
    scores: Sequence[dict[str, float]]
    version: str = "n/a"
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the shape of the output.

        Raises:
            ValueError: If a score mapping names an unknown class, or a binary
                system emits a value other than 0 or 1. Both would silently
                distort every metric computed afterwards.
        """
        for index, row in enumerate(self.scores):
            unknown = set(row) - set(LABELS)
            if unknown:
                raise ValueError(f"{self.name}: unit {index} scores unknown classes: {unknown}")
            if self.kind == "binary":
                for label, value in row.items():
                    if value not in (0.0, 1.0):
                        raise ValueError(
                            f"{self.name}: binary systems must emit 0.0 or 1.0, "
                            f"got {value!r} for {label!r} on unit {index}"
                        )

    def redaction_scores(self) -> list[float]:
        """Return the per-unit score for the main task, "should this be redacted".

        Returns:
            For each unit, the greatest score across the secret classes. The
            main task is binary (plan §3.2 asks for a redaction decision), and a
            unit is a secret if it is a secret of *any* class, so the maximum is
            the right reduction — summing would let several weak class scores add
            up to a confident redaction.
        """
        return [
            max((row.get(label, 0.0) for label in SECRET_LABELS), default=0.0)
            for row in self.scores
        ]

    def predicted_class(self, index: int) -> str | None:
        """Return the highest-scoring class for one unit.

        Args:
            index: Position of the unit.

        Returns:
            The class with the greatest score, or ``None`` when every score is
            zero — which is a real state for a rule system that matched nothing,
            and distinct from "it chose none".
        """
        row = self.scores[index]
        if not row or all(value == 0.0 for value in row.values()):
            return None
        return max(row, key=lambda label: (row[label], label))


@runtime_checkable
class System(Protocol):
    """A system that can be evaluated."""

    #: Identifier used on the command line and in reports.
    name: str

    #: Whether the system emits scores or a single decision.
    kind: SystemKind

    def version(self) -> str:
        """Return the system's version, for the report."""
        ...

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score a sequence of units.

        Args:
            units: The units to score, in order.

        Returns:
            The system's output, with one score mapping per input unit.
        """
        ...


def empty_scores(count: int) -> list[dict[str, float]]:
    """Return all-zero scores for a given number of units.

    Args:
        count: How many units.

    Returns:
        One empty mapping per unit.
    """
    return [{} for _ in range(count)]
