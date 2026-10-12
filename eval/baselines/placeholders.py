"""Interface placeholders for control groups that later work orders fill in.

NR-0002 §2.2 asks for Laya (control 4) and the student models (control 6) to
exist as interfaces now and be connected later. A placeholder that *runs* and
reports nothing is better than an unregistered name: it proves the framework
accepts the system, it appears in ``--systems`` and in the report's list of
known systems, and when NR-0003 or NR-0004 swaps in the real thing the framework
needs no change.

It scores nothing rather than scoring randomly. A placeholder that emitted
plausible numbers would be indistinguishable in a report from a real result,
which is exactly the kind of mistake a go/no-go gate must not be exposed to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from datagen.unit import DetectionUnit

from eval.system import SystemOutput


@dataclass(slots=True)
class PlaceholderSystem:
    """A registered system that is not implemented yet.

    Attributes:
        name: System name in reports.
        work_order: The work order that will implement it.
        reason: One line on what it will be.
        kind: ``continuous``, matching what the real system will emit, so the
            framework's continuous path is exercised.
    """

    name: str
    work_order: str
    reason: str
    kind: str = "continuous"

    def version(self) -> str:
        """Return a version string that cannot be mistaken for a real one."""
        return f"not implemented (see {self.work_order})"

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Return empty scores for every unit.

        Args:
            units: The units that would be scored.

        Returns:
            Output with no findings, and notes saying why, so a reader of the
            report cannot mistake zero recall for a measured result.
        """
        return SystemOutput(
            name=self.name,
            kind=self.kind,
            scores=[{} for _ in units],
            version=self.version(),
            notes={
                "status": f"placeholder; implemented in {self.work_order}",
                "reason": self.reason,
                "warning": "scores nothing; zero recall here is not a measurement",
            },
        )
