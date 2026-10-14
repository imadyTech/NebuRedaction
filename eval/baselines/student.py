"""Control group 6: the student models, and "rules + student" (NR-0004 §2 item 4).

Two systems, both continuous so the framework can choose their thresholds on
the selection set exactly as it does for ``rules+entropy`` and compare them at
the 0.1% false-positive operating point G2 is judged at.

**The rule half of "rules + student" is the value patterns only, and that is a
measurement rather than a preference.** NR-0002 measured the full rule baseline
at an 11.17% false-positive rate — 1,296 false positives, every one of them
traced to a key name or a template — while ``rules-values-only`` produced
**zero** false positives at 0.593 recall. A union with the full rule set
inherits its false-positive rate, so "rules + student" could never be *operated*
at 0.1%: the gate's own operating point would be unreachable by construction,
and the comparison G2 asks for would be impossible to make. Unioning with the
value patterns instead adds 0.593 recall at no measured false-positive cost on
synthetic data.

That last qualifier matters. On CredData the same component measured 1.22%, so
the union's cost is not zero everywhere, and the report gives both.

**Scoring goes through the saved artefacts, not the trainer.** The model is
loaded from disk and the calibration read from its JSON, so what the gate
measures is what would ship — the same reason NR-0003's control group 4 replays
from a file and model A scores from its int8 arrays.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS, DetectionUnit

from eval.baselines.rules import VALUE_RULES
from eval.system import SystemOutput

VERSION = "0.1.0"


def _load_calibration(path: Path | None) -> Any:
    """Load a saved calibration, or None.

    Args:
        path: A file written by ``training.calibration.save``, or None.

    Returns:
        A callable mapping raw scores to calibrated ones, or None when no
        calibration is supplied — in which case the raw score is used and the
        report says so, because an uncalibrated score still *ranks* correctly
        and every threshold-free metric is unaffected.
    """
    if path is None or not path.is_file():
        return None

    from training.calibration import Isotonic, Platt, log_odds

    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    if payload["method"] == "platt":
        fitted = Platt(slope=parameters["slope"], intercept=parameters["intercept"])
    else:
        fitted = Isotonic(
            thresholds=np.array(parameters["thresholds"]),
            values=np.array(parameters["values"]),
        )

    def calibrate(scores: np.ndarray) -> np.ndarray:
        return fitted.apply(log_odds(scores))

    return calibrate


@dataclass
class StudentSystem:
    """A trained student model as an evaluation system.

    Attributes:
        model_dir: Directory holding the saved model.
        name: System name in reports.
        kind: ``continuous``.
        calibration_path: Optional calibration JSON.
        variant: ``a`` or ``b``, selecting which loader to use.
    """

    model_dir: Path
    name: str = "student-a"
    kind: str = "continuous"
    calibration_path: Path | None = None
    variant: str = "a"
    _model: Any = field(default=None, init=False, repr=False)
    _calibrate: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the model and its calibration.

        Raises:
            FileNotFoundError: If the model is not where it is expected.
            ValueError: If the variant is unknown.
        """
        if self.variant == "a":
            from training.model_a import load

            self._model = load(self.model_dir)
        elif self.variant == "b":
            from training.model_b import TreeModel

            from neburedaction.features import FEATURE_VERSION

            booster = self.model_dir / "model-b.txt"
            columns = self.model_dir / "model-b-columns.npy"
            metadata = self.model_dir / "model-b.json"
            for path in (booster, columns, metadata):
                if not path.is_file():
                    raise FileNotFoundError(path)
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            self._model = TreeModel(
                booster_text=booster.read_text(encoding="utf-8"),
                columns=np.load(columns),
                trees=int(saved["trees"]),
                leaves=int(saved["leaves"]),
                compact={
                    "internal_nodes": int(saved["internal_nodes"]),
                    "leaf_nodes": int(saved["leaf_nodes"]),
                    "bytes": int(saved["size_bytes"]) - int(np.load(columns).size) * 2,
                },
                buckets=int(saved["buckets"]),
                feature_version=FEATURE_VERSION,
                labels=tuple(saved["labels"]),
            )
        else:
            raise ValueError(f"unknown student variant {self.variant!r}")

        self._calibrate = _load_calibration(self.calibration_path)

    def version(self) -> str:
        """Return the adapter's version."""
        return VERSION

    def class_probabilities(self, units: Sequence[DetectionUnit]) -> np.ndarray:
        """Score units, returning the full class distribution."""
        if self.variant == "a":
            from training.model_a import score_units

            return score_units(self._model, units)

        from training.dataset import featurise_units
        from training.model_b import predict_proba

        return predict_proba(self._model, featurise_units(units, self._model.buckets))

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score a sequence of units.

        Args:
            units: The units to score.

        Returns:
            Continuous per-class scores. The redaction score the framework
            derives from them is the mass on any class but ``none``, and the
            calibration is applied to that rather than per class: a threshold
            operates on "redact or not", which is the main task.
        """
        probabilities = self.class_probabilities(units)
        none_index = LABELS.index("none")
        redaction = 1.0 - probabilities[:, none_index]

        if self._calibrate is not None:
            calibrated = self._calibrate(redaction)
            # Rescale the per-class mass so the framework's own derivation of
            # the redaction score agrees with the calibrated number, instead of
            # carrying two different answers for the same quantity.
            scale = np.divide(
                calibrated,
                np.maximum(redaction, 1e-12),
                out=np.ones_like(calibrated),
                where=redaction > 1e-12,
            )
            adjusted = probabilities.copy()
            for index in range(len(LABELS)):
                if index != none_index:
                    adjusted[:, index] *= scale
            adjusted[:, none_index] = 1.0 - calibrated
            probabilities = adjusted

        return SystemOutput(
            name=self.name,
            kind="continuous",
            scores=[
                {label: float(row[index]) for index, label in enumerate(LABELS)}
                for row in probabilities
            ],
            version=self.version(),
            notes={
                "variant": self.variant,
                "model_dir": self.model_dir.name,
                "size_bytes": str(self._model.size_bytes),
                "calibrated": "yes" if self._calibrate is not None else "no (raw scores)",
                "feature_version": self._model.feature_version,
            },
        )


@dataclass
class RulesPlusStudent:
    """G2's candidate: the value-pattern rules unioned with a student.

    Attributes:
        student: The student system to combine.
        name: System name in reports.
        kind: ``continuous``.
        matched: Per-unit record of which value rule fired, for error analysis.
    """

    student: StudentSystem
    name: str = "rules+student-a"
    kind: str = "continuous"
    matched: list[str | None] = field(default_factory=list)

    def version(self) -> str:
        """Return the adapter's version."""
        return VERSION

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score units as the union of value rules and the student.

        Args:
            units: The units to score.

        Returns:
            Continuous output. A unit a value pattern matched scores 1.0, so
            every threshold below 1.0 includes all of them; otherwise the
            student's score stands. That makes the union expressible as one
            continuous system, which is what lets the framework pick its
            threshold on the selection set and compare it with everything else
            at the same operating point.
        """
        student_output = self.student.score(units)
        scores = [dict(row) for row in student_output.scores]
        self.matched = []

        for index, unit in enumerate(units):
            fired: str | None = None
            for rule in VALUE_RULES:
                if rule.pattern.search(unit.value):
                    fired = rule.name
                    # The rule's class is evidence about the class too, not only
                    # about whether to redact, so it takes the whole mass.
                    scores[index] = dict.fromkeys(LABELS, 0.0)
                    scores[index][rule.label] = 1.0
                    break
            self.matched.append(fired)

        return SystemOutput(
            name=self.name,
            kind="continuous",
            scores=scores,
            version=self.version(),
            notes={
                **student_output.notes,
                "rule_half": "value patterns only",
                "rule_half_reason": (
                    "NR-0002 measured the full rule baseline at an 11.17% false-positive "
                    "rate while rules-values-only produced zero. A union with the full set "
                    "inherits its rate, so the 0.1% operating point G2 is judged at would "
                    "be unreachable by construction."
                ),
                "rules_fired": str(sum(1 for name in self.matched if name is not None)),
            },
        )
