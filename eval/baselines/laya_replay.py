"""Control group 4: the Laya teacher, zero-shot (NR-0003 §2.2 item 5).

**A replay adapter, not a live model.** The teacher takes about 0.135 seconds
per unit on this machine, so scoring the 20,000-unit synthetic test set inside
``eval.run`` would take 45 minutes and the 49,568-unit CredData set nearly two
hours — every time anyone ran the evaluation. Worse, it would make the
evaluation non-deterministic in wall-clock terms and couple the report to a
model checkpoint being present.

So the probabilities are computed once by ``labeling.teacher.label_corpus`` and
replayed here. The framework sees an ordinary continuous system; what it is
reading is a file.

**Continuous, not binary.** The teacher reports a distribution, so a threshold
can be chosen for it on the selection set exactly as for ``rules+entropy``,
and control group 4 can be compared with the others at the same 0.1%
false-positive operating point G2 is judged at. Declaring it binary would throw
that away and flatter it, since its own argmax operating point has an
enormous false-positive rate — 81.45% on the adversarial slice.

**Replaying CredData scores is not labelling CredData.** §2.2 item 5 forbids
control-group results flowing back into labels and plan §5.1 forbids a label
derived from third-party data existing at all. This module only ever produces a
``SystemOutput`` for a report; nothing in ``labeling/`` reads it, and the file
it replays is written with the ``evaluation_only`` marker.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from datagen.unit import LABELS, DetectionUnit

from eval.system import SystemOutput

VERSION = "0.1.0"


@dataclass
class LayaReplay:
    """Replays precomputed teacher probabilities as an evaluation system.

    Attributes:
        labels_path: A JSON Lines file written by
            ``labeling.teacher.label_corpus``.
        name: System name in reports.
        kind: ``continuous`` — the teacher reports a distribution.
        temperature: Recalibration temperature to apply. 1.0 reports the raw
            probabilities, which is what the report should show for a zero-shot
            control group: recalibration is part of *using* the teacher as a
            label source, not part of what it is.
        model: Recorded in the report's notes.
        prompt_version: Recorded in the report's notes, because a label and a
            score are both only interpretable with the wording that produced
            them.
    """

    labels_path: Path
    name: str = "laya"
    kind: str = "continuous"
    temperature: float = 1.0
    model: str = ""
    prompt_version: str = ""
    _probabilities: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Load the precomputed probabilities.

        Raises:
            FileNotFoundError: If the labels have not been computed.
        """
        if not self.labels_path.is_file():
            raise FileNotFoundError(
                f"{self.labels_path} not found; compute it with "
                "`python -m labeling.teacher.label_corpus --units <corpus> --out "
                f"{self.labels_path}`"
            )
        with self.labels_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._probabilities[record["unit_id"]] = record["probabilities"]
                self.model = self.model or record.get("model", "")
                self.prompt_version = self.prompt_version or record.get("prompt_version", "")

    def version(self) -> str:
        """Return the adapter's version."""
        return VERSION

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score units by looking up their precomputed distributions.

        Args:
            units: The units to score.

        Returns:
            Continuous output: the teacher's probability per class.

        Raises:
            KeyError: If any unit has no precomputed distribution. This is a
                hard failure rather than a zero, because a unit scored 0.0 for
                every class counts as a confident negative and would silently
                improve the system's false-positive rate in proportion to how
                much of the test set was missing. The caller must restrict the
                test set to the units that were labelled, and declare the
                sampling ratio.
        """
        missing = [unit.id for unit in units if unit.id not in self._probabilities]
        if missing:
            raise KeyError(
                f"{len(missing)} of {len(units)} units have no precomputed teacher "
                f"probability (first: {missing[0]!r}). Restrict the test set to the "
                "labelled sample rather than letting unscored units count as negatives."
            )

        from labeling.teacher.calibration import apply_temperature

        scores: list[dict[str, float]] = []
        for unit in units:
            distribution = self._probabilities[unit.id]
            if self.temperature != 1.0:
                distribution = apply_temperature(distribution, self.temperature)
            scores.append({label: float(distribution.get(label, 0.0)) for label in LABELS})

        return SystemOutput(
            name=self.name,
            kind="continuous",
            scores=scores,
            version=self.version(),
            notes={
                "model": self.model,
                "prompt_version": self.prompt_version,
                "temperature": f"{self.temperature:g}",
                "replayed_from": str(self.labels_path.name),
                "zero_shot": "yes",
            },
        )
