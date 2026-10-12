"""Error analysis for the rule baseline (NR-0002 §6).

The acceptance criterion asks for the principal false-negative and
false-positive *types* on each slice, with three synthetic examples of each. The
counts in the report say how badly a system does; this says what it does wrong,
which is what the next work order acts on.

Only synthetic examples are ever printed. CredData's values must not leave the
local machine (plan §5.1), so third-party units are counted but never quoted —
the grouping is reported for them, the examples come from the synthetic set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datagen.loader import iter_units
from datagen.unit import DetectionUnit

from eval.baselines import build

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: How many examples to show per error type.
EXAMPLES_PER_TYPE = 3

#: Longest value printed in an example. Long PEM bodies add nothing to a report.
MAX_VALUE_LENGTH = 70


@dataclass
class ErrorGroup:
    """One kind of mistake.

    Attributes:
        key: What groups these errors — a family, a matched rule, a context.
        count: How many units fell in this group.
        examples: Up to :data:`EXAMPLES_PER_TYPE` synthetic examples.
    """

    key: str
    count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, unit: DetectionUnit, extra: dict[str, Any] | None = None) -> None:
        """Record a unit in this group.

        Args:
            unit: The misclassified unit.
            extra: Anything else worth showing, such as the rule that fired.
        """
        self.count += 1
        if len(self.examples) < EXAMPLES_PER_TYPE and unit.source == "synth":
            value = (
                unit.value
                if len(unit.value) <= MAX_VALUE_LENGTH
                else (unit.value[:MAX_VALUE_LENGTH] + "…")
            )
            self.examples.append(
                {
                    "id": unit.id,
                    "key": unit.key,
                    "value": value.replace("\n", "\\n"),
                    "label": unit.label,
                    "family": unit.family,
                    "template": unit.template,
                    **(extra or {}),
                }
            )


def analyse(system_name: str, units: Sequence[DetectionUnit]) -> dict[str, Any]:
    """Group a rule system's mistakes by cause.

    Args:
        system_name: A registered system name; must be a binary rule system, as
            the grouping is by which rule fired.
        units: The labelled units to analyse.

    Returns:
        False negatives grouped by family and false positives grouped by the
        rule that fired, each with synthetic examples, plus totals.

    Raises:
        ValueError: If the system is not binary. A continuous system has no
            single operating point to attribute an error to, so grouping its
            mistakes this way would be meaningless without a threshold.
    """
    system = build(system_name)
    output = system.score(list(units))
    if output.kind != "binary":
        raise ValueError(
            f"{system_name} is {output.kind}; error analysis by matched rule needs a binary system"
        )

    matched = getattr(system, "matched", [None] * len(units))
    scores = output.redaction_scores()

    false_negatives: dict[str, ErrorGroup] = {}
    false_positives: dict[str, ErrorGroup] = {}
    confusions: Counter[str] = Counter()

    for index, unit in enumerate(units):
        flagged = scores[index] >= 1.0
        rule = matched[index] if index < len(matched) else None

        if unit.should_redact and not flagged:
            key = unit.family or "(no family)"
            false_negatives.setdefault(key, ErrorGroup(key)).add(unit)
        elif not unit.should_redact and flagged:
            key = rule or "(unknown rule)"
            false_positives.setdefault(key, ErrorGroup(key)).add(unit, {"matched_rule": rule})
        elif unit.should_redact and flagged:
            predicted = output.predicted_class(index)
            if predicted and predicted != unit.label:
                confusions[f"{unit.label} -> {predicted}"] += 1

    def _render(groups: dict[str, ErrorGroup]) -> list[dict[str, Any]]:
        """Sort groups by count and serialise."""
        return [
            {"group": group.key, "count": group.count, "examples": group.examples}
            for group in sorted(groups.values(), key=lambda g: (-g.count, g.key))
        ]

    positives = sum(1 for unit in units if unit.should_redact)
    negatives = len(units) - positives
    return {
        "system": system_name,
        "units": len(units),
        "positives": positives,
        "negatives": negatives,
        "false_negatives": {
            "total": sum(g.count for g in false_negatives.values()),
            "note": "Grouped by format family: which formats the rules do not recognise.",
            "groups": _render(false_negatives),
        },
        "false_positives": {
            "total": sum(g.count for g in false_positives.values()),
            "note": (
                "Grouped by the rule that fired: which rule is responsible. A 'key:' group "
                "means a key name triggered it, which is the adversarial failure mode."
            ),
            "groups": _render(false_positives),
        },
        "class_confusions": {
            "note": "Flagged correctly but attributed to the wrong class.",
            "counts": dict(confusions.most_common(20)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the error analysis and write it as JSON.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=["rules", "rules-values-only"])
    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synth" / "v0" / "test.jsonl",
        help="labelled units to analyse",
    )
    parser.add_argument("--out", type=Path, default=None, help="output JSON file")
    args = parser.parse_args(argv)

    units = [unit for unit in iter_units(args.data) if unit.label is not None]
    result = {
        "data": str(args.data).replace("\\", "/"),
        "examples_per_type": EXAMPLES_PER_TYPE,
        "note": (
            "Examples are synthetic only. CredData values must not leave the local machine "
            "(plan §5.1), so third-party units are counted but never quoted."
        ),
        "analyses": [analyse(name, units) for name in args.systems],
    }

    out = args.out or (REPOSITORY_ROOT / "runs" / "eval" / "error-analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for analysis in result["analyses"]:
        print(f"\n=== {analysis['system']} on {len(units)} units ===")
        print(f"false negatives: {analysis['false_negatives']['total']}")
        for group in analysis["false_negatives"]["groups"][:6]:
            print(f"   {group['count']:>6}  {group['group']}")
        print(f"false positives: {analysis['false_positives']['total']}")
        for group in analysis["false_positives"]["groups"][:6]:
            print(f"   {group['count']:>6}  {group['group']}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
