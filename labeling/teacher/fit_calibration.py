"""Fit the teacher's temperature on the synthetic validation split (§2.2 item 6).

Usage:
    python -m labeling.teacher.fit_calibration

Writes ``data/labels/calibration.json``, which the merge refuses to run without.

Fitted on the synthetic validation split and nowhere else. EXECUTION.md §8
forbids selecting anything on third-party data, and NR-0002 already pinned the
validation split as the project's designated selection set, so using it here
keeps one rule rather than inventing a second.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training

from labeling.teacher.calibration import fit_temperature

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

SYNTH_VAL = REPOSITORY_ROOT / "data" / "synth" / "v0" / "val.jsonl"
TEACHER_VAL = REPOSITORY_ROOT / "data" / "labels" / "teacher-val.jsonl"
DEFAULT_OUT = REPOSITORY_ROOT / "data" / "labels" / "calibration.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Fit and write the temperature.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when an input is missing or nothing overlaps.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=SYNTH_VAL)
    parser.add_argument("--teacher", type=Path, default=TEACHER_VAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    for path in (args.units, args.teacher):
        if not path.is_file():
            print(f"FAILED: {path} not found", file=sys.stderr)
            if path == args.teacher:
                print(
                    "        run: python -m labeling.teacher.label_corpus "
                    "--units data/synth/v0/val.jsonl --out data/labels/teacher-val.jsonl "
                    "--sample 5000",
                    file=sys.stderr,
                )
            return 2

    truths = {unit.id: unit.label for unit in load_for_training([args.units])}
    distributions = []
    labels = []
    with args.teacher.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            truth = truths.get(record["unit_id"])
            if truth is None:
                continue
            distributions.append(record["probabilities"])
            labels.append(truth)

    if not distributions:
        print(
            f"FAILED: no unit in {args.teacher} matches {args.units}; the teacher pass and "
            "the corpus do not correspond",
            file=sys.stderr,
        )
        return 2

    result = fit_temperature(distributions, labels)
    payload = {
        "work_order": "NR-0003",
        "section": "2.2 item 6",
        "fitted_on": str(args.units).replace("\\", "/"),
        "teacher_labels": str(args.teacher).replace("\\", "/"),
        **result.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if result.at_bound:
        print(
            "\nWARNING: the fit hit a search bound, so this is a clamp rather than an "
            "optimum. The teacher is more miscalibrated than temperature scaling can "
            "express, which is itself the finding.",
            file=sys.stderr,
        )
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
