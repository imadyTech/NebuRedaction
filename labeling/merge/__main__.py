"""Produce the training files (NR-0003 §2.5).

Usage:
    python -m labeling.merge

Reads the rule labels, the teacher labels, the fitted temperature, the
CredSweeper opinions and the review decisions, applies the §2.3 table and writes
``data/labels/train_v0.jsonl``, ``val_v0.jsonl`` and ``merge-statistics.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training

from labeling.merge import MergedLabel, merge_unit, resolve_decisions, statistics, write_merged
from labeling.rules import read_labels

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

OWN_UNITS = REPOSITORY_ROOT / "data" / "own" / "own_v0.jsonl"
SYNTH_TRAIN = REPOSITORY_ROOT / "data" / "synth" / "v0" / "train.jsonl"
SYNTH_VAL = REPOSITORY_ROOT / "data" / "synth" / "v0" / "val.jsonl"
LABELS_DIR = REPOSITORY_ROOT / "data" / "labels"


def main(argv: Sequence[str] | None = None) -> int:
    """Merge every label source into the training files.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when an input is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_DIR)
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="merge without a fitted temperature; §2.2 item 6 forbids this for a real run",
    )
    args = parser.parse_args(argv)

    teacher_path = args.labels / "teacher_v0.jsonl"
    for path in (OWN_UNITS, SYNTH_TRAIN, SYNTH_VAL, teacher_path):
        if not path.is_file():
            print(f"FAILED: {path} not found", file=sys.stderr)
            return 2

    from labeling.teacher.calibration import apply_temperature

    calibration_path = args.labels / "calibration.json"
    if calibration_path.is_file():
        temperature = float(json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"])
        print(f"applying fitted temperature {temperature:.4f}")
    elif args.allow_uncalibrated:
        temperature = 1.0
        print("WARNING: merging on raw probabilities (--allow-uncalibrated)", file=sys.stderr)
    else:
        print(
            f"FAILED: {calibration_path} not found. §2.2 item 6 requires temperature "
            "recalibration on the synthetic validation split before any automatic "
            "adoption, because an uncalibrated 0.9 is not a 90% chance of being right "
            "— this run measured OrderId=29 scored `token` at 0.979. Pass "
            "--allow-uncalibrated only to inspect the pipeline.",
            file=sys.stderr,
        )
        return 2

    teacher: dict[str, dict[str, float]] = {}
    with teacher_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            teacher[record["unit_id"]] = apply_temperature(record["probabilities"], temperature)

    credsweeper: dict[str, bool] = {}
    sweeper_path = args.labels / "credsweeper-own.json"
    if sweeper_path.is_file():
        credsweeper = {
            unit_id: bool(secret)
            for unit_id, secret in json.loads(sweeper_path.read_text(encoding="utf-8")).items()
        }
    else:
        print(
            "WARNING: no credsweeper-own.json; §2.3 requires a second opinion before an "
            "own-log label is auto-adopted, so every own-log unit will route to review.",
            file=sys.stderr,
        )

    rules = {}
    for name in ("rules-own.jsonl", "rules-synth-train.jsonl", "rules-synth-val.jsonl"):
        path = args.labels / name
        if path.is_file():
            rules.update({label.unit_id: label for label in read_labels(path)})

    decisions = resolve_decisions(args.labels / "review_v0.jsonl")
    print(f"{len(teacher)} teacher labels, {len(decisions)} review decisions")

    omitted: list[str] = []

    def merge(paths: list[Path]) -> list[MergedLabel]:
        """Merge the units in some corpora, dropping the unlabelable."""
        merged: list[MergedLabel] = []
        for unit in load_for_training(paths):
            result = merge_unit(
                unit,
                rules.get(unit.id),
                teacher.get(unit.id),
                credsweeper.get(unit.id),
                decisions.get(unit.id),
            )
            if result is None:
                omitted.append(unit.id)
            else:
                merged.append(result)
        return merged

    # Own logs go to validation, not training: no positives, so their training
    # value is a handful of negatives while their measurement value is the only
    # in-distribution read the project has on the false-positive rate. They
    # carry source="own" so the evaluation can slice them apart.
    train = merge([SYNTH_TRAIN])
    validation = merge([SYNTH_VAL, OWN_UNITS])

    train_path = args.labels / "train_v0.jsonl"
    val_path = args.labels / "val_v0.jsonl"
    print(f"{write_merged(train, train_path)} -> {train_path}")
    print(f"{write_merged(validation, val_path)} -> {val_path}")

    summary = {
        "work_order": "NR-0003",
        "section": "2.5",
        "temperature": temperature,
        "credsweeper_consulted": bool(credsweeper),
        "omitted_unlabelable": {
            "count": len(omitted),
            "note": (
                "Own-log units §2.3 routed to review that no human decided — outside the "
                "500-unit cap, or skipped. Omitted rather than defaulted: these are the "
                "units where the teacher saw a credential, so defaulting them to `none` "
                "would teach the student to ignore exactly the cases a human was asked "
                "to look at."
            ),
            "unit_ids": omitted[:100],
        },
        "train": statistics(train, decisions.values()),
        "val": statistics(validation, decisions.values()),
    }
    summary_path = args.labels / "merge-statistics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{summary_path}")
    if omitted:
        print(f"{len(omitted)} own-log unit(s) omitted as unlabelable; see the statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
