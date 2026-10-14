"""Fit a student's calibration and its threshold presets (NR-0004 §2 item 3).

Usage:
    python -m training.fit_student_calibration --model-dir runs/NR-0004/<ts>/C-0.2

Writes ``calibration.json`` beside the model, which ``eval.students`` reads.

Fitted on the **validation** split and nowhere else (EXECUTION.md §8), from the
scores the saved model produces — not from the estimator the trainer held in
memory — so the calibration corrects what would ship rather than something
upstream of quantisation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from datagen.unit import LABELS
from training.calibration import fit, log_odds, save
from training.dataset import VAL_LABELS, read_records
from training.model_a import load, score_units

from neburedaction.features import DEFAULT_BUCKETS

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Fit and write the calibration.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when an input is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=VAL_LABELS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.val.is_file():
        print(f"FAILED: {args.val} not found", file=sys.stderr)
        return 2
    try:
        model = load(args.model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    records = list(read_records(args.val))
    if args.limit is not None:
        records = records[: args.limit]
    print(f"scoring {len(records)} validation units with the saved model ...", flush=True)

    started = time.perf_counter()
    probabilities = score_units(model, records)
    elapsed = time.perf_counter() - started
    none_index = LABELS.index("none")
    scores = 1.0 - probabilities[:, none_index]
    truth = np.array([record.label != "none" for record in records])

    print(
        f"scored in {elapsed:.1f}s ({elapsed / len(records) * 1e6:.1f} us/unit), "
        f"{int(truth.sum())} positives of {len(records)}",
        flush=True,
    )

    calibration = fit(log_odds(scores), truth)
    out = args.out or (args.model_dir / "calibration.json")
    save(calibration, out)

    print(json.dumps(calibration.to_dict(), indent=2, ensure_ascii=False))
    print(f"\n{out}")

    # A calibration fitted where the model has no headroom is worth flagging:
    # if neither preset is attainable the thresholds are None and every
    # downstream operating point will be missing, which is a result rather than
    # a crash but should not be discovered three steps later.
    unattainable = [name for name, value in calibration.thresholds.items() if value is None]
    if unattainable:
        print(
            f"\nWARNING: no threshold achieves {', '.join(unattainable)} on this split. "
            "The model cannot be operated at that rate; eval.students will report it.",
            file=sys.stderr,
        )
    assert DEFAULT_BUCKETS  # the import documents which bucket count the model assumes
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
