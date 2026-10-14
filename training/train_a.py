"""Fit model A over a penalty grid and keep the diagnostics (NR-0004 §2 item 2).

Usage:
    python -m training.train_a --dry-run
    python -m training.train_a                       # the full grid
    python -m training.train_a --limit 20000 --grid 0.5

Hyperparameters are selected on the **validation** split and nowhere else
(§2 item 2). The training split is only ever fitted on, and the test split is
not touched by this module at all — G2 is judged by ``eval.run``, after the
model is fixed.

**Convergence is not optional for an L1 fit.** Measured on 5,000 rows, with
everything else held constant:

    max_iter  iterations used  converged  sparsity  size
         100              100      False     0.887  28.9 KB
         400              400      False     0.937  17.5 KB
        1200              481       True     0.943  16.2 KB

The sparsity pattern is still moving when the solver stops early, and it moves
in the direction that *overstates* the model's size — so a run capped too low
reports a harder size budget than the model really has, and reports a sparsity
figure that is not a measurement of anything. Hence a cap high enough to reach
convergence, with ``converged`` recorded per fit so a run that did not can
never be mistaken for one that did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS
from training.dataset import DEFAULT_SOFT_LABEL_WEIGHT, Dataset, load_splits
from training.model_a import dense_feature_report, save, train

from neburedaction.features import DEFAULT_BUCKETS, MAX_BUCKETS, MIN_BUCKETS

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: G2's size limit, in bytes.
SIZE_LIMIT_BYTES = 100 * 1024

#: Penalty grid. Smaller C means more L1 pressure, a sparser and smaller model,
#: and usually less recall — which is the trade the gate is about, so the grid
#: spans two orders of magnitude rather than hedging near one value.
DEFAULT_GRID: tuple[float, ...] = (0.05, 0.2, 0.5, 2.0, 8.0)

#: Iteration cap. Chosen from the measurement in the module docstring, with
#: headroom for the full split, which needs more iterations than 5,000 rows did.
MAX_ITERATIONS = 3000


def evaluate(model: Any, dataset: Dataset) -> dict[str, Any]:
    """Score a fitted model on a split, for hyperparameter selection.

    Args:
        model: The quantised model.
        dataset: The split to score.

    Returns:
        Redaction recall at the measured false-positive rates that matter, plus
        class accuracy.

    Deliberately **not** the project's evaluation. `eval.run` owns the metrics
    G2 is judged on, with its confidence intervals, slices and pre-fixed
    comparison rule; duplicating that here would invite two numbers for the
    same quantity. This is only rich enough to choose between fits, and the
    report says so.
    """
    probabilities = model.predict_proba(dataset.features)
    none_index = LABELS.index("none")
    # Redaction score: the mass on any class but `none`.
    scores = 1.0 - probabilities[:, none_index]
    truth = dataset.hard != none_index

    positives = int(truth.sum())
    negatives = int((~truth).sum())
    # Ties are broken **pessimistically**: at an equal score a negative is
    # ranked ahead of a positive. This is the convention NR-0002 fixed for
    # PR-AUC, and it is not cosmetic. A single threshold cannot separate units
    # that share a score, so counting the positives first would claim a recall
    # the threshold cannot deliver — with every score equal, the optimistic
    # order reports full recall at a 0.01% false-positive rate while the
    # threshold in fact flags everything. int8 quantisation makes exact ties
    # common, so this is the normal case rather than a corner one.
    order = np.lexsort((truth, -scores))
    sorted_truth = truth[order]

    def recall_at_fpr(target: float) -> dict[str, Any]:
        """Recall at the largest threshold whose false-positive rate fits."""
        allowed = int(target * negatives)
        false_positives = np.cumsum(~sorted_truth)
        within = np.nonzero(false_positives <= allowed)[0]
        if within.size == 0:
            return {"target_fpr": target, "attainable": False}
        cut = within[-1]
        caught = int(np.cumsum(sorted_truth)[cut])
        return {
            "target_fpr": target,
            "attainable": True,
            "recall": round(caught / positives, 6) if positives else None,
            "measured_fpr": round(int(false_positives[cut]) / negatives, 6) if negatives else None,
            "threshold": round(float(scores[order][cut]), 6),
        }

    predicted = probabilities.argmax(axis=1)
    return {
        "rows": int(dataset.features.shape[0]),
        "positives": positives,
        "negatives": negatives,
        "class_accuracy": round(float((predicted == dataset.hard).mean()), 6),
        "redaction_accuracy": round(float(((predicted != none_index) == truth).mean()), 6),
        "operating_points": [recall_at_fpr(target) for target in (0.001, 0.01)],
        "note": (
            "Selection metrics only. G2 is judged by eval.run, which owns the "
            "confidence intervals, the slices and the comparison rule fixed in "
            "eval/README.md before any model existed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the grid and write the best model.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the merged labels are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--soft-label-weight",
        type=float,
        default=DEFAULT_SOFT_LABEL_WEIGHT,
        help="how much of a positive's target comes from the teacher; 0.0 is the control",
    )
    parser.add_argument("--limit", type=int, default=None, help="rows per split, for a quick run")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0004")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not MIN_BUCKETS <= args.buckets <= MAX_BUCKETS:
        parser.error(f"--buckets must be in [{MIN_BUCKETS}, {MAX_BUCKETS}] (plan §4.2)")

    print(f"loading and featurising (buckets={args.buckets}, limit={args.limit}) ...", flush=True)
    started = time.perf_counter()
    try:
        train_set, validation_set = load_splits(
            buckets=args.buckets, soft_label_weight=args.soft_label_weight, limit=args.limit
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    featurise_seconds = time.perf_counter() - started
    print(f"featurised in {featurise_seconds:.0f}s", flush=True)
    print(json.dumps(train_set.statistics(), indent=2), flush=True)

    if args.dry_run:
        print("\ndry run: nothing was fitted")
        return 0

    stamped = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stamped.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    for penalty_strength in args.grid:
        print(f"\nfitting C={penalty_strength} ...", flush=True)
        model, diagnostics = train(
            train_set,
            penalty_strength=penalty_strength,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
        selection = evaluate(model, validation_set)
        within_budget = model.size_bytes <= SIZE_LIMIT_BYTES
        record = {
            "C": penalty_strength,
            "within_size_limit": within_budget,
            "diagnostics": diagnostics,
            "validation": selection,
        }
        results.append(record)

        point = selection["operating_points"][0]
        recall = point.get("recall") if point.get("attainable") else None
        print(
            f"  {diagnostics['size_kib']} KB, sparsity {diagnostics['sparsity']:.4f}, "
            f"converged={diagnostics['converged']}, "
            f"recall@0.1%={recall}, class acc {selection['class_accuracy']:.4f}",
            flush=True,
        )

        # Selection rule, fixed before the numbers were seen: among fits inside
        # the size limit, the highest recall at the 0.1% false-positive target
        # G2 is judged at. A fit that cannot be operated there, or that does not
        # fit, is not a candidate however good it looks elsewhere.
        is_candidate = within_budget and point.get("attainable") and recall is not None
        if is_candidate and (best is None or recall > best[0]):
            best = (recall, model, record)

    summary: dict[str, Any] = {
        "work_order": "NR-0004",
        "section": "2 item 2",
        "model": "A (hashed features + multinomial logistic regression, int8)",
        "buckets": args.buckets,
        "soft_label_weight": args.soft_label_weight,
        "grid": list(args.grid),
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "size_limit_bytes": SIZE_LIMIT_BYTES,
        "featurise_seconds": round(featurise_seconds, 1),
        "train_statistics": train_set.statistics(),
        "validation_statistics": validation_set.statistics(),
        "selection_rule": (
            "Among fits within the 100 KB limit, the highest recall at the 0.1% "
            "false-positive target. Fixed before the numbers were seen."
        ),
        "results": results,
    }

    if best is None:
        summary["selected"] = None
        print(
            "\nNo fit was both within the size limit and operable at 0.1% false "
            "positives. That is a result, not a crash: it is recorded and G2 will "
            "report it (plan §6.4 makes G2 a recording gate as of 2026-10-14).",
            file=sys.stderr,
        )
    else:
        recall, model, record = best
        summary["selected"] = {"C": record["C"], "recall_at_0.001": recall}
        paths = save(model, record["diagnostics"], stamped)
        summary["model_paths"] = {name: str(path) for name, path in paths.items()}
        summary["dense_feature_weights"] = dense_feature_report(model)
        print(f"\nselected C={record['C']}, recall@0.1%={recall}, {model.size_bytes / 1024:.1f} KB")
        print(f"{paths['arrays']}\n{paths['metadata']}")

    report = stamped / "model-a-grid.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
