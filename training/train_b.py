"""Train model B over a small tree grid (NR-0004 §11.1).

Usage:
    python -m training.train_b

**The selection rule is the one model A used, and it is written here before any
number is seen** (§11.1): among fits inside the 100 KB limit, the highest
recall at the balanced false-positive target, with the smaller model breaking a
near-tie. Plan §6.3 exists to stop a metric being chosen after the results are
in, and that applies to picking between a project's own fits as much as to the
gate.

A tree model's size cannot be pruned to a byte target the way a linear model's
coefficients can — the only levers are the number of trees and the leaves per
tree, and both must be set before fitting. So the budget is explored by the
grid rather than applied afterwards.
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
from training.model_b import DEFAULT_TOP_BUCKETS, predict_proba, save, train

from neburedaction.features import DEFAULT_BUCKETS

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: G2's size limit, in bytes.
SIZE_LIMIT_BYTES = 100 * 1024

#: The balanced preset's target, which selection reads (plan §4.5).
BALANCED_TARGET_FPR = 0.001

#: `(trees per class, leaves per tree)`. Chosen around the 96.2 KB measurement
#: taken at 20,000 rows, since the full split produces fuller trees and so more
#: nodes for the same settings.
DEFAULT_GRID: tuple[tuple[int, int], ...] = ((40, 15), (70, 15), (40, 31), (100, 10))


def operating_points(scores: np.ndarray, truth: np.ndarray) -> list[dict[str, Any]]:
    """Recall at the two targets, with ties broken pessimistically.

    Args:
        scores: Redaction scores.
        truth: Whether each unit should be redacted.

    Returns:
        One entry per target false-positive rate.

    The tie rule matters here more than it did for model A: a boosted ensemble
    puts many units on identical scores, because a leaf value is shared by
    every unit that reaches it.
    """
    negatives = int((~truth).sum())
    positives = int(truth.sum())
    order = np.lexsort((truth, -scores))
    sorted_truth = truth[order]
    false_positives = np.cumsum(~sorted_truth)
    true_positives = np.cumsum(sorted_truth)

    results = []
    for target in (BALANCED_TARGET_FPR, 0.01):
        within = np.nonzero(false_positives <= int(target * negatives))[0]
        if within.size == 0:
            results.append({"target_fpr": target, "attainable": False})
            continue
        cut = int(within[-1])
        results.append(
            {
                "target_fpr": target,
                "attainable": True,
                "recall": round(float(true_positives[cut]) / positives, 6),
                "measured_fpr": round(float(false_positives[cut]) / negatives, 6),
            }
        )
    return results


def evaluate(model: Any, dataset: Dataset) -> dict[str, Any]:
    """Score a fitted tree model on a split, for selection only.

    Args:
        model: The fitted model.
        dataset: The split to score.

    Returns:
        Selection metrics. G2 is judged by ``eval.students``; duplicating its
        metrics here would invite two numbers for one quantity.
    """
    started = time.perf_counter()
    probabilities = predict_proba(model, dataset.features)
    elapsed = time.perf_counter() - started
    none_index = LABELS.index("none")
    scores = 1.0 - probabilities[:, none_index]
    truth = dataset.hard != none_index
    predicted = probabilities.argmax(axis=1)
    return {
        "rows": int(dataset.features.shape[0]),
        "class_accuracy": round(float((predicted == dataset.hard).mean()), 6),
        "operating_points": operating_points(scores, truth),
        "scoring_seconds": round(elapsed, 2),
        "microseconds_per_unit": round(elapsed / dataset.features.shape[0] * 1e6, 1),
        "note": "Selection metrics only; G2 is judged by eval.students.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the tree grid and write the best model.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the merged labels are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-buckets", type=int, default=DEFAULT_TOP_BUCKETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0004")
    args = parser.parse_args(argv)

    print("loading and featurising ...", flush=True)
    started = time.perf_counter()
    try:
        train_set, validation_set = load_splits(
            buckets=DEFAULT_BUCKETS,
            soft_label_weight=DEFAULT_SOFT_LABEL_WEIGHT,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"featurised in {time.perf_counter() - started:.0f}s", flush=True)

    stamped = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stamped.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best: tuple[float, Any, dict[str, Any]] | None = None
    for trees, leaves in DEFAULT_GRID:
        print(f"\nfitting trees={trees} leaves={leaves} ...", flush=True)
        model, _, diagnostics = train(
            train_set,
            top_buckets=args.top_buckets,
            trees=trees,
            leaves=leaves,
            seed=args.seed,
        )
        selection = evaluate(model, validation_set)
        within_budget = model.size_bytes <= SIZE_LIMIT_BYTES
        fit_paths = save(model, diagnostics, stamped / f"trees{trees}-leaves{leaves}")
        point = selection["operating_points"][0]
        recall = point.get("recall") if point.get("attainable") else None
        results.append(
            {
                "trees": trees,
                "leaves": leaves,
                "within_size_limit": within_budget,
                "diagnostics": diagnostics,
                "validation": selection,
                "paths": {name: str(path) for name, path in fit_paths.items()},
            }
        )
        print(
            f"  {diagnostics['size_kib']} KB (text {diagnostics['text_size_kib']} KB), "
            f"recall@0.1%={recall}, class acc {selection['class_accuracy']:.4f}, "
            f"within budget={within_budget}",
            flush=True,
        )

        is_candidate = within_budget and recall is not None
        if is_candidate and (
            best is None
            or recall > best[0] * 1.01
            or (abs(recall - best[0]) <= best[0] * 0.01 and model.size_bytes < best[1].size_bytes)
        ):
            best = (recall, model, results[-1])

    summary: dict[str, Any] = {
        "work_order": "NR-0004",
        "section": "11.1",
        "model": "B (LightGBM, dense features + top hashed buckets)",
        "grid": [list(pair) for pair in DEFAULT_GRID],
        "top_buckets": args.top_buckets,
        "seed": args.seed,
        "size_limit_bytes": SIZE_LIMIT_BYTES,
        "selection_rule": (
            "Among fits within 100 KB, the highest recall at the balanced target; a "
            "near-tie within 1% goes to the smaller model. Written before any number "
            "was seen (§11.1, plan §6.3)."
        ),
        "results": results,
    }

    if best is None:
        summary["selected"] = None
        print("\nNo fit was both within the size limit and operable at 0.1%.", file=sys.stderr)
    else:
        recall, model, record = best
        summary["selected"] = {
            "trees": record["trees"],
            "leaves": record["leaves"],
            "recall_at_0.001": recall,
            "size_bytes": model.size_bytes,
        }
        paths = save(model, record["diagnostics"], stamped)
        summary["model_paths"] = {name: str(path) for name, path in paths.items()}
        print(
            f"\nselected trees={record['trees']} leaves={record['leaves']}, "
            f"recall@0.1%={recall}, {model.size_bytes / 1024:.1f} KB"
        )

    report = stamped / "model-b-grid.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
