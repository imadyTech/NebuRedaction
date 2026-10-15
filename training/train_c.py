"""Train model C, the character CNN control (NR-0004 §2 item 2).

Usage:
    python -m training.train_c --epochs 3

C exists to answer one question: **how much of the task is the features and how
much is the model?** A and B read 31 hand-designed morphological features plus
hashed n-grams; C reads the value's characters and nothing else. If C clearly
wins, the features are leaving information on the table and NR-0005's .NET port
is carrying a hand-built pipeline it did not need. If it does not, the features
are where the work is — which is what makes a 40 KB linear model a sensible
thing to ship.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from datagen.unit import LABELS
from training.dataset import DEFAULT_SOFT_LABEL_WEIGHT, load_splits
from training.model_c import predict_proba, save, train

from neburedaction.features import DEFAULT_BUCKETS

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    """Train model C and report its operating point.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the merged labels are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0004")
    args = parser.parse_args(argv)

    print("loading and featurising ...", flush=True)
    try:
        # C does not use the engineered features, but it is trained on the same
        # soft targets as A and B so the comparison is about the model rather
        # than about what each was asked to predict.
        train_set, validation_set = load_splits(
            buckets=DEFAULT_BUCKETS,
            soft_label_weight=DEFAULT_SOFT_LABEL_WEIGHT,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    model, diagnostics = train(
        train_set,
        validation_set,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
        seed=args.seed,
    )

    # The same operating-point summary the model A grid reports, so the two are
    # read on the same scale. G2 itself is judged by eval.students.
    started = time.perf_counter()
    probabilities = predict_proba(model, [record.value for record in validation_set.records])
    scoring_seconds = time.perf_counter() - started
    none_index = LABELS.index("none")
    scores = 1.0 - probabilities[:, none_index]
    truth = validation_set.hard != none_index

    order = np.lexsort((truth, -scores))
    sorted_truth = truth[order]
    false_positives = np.cumsum(~sorted_truth)
    true_positives = np.cumsum(sorted_truth)
    negatives = int((~truth).sum())
    positives = int(truth.sum())

    operating = []
    for target in (0.001, 0.01):
        within = np.nonzero(false_positives <= int(target * negatives))[0]
        if within.size == 0:
            operating.append({"target_fpr": target, "attainable": False})
            continue
        cut = int(within[-1])
        operating.append(
            {
                "target_fpr": target,
                "attainable": True,
                "recall": round(float(true_positives[cut]) / positives, 6),
                "measured_fpr": round(float(false_positives[cut]) / negatives, 6),
            }
        )

    stamped = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    diagnostics["validation"] = {
        "rows": len(validation_set.records),
        "operating_points": operating,
        "scoring_seconds": round(scoring_seconds, 1),
        "microseconds_per_unit": round(scoring_seconds / len(validation_set.records) * 1e6, 1),
        "note": "Selection metrics on the validation split; G2 is judged by eval.students.",
    }
    paths = save(model, diagnostics, stamped)

    print(json.dumps({k: v for k, v in diagnostics.items() if k != "history"}, indent=2))
    print(f"\n{paths['arrays']}\n{paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
