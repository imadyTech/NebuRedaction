"""Metric computation for the evaluation framework (NR-0002 §2.1 item 2).

Everything here is deterministic and dependency-free: no numpy, no scikit-learn.
That is deliberate. The whole point of fixing the evaluation before training
(plan §6.3) is that the numbers are auditable, and a reader who wants to check
whether "recall at 0.1% false-positive rate" means what they think it means can
read forty lines of Python instead of tracing a library's conventions.

Two conventions worth stating because they are easy to get wrong:

**The false-positive rate is per negative unit**, not per record flagged —
``FP / (FP + TN)``. Plan §4.5 sets the target "误报率 ≤ 0.1% (以属性计)", and an
attribute is a unit, so the denominator is the negatives in the slice.

**A threshold chosen at a target FPR is chosen on a validation set, never on the
set it is reported against.** EXECUTION.md §8 forbids selecting thresholds on
third-party data at all. So the caller selects on synthetic validation data and
reports the *measured* FPR on each test set alongside recall — the two can
differ, and when they do, that difference is the finding.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Counts at one operating point.

    Attributes:
        true_positives: Secrets scored at or above the threshold.
        false_negatives: Secrets scored below it.
        false_positives: Non-secrets scored at or above it.
        true_negatives: Non-secrets scored below it.
    """

    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int

    @property
    def positives(self) -> int:
        """Number of secret units."""
        return self.true_positives + self.false_negatives

    @property
    def negatives(self) -> int:
        """Number of non-secret units."""
        return self.false_positives + self.true_negatives

    @property
    def recall(self) -> float:
        """Share of secrets caught, or 0.0 when there are no secrets."""
        return self.true_positives / self.positives if self.positives else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of non-secrets wrongly flagged, or 0.0 when there are none."""
        return self.false_positives / self.negatives if self.negatives else 0.0

    @property
    def precision(self) -> float:
        """Share of flagged units that really are secrets."""
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the report.

        Returns:
            Counts and the three derived rates, rounded for stable comparison.
        """
        return {
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "positives": self.positives,
            "negatives": self.negatives,
            "recall": round(self.recall, 6),
            "false_positive_rate": round(self.false_positive_rate, 8),
            "precision": round(self.precision, 6),
        }


def confusion_at(
    scores: Sequence[float], truth: Sequence[bool], threshold: float
) -> ConfusionCounts:
    """Count outcomes at a threshold.

    Args:
        scores: One score per unit.
        truth: True where the unit is a secret.
        threshold: Units scoring ``>= threshold`` are flagged.

    Returns:
        The counts.

    Raises:
        ValueError: If the inputs differ in length.
    """
    if len(scores) != len(truth):
        raise ValueError(f"scores and truth differ in length: {len(scores)} vs {len(truth)}")
    tp = fn = fp = tn = 0
    for score, is_secret in zip(scores, truth, strict=True):
        flagged = score >= threshold
        if is_secret:
            if flagged:
                tp += 1
            else:
                fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return ConfusionCounts(tp, fn, fp, tn)


def threshold_for_false_positive_rate(
    scores: Sequence[float], truth: Sequence[bool], target_rate: float
) -> float | None:
    """Find the lowest threshold whose false-positive rate is within a target.

    Lowest, not any: among thresholds meeting the target, the lowest maximises
    recall, which is the operating point a deployment would actually pick.

    Args:
        scores: One score per unit, from the **selection** set.
        truth: True where the unit is a secret.
        target_rate: The false-positive rate not to exceed, e.g. ``0.001``.

    Returns:
        The threshold, or ``None`` when no threshold achieves the target —
        which happens when even flagging nothing but the very top score exceeds
        it. Returning ``None`` rather than a sentinel forces the caller to say
        what it means in the report instead of silently reporting zero recall.
    """
    negative_scores = sorted((s for s, t in zip(scores, truth, strict=True) if not t), reverse=True)
    if not negative_scores:
        # Nothing to be wrong about: any threshold has a 0% false-positive rate.
        return min(scores) if scores else None

    allowed = int(len(negative_scores) * target_rate)
    if allowed >= len(negative_scores):
        return min(scores) if scores else None

    # The threshold must exceed the (allowed+1)-th highest negative score so that
    # at most `allowed` negatives are flagged. Nudge above it so equal-scoring
    # negatives fall below rather than on the line.
    boundary = negative_scores[allowed]
    candidates = sorted({s for s in scores if s > boundary})
    return candidates[0] if candidates else None


def average_precision(scores: Sequence[float], truth: Sequence[bool]) -> float:
    """Compute PR-AUC as average precision.

    Average precision is used rather than trapezoidal area under an interpolated
    curve: it needs no interpolation choice, so two implementations agree, and it
    is the standard definition for a ranking.

    Args:
        scores: One score per unit.
        truth: True where the unit is a secret.

    Returns:
        Average precision in [0, 1], or 0.0 when there are no secrets.
    """
    total_positives = sum(1 for t in truth if t)
    if not total_positives:
        return 0.0

    # Descending by score; ties are broken by putting negatives first, which is
    # the pessimistic reading and stops a system scoring everything identically
    # from looking good.
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], truth[i]))

    seen_positives = 0
    summed = 0.0
    for rank, index in enumerate(order, start=1):
        if truth[index]:
            seen_positives += 1
            summed += seen_positives / rank
    return summed / total_positives


def bootstrap_interval(
    values: Sequence[float],
    statistic: str = "mean",
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261013,
) -> dict[str, float]:
    """Estimate a confidence interval for a per-unit statistic by resampling.

    Args:
        values: Per-unit indicator values, e.g. 1.0 for each caught secret.
        statistic: Currently only ``mean``, which covers recall, false-positive
            rate and precision when the caller supplies the right indicators.
        resamples: Number of bootstrap resamples (NR-0002 §2.1 item 3: 1,000).
        confidence: Interval width, e.g. 0.95.
        seed: Fixed so the interval is reproducible (EXECUTION.md §2 item 7).

    Returns:
        ``point``, ``low`` and ``high``. An empty input yields zeros rather than
        an error: a slice can legitimately be empty, and the report says so.

    For 0/1 indicator input — which is what recall and false-positive rate are —
    the resampling is performed by drawing from the equivalent binomial
    distribution rather than by resampling each item. See the implementation
    comment; the distribution is identical and it is around four orders of
    magnitude faster on a large slice.

    Raises:
        ValueError: If an unsupported statistic is requested.
    """
    if statistic != "mean":
        raise ValueError(f"unsupported statistic: {statistic!r}")
    if not values:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}

    point = sum(values) / len(values)
    rng = random.Random(seed)
    size = len(values)

    # Fast path for indicator values, which is what every caller here passes:
    # recall and false-positive rate are means of 0/1 indicators.
    #
    # Resampling n items with replacement and taking the mean of 0/1 values is
    # the same as drawing the *count* of ones from Binomial(n, p̂) and dividing
    # by n — the individual draws are never needed. This is the identical
    # distribution, not an approximation, and it turns 1,000 * n random draws
    # into 1,000. On the 49,568-unit CredData slice that is ~50 million draws
    # replaced by a thousand, which is the difference between a report that
    # takes minutes and one that takes an afternoon.
    if all(value in (0.0, 1.0) for value in values):
        successes = round(point * size)
        probability = successes / size
        means = sorted(rng.binomialvariate(size, probability) / size for _ in range(resamples))
    else:
        means = []
        for _ in range(resamples):
            total = 0.0
            for _ in range(size):
                total += values[rng.randrange(size)]
            means.append(total / size)
        means.sort()

    tail = (1.0 - confidence) / 2.0
    low = means[min(len(means) - 1, max(0, math.floor(tail * len(means))))]
    high = means[min(len(means) - 1, math.ceil((1.0 - tail) * len(means)) - 1)]
    return {
        "point": round(point, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "n": size,
    }


def recall_indicators(
    scores: Sequence[float], truth: Sequence[bool], threshold: float
) -> list[float]:
    """Return a 1/0 indicator per secret unit: was it caught at this threshold.

    Args:
        scores: One score per unit.
        truth: True where the unit is a secret.
        threshold: The operating point.

    Returns:
        One indicator per secret unit, suitable for :func:`bootstrap_interval`.
    """
    return [
        1.0 if score >= threshold else 0.0
        for score, is_secret in zip(scores, truth, strict=True)
        if is_secret
    ]


def false_positive_indicators(
    scores: Sequence[float], truth: Sequence[bool], threshold: float
) -> list[float]:
    """Return a 1/0 indicator per non-secret unit: was it wrongly flagged.

    Args:
        scores: One score per unit.
        truth: True where the unit is a secret.
        threshold: The operating point.

    Returns:
        One indicator per non-secret unit.
    """
    return [
        1.0 if score >= threshold else 0.0
        for score, is_secret in zip(scores, truth, strict=True)
        if not is_secret
    ]


def per_class_recall(
    scores: Sequence[dict[str, float]],
    labels: Sequence[str | None],
    redaction_scores: Sequence[float],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Compute recall per secret class at an operating point.

    Reported two ways, because they answer different questions:

    * ``recall`` — was the unit flagged for redaction at all? This is what
      matters operationally: a token masked as an ``api_key`` is still masked.
    * ``exact_class_recall`` — was it flagged *and* attributed to the right
      class? This is what matters for the masking fingerprint's class label and
      for diagnosing confusions.

    Args:
        scores: Per-unit class scores.
        labels: The true label per unit, or ``None`` when unlabelled.
        redaction_scores: Per-unit main-task score.
        threshold: The operating point.

    Returns:
        A mapping from class to its counts and both recalls.
    """
    buckets: dict[str, dict[str, int]] = {}
    for index, label in enumerate(labels):
        if label is None or label == "none":
            continue
        bucket = buckets.setdefault(label, {"total": 0, "flagged": 0, "exact": 0})
        bucket["total"] += 1
        if redaction_scores[index] >= threshold:
            bucket["flagged"] += 1
            row = scores[index]
            if row and max(row, key=lambda name: (row[name], name)) == label:
                bucket["exact"] += 1

    return {
        label: {
            "total": bucket["total"],
            "flagged": bucket["flagged"],
            "recall": round(bucket["flagged"] / bucket["total"], 6) if bucket["total"] else 0.0,
            "exact_class_recall": (
                round(bucket["exact"] / bucket["total"], 6) if bucket["total"] else 0.0
            ),
        }
        for label, bucket in sorted(buckets.items())
    }
