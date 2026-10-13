"""Temperature recalibration of the teacher's probabilities (NR-0003 §2.2 item 6).

§2.2 item 6 requires this before any automatic adoption, and NR-0000 supplies
the reason: the teacher labelled a user name as a password at 0.972 confidence.
This session's run reproduced it in a harder form within the first 690 records —
``OrderId = 29`` called a bearer token at 0.979. An uncalibrated 0.9 is
therefore not "a 90% chance of being right", and the merge strategy's 0.9
auto-adopt threshold applied to raw probabilities would adopt precisely the
confident false positives the project exists to prevent.

**What this can and cannot fix.** Temperature scaling is monotone: it sharpens
or flattens every distribution by the same exponent, so it changes *how
confident* the teacher sounds and never *what it picks*. The argmax is
invariant, so recalibration cannot turn a wrong answer into a right one. What it
buys is that a reported 0.9 means something closer to 90%, which is exactly and
only what a threshold needs. A teacher that is confidently wrong in a
*systematic* way — the template-hint failure — is not a calibration problem and
will not be fixed here; it is measured instead, by §2.2 item 6's adversarial
set.

Fitting is on the synthetic validation split and nowhere else. EXECUTION.md §8
forbids selecting anything on third-party data, and the validation split is
already the project's designated selection set (NR-0002 pinned it as role
``selection``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from datagen.unit import LABELS

#: Search bounds for the temperature. Below 1 sharpens, above 1 flattens; a
#: miscalibrated-overconfident model needs T > 1. The range is wide enough that
#: hitting a bound is itself a finding rather than a silent clamp.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0

#: Bins for the expected calibration error.
ECE_BINS = 10

#: Floor applied inside logarithms, so a reported probability of exactly zero
#: does not make the objective infinite.
EPSILON = 1e-12


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """Rescale a distribution by a temperature.

    Args:
        probabilities: A distribution over :data:`datagen.unit.LABELS`.
        temperature: The temperature. 1.0 returns the input unchanged, greater
            than 1.0 flattens it, less than 1.0 sharpens it.

    Returns:
        The rescaled distribution, summing to 1.

    Raises:
        ValueError: If the temperature is not positive.

    Because the teacher reports probabilities rather than logits, the logits are
    recovered as their logarithm — exact up to the additive constant that
    softmax ignores — so this is ordinary temperature scaling and not an
    approximation of it.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = {
        label: math.log(max(value, EPSILON)) / temperature for label, value in probabilities.items()
    }
    largest = max(logits.values())
    exponentiated = {label: math.exp(value - largest) for label, value in logits.items()}
    total = sum(exponentiated.values())
    return {label: value / total for label, value in exponentiated.items()}


def negative_log_likelihood(
    distributions: Sequence[dict[str, float]], truths: Sequence[str], temperature: float
) -> float:
    """Mean negative log likelihood of the true classes at a temperature.

    Args:
        distributions: One distribution per unit.
        truths: The true class per unit.
        temperature: The temperature to evaluate.

    Returns:
        The mean NLL, in nats. This is the objective temperature scaling is
        conventionally fitted on, and it is a proper scoring rule: it is
        minimised by reporting the true probabilities, so it cannot be gamed by
        being vague.
    """
    total = 0.0
    for distribution, truth in zip(distributions, truths, strict=True):
        scaled = apply_temperature(distribution, temperature)
        total -= math.log(max(scaled.get(truth, 0.0), EPSILON))
    return total / len(truths) if truths else 0.0


def expected_calibration_error(
    distributions: Sequence[dict[str, float]],
    truths: Sequence[str],
    temperature: float = 1.0,
    bins: int = ECE_BINS,
) -> float:
    """Expected calibration error of the top-class probability.

    Args:
        distributions: One distribution per unit.
        truths: The true class per unit.
        temperature: Temperature to apply first.
        bins: Number of equal-width confidence bins.

    Returns:
        The ECE: the average gap between confidence and accuracy, weighted by
        bin population. Zero means a reported 0.9 is right 90% of the time.

    Reported alongside the NLL because the NLL alone can improve while the
    quantity the threshold actually depends on does not. The threshold reads the
    top probability, so that is what is binned.
    """
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for distribution, truth in zip(distributions, truths, strict=True):
        scaled = apply_temperature(distribution, temperature)
        best = max(scaled, key=lambda label: scaled[label])
        confidence = scaled[best]
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, best == truth))

    total = sum(len(bucket) for bucket in buckets)
    if not total:
        return 0.0
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, correct in bucket if correct) / len(bucket)
        error += len(bucket) / total * abs(mean_confidence - accuracy)
    return error


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted temperature and what it achieved.

    Attributes:
        temperature: The fitted temperature.
        units: How many units it was fitted on.
        nll_before: Mean NLL at T = 1.
        nll_after: Mean NLL at the fitted temperature.
        ece_before: ECE at T = 1.
        ece_after: ECE at the fitted temperature.
        accuracy: Top-class accuracy, which temperature scaling cannot change.
        at_bound: Whether the fit hit a search bound, which means the model is
            more miscalibrated than the range allows for and the number should
            not be trusted as an optimum.
    """

    temperature: float
    units: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    accuracy: float
    at_bound: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the execution report."""
        return {
            "temperature": round(self.temperature, 4),
            "units": self.units,
            "nll": {"before": round(self.nll_before, 4), "after": round(self.nll_after, 4)},
            "ece": {"before": round(self.ece_before, 4), "after": round(self.ece_after, 4)},
            "top_class_accuracy": round(self.accuracy, 4),
            "at_search_bound": self.at_bound,
            "note": (
                "Temperature scaling is monotone, so accuracy is unchanged by construction. "
                "It makes a reported probability mean what it says; it does not make the "
                "teacher right."
            ),
        }


def fit_temperature(
    distributions: Sequence[dict[str, float]],
    truths: Sequence[str],
    *,
    tolerance: float = 1e-4,
) -> Calibration:
    """Fit a temperature by minimising the NLL on labelled data.

    Args:
        distributions: One distribution per unit, from the teacher.
        truths: The true class per unit. Must be synthetic validation labels:
            EXECUTION.md §8 forbids selecting on third-party data.
        tolerance: Width of the bracket at which the search stops.

    Returns:
        The fitted calibration.

    Raises:
        ValueError: If there is nothing to fit on, or a truth is not one of the
            seven classes.

    Golden-section search rather than a gradient method or a dependency. The
    objective is one-dimensional and, for temperature scaling of a fixed set of
    distributions, convex in ``log T``; a bracketed search therefore finds the
    optimum without needing derivatives, and it adds no package to the `eval`
    group for the sake of one scalar.
    """
    if not distributions:
        raise ValueError("nothing to calibrate on")
    unknown = sorted(set(truths) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown classes in the truth labels: {unknown}")

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    left = high - golden * (high - low)
    right = low + golden * (high - low)
    objective_left = negative_log_likelihood(distributions, truths, math.exp(left))
    objective_right = negative_log_likelihood(distributions, truths, math.exp(right))

    while high - low > tolerance:
        if objective_left < objective_right:
            high, right, objective_right = right, left, objective_left
            left = high - golden * (high - low)
            objective_left = negative_log_likelihood(distributions, truths, math.exp(left))
        else:
            low, left, objective_left = left, right, objective_right
            right = low + golden * (high - low)
            objective_right = negative_log_likelihood(distributions, truths, math.exp(right))

    temperature = math.exp((low + high) / 2.0)
    correct = 0
    for distribution, truth in zip(distributions, truths, strict=True):
        if max(distribution, key=lambda label: distribution[label]) == truth:
            correct += 1

    return Calibration(
        temperature=temperature,
        units=len(distributions),
        nll_before=negative_log_likelihood(distributions, truths, 1.0),
        nll_after=negative_log_likelihood(distributions, truths, temperature),
        ece_before=expected_calibration_error(distributions, truths, 1.0),
        ece_after=expected_calibration_error(distributions, truths, temperature),
        accuracy=correct / len(distributions),
        at_bound=(temperature <= MIN_TEMPERATURE * 1.01 or temperature >= MAX_TEMPERATURE * 0.99),
    )
