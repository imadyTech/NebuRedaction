"""Calibration and the two threshold presets (NR-0004 §2 item 3, plan §4.5).

Plan §4.5 asks for Platt or isotonic calibration of the student's output, a
default threshold chosen on the validation split for a false-positive rate at
or below 0.1% per attribute, and "conservative / balanced" presets.

**What "conservative" means here is taken from the project's own usage, not
invented.** NR-0002 §2.4 describes holding the student to a *stricter*
false-positive budget than the baseline it must beat as "方向保守 ... 是刻意的"
— conservative is the direction that redacts less and damages fewer logs. So:

===============  ==================  ====================================
preset           target FPR          what it is for
===============  ==================  ====================================
``conservative``  0.01%              ten times stricter than the default;
                                     fewer masked non-secrets, lower recall
``balanced``      0.1%               plan §4.5's default
===============  ==================  ====================================

Both are chosen on the **validation** split and nowhere else (EXECUTION.md §8),
and the measured rate on every test set is reported next to them, because
NR-0002 finding 5 measured a threshold picked for 0.1% on synthetic validation
data coming out at 1.223% on CredData — twenty times its target. A preset is a
promise about the data it was fitted on, not about the world.

**Calibration cannot change which units are flagged at a given rank.** Both
methods here are monotone in the score, so the ranking is untouched and so is
every threshold-free metric. What they change is what a reported probability
*means*, which is the whole of what a threshold needs — the same boundary
NR-0003's temperature recalibration ran into, where recalibrating the teacher
left its 81.45% adversarial false-positive rate exactly where it was while
removing every confident error.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The presets, as (name, target false-positive rate).
PRESETS: tuple[tuple[str, float], ...] = (("conservative", 0.0001), ("balanced", 0.001))

#: Plan §4.5's default, which `balanced` implements.
DEFAULT_TARGET_FPR = 0.001

#: Largest number of knots an isotonic calibration may keep.
#:
#: Fitted without a cap on the validation split, isotonic produced **16,825
#: knots — 263 KB**, against a model of 40.3 KB and a G2 budget of 100 KB for
#: the whole thing. A model cannot ship without its calibration, so the
#: calibration is part of the size, and a 263 KB correction to a 40 KB model is
#: the tail wagging the dog.
#:
#: A monotone step function does not need one knot per training row. Thinning
#: to this many, by taking an evenly spaced subset in rank order, costs a
#: fraction of the Brier gain and brings the stored size to about 2 KB in
#: float32. The chosen count is recorded with the fit so the trade is visible.
MAX_ISOTONIC_KNOTS = 256

#: Bytes per retained knot: a float32 threshold and a float32 value.
BYTES_PER_KNOT = 8

#: Bytes a Platt calibration costs: two float32s.
PLATT_BYTES = 8


@dataclass(frozen=True, slots=True)
class Platt:
    """A Platt scaling of a redaction score.

    Attributes:
        slope: Multiplier on the score.
        intercept: Offset.

    Two floats, which is why it is preferred where it is competitive: NR-0005
    has to reimplement whatever is chosen, and two floats port to .NET as one
    line where an isotonic step function ports as a binary search over knots.
    """

    slope: float
    intercept: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities."""
        return 1.0 / (1.0 + np.exp(-(self.slope * scores + self.intercept)))

    @property
    def size_bytes(self) -> int:
        """Stored size: two float32s."""
        return PLATT_BYTES

    def to_dict(self) -> dict[str, Any]:
        """Serialise."""
        return {
            "kind": "platt",
            "slope": self.slope,
            "intercept": self.intercept,
            "bytes": PLATT_BYTES,
        }


@dataclass(frozen=True, slots=True)
class Isotonic:
    """An isotonic (step-function) calibration.

    Attributes:
        thresholds: Ascending score knots.
        values: The calibrated probability at and above each knot.
    """

    thresholds: np.ndarray
    values: np.ndarray

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities."""
        indices = np.searchsorted(self.thresholds, scores, side="right") - 1
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]

    @property
    def size_bytes(self) -> int:
        """Stored size in float32, which is what would ship."""
        return int(self.thresholds.size) * BYTES_PER_KNOT

    def to_dict(self) -> dict[str, Any]:
        """Serialise, with the knot count since the arrays can be long."""
        return {
            "kind": "isotonic",
            "knots": int(self.thresholds.size),
            "bytes": self.size_bytes,
        }


def brier_score(probabilities: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error of a probabilistic prediction.

    Args:
        probabilities: Predicted probability of the positive class.
        truth: Boolean outcomes.

    Returns:
        The Brier score, lower being better.

    Used to choose between the two calibrations. It is a proper scoring rule,
    so it cannot be improved by being vague, and unlike log-loss it does not
    become infinite on a confident mistake — which matters because a student
    that outputs an exact 0.0 or 1.0 is common after int8 quantisation.
    """
    return float(np.mean((probabilities - truth.astype(np.float64)) ** 2))


def fit_platt(scores: np.ndarray, truth: np.ndarray, iterations: int = 200) -> Platt:
    """Fit a Platt scaling by Newton's method.

    Args:
        scores: Raw redaction scores on the validation split.
        truth: Whether each unit should be redacted.
        iterations: Maximum Newton steps.

    Returns:
        The fitted scaling.

    Raises:
        ValueError: If there is nothing to fit on, or only one outcome — a
            calibration fitted on all-positive or all-negative data is a
            constant, and silently returning one would make every threshold
            meaningless.

    Implemented here rather than taken from scikit-learn because NR-0005 has to
    port it: two parameters fitted by a documented six-line update are
    transcribable, where a dependency is not.
    """
    if scores.size == 0:
        raise ValueError("nothing to calibrate on")
    if truth.all() or not truth.any():
        raise ValueError("validation split has only one outcome; a calibration would be constant")

    target = truth.astype(np.float64)
    slope, intercept = 1.0, 0.0
    for _ in range(iterations):
        logits = slope * scores + intercept
        predicted = 1.0 / (1.0 + np.exp(-logits))
        error = predicted - target
        weight = np.maximum(predicted * (1.0 - predicted), 1e-12)

        gradient = np.array([float((error * scores).sum()), float(error.sum())])
        hessian = np.array(
            [
                [float((weight * scores * scores).sum()), float((weight * scores).sum())],
                [float((weight * scores).sum()), float(weight.sum())],
            ]
        )
        hessian += np.eye(2) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        slope -= float(step[0])
        intercept -= float(step[1])
        if max(abs(step[0]), abs(step[1])) < 1e-10:
            break
    return Platt(slope=slope, intercept=intercept)


def fit_isotonic(
    scores: np.ndarray, truth: np.ndarray, max_knots: int | None = MAX_ISOTONIC_KNOTS
) -> Isotonic:
    """Fit an isotonic calibration by pool-adjacent-violators.

    Args:
        scores: Raw redaction scores on the validation split.
        truth: Whether each unit should be redacted.
        max_knots: Largest number of knots to keep, or None for all of them.

    Returns:
        The fitted step function, thinned to at most ``max_knots`` knots.

    Raises:
        ValueError: If there is nothing to fit on.

    Also implemented here rather than imported, for the same portability
    reason, and because the algorithm is a dozen lines.
    """
    if scores.size == 0:
        raise ValueError("nothing to calibrate on")

    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_truth = truth[order].astype(np.float64)

    # Pool adjacent violators: merge neighbouring blocks until the block means
    # are non-decreasing.
    values: list[float] = []
    weights: list[float] = []
    starts: list[float] = []
    for score, outcome in zip(ordered_scores, ordered_truth, strict=True):
        values.append(outcome)
        weights.append(1.0)
        starts.append(float(score))
        while len(values) > 1 and values[-2] > values[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total_weight
            values[-2:] = [merged]
            weights[-2:] = [total_weight]
            starts[-2:] = [starts[-2]]

    thresholds_array = np.array(starts, dtype=np.float64)
    values_array = np.array(values, dtype=np.float64)

    # Thin to a bounded number of knots. Without this the function has one knot
    # per distinct pooled block — 16,825 on the validation split, 263 KB — for a
    # model of 40 KB inside a 100 KB budget. An evenly spaced subset in rank
    # order preserves the shape of a monotone step function closely, and the
    # first and last knots are always kept so the domain is not clipped.
    if max_knots is not None and thresholds_array.size > max_knots:
        # Keep the knots where the function actually *moves*, not every Nth one.
        # An evenly spaced subset was tried first and was much worse than no
        # calibration at all (Brier 0.0305 against 0.0277 uncalibrated): a
        # monotone step function concentrates almost all of its rise in a few
        # places, and sampling uniformly in rank order misses exactly those.
        steps = np.abs(np.diff(values_array, prepend=values_array[0]))
        keep = np.argsort(-steps)[: max_knots - 2]
        picks = np.unique(np.concatenate([keep, np.array([0, thresholds_array.size - 1])]))
        thresholds_array = thresholds_array[picks]
        values_array = values_array[picks]

    return Isotonic(thresholds=thresholds_array, values=values_array)


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted calibration and its thresholds.

    Attributes:
        method: ``platt`` or ``isotonic``.
        platt: The Platt parameters, when that method won.
        isotonic: The isotonic step function, when that method won.
        thresholds: Preset name to calibrated-probability threshold.
        diagnostics: Brier scores and the measured rates.
    """

    method: str
    platt: Platt | None
    isotonic: Isotonic | None
    thresholds: dict[str, float | None]
    diagnostics: dict[str, Any]

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate raw scores."""
        if self.method == "platt":
            assert self.platt is not None
            return self.platt.apply(scores)
        assert self.isotonic is not None
        return self.isotonic.apply(scores)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the model metadata."""
        fitted = self.platt if self.method == "platt" else self.isotonic
        assert fitted is not None
        return {
            "method": self.method,
            "parameters": fitted.to_dict(),
            "thresholds": self.thresholds,
            "diagnostics": self.diagnostics,
        }


def threshold_for_fpr(
    probabilities: np.ndarray, truth: np.ndarray, target: float
) -> tuple[float | None, float | None, float | None]:
    """Find the threshold whose false-positive rate fits a target.

    Args:
        probabilities: Calibrated probabilities.
        truth: Whether each unit should be redacted.
        target: The false-positive rate to stay at or below.

    Returns:
        ``(threshold, measured_fpr, recall)``, all ``None`` when no threshold
        achieves the target — which is a real outcome and not an error.
        NR-0003's control group 4 could not be operated at either 0.1% or 1%,
        and reporting that is more useful than returning a threshold that does
        not do what it says.

    The threshold is the *lowest* score that stays within budget, so recall is
    the highest available at that rate, and ties are broken pessimistically so
    that the rate is one the threshold can actually hold.
    """
    negatives = int((~truth).sum())
    positives = int(truth.sum())
    if negatives == 0 or positives == 0:
        return None, None, None

    # Ties are broken **pessimistically**: at an equal score a negative is
    # ranked ahead of a positive. This is the convention NR-0002 fixed for
    # PR-AUC, and it is not cosmetic. A single threshold cannot separate units
    # that share a score, so counting the positives first would claim a recall
    # the threshold cannot deliver — with every score equal, the optimistic
    # order reports full recall at a 0.01% false-positive rate while the
    # threshold in fact flags everything. int8 quantisation makes exact ties
    # common, so this is the normal case rather than a corner one.
    order = np.lexsort((truth, -probabilities))
    sorted_truth = truth[order]
    false_positives = np.cumsum(~sorted_truth)
    true_positives = np.cumsum(sorted_truth)

    allowed = int(target * negatives)
    within = np.nonzero(false_positives <= allowed)[0]
    if within.size == 0:
        return None, None, None
    cut = int(within[-1])
    return (
        float(probabilities[order][cut]),
        float(false_positives[cut]) / negatives,
        float(true_positives[cut]) / positives,
    )


def fit(
    scores: np.ndarray, truth: np.ndarray, presets: Sequence[tuple[str, float]] = PRESETS
) -> Calibration:
    """Fit both calibrations, keep the better, and choose the preset thresholds.

    Args:
        scores: Raw redaction scores on the **validation** split.
        truth: Whether each unit should be redacted.
        presets: Preset names and their target false-positive rates.

    Returns:
        The calibration, with a threshold per preset.

    Raises:
        ValueError: If the inputs are unusable for calibration.

    Chosen by Brier score on the same split the calibration was fitted on,
    which overstates both methods equally; isotonic is the one that can
    overfit, so the choice is recorded with both scores rather than asserted,
    and a tie goes to Platt because it is two floats to port instead of a step
    function.
    """
    platt = fit_platt(scores, truth)
    isotonic = fit_isotonic(scores, truth)

    platt_brier = brier_score(platt.apply(scores), truth)
    isotonic_brier = brier_score(isotonic.apply(scores), truth)
    raw_brier = brier_score(np.clip(scores, 0.0, 1.0), truth)

    # **Chosen by recall at the balanced target, not by Brier score.** This rule
    # has been wrong twice, and both corrections are worth keeping visible.
    #
    # It first ignored size: unthinned isotonic won on Brier while costing
    # 263 KB against a 40 KB model inside a 100 KB budget, and nothing noticed.
    # A calibration ships with its model, so its bytes are the model's bytes.
    #
    # It then still selected on Brier, which is not what this project measures.
    # Plan §4.5 and G2 both operate the model at a false-positive target and ask
    # for recall there. Measured on the validation split, isotonic beat Platt on
    # Brier (0.02234 against 0.02340) and *lost* on recall at 0.1% (0.8042
    # against 0.8207) — because a step function maps whole ranges to one value,
    # which creates score ties by construction, and ties cost recall under the
    # pessimistic tie-breaking a threshold has to use. Optimising the
    # probabilistic score was choosing the worse operating point.
    #
    # So: recall at the balanced target decides, size breaks a near-tie, and
    # Brier is reported because a calibrated probability is still what the
    # caller sees.
    def operating_recall(fitted: Platt | Isotonic) -> float:
        applied = fitted.apply(scores)
        _, _, recall = threshold_for_fpr(applied, truth, DEFAULT_TARGET_FPR)
        return recall if recall is not None else -1.0

    platt_recall = operating_recall(platt)
    isotonic_recall = operating_recall(isotonic)
    method = "isotonic" if isotonic_recall > platt_recall * 1.01 else "platt"
    chosen = platt if method == "platt" else isotonic
    calibrated = chosen.apply(scores)

    thresholds: dict[str, float | None] = {}
    measured: dict[str, Any] = {}
    for name, target in presets:
        threshold, rate, recall = threshold_for_fpr(calibrated, truth, target)
        thresholds[name] = threshold
        measured[name] = {
            "target_fpr": target,
            "attainable": threshold is not None,
            "measured_fpr": round(rate, 6) if rate is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
        }

    return Calibration(
        method=method,
        platt=platt if method == "platt" else None,
        isotonic=isotonic if method == "isotonic" else None,
        thresholds=thresholds,
        diagnostics={
            "brier": {
                "uncalibrated": round(raw_brier, 6),
                "platt": round(platt_brier, 6),
                "isotonic": round(isotonic_brier, 6),
            },
            "recall_at_balanced_target": {
                "platt": round(platt_recall, 6),
                "isotonic": round(isotonic_recall, 6),
            },
            "chosen_because": (
                "Platt matches or beats isotonic on recall at the balanced target, "
                "and costs 8 bytes against 2 KB"
                if method == "platt"
                else "isotonic beat Platt by more than 1% of recall at the balanced target"
            ),
            "selection_rule": (
                "Recall at the balanced false-positive target decides, size breaks a "
                "near-tie. Brier is reported but does not select: plan §4.5 and G2 both "
                "operate at a false-positive target, and isotonic's constant regions "
                "create score ties that cost recall there even while improving Brier."
            ),
            "isotonic_knots": int(isotonic.thresholds.size),
            "size_bytes": {"platt": platt.size_bytes, "isotonic": isotonic.size_bytes},
            "chosen_size_bytes": (platt if method == "platt" else isotonic).size_bytes,
            "presets": measured,
            "fitted_on": "validation split only (EXECUTION.md §8)",
            "caveat": (
                "A preset is a promise about the split it was fitted on. NR-0002 "
                "finding 5 measured a threshold chosen for 0.1% on synthetic "
                "validation data coming out at 1.223% on CredData."
            ),
        },
    )


def save(calibration: Calibration, path: Path) -> Path:
    """Write a calibration beside its model.

    Args:
        calibration: The fitted calibration.
        path: Output JSON file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration.to_dict()
    if calibration.method == "isotonic":
        assert calibration.isotonic is not None
        payload["parameters"]["thresholds"] = calibration.isotonic.thresholds.tolist()
        payload["parameters"]["values"] = calibration.isotonic.values.tolist()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def log_odds(probabilities: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Convert probabilities to log-odds, for Platt's input.

    Args:
        probabilities: Values in [0, 1].
        floor: Clamp keeping the logarithm finite.

    Returns:
        The log-odds.

    Platt is conventionally fitted on a score, not on a probability, and
    feeding it a probability that is already squashed into [0, 1] leaves it
    very little to correct. The student's redaction score *is* a probability,
    so it is mapped back to log-odds first — which is also what makes the
    fitted slope interpretable as a sharpening or flattening, exactly like
    NR-0003's temperature.
    """
    clipped = np.clip(probabilities, floor, 1.0 - floor)
    return np.log(clipped / (1.0 - clipped))
