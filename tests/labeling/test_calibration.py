"""Tests for temperature recalibration (NR-0003 §2.2 item 6)."""

from __future__ import annotations

import math

import pytest
from datagen.unit import LABELS
from labeling.teacher.calibration import (
    MAX_TEMPERATURE,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
)


def distribution(top: str, confidence: float) -> dict[str, float]:
    """Build a distribution with a given top class and confidence."""
    rest = [label for label in LABELS if label != top]
    return {top: confidence, **{label: (1.0 - confidence) / len(rest) for label in rest}}


def test_temperature_one_changes_nothing() -> None:
    """The identity case, which the auto-adopt path depends on being exact."""
    original = distribution("token", 0.8)
    scaled = apply_temperature(original, 1.0)
    for label in LABELS:
        assert scaled[label] == pytest.approx(original[label], abs=1e-9)


def test_a_high_temperature_flattens_and_a_low_one_sharpens() -> None:
    """Which is the whole mechanism: it moves confidence, not the choice."""
    original = distribution("token", 0.8)
    assert apply_temperature(original, 5.0)["token"] < 0.8
    assert apply_temperature(original, 0.2)["token"] > 0.8


def test_the_result_is_always_a_distribution() -> None:
    """Downstream code assumes seven keys summing to one."""
    for temperature in (0.1, 0.5, 1.0, 3.0, 19.0):
        scaled = apply_temperature(distribution("api_key", 0.95), temperature)
        assert set(scaled) == set(LABELS)
        assert sum(scaled.values()) == pytest.approx(1.0)


def test_the_argmax_is_invariant() -> None:
    """Temperature scaling is monotone, so it cannot fix a wrong answer.

    This is the property that bounds what §2.2 item 6 can achieve, and the
    reason the adversarial spot-check set exists alongside the recalibration:
    calibration makes 0.9 mean 90%, it does not stop `OrderId = 29` being
    called a token.
    """
    original = distribution("token", 0.4)
    for temperature in (0.1, 1.0, 10.0):
        scaled = apply_temperature(original, temperature)
        assert max(scaled, key=lambda label: scaled[label]) == "token"


def test_a_non_positive_temperature_is_refused() -> None:
    """Zero would divide by zero and a negative would invert the ordering."""
    with pytest.raises(ValueError, match="must be positive"):
        apply_temperature(distribution("none", 0.5), 0.0)


def test_an_overconfident_teacher_is_fitted_a_temperature_above_one() -> None:
    """The case NR-0000 and this run both measured.

    Every answer is reported at 0.95 but only 60% are right, so the
    distributions must be flattened: T > 1.
    """
    distributions, truths = [], []
    for index in range(100):
        distributions.append(distribution("token", 0.95))
        truths.append("token" if index < 60 else "none")
    result = fit_temperature(distributions, truths)
    assert result.temperature > 1.0
    assert result.nll_after < result.nll_before
    assert result.ece_after < result.ece_before
    assert result.accuracy == pytest.approx(0.60)


def test_an_underconfident_teacher_is_fitted_a_temperature_below_one() -> None:
    """The symmetric case, so the search is not biased in one direction."""
    distributions, truths = [], []
    for index in range(100):
        distributions.append(distribution("token", 0.45))
        truths.append("token" if index < 95 else "none")
    result = fit_temperature(distributions, truths)
    assert result.temperature < 1.0
    assert result.nll_after <= result.nll_before


def test_a_well_calibrated_teacher_is_left_near_one() -> None:
    """Nothing to fix, so the fit should not invent a correction."""
    distributions, truths = [], []
    for index in range(1000):
        distributions.append(distribution("token", 0.7))
        truths.append("token" if index % 10 < 7 else "none")
    result = fit_temperature(distributions, truths)
    assert 0.6 < result.temperature < 1.6
    assert result.ece_after < 0.1


def test_accuracy_is_unchanged_by_the_fit() -> None:
    """Stated in the report's note, so it had better be true."""
    distributions = [distribution("token", 0.9) for _ in range(50)]
    truths = ["token" if index < 20 else "none" for index in range(50)]
    result = fit_temperature(distributions, truths)

    correct = 0
    for original, truth in zip(distributions, truths, strict=True):
        scaled = apply_temperature(original, result.temperature)
        if max(scaled, key=lambda label: scaled[label]) == truth:
            correct += 1
    assert correct / len(truths) == pytest.approx(result.accuracy)


def test_hitting_a_search_bound_is_reported_not_hidden() -> None:
    """A clamped optimum is reported as clamped.

    A report that said 20.0 without saying "at the bound" would be read as a
    measurement.
    """
    # Every answer confidently wrong: the best the search can do is flatten as
    # far as it is allowed.
    distributions = [distribution("token", 0.999) for _ in range(50)]
    truths = ["none"] * 50
    result = fit_temperature(distributions, truths)
    assert result.at_bound is True
    assert result.temperature >= MAX_TEMPERATURE * 0.99


def test_the_objective_is_minimised_at_the_fitted_temperature() -> None:
    """A sanity check on the search itself, against a brute-force scan."""
    distributions, truths = [], []
    for index in range(200):
        distributions.append(distribution("api_key", 0.85))
        truths.append("api_key" if index < 130 else "none")
    result = fit_temperature(distributions, truths)
    best = negative_log_likelihood(distributions, truths, result.temperature)
    for exponent in range(-20, 21):
        candidate = result.temperature * math.exp(exponent / 10.0)
        if candidate <= 0:
            continue
        assert negative_log_likelihood(distributions, truths, candidate) >= best - 1e-9


def test_a_perfectly_calibrated_set_has_zero_calibration_error() -> None:
    """The ECE's own sanity check."""
    distributions, truths = [], []
    for index in range(100):
        distributions.append(distribution("token", 0.65))
        truths.append("token" if index % 20 < 13 else "none")
    assert expected_calibration_error(distributions, truths) < 0.02


def test_calibrating_on_nothing_is_an_error() -> None:
    """An empty validation split means the threshold has no basis at all."""
    with pytest.raises(ValueError, match="nothing to calibrate"):
        fit_temperature([], [])


def test_an_unknown_truth_class_is_an_error() -> None:
    """A truth outside the seven means the label sets have drifted apart."""
    with pytest.raises(ValueError, match="unknown classes"):
        fit_temperature([distribution("token", 0.5)], ["banana"])
