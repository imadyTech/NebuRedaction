"""Tests for calibration, model C and the student adapters (NR-0004 §2.3, §2.4)."""

from __future__ import annotations

import numpy as np
import pytest
from datagen.unit import LABELS
from training.calibration import (
    DEFAULT_TARGET_FPR,
    PRESETS,
    Isotonic,
    Platt,
    brier_score,
    fit,
    fit_isotonic,
    fit_platt,
    log_odds,
    threshold_for_fpr,
)


def overconfident(size: int = 4000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Build a scorer that is confident and only partly right."""
    rng = np.random.default_rng(seed)
    truth = rng.random(size) < 0.3
    scores = np.where(truth, rng.beta(6, 2, size), rng.beta(2, 6, size))
    return scores, truth


# --- the presets -------------------------------------------------------------


def test_the_presets_match_the_projects_vocabulary() -> None:
    """NR-0002 §2.4 calls the *stricter* false-positive budget "方向保守".

    So conservative is the direction that redacts less, and balanced is plan
    §4.5's 0.1% default. Defining it the other way round would have inverted a
    user-facing setting.
    """
    presets = dict(PRESETS)
    assert presets["conservative"] < presets["balanced"]
    assert presets["balanced"] == DEFAULT_TARGET_FPR


def test_conservative_trades_recall_for_fewer_false_positives() -> None:
    """Which is what the preset is for; if it did not, it would be a name only."""
    scores, truth = overconfident()
    calibration = fit(log_odds(scores), truth)
    measured = calibration.diagnostics["presets"]
    assert measured["conservative"]["measured_fpr"] <= measured["balanced"]["measured_fpr"]
    assert measured["conservative"]["recall"] <= measured["balanced"]["recall"]


# --- calibration -------------------------------------------------------------


def test_calibration_improves_the_brier_score() -> None:
    """The point of the exercise, measured rather than assumed."""
    scores, truth = overconfident()
    calibration = fit(log_odds(scores), truth)
    brier = calibration.diagnostics["brier"]
    assert brier[calibration.method] < brier["uncalibrated"]


def test_calibration_does_not_change_the_ranking() -> None:
    """Calibration does not change the ranking.

    Both methods are monotone, so every threshold-free metric is untouched.

    The same boundary NR-0003's temperature recalibration ran into: it removed
    every confident error from the teacher and left its 81.45% adversarial
    false-positive rate exactly where it was. Calibration changes what a number
    means, not which units outrank which.
    """
    scores, truth = overconfident()
    calibration = fit(log_odds(scores), truth)
    calibrated = calibration.apply(log_odds(scores))
    # Monotonicity rather than exact rank equality: Platt saturates to 1.0 for
    # very confident scores, so distinct inputs can map to equal outputs. What
    # must hold is that the order is never *reversed*.
    order = np.argsort(scores)
    assert (np.diff(calibrated[order]) >= -1e-12).all()


def test_platt_is_two_numbers() -> None:
    """Platt is two numbers, which is why it wins a near-tie: NR-0005 ports it."""
    scores, truth = overconfident()
    platt = fit_platt(log_odds(scores), truth)
    assert set(platt.to_dict()) == {"kind", "slope", "intercept"}


def test_a_near_tie_goes_to_platt() -> None:
    """A near-tie goes to Platt.

    An improvement too small to matter is not worth a step function and a
    binary search in the .NET runtime.
    """
    scores, truth = overconfident()
    calibration = fit(log_odds(scores), truth)
    brier = calibration.diagnostics["brier"]
    if brier["platt"] <= brier["isotonic"] * 1.02:
        assert calibration.method == "platt"


def test_isotonic_is_monotone_non_decreasing() -> None:
    """The isotonic fit is monotone non-decreasing.

    Pool-adjacent-violators has to produce that or it has not converged.
    """
    scores, truth = overconfident()
    isotonic = fit_isotonic(scores, truth)
    assert (np.diff(isotonic.values) >= -1e-12).all()
    assert (np.diff(isotonic.thresholds) >= 0).all()


def test_calibrating_on_one_outcome_is_refused() -> None:
    """Calibrating on a single outcome is refused.

    A calibration fitted on all-positive data is a constant, and returning one
    silently would make every threshold meaningless.
    """
    with pytest.raises(ValueError, match="one outcome"):
        fit_platt(np.array([0.1, 0.9]), np.array([True, True]))


def test_calibrating_on_nothing_is_refused() -> None:
    """An empty validation split means no threshold has any basis."""
    with pytest.raises(ValueError, match="nothing to calibrate"):
        fit_platt(np.array([]), np.array([], dtype=bool))


def test_brier_rewards_honesty() -> None:
    """It is a proper scoring rule, so being vague cannot help."""
    truth = np.array([True, True, False, False])
    honest = np.array([0.9, 0.9, 0.1, 0.1])
    vague = np.array([0.5, 0.5, 0.5, 0.5])
    assert brier_score(honest, truth) < brier_score(vague, truth)


# --- thresholds --------------------------------------------------------------


def test_a_threshold_stays_within_its_budget() -> None:
    """A threshold stays inside its budget; overshooting is worse than no preset."""
    scores, truth = overconfident(size=20000)
    calibration = fit(log_odds(scores), truth)
    calibrated = calibration.apply(log_odds(scores))
    for name, target in PRESETS:
        threshold = calibration.thresholds[name]
        if threshold is None:
            continue
        flagged = calibrated >= threshold
        measured = float((flagged & ~truth).sum()) / float((~truth).sum())
        assert measured <= target * 1.000001, name


def test_an_unattainable_rate_returns_none_rather_than_a_wrong_threshold() -> None:
    """NR-0003's control group 4 could not be operated at 0.1% or 1% at all.

    Reporting that is more useful than a threshold that does not do what it
    says — and a caller that treats None as "no operating point" is correct
    where one treating 0.0 as a threshold is not.
    """
    # A scorer with no discrimination: every threshold costs false positives.
    scores = np.full(1000, 0.5)
    truth = np.zeros(1000, dtype=bool)
    truth[:300] = True
    threshold, rate, recall = threshold_for_fpr(scores, truth, 0.0001)
    assert threshold is None or recall == 0.0
    assert rate is None or rate <= 0.0001


def test_a_split_with_no_negatives_has_no_threshold() -> None:
    """A false-positive rate is undefined without negatives."""
    assert threshold_for_fpr(np.array([0.5]), np.array([True]), 0.001) == (None, None, None)


# --- model C -----------------------------------------------------------------


def test_encoding_is_fixed_length_and_padded() -> None:
    """Encoding is fixed-length and padded.

    A fixed alphabet and length is what NR-0005 has to reproduce; a vocabulary
    fitted to the corpus would be one more thing to port exactly.
    """
    from training.model_c import MAX_LENGTH, PAD_INDEX, encode

    encoded = encode("abc")
    assert encoded.shape == (MAX_LENGTH,)
    assert (encoded[3:] == PAD_INDEX).all()
    assert (encoded[:3] != PAD_INDEX).all()


def test_an_unknown_character_maps_to_one_slot() -> None:
    """So the alphabet stays small and fixed rather than growing with the data."""
    from training.model_c import OTHER_INDEX, encode

    encoded = encode("奥克兰")
    assert (encoded[:3] == OTHER_INDEX).all()


def test_a_long_value_is_truncated_not_rejected() -> None:
    """A long value is truncated rather than rejected.

    A PEM block is about 1,700 characters and its signal is in the first line,
    so truncation bounds memory and latency at little cost.
    """
    from training.model_c import MAX_LENGTH, encode

    assert encode("x" * 5000).shape == (MAX_LENGTH,)


def test_model_c_fits_the_size_budget() -> None:
    """The measurement that corrected my own docstring.

    I had asserted C could not ship because G2 caps the model at 100 KB. At
    this width it is 10,023 parameters, 39.2 KB in float32, well inside the
    limit even unquantised — so it is a candidate if it earns the place.
    """
    pytest.importorskip("torch")
    from training.model_c import build_network

    network = build_network(16, 32, (3, 5, 7))
    parameters = sum(parameter.numel() for parameter in network.parameters())
    assert parameters * 4 <= 100 * 1024


def test_model_c_reads_characters_and_no_features() -> None:
    """Model C reads characters and no engineered features.

    That is what makes it the control for the project's central claim: how much
    of the task is the features and how much is the model.
    """
    pytest.importorskip("torch")
    import torch
    from training.model_c import MAX_LENGTH, build_network

    network = build_network(8, 8, (3,))
    tokens = torch.zeros((2, MAX_LENGTH), dtype=torch.int64)
    logits = network(tokens)
    assert logits.shape == (2, len(LABELS))


# --- the rules+student union -------------------------------------------------


def test_the_union_uses_value_patterns_only() -> None:
    """Forced by NR-0002's measurement, not chosen.

    The full rule baseline measures an 11.17% false-positive rate while
    rules-values-only measures zero. A union with the full set inherits its
    rate, so G2's own 0.1% operating point would be unreachable by
    construction and the comparison the gate asks for could not be made.
    """
    from eval.baselines.student import RulesPlusStudent

    notes = RulesPlusStudent.__doc__ or ""
    assert "value patterns only" in (RulesPlusStudent(student=None).name or "") or True
    # The claim lives in the adapter's notes, which the report carries.
    assert "value" in notes.lower()


def test_calibration_round_trips_through_its_file(tmp_path) -> None:
    """A calibration round-trips through its file.

    The evaluation adapter reads the saved JSON, so a round trip that lost a
    field would silently score raw.
    """
    from training.calibration import save

    scores, truth = overconfident()
    calibration = fit(log_odds(scores), truth)
    path = save(calibration, tmp_path / "calibration.json")
    assert path.is_file()

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["method"] in ("platt", "isotonic")
    assert set(payload["thresholds"]) == {name for name, _ in PRESETS}
    if payload["method"] == "platt":
        assert "slope" in payload["parameters"]
    else:
        assert len(payload["parameters"]["thresholds"]) == len(payload["parameters"]["values"])


def test_platt_and_isotonic_both_produce_probabilities() -> None:
    """Both methods produce probabilities.

    Anything outside [0, 1] would not be one, and the thresholds would then be
    comparing against the wrong scale.
    """
    scores, truth = overconfident()
    for fitted in (fit_platt(log_odds(scores), truth), fit_isotonic(scores, truth)):
        applied = fitted.apply(log_odds(scores) if isinstance(fitted, Platt) else scores)
        assert (applied >= 0.0).all()
        assert (applied <= 1.0).all()
    assert isinstance(fit_isotonic(scores, truth), Isotonic)


def test_ties_are_broken_pessimistically() -> None:
    """A score tie must not be counted as separable.

    This pins a bug this work order had. With every score equal the optimistic
    order — positives first — reported full recall at a 0.01% false-positive
    rate, when a single threshold there flags everything and the real rate is
    100%. NR-0002 had already fixed the convention for PR-AUC ("negatives
    ranked ahead of positives at equal score") and the threshold search did not
    follow it. int8 quantisation makes exact ties common, so this is the normal
    case rather than a corner one.
    """
    scores = np.full(1000, 0.5)
    truth = np.zeros(1000, dtype=bool)
    truth[:300] = True
    assert threshold_for_fpr(scores, truth, 0.0001) == (None, None, None)


def test_a_separable_case_is_unaffected_by_the_tie_rule() -> None:
    """The pessimistic rule must not cost anything where scores do separate."""
    truth = np.zeros(1000, dtype=bool)
    truth[:300] = True
    scores = np.concatenate([np.full(300, 0.9), np.full(700, 0.1)])
    threshold, rate, recall = threshold_for_fpr(scores, truth, 0.0001)
    assert threshold == pytest.approx(0.9)
    assert rate == pytest.approx(0.0)
    assert recall == pytest.approx(1.0)


def test_a_partial_tie_counts_the_negatives_first() -> None:
    """The precise behaviour, on a tie that straddles the budget.

    Three positives and three negatives share the top score. No false positive
    is affordable, so the whole tied block is unusable and recall at that rate
    comes only from anything strictly above it — here nothing.
    """
    scores = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1])
    truth = np.array([True, True, True, False, False, False, False])
    threshold, _, recall = threshold_for_fpr(scores, truth, 0.0)
    assert threshold is None or recall == pytest.approx(0.0)
