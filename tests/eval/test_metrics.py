"""Known-answer tests for the metrics (NR-0002 §6).

Every case here is small enough to verify by hand, which is the point: a metric
implementation that is only tested against itself can be confidently wrong. The
expected values are worked out in the comments rather than copied from a run.
"""

from __future__ import annotations

import math

import pytest
from eval import metrics


class TestConfusion:
    def test_counts_on_a_hand_checked_case(self):
        # scores:  0.9  0.8  0.4  0.2
        # truth:   T    F    T    F
        # threshold 0.5 flags the first two.
        counts = metrics.confusion_at([0.9, 0.8, 0.4, 0.2], [True, False, True, False], 0.5)
        assert (counts.true_positives, counts.false_positives) == (1, 1)
        assert (counts.false_negatives, counts.true_negatives) == (1, 1)
        assert counts.recall == 0.5
        assert counts.false_positive_rate == 0.5
        assert counts.precision == 0.5

    def test_threshold_is_inclusive(self):
        counts = metrics.confusion_at([0.5], [True], 0.5)
        assert counts.true_positives == 1

    def test_no_positives_gives_zero_recall_not_an_error(self):
        counts = metrics.confusion_at([0.9, 0.1], [False, False], 0.5)
        assert counts.recall == 0.0
        assert counts.false_positive_rate == 0.5

    def test_no_negatives_gives_zero_false_positive_rate(self):
        counts = metrics.confusion_at([0.9], [True], 0.5)
        assert counts.false_positive_rate == 0.0

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            metrics.confusion_at([0.1, 0.2], [True], 0.5)

    def test_to_dict_is_json_friendly(self):
        import json

        json.dumps(metrics.confusion_at([0.9], [True], 0.5).to_dict())


class TestThresholdSelection:
    def test_picks_the_lowest_threshold_meeting_the_target(self):
        # 100 negatives scoring 0.0..0.99 and one positive at 1.0.
        # A 1% target allows 1 false positive, so the threshold may sit just
        # above the second-highest negative (0.98) — catching 0.99 and 1.0.
        scores = [i / 100 for i in range(100)] + [1.0]
        truth = [False] * 100 + [True]
        threshold = metrics.threshold_for_false_positive_rate(scores, truth, 0.01)
        assert threshold is not None
        counts = metrics.confusion_at(scores, truth, threshold)
        assert counts.false_positive_rate <= 0.01
        assert counts.recall == 1.0

    def test_zero_target_allows_no_false_positives(self):
        scores = [0.1, 0.2, 0.9]
        truth = [False, False, True]
        threshold = metrics.threshold_for_false_positive_rate(scores, truth, 0.0)
        assert threshold is not None
        assert metrics.confusion_at(scores, truth, threshold).false_positives == 0

    def test_returns_none_when_the_target_is_unreachable(self):
        # Every negative outscores every positive: no threshold separates them
        # without a false positive, so there is nothing honest to return.
        scores = [0.9, 0.9, 0.1]
        truth = [False, False, True]
        assert metrics.threshold_for_false_positive_rate(scores, truth, 0.0) is None

    def test_all_positives_means_any_threshold_works(self):
        threshold = metrics.threshold_for_false_positive_rate([0.3, 0.7], [True, True], 0.001)
        assert threshold == 0.3


class TestAveragePrecision:
    def test_perfect_ranking_scores_one(self):
        assert metrics.average_precision([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0

    def test_worst_ranking_on_a_hand_checked_case(self):
        # Negatives ranked first: the positives land at ranks 3 and 4, so
        # average precision = (1/3 + 2/4) / 2 = (0.3333 + 0.5) / 2 = 0.41667.
        value = metrics.average_precision([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
        assert value == pytest.approx((1 / 3 + 2 / 4) / 2, abs=1e-9)

    def test_interleaved_on_a_hand_checked_case(self):
        # T F T F -> positives at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.83333.
        value = metrics.average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
        assert value == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-9)

    def test_no_positives_scores_zero(self):
        assert metrics.average_precision([0.5, 0.5], [False, False]) == 0.0

    def test_ties_are_broken_pessimistically(self):
        # All scores equal. Negatives are ranked first, so a system that cannot
        # discriminate cannot score well by luck.
        pessimistic = metrics.average_precision([0.5] * 4, [True, True, False, False])
        assert pessimistic < 1.0


class TestBootstrap:
    def test_point_estimate_is_the_mean(self):
        result = metrics.bootstrap_interval([1.0, 1.0, 0.0, 0.0], resamples=200)
        assert result["point"] == 0.5
        assert result["n"] == 4

    def test_interval_brackets_the_point(self):
        result = metrics.bootstrap_interval([1.0] * 60 + [0.0] * 40, resamples=500)
        assert result["low"] <= result["point"] <= result["high"]

    def test_is_reproducible_for_a_fixed_seed(self):
        values = [1.0] * 30 + [0.0] * 70
        first = metrics.bootstrap_interval(values, resamples=300, seed=7)
        second = metrics.bootstrap_interval(values, resamples=300, seed=7)
        assert first == second

    def test_a_different_seed_moves_the_interval(self):
        values = [1.0] * 30 + [0.0] * 70
        a = metrics.bootstrap_interval(values, resamples=300, seed=1)
        b = metrics.bootstrap_interval(values, resamples=300, seed=2)
        assert (a["low"], a["high"]) != (b["low"], b["high"])

    def test_constant_input_gives_a_degenerate_interval(self):
        result = metrics.bootstrap_interval([1.0] * 20, resamples=100)
        assert result["low"] == result["high"] == 1.0

    def test_empty_input_is_reported_not_raised(self):
        assert metrics.bootstrap_interval([]) == {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}

    def test_unsupported_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported statistic"):
            metrics.bootstrap_interval([1.0], statistic="median")

    def test_interval_narrows_as_the_sample_grows(self):
        small = metrics.bootstrap_interval([1.0, 0.0] * 10, resamples=400, seed=3)
        large = metrics.bootstrap_interval([1.0, 0.0] * 500, resamples=400, seed=3)
        assert (large["high"] - large["low"]) < (small["high"] - small["low"])


class TestIndicators:
    def test_recall_indicators_cover_only_positives(self):
        indicators = metrics.recall_indicators([0.9, 0.1, 0.8], [True, True, False], 0.5)
        assert indicators == [1.0, 0.0]

    def test_false_positive_indicators_cover_only_negatives(self):
        indicators = metrics.false_positive_indicators([0.9, 0.1, 0.8], [True, False, False], 0.5)
        assert indicators == [0.0, 1.0]

    def test_indicator_means_match_the_confusion_counts(self):
        scores = [0.9, 0.6, 0.4, 0.1]
        truth = [True, False, True, False]
        counts = metrics.confusion_at(scores, truth, 0.5)
        recall = metrics.recall_indicators(scores, truth, 0.5)
        fpr = metrics.false_positive_indicators(scores, truth, 0.5)
        assert sum(recall) / len(recall) == counts.recall
        assert sum(fpr) / len(fpr) == counts.false_positive_rate


class TestPerClassRecall:
    def test_separates_flagged_from_correctly_attributed(self):
        # A token flagged as an api_key is redacted (recall) but misattributed
        # (exact_class_recall) — the distinction the project needs.
        scores = [{"api_key": 0.9}, {"token": 0.9}]
        labels = ["token", "token"]
        redaction = [0.9, 0.9]
        result = metrics.per_class_recall(scores, labels, redaction, 0.5)
        assert result["token"]["recall"] == 1.0
        assert result["token"]["exact_class_recall"] == 0.5

    def test_ignores_negatives_and_unlabelled(self):
        result = metrics.per_class_recall(
            [{"api_key": 1.0}, {}, {}], ["api_key", "none", None], [1.0, 0.0, 0.0], 0.5
        )
        assert set(result) == {"api_key"}

    def test_missed_secrets_lower_recall(self):
        result = metrics.per_class_recall(
            [{"api_key": 0.9}, {}], ["api_key", "api_key"], [0.9, 0.0], 0.5
        )
        assert result["api_key"] == {
            "total": 2,
            "flagged": 1,
            "recall": 0.5,
            "exact_class_recall": 0.5,
        }


def test_entropy_of_uniform_alphabet_is_log2_of_its_size():
    from eval.baselines.rules_entropy import shannon_entropy

    # Four distinct characters, each once: entropy is exactly 2 bits.
    assert shannon_entropy("abcd") == pytest.approx(2.0, abs=1e-12)
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("ab") == pytest.approx(math.log2(2), abs=1e-12)


class TestBootstrapEquivalence:
    """The fast path must be the same distribution, not merely a fast one.

    Resampling n items with replacement and taking the mean of 0/1 values is
    distributionally identical to drawing the count of ones from Binomial(n, p)
    and dividing by n. These tests pin that claim, because the implementation
    relies on it for a four-order-of-magnitude speedup and a silent divergence
    would quietly corrupt every confidence interval in every report.
    """

    @staticmethod
    def _explicit_resample(values, resamples, seed):
        """Resample item by item, the textbook way."""
        import random as _random

        rng = _random.Random(seed)
        size = len(values)
        means = sorted(
            sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(resamples)
        )
        lower = means[int(0.025 * resamples)]
        upper = means[int(0.975 * resamples) - 1]
        return lower, upper

    @pytest.mark.parametrize("share", [0.05, 0.3, 0.5, 0.9])
    def test_interval_matches_explicit_resampling(self, share):
        size = 2000
        values = [1.0] * int(size * share) + [0.0] * (size - int(size * share))
        fast = metrics.bootstrap_interval(values, resamples=600, seed=11)
        low, high = self._explicit_resample(values, 600, 11)
        # Both are random estimates of the same interval, so they agree to
        # within sampling noise rather than exactly.
        assert fast["low"] == pytest.approx(low, abs=0.02)
        assert fast["high"] == pytest.approx(high, abs=0.02)

    def test_non_indicator_input_uses_the_explicit_path(self):
        """Values that are not 0/1 must not take the binomial shortcut."""
        values = [0.25, 0.5, 0.75, 1.0]
        result = metrics.bootstrap_interval(values, resamples=200, seed=5)
        assert result["point"] == pytest.approx(0.625)
        assert result["low"] <= result["point"] <= result["high"]

    def test_fast_path_is_actually_fast(self):
        """A regression here would return the report to taking an afternoon."""
        import time

        values = [1.0] * 20000 + [0.0] * 30000
        started = time.perf_counter()
        metrics.bootstrap_interval(values, resamples=1000, seed=1)
        assert time.perf_counter() - started < 1.0
