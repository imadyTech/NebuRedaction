"""Tests for the training dataset and model A (NR-0004 §2 item 2).

The things worth pinning here are the ones that would otherwise fail silently:
the soft-label policy, the distillation identity, the mask not leaking from
validation into training, and the size accounting G2 is judged on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from datagen.unit import LABELS
from neburedaction.features import DENSE_FEATURES, MIN_BUCKETS

from training.dataset import Record, build, read_records, soft_target
from training.model_a import (
    BYTES_PER_COEFFICIENT,
    apply_dense_scaling,
    expand_soft_targets,
    fit_dense_scales,
    quantise,
)


def record(
    label: str = "api_key",
    teacher: dict[str, float] | None = None,
    key: str = "ApiKey",
    value: str = "abc123def456",
    source: str = "synth",
) -> Record:
    """Build a record for a test case."""
    return Record(
        unit_id=f"u-{label}-{key}",
        key=key,
        value=value,
        template=None,
        scope="otel.attribute",
        label=label,
        teacher=teacher,
        source=source,
        label_source="generator",
        family="aws.x",
    )


def distribution(top: str, confidence: float) -> dict[str, float]:
    """Build a teacher distribution."""
    rest = [name for name in LABELS if name != top]
    return {top: confidence, **{name: (1.0 - confidence) / len(rest) for name in rest}}


# --- the soft-label policy ---------------------------------------------------


def test_a_negative_never_takes_a_soft_label() -> None:
    """Frank's decision, 2026-10-14, and NR-0003's measurement is the reason.

    The teacher agrees with the generator on 0.807 of positives but 0.342 of
    negatives, and its negative errors are concentrated rather than diffuse:
    `private_key` was predicted 33,708 times against 813 true instances, with
    25,096 of those on true negatives spread near-uniformly across uuids,
    Windows paths, SSH *public* keys, digests, e-mail addresses and IPv4
    addresses. Distilling that would teach the student all of those are private
    keys.
    """
    target = soft_target(record(label="none", teacher=distribution("private_key", 0.9)), 0.5)
    assert target[LABELS.index("none")] == 1.0
    assert target[LABELS.index("private_key")] == 0.0


def test_a_positive_mixes_hard_and_teacher() -> None:
    """The target is a mixture, not a replacement: the hard label keeps most of
    the mass and stays the argmax."""
    target = soft_target(record(label="api_key", teacher=distribution("none", 0.9)), 0.3)
    assert target[LABELS.index("api_key")] == pytest.approx(0.7 + 0.3 * (0.1 / 6))
    assert int(np.argmax(target)) == LABELS.index("api_key")
    assert target.sum() == pytest.approx(1.0)


def test_weight_zero_is_the_supervised_control() -> None:
    """The control the report compares distillation against: if distillation
    does not beat it, the teacher added nothing."""
    target = soft_target(record(teacher=distribution("none", 0.95)), 0.0)
    assert target[LABELS.index("api_key")] == 1.0
    assert target.sum() == pytest.approx(1.0)


def test_a_positive_without_a_teacher_opinion_is_kept_not_dropped() -> None:
    """About 45,000 of the 200,000 training rows have none, because the
    teacher's scope sampled the rule-high-confidence pool. Their hard labels
    are still guaranteed by construction."""
    target = soft_target(record(teacher=None), 0.3)
    assert target[LABELS.index("api_key")] == 1.0


def test_every_target_is_a_distribution() -> None:
    """The distillation identity in model_a depends on it."""
    for weight in (0.0, 0.3, 1.0):
        for label in LABELS:
            target = soft_target(record(label=label, teacher=distribution("token", 0.4)), weight)
            assert target.sum() == pytest.approx(1.0)
            assert (target >= 0).all()


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_an_out_of_range_weight_is_refused(weight: float) -> None:
    """A weight above one would put negative mass on the hard label."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        soft_target(record(), weight)


def test_an_unknown_hard_label_is_refused() -> None:
    """A drifted class set must not reach the target matrix."""
    with pytest.raises(ValueError, match="unknown label"):
        soft_target(record(label="secret"), 0.3)


def test_a_record_without_a_provenance_is_refused(tmp_path: Path) -> None:
    """NR-0003 writes meta.label_source on every record precisely so the
    trainer can tell a construction-guaranteed label from a model's guess.
    Training on the two as though they were alike is the difference between
    distillation and self-training."""
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"id": "a", "key": "k", "value": "v", "source": "synth", "label": "none"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label_source"):
        list(read_records(path))


# --- the rare-bucket mask ----------------------------------------------------


def test_the_mask_is_computed_on_training_data_and_reused() -> None:
    """Computing it from the validation split would let that split shape the
    model's feature space — a leak, even though it looks like housekeeping."""
    training = [record(value=f"value-{index}") for index in range(4)]
    train_set = build(training, buckets=MIN_BUCKETS, min_bucket_occurrences=2)

    validation = [record(value="completely-different-value")]
    val_set = build(validation, buckets=MIN_BUCKETS, mask=train_set.mask)
    assert val_set.mask == train_set.mask


def test_both_splits_have_the_same_width() -> None:
    """A model fitted on one and scored on the other requires it."""
    train_set = build([record(value="aaa")], buckets=MIN_BUCKETS)
    val_set = build([record(value="bbb")], buckets=MIN_BUCKETS, mask=train_set.mask)
    assert train_set.features.shape[1] == val_set.features.shape[1]
    assert train_set.features.shape[1] == len(DENSE_FEATURES) + MIN_BUCKETS


# --- the distillation identity -----------------------------------------------


def test_expansion_reproduces_the_target_mass() -> None:
    """The identity model A relies on, stated as a test.

    Cross-entropy against a distribution is a weighted sum of cross-entropies
    against one-hot targets, so a row with target p is equivalent to that row
    repeated once per class with sample_weight p_k. This checks the weights
    reconstruct p exactly.
    """
    features = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    targets = np.array([[0.7, 0.3, 0.0], [0.0, 1.0, 0.0]])
    expanded, classes, weights = expand_soft_targets(features, targets)

    assert expanded.shape[0] == 3  # two classes for row 0, one for row 1
    reconstructed = np.zeros_like(targets)
    for position, (class_index, weight) in enumerate(zip(classes, weights, strict=True)):
        original = 0 if position < 2 else 1
        reconstructed[original, class_index] = weight
    assert reconstructed == pytest.approx(targets)


def test_a_negative_expands_to_one_row() -> None:
    """So distillation costs rows only where it is actually applied."""
    features = sparse.csr_matrix(np.array([[1.0]]))
    expanded, _, weights = expand_soft_targets(features, np.array([[1.0, 0.0]]))
    assert expanded.shape[0] == 1
    assert weights == pytest.approx([1.0])


def test_a_target_that_is_not_a_distribution_is_refused() -> None:
    """The identity would no longer hold, and the fit would be silently wrong."""
    features = sparse.csr_matrix(np.array([[1.0]]))
    with pytest.raises(ValueError, match="sums to"):
        expand_soft_targets(features, np.array([[0.5, 0.2]]))


# --- dense scaling -----------------------------------------------------------


def test_scaling_preserves_sparsity() -> None:
    """Centring would make every zero non-zero: 200,000 by 16,415 float64s is
    26 GB. Scaling without centring leaves zeros at zero."""
    features = sparse.csr_matrix(np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 1.0]]))
    scales = np.array([0.5, 1.0, 1.0], dtype=np.float32)
    scaled = apply_dense_scaling(features, scales)
    assert scaled.nnz == features.nnz
    assert scaled.toarray()[0, 0] == pytest.approx(1.0)


def test_a_zero_variance_feature_gets_a_finite_scale() -> None:
    """Otherwise the multiplier is an infinity and every coefficient is NaN."""
    features = sparse.csr_matrix(np.ones((5, len(DENSE_FEATURES) + MIN_BUCKETS)))
    scales = fit_dense_scales(features)
    assert np.isfinite(scales).all()
    assert (scales == 1.0).all()


def test_scaling_leaves_the_hashed_block_alone() -> None:
    """The hashed features are signed counts in a comparable range already."""
    width = len(DENSE_FEATURES) + MIN_BUCKETS
    row = np.zeros((1, width))
    row[0, 0] = 4.0
    row[0, len(DENSE_FEATURES)] = -1.0
    scaled = apply_dense_scaling(sparse.csr_matrix(row), np.full(len(DENSE_FEATURES), 0.25))
    dense_array = scaled.toarray()
    assert dense_array[0, 0] == pytest.approx(1.0)
    assert dense_array[0, len(DENSE_FEATURES)] == pytest.approx(-1.0)


# --- quantisation and the size budget ----------------------------------------


def coefficients(width: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Build a coefficient matrix with a wide dynamic range across classes."""
    matrix = np.zeros((len(LABELS), width))
    matrix[LABELS.index("private_key"), 3] = 12.0
    matrix[LABELS.index("password"), 4] = 0.02
    matrix[LABELS.index("api_key"), 5] = -3.0
    matrix[LABELS.index("none"), 6] = 1e-9  # below the prune threshold
    return matrix, np.zeros(len(LABELS))


def test_quantisation_is_scaled_per_class() -> None:
    """One class's coefficients can be orders of magnitude larger than
    another's — private_key has unmistakable markers and password has none —
    so a single global scale would quantise the small class to zeros."""
    matrix, intercepts = coefficients()
    model = quantise(matrix, intercepts, MIN_BUCKETS, np.ones(len(DENSE_FEATURES), dtype=np.float32))
    assert model.scales[LABELS.index("private_key")] > model.scales[LABELS.index("password")]
    # The small class survives rather than vanishing.
    password_entries = model.classes == LABELS.index("password")
    assert password_entries.any()


def test_coefficients_below_the_threshold_are_dropped() -> None:
    """They contribute nothing while costing three bytes each."""
    matrix, intercepts = coefficients()
    model = quantise(matrix, intercepts, MIN_BUCKETS, np.ones(len(DENSE_FEATURES), dtype=np.float32))
    assert not (model.classes == LABELS.index("none")).any()


def test_the_size_is_computed_from_the_stored_arrays() -> None:
    """G2 is judged on this number, so it is measured and not estimated."""
    matrix, intercepts = coefficients()
    model = quantise(matrix, intercepts, MIN_BUCKETS, np.ones(len(DENSE_FEATURES), dtype=np.float32))
    expected = (
        model.stored_coefficients * BYTES_PER_COEFFICIENT
        + model.scales.nbytes
        + model.intercepts.nbytes
        + model.dense_scales.nbytes
    )
    assert model.size_bytes == expected


def test_predictions_are_distributions() -> None:
    """Scored from the stored int8 arrays, not from the scikit-learn estimator,
    so what is measured is what would ship."""
    matrix, intercepts = coefficients(width=len(DENSE_FEATURES) + MIN_BUCKETS)
    model = quantise(matrix, intercepts, MIN_BUCKETS, np.ones(len(DENSE_FEATURES), dtype=np.float32))
    features = sparse.csr_matrix(np.zeros((3, len(DENSE_FEATURES) + MIN_BUCKETS)))
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (3, len(LABELS))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3))


def test_the_label_order_is_the_projects() -> None:
    """scikit-learn orders coefficients by the classes it saw; a model stored
    in that order would be silently misread by anything using LABELS."""
    matrix, intercepts = coefficients()
    model = quantise(matrix, intercepts, MIN_BUCKETS, np.ones(len(DENSE_FEATURES), dtype=np.float32))
    assert model.labels == tuple(LABELS)
