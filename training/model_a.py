"""Model A: hashed features plus multinomial logistic regression (NR-0004 §2 item 2).

Plan §4.3 makes A the preferred model — smallest, fastest, easiest to explain —
and budgets it at 16-64 KB int8. G2 caps the shipped model at 100 KB.

**The size budget forces sparsity, not just quantisation.** With seven classes
and the plan's *minimum* 2^14 buckets, a dense int8 coefficient matrix is
(31 + 16384) x 7 = 114,905 bytes = 112 KB, already over the limit before any
larger bucket count is considered. Stored sparsely at three bytes an entry —
a 16-bit index and an int8 value — 100 KB buys about 30% non-zero and 67 KB
buys 20%. So L1 regularisation is a requirement of the gate rather than a
preference, and that is why the solver is ``saga``: it is the only multinomial
solver in scikit-learn that supports an L1 penalty.

**Distillation by sample replication.** scikit-learn's ``y`` takes a class, not
a distribution, so soft targets cannot be passed directly. They do not need a
custom optimiser either: cross-entropy against a distribution is exactly a
weighted sum of cross-entropies against one-hot targets,

    -sum_k p_k log q_k  =  sum_k p_k * (-log q_k)

so a row with target ``p`` is equivalent to that row repeated once per class
with ``sample_weight = p_k``. This is an identity, not an approximation, and it
buys a well-tested L1 solver instead of a hand-rolled one. Classes with zero
weight are skipped, so a negative stays one row and only distilled positives
expand.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS
from scipy import sparse
from training.dataset import Dataset

from neburedaction.features import (
    DENSE_FEATURES,
    FEATURE_VERSION,
)

#: Smallest absolute coefficient kept. Below this a coefficient is noise that
#: costs three bytes to store, and the gate is a size limit.
PRUNE_THRESHOLD = 1e-4

#: Bytes per stored coefficient: a 16-bit column index plus an int8 value.
BYTES_PER_COEFFICIENT = 3


@dataclass(frozen=True, slots=True)
class QuantisedModel:
    """An int8 multinomial logistic regression, stored sparsely.

    Attributes:
        columns: Column index per stored coefficient.
        classes: Class index per stored coefficient.
        values: int8 coefficient values.
        scales: Per-class dequantisation scale.
        intercepts: Per-class intercept, kept in float32 — seven numbers, so
            quantising them would save 21 bytes and cost accuracy.
        dense_scales: Per-dense-feature multiplier applied before scoring. 31
            float32s, and what makes the L1 penalty comparable across features
            whose natural magnitudes differ by an order of magnitude.
        buckets: Number of hash buckets the features used.
        feature_version: The feature definition this was fitted under.
        labels: Class order.
    """

    columns: np.ndarray
    classes: np.ndarray
    values: np.ndarray
    scales: np.ndarray
    intercepts: np.ndarray
    dense_scales: np.ndarray
    buckets: int
    feature_version: str
    labels: tuple[str, ...]

    @property
    def stored_coefficients(self) -> int:
        """How many coefficients survive pruning."""
        return int(self.values.size)

    @property
    def size_bytes(self) -> int:
        """Serialised size, as G2 measures it.

        Counts the sparse coefficient store plus the per-class scales and
        intercepts. This is the number the gate is judged on, so it is computed
        from the arrays rather than estimated.
        """
        return (
            self.stored_coefficients * BYTES_PER_COEFFICIENT
            + self.scales.nbytes
            + self.intercepts.nbytes
            + self.dense_scales.nbytes
        )

    def predict_proba(self, features: sparse.csr_matrix) -> np.ndarray:
        """Score rows with the dequantised coefficients.

        Args:
            features: Sparse row matrix with the same width as training.

        Returns:
            Row-normalised class probabilities.

        Deliberately implemented from the stored arrays rather than by keeping
        the scikit-learn estimator, so what is measured is what would ship.
        A difference between this and the estimator is a quantisation error
        worth seeing, and hiding it behind the estimator would mean the
        reported numbers belonged to a model that was never exported.
        """
        features = apply_dense_scaling(features, self.dense_scales)
        weights = np.zeros((features.shape[1], len(self.labels)), dtype=np.float64)
        weights[self.columns, self.classes] = self.values.astype(np.float64)
        weights *= self.scales[np.newaxis, :]
        logits = features @ weights + self.intercepts[np.newaxis, :]
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metadata, without the coefficient arrays."""
        return {
            "kind": "multinomial_logistic_regression_int8",
            "feature_version": self.feature_version,
            "buckets": self.buckets,
            "labels": list(self.labels),
            "stored_coefficients": self.stored_coefficients,
            "size_bytes": self.size_bytes,
            "size_kib": round(self.size_bytes / 1024, 1),
            "bytes_per_coefficient": BYTES_PER_COEFFICIENT,
        }


def fit_dense_scales(features: sparse.csr_matrix) -> np.ndarray:
    """Compute a multiplier per dense feature from the training split.

    Args:
        features: The training split's sparse row matrix.

    Returns:
        One float32 multiplier per entry of
        :data:`neburedaction.features.DENSE_FEATURES`.

    **Scaled but not centred, which is a sparsity decision as much as a
    statistical one.** Subtracting a mean makes every zero non-zero and turns a
    1.7%-dense matrix into a full one: 200,000 rows by 16,415 columns of
    float64 is 26 GB. Dividing by the standard deviation leaves zeros at zero
    while still bringing `value_length_log` (roughly 0 to 7) and `ratio_digit`
    (0 to 1) onto comparable footing, which is what the L1 penalty needs. An
    unscaled L1 penalises a small-magnitude feature far more than a large one
    for the same amount of evidence, so the sparsity pattern would otherwise
    reflect the units the features happen to be measured in.

    A zero-variance feature gets a multiplier of 1.0 rather than an infinity.
    """
    dense_count = len(DENSE_FEATURES)
    block = features[:, :dense_count]
    squares = np.asarray(block.multiply(block).mean(axis=0)).ravel()
    means = np.asarray(block.mean(axis=0)).ravel()
    variances = np.maximum(squares - means**2, 0.0)
    deviations = np.sqrt(variances)
    scales = np.where(deviations > 1e-12, 1.0 / np.maximum(deviations, 1e-12), 1.0)
    return scales.astype(np.float32)


def apply_dense_scaling(features: sparse.csr_matrix, scales: np.ndarray) -> sparse.csr_matrix:
    """Multiply the dense feature block by its stored scales.

    Args:
        features: A sparse row matrix.
        scales: Multipliers from :func:`fit_dense_scales`.

    Returns:
        A new matrix with the dense block scaled and the hashed block
        untouched. The hashed features are signed counts in a comparable range
        already, so scaling them would only blur that.
    """
    dense_count = len(DENSE_FEATURES)
    multipliers = np.ones(features.shape[1], dtype=np.float64)
    multipliers[:dense_count] = scales.astype(np.float64)
    return features @ sparse.diags(multipliers)


def expand_soft_targets(
    features: sparse.csr_matrix, targets: np.ndarray, minimum_weight: float = 1e-6
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Expand soft targets into weighted one-hot rows.

    Args:
        features: Sparse row matrix.
        targets: Soft target matrix, one probability vector per row.
        minimum_weight: Target mass below which a class is skipped.

    Returns:
        ``(expanded_features, classes, weights)``.

    Raises:
        ValueError: If a target row does not sum to one. A row that does not is
            not a distribution, and the equivalence this relies on would no
            longer hold.

    The identity is in the module docstring. The row *indices* are duplicated
    rather than the data, so the expansion costs index memory and not a copy of
    every feature value.
    """
    sums = targets.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        worst = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(f"target row {worst} sums to {sums[worst]}, not 1.0")

    rows: list[int] = []
    classes: list[int] = []
    weights: list[float] = []
    for row in range(targets.shape[0]):
        for index in np.nonzero(targets[row] > minimum_weight)[0]:
            rows.append(row)
            classes.append(int(index))
            weights.append(float(targets[row, index]))

    return (
        features[np.array(rows, dtype=np.int64)],
        np.array(classes, dtype=np.int64),
        np.array(weights, dtype=np.float64),
    )


def quantise(
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    buckets: int,
    dense_scales: np.ndarray,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> QuantisedModel:
    """Prune and quantise a fitted coefficient matrix to int8.

    Args:
        coefficients: Shape ``(classes, features)``, as scikit-learn returns.
        intercepts: One per class.
        buckets: Number of hash buckets.
        dense_scales: Multipliers from :func:`fit_dense_scales`.
        prune_threshold: Smallest absolute coefficient kept.

    Returns:
        The quantised model.

    Scaled **per class**, not globally. One class's coefficients can be an
    order of magnitude larger than another's — `private_key` has unmistakable
    markers and `password` has none — and a single global scale would quantise
    the small class to zeros. Seven extra floats is a trivial price.
    """
    classes, _ = coefficients.shape
    kept_columns: list[int] = []
    kept_classes: list[int] = []
    kept_values: list[int] = []
    scales = np.zeros(classes, dtype=np.float32)

    for class_index in range(classes):
        row = coefficients[class_index]
        significant = np.nonzero(np.abs(row) >= prune_threshold)[0]
        if significant.size == 0:
            scales[class_index] = 1.0
            continue
        largest = float(np.abs(row[significant]).max())
        scale = largest / 127.0 if largest > 0 else 1.0
        scales[class_index] = np.float32(scale)
        quantised = np.clip(np.rint(row[significant] / scale), -127, 127).astype(np.int8)
        # A coefficient that quantises to zero would contribute nothing while
        # still costing three bytes.
        surviving = np.nonzero(quantised != 0)[0]
        kept_columns.extend(significant[surviving].tolist())
        kept_classes.extend([class_index] * surviving.size)
        kept_values.extend(quantised[surviving].tolist())

    return QuantisedModel(
        columns=np.array(kept_columns, dtype=np.int32),
        classes=np.array(kept_classes, dtype=np.int8),
        values=np.array(kept_values, dtype=np.int8),
        scales=scales,
        intercepts=intercepts.astype(np.float32),
        dense_scales=dense_scales.astype(np.float32),
        buckets=buckets,
        feature_version=FEATURE_VERSION,
        labels=tuple(LABELS),
    )


def train(
    dataset: Dataset,
    penalty_strength: float = 0.5,
    max_iterations: int = 200,
    prune_threshold: float = PRUNE_THRESHOLD,
    seed: int = 20261014,
) -> tuple[QuantisedModel, dict[str, Any]]:
    """Fit model A on a featurised split.

    Args:
        dataset: The training split.
        penalty_strength: scikit-learn's ``C``; smaller means more L1 pressure
            and a smaller model.
        max_iterations: Solver iteration cap.
        prune_threshold: Smallest absolute coefficient kept.
        seed: Solver seed, so a fit is reproducible.

    Returns:
        ``(model, diagnostics)``.

    Raises:
        ValueError: If the dataset's soft targets are malformed.
    """
    from sklearn.linear_model import LogisticRegression

    dense_scales = fit_dense_scales(dataset.features)
    scaled = apply_dense_scaling(dataset.features, dense_scales).tocsr()

    started = time.perf_counter()
    expanded, classes, weights = expand_soft_targets(scaled, dataset.targets)
    expansion_seconds = time.perf_counter() - started

    # l1_ratio=1.0 is pure L1. `penalty="l1"` was deprecated in scikit-learn
    # 1.8 and, as its own warning says, is *inconsistent* with the l1_ratio
    # default of 0.0 — so passing it silently fitted an L2 model whose apparent
    # sparsity came from the pruning step rather than from the penalty. That is
    # the kind of defect that leaves every reported number plausible, so the
    # ratio is set explicitly and recorded in the diagnostics.
    estimator = LogisticRegression(
        C=penalty_strength,
        l1_ratio=1.0,
        solver="saga",
        max_iter=max_iterations,
        random_state=seed,
        tol=1e-3,
    )
    fit_started = time.perf_counter()
    estimator.fit(expanded, classes, sample_weight=weights)
    fit_seconds = time.perf_counter() - fit_started

    # scikit-learn orders coefficients by the classes it saw. Reindex to
    # LABELS order so the stored model matches the project's class order rather
    # than whichever classes happened to appear.
    coefficients = np.zeros((len(LABELS), dataset.features.shape[1]), dtype=np.float64)
    intercepts = np.zeros(len(LABELS), dtype=np.float64)
    for position, class_index in enumerate(estimator.classes_):
        coefficients[int(class_index)] = estimator.coef_[position]
        intercepts[int(class_index)] = estimator.intercept_[position]

    model = quantise(coefficients, intercepts, dataset.buckets, dense_scales, prune_threshold)
    nonzero = int((np.abs(coefficients) >= prune_threshold).sum())
    total = coefficients.size

    return model, {
        "penalty": "l1 (l1_ratio=1.0)",
        "C": penalty_strength,
        "solver": "saga",
        "max_iter": max_iterations,
        "iterations_used": int(np.max(estimator.n_iter_)),
        "converged": bool(np.max(estimator.n_iter_) < max_iterations),
        "seed": seed,
        "expanded_rows": int(expanded.shape[0]),
        "original_rows": int(dataset.features.shape[0]),
        "expansion_seconds": round(expansion_seconds, 1),
        "fit_seconds": round(fit_seconds, 1),
        "nonzero_before_quantisation": nonzero,
        "total_coefficients": total,
        "sparsity": round(1.0 - nonzero / total, 6),
        **model.to_dict(),
    }


def save(model: QuantisedModel, diagnostics: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Write the model and its diagnostics.

    Args:
        model: The quantised model.
        diagnostics: The training diagnostics.
        directory: Where to write.

    Returns:
        The paths written.

    The arrays go in a single ``.npz``; the metadata goes beside it as JSON so
    the model's shape, size and feature version can be read without loading
    numpy. NR-0005 defines the shipped ``.nrm`` container, so this is an
    interchange format and not that.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays = directory / "model-a.npz"
    np.savez_compressed(
        arrays,
        columns=model.columns,
        classes=model.classes,
        values=model.values,
        scales=model.scales,
        intercepts=model.intercepts,
        dense_scales=model.dense_scales,
    )
    metadata = directory / "model-a.json"
    metadata.write_text(
        json.dumps({**model.to_dict(), "diagnostics": diagnostics}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"arrays": arrays, "metadata": metadata}


def dense_feature_report(model: QuantisedModel) -> list[dict[str, Any]]:
    """Report the dense features' weights, for interpretability.

    Args:
        model: The quantised model.

    Returns:
        One entry per dense feature with its weight per class.

    Plan §4.3 makes "easiest to explain" part of why A is preferred, and the 31
    dense features are the explainable half: unlike a hash bucket, each has a
    name and a meaning. This is what lets the report say *what the model
    learned* rather than only how well it scored.
    """
    rows: list[dict[str, Any]] = []
    for column, name in enumerate(DENSE_FEATURES):
        selected = model.columns == column
        if not selected.any():
            continue
        weights = {
            LABELS[int(class_index)]: round(float(value) * float(model.scales[int(class_index)]), 5)
            for class_index, value in zip(
                model.classes[selected], model.values[selected], strict=True
            )
        }
        rows.append({"feature": name, "weights": weights})
    return rows


def load(directory: Path) -> QuantisedModel:
    """Load a saved model from disk.

    Args:
        directory: A directory written by :func:`save`.

    Returns:
        The model, ready to score with :meth:`QuantisedModel.predict_proba`.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If the saved feature version is not the one this build of
            the extractor produces. A model read under a different feature
            definition is not slightly wrong, it is reading different columns —
            so this refuses rather than scoring nonsense.
    """
    arrays_path = directory / "model-a.npz"
    metadata_path = directory / "model-a.json"
    for path in (arrays_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"model was fitted under feature version {metadata['feature_version']!r} but this "
            f"build produces {FEATURE_VERSION!r}; the columns would not line up"
        )

    with np.load(arrays_path) as arrays:
        return QuantisedModel(
            columns=arrays["columns"],
            classes=arrays["classes"],
            values=arrays["values"],
            scales=arrays["scales"],
            intercepts=arrays["intercepts"],
            dense_scales=arrays["dense_scales"],
            buckets=int(metadata["buckets"]),
            feature_version=metadata["feature_version"],
            labels=tuple(metadata["labels"]),
        )


def score_units(model: QuantisedModel, units: Sequence[Any]) -> np.ndarray:
    """Score detection units directly, as inference would.

    Args:
        model: A loaded model.
        units: Objects with ``key``, ``value``, ``template`` and ``scope``.

    Returns:
        Class probabilities, one row per unit.

    This is the end-to-end path — features extracted from the unit, scored by
    the stored int8 coefficients — so it exercises what NR-0005 will
    reimplement rather than anything that only exists inside the trainer.
    """
    from training.dataset import featurise_units

    return model.predict_proba(featurise_units(units, model.buckets))
