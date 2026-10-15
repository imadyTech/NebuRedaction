"""Model B: gradient-boosted trees (NR-0004 §2 item 2, plan §4.3).

Plan §4.3 describes B as trees over "dense features plus a few hashed
features", budgeted at 50-200 KB, as the stronger-non-linearity control against
A. G2 caps the shipped model at 100 KB, so the upper half of that budget is not
available to a model that wants to be the default.

**"A few hashed features" is doing real work in that sentence.** A tree has to
evaluate a split per feature it considers, and 16,384 hash buckets that are
each non-zero in a fraction of a percent of rows give it 16,384 nearly-constant
columns to search: almost every split it finds is a split on one value of one
rare n-gram, which is memorisation with extra steps. So B gets the 31 dense
features plus the ``top_buckets`` most frequently non-zero buckets of the
training split, and nothing else. Selecting them by *frequency* rather than by
correlation with the label is deliberate — frequency is a property of the
inputs alone, so it cannot leak the target the way a correlation filter would.

**Size is controlled by the tree budget, not after the fact.** A fitted
LightGBM model cannot be pruned to a byte target the way a linear model's
coefficients can; the only levers are the number of trees and the leaves per
tree, and both have to be set before fitting. They are therefore part of the
grid rather than a post-processing step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS
from scipy import sparse
from training.dataset import Dataset
from training.model_a import expand_soft_targets

from neburedaction.features import DENSE_FEATURES, FEATURE_VERSION

#: How many hashed buckets B is allowed, on top of the 31 dense features.
DEFAULT_TOP_BUCKETS = 256


def select_buckets(features: sparse.csr_matrix, count: int) -> np.ndarray:
    """Choose the most frequently non-zero hashed buckets.

    Args:
        features: The training split's sparse row matrix.
        count: How many buckets to keep.

    Returns:
        The selected column indices, sorted, as absolute columns of the full
        matrix.

    Selected by how often a bucket is non-zero, which depends only on the
    inputs. Choosing by correlation with the label would make the feature set a
    function of the training targets, and the validation split would then be
    scored with features that had already seen the answer.
    """
    dense_count = len(DENSE_FEATURES)
    hashed = features[:, dense_count:]
    occurrences = np.asarray((hashed != 0).sum(axis=0)).ravel()
    best = np.argsort(-occurrences, kind="stable")[:count]
    return np.sort(best) + dense_count


def project(features: sparse.csr_matrix, columns: np.ndarray) -> np.ndarray:
    """Reduce the feature matrix to the dense block plus selected buckets.

    Args:
        features: A sparse row matrix.
        columns: Hashed columns to keep, from :func:`select_buckets`.

    Returns:
        A dense float32 array. Dense is correct here and not a mistake: after
        projection the width is 31 + a few hundred, so the array is small, and
        LightGBM is faster on dense input at that width.
    """
    kept = np.concatenate([np.arange(len(DENSE_FEATURES)), columns])
    return np.asarray(features[:, kept].todense(), dtype=np.float32)


#: Bytes per internal node in the compact encoding: a uint16 feature index, a
#: float32 threshold and two int16 child slots.
BYTES_PER_INTERNAL_NODE = 10

#: Bytes per leaf: one float32 value.
BYTES_PER_LEAF = 4


def compact_size(booster: object) -> dict[str, int]:
    """Measure the trees in a compact binary encoding.

    Args:
        booster: A fitted LightGBM booster.

    Returns:
        Node counts and the encoded size in bytes.

    **Why not just use LightGBM's own serialisation.** Its text format is about
    eight times larger than the information in the trees: 120 trees over seven
    classes came to 2,929 KB of text against 356 KB encoded. G2 caps the
    shipped model at 100 KB, and judging B on a text format it would never ship
    in would disqualify it for a serialisation artefact rather than for
    anything about the model. NR-0005 defines the shipped `.nrm` container and
    would encode trees compactly.

    So this walks the dumped tree structure and counts what a compact encoding
    costs: ten bytes per internal node (uint16 feature index, float32
    threshold, two int16 child slots) and four per leaf. That is a *measured*
    count of real nodes rather than an estimate from the tree count, because a
    boosted tree is not always full — LightGBM stops splitting where a leaf
    would fall below `min_data_in_leaf`, so the node count has to be counted.

    It is still a number for a format that does not exist yet, and the report
    says so. What it supports is a decision about trees and leaves; NR-0005
    will produce the figure that counts.
    """
    dumped = booster.dump_model()  # type: ignore[attr-defined]
    internal = 0
    leaves = 0

    def walk(node: dict) -> None:
        nonlocal internal, leaves
        if "leaf_value" in node or "leaf_index" in node:
            leaves += 1
            return
        internal += 1
        for side in ("left_child", "right_child"):
            child = node.get(side)
            if isinstance(child, dict):
                walk(child)

    for tree in dumped.get("tree_info", []):
        structure = tree.get("tree_structure")
        if isinstance(structure, dict):
            walk(structure)

    return {
        "internal_nodes": internal,
        "leaf_nodes": leaves,
        "bytes": internal * BYTES_PER_INTERNAL_NODE + leaves * BYTES_PER_LEAF,
    }


@dataclass(frozen=True, slots=True)
class TreeModel:
    """A fitted LightGBM multiclass model.

    Attributes:
        booster_text: The model in LightGBM's own text format, which is what
            would be shipped and therefore what is measured.
        columns: The hashed columns selected, needed to reproduce the
            projection at inference.
        trees: Number of boosting rounds actually used.
        leaves: Leaves per tree.
        compact: Measured node counts and compact encoding size.
        buckets: The hash bucket count the features were built with.
        feature_version: The feature definition fitted under.
        labels: Class order.
    """

    booster_text: str
    columns: np.ndarray
    trees: int
    leaves: int
    compact: dict[str, int]
    buckets: int
    feature_version: str
    labels: tuple[str, ...]

    @property
    def text_size_bytes(self) -> int:
        """Size in LightGBM's own text format, which is not what would ship."""
        return len(self.booster_text.encode("utf-8")) + self.columns.size * 2

    @property
    def size_bytes(self) -> int:
        """Serialised size in a compact encoding, as G2 is judged on.

        The measured node encoding plus two bytes per selected hashed column,
        which the inference side needs to rebuild the same projection. See
        :func:`compact_size` for why the text format is not the number used.
        """
        return self.compact["bytes"] + self.columns.size * 2

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metadata, without the booster text."""
        return {
            "kind": "lightgbm_multiclass",
            "feature_version": self.feature_version,
            "buckets": self.buckets,
            "selected_hashed_columns": int(self.columns.size),
            "trees": self.trees,
            "leaves": self.leaves,
            "labels": list(self.labels),
            "internal_nodes": self.compact["internal_nodes"],
            "leaf_nodes": self.compact["leaf_nodes"],
            "size_bytes": self.size_bytes,
            "size_kib": round(self.size_bytes / 1024, 1),
            "text_size_kib": round(self.text_size_bytes / 1024, 1),
            "size_note": (
                "Compact encoding (10 bytes per internal node, 4 per leaf) plus 2 bytes "
                "per selected column. LightGBM's text format is about 8x larger and is "
                "not what would ship; NR-0005 defines the real container."
            ),
        }


def train(
    dataset: Dataset,
    top_buckets: int = DEFAULT_TOP_BUCKETS,
    trees: int = 120,
    leaves: int = 31,
    learning_rate: float = 0.1,
    min_data_in_leaf: int = 50,
    seed: int = 20261014,
) -> tuple[TreeModel, np.ndarray, dict[str, Any]]:
    """Fit model B.

    Args:
        dataset: The training split.
        top_buckets: How many hashed buckets to admit.
        trees: Boosting rounds.
        leaves: Leaves per tree.
        learning_rate: Shrinkage.
        min_data_in_leaf: Smallest leaf, which also limits memorisation: a leaf
            holding one row is that row's value looked up.
        seed: Seed, so a fit is reproducible.

    Returns:
        ``(model, selected_columns, diagnostics)``.
    """
    import lightgbm

    columns = select_buckets(dataset.features, top_buckets)
    projected = project(dataset.features, columns)

    # The same replication identity model A uses: cross-entropy against a
    # distribution is a weighted sum of cross-entropies against one-hot
    # targets, so LightGBM's integer labels plus sample weights express the
    # soft targets exactly.
    sparse_projected = sparse.csr_matrix(projected)
    expanded, classes, weights = expand_soft_targets(sparse_projected, dataset.targets)

    started = time.perf_counter()
    booster = lightgbm.train(
        {
            "objective": "multiclass",
            "num_class": len(LABELS),
            "num_leaves": leaves,
            "learning_rate": learning_rate,
            "min_data_in_leaf": min_data_in_leaf,
            "verbosity": -1,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
        },
        lightgbm.Dataset(
            np.asarray(expanded.todense(), dtype=np.float32),
            label=classes,
            weight=weights,
        ),
        num_boost_round=trees,
    )
    fit_seconds = time.perf_counter() - started

    model = TreeModel(
        booster_text=booster.model_to_string(),
        columns=columns,
        trees=booster.num_trees() // len(LABELS),
        leaves=leaves,
        compact=compact_size(booster),
        buckets=dataset.buckets,
        feature_version=FEATURE_VERSION,
        labels=tuple(LABELS),
    )
    return (
        model,
        columns,
        {
            "top_buckets": top_buckets,
            "requested_trees": trees,
            "leaves": leaves,
            "learning_rate": learning_rate,
            "min_data_in_leaf": min_data_in_leaf,
            "seed": seed,
            "expanded_rows": int(expanded.shape[0]),
            "original_rows": int(dataset.features.shape[0]),
            "projected_width": int(projected.shape[1]),
            "fit_seconds": round(fit_seconds, 1),
            **model.to_dict(),
        },
    )


def predict_proba(model: TreeModel, features: sparse.csr_matrix) -> np.ndarray:
    """Score rows with a fitted tree model.

    Args:
        model: The fitted model.
        features: A sparse row matrix of the full training width.

    Returns:
        Class probabilities, one row per input row.

    Reconstructs the booster from its serialised text, so what is scored is
    what would ship — the same reason model A scores from its int8 arrays
    rather than from the scikit-learn estimator.
    """
    import lightgbm

    booster = lightgbm.Booster(model_str=model.booster_text)
    projected = project(features, model.columns)
    return np.asarray(booster.predict(projected), dtype=np.float64)


def save(model: TreeModel, diagnostics: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Write the model and its diagnostics.

    Args:
        model: The fitted model.
        diagnostics: Training diagnostics.
        directory: Where to write.

    Returns:
        The paths written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    booster = directory / "model-b.txt"
    booster.write_text(model.booster_text, encoding="utf-8")
    columns = directory / "model-b-columns.npy"
    np.save(columns, model.columns)
    metadata = directory / "model-b.json"
    metadata.write_text(
        json.dumps({**model.to_dict(), "diagnostics": diagnostics}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"booster": booster, "columns": columns, "metadata": metadata}
