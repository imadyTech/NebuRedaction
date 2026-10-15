"""Reading a `.nrm` v1 model (NR-0005 §2.1, §2.3).

The format is specified in ``docs/spec/nrm-v1.md``. This module and the .NET
reader are the two implementations plan §6.4's G3 requires to agree to within
1e-6 per sample, so everything here that could differ between them is stated
rather than left to the language: the accumulation width, the summation order,
the comparison direction at a split, and the clamp inside the logit.

Standard library only, like the rest of this package.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neburedaction.features import FEATURE_VERSION, dense_features, hashed_features

MAGIC = b"NRM1"
SUPPORTED_FORMAT_VERSION = 1

#: Clamp applied inside the logit, so a probability of exactly 0 or 1 does not
#: produce an infinity. Stated here because the .NET reader must use the same
#: number: a different clamp is a different score for a saturated unit.
LOGIT_EPSILON = 1e-9


class NrmError(Exception):
    """A `.nrm` file could not be loaded.

    Deliberately not a ``ValueError``: every case this covers is a refusal to
    use a file, not a complaint about an argument, and a caller catching
    ``ValueError`` around a parse should not swallow "this model was fitted
    under a different feature definition".
    """


@dataclass(frozen=True, slots=True)
class Tree:
    """One decision tree.

    Attributes:
        class_index: The class this tree contributes to.
        features: Feature index per internal node.
        thresholds: Split threshold per internal node.
        left: Left child per internal node; negative is ``~leaf``.
        right: Right child per internal node.
        leaves: Leaf values.
    """

    class_index: int
    features: tuple[int, ...]
    thresholds: tuple[float, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    leaves: tuple[float, ...]

    def evaluate(self, values: list[float]) -> float:
        """Walk the tree for one projected feature vector.

        Args:
            values: The projected feature vector.

        Returns:
            The leaf value reached.

        The comparison is ``value <= threshold`` goes left, which is
        LightGBM's convention and the one the exporter preserved. Getting the
        direction wrong would be invisible on most units and wrong on the ones
        that sit exactly on a split.
        """
        if not self.features:
            return self.leaves[0]
        node = 0
        while True:
            child = (
                self.left[node]
                if values[self.features[node]] <= self.thresholds[node]
                else self.right[node]
            )
            if child < 0:
                return self.leaves[~child]
            node = child


@dataclass(frozen=True, slots=True)
class NrmModel:
    """A loaded `.nrm` model.

    Attributes:
        metadata: The file's metadata document.
        labels: Class order.
        trees: The ensemble, in file order.
        dense_feature_count: Entries taken from the dense feature block.
        bucket_ids: The hash buckets the model reads, in projection order.
        bucket_count: The hashing space the buckets index.
        slope: Platt slope.
        intercept: Platt intercept.
        thresholds: Preset name to calibrated-probability threshold.
    """

    metadata: dict[str, Any]
    labels: tuple[str, ...]
    trees: tuple[Tree, ...]
    dense_feature_count: int
    bucket_ids: tuple[int, ...]
    bucket_count: int
    slope: float
    intercept: float
    thresholds: dict[str, float]

    @property
    def none_index(self) -> int:
        """Index of the ``none`` class in :attr:`labels`."""
        return self.labels.index("none")

    def project(self, key: str, value: str, template: str | None, scope: str | None) -> list[float]:
        """Build the projected feature vector for one unit.

        Args:
            key: Attribute name.
            value: Attribute value.
            template: Message template, if any.
            scope: Scope name, if any.

        Returns:
            ``dense_feature_count + len(bucket_ids)`` values, the dense block
            first and then the selected buckets in stored order.
        """
        dense = dense_features(key, value, template, scope)
        hashed = hashed_features(key, value, template, scope, self.bucket_count)
        projected = list(dense[: self.dense_feature_count])
        projected.extend(hashed.get(bucket, 0.0) for bucket in self.bucket_ids)
        return projected

    def class_probabilities(self, unit: Any) -> list[float]:
        """Score one unit, returning a probability per class.

        Args:
            unit: Anything with ``key``, ``value`` and optionally ``template``
                and ``scope``.

        Returns:
            One probability per entry of :attr:`labels`.

        Accumulates in Python floats, which are 64-bit, and sums trees in file
        order. Both are required by the specification: 32-bit accumulation
        loses G3's 1e-6 agreement, and floating-point addition is not
        associative, so a different order is a different answer.
        """
        projected = self.project(
            unit.key,
            unit.value,
            getattr(unit, "template", None),
            getattr(unit, "scope", None),
        )
        scores = [0.0] * len(self.labels)
        for tree in self.trees:
            scores[tree.class_index] += tree.evaluate(projected)

        largest = max(scores)
        exponentiated = [math.exp(score - largest) for score in scores]
        total = sum(exponentiated)
        return [value / total for value in exponentiated]

    def redaction_score(self, probabilities: list[float]) -> float:
        """Calibrate the probability that a unit should be redacted.

        Args:
            probabilities: Per-class probabilities.

        Returns:
            The calibrated redaction probability.
        """
        raw = 1.0 - probabilities[self.none_index]
        clamped = min(max(raw, LOGIT_EPSILON), 1.0 - LOGIT_EPSILON)
        logit = math.log(clamped / (1.0 - clamped))
        return 1.0 / (1.0 + math.exp(-(self.slope * logit + self.intercept)))


def _read(handle: Any, size: int, what: str) -> bytes:
    """Read exactly ``size`` bytes or fail with what was being read."""
    chunk = handle.read(size)
    if len(chunk) != size:
        raise NrmError(f"file truncated while reading {what}")
    return chunk


def load_nrm(path: Path, feature_version: str = FEATURE_VERSION) -> NrmModel:
    """Load a `.nrm` v1 file.

    Args:
        path: The file to read.
        feature_version: The feature definition this build produces.

    Returns:
        The loaded model.

    Raises:
        NrmError: If the magic, the format version, the model type or the
            feature version is not one this reader implements. All four are
            checked before any weight is read, because a file that fails any of
            them would otherwise load and classify plausibly.
    """
    with path.open("rb") as handle:
        if _read(handle, 4, "magic") != MAGIC:
            raise NrmError(f"{path} is not a .nrm file: wrong magic")
        (version,) = struct.unpack("<H", _read(handle, 2, "format version"))
        if version != SUPPORTED_FORMAT_VERSION:
            raise NrmError(
                f"{path} is format version {version}; this reader implements "
                f"{SUPPORTED_FORMAT_VERSION}"
            )
        (length,) = struct.unpack("<I", _read(handle, 4, "metadata length"))
        metadata = json.loads(_read(handle, length, "metadata").decode("utf-8"))

        if metadata.get("model_type") != "gbdt":
            raise NrmError(
                f"{path} has model_type {metadata.get('model_type')!r}; this reader "
                "implements 'gbdt' only (NR-0004 §11.2 selected model B)"
            )
        if metadata.get("feature_version") != feature_version:
            raise NrmError(
                f"{path} was fitted under feature version "
                f"{metadata.get('feature_version')!r} but this build produces "
                f"{feature_version!r}; the columns would not line up"
            )

        header = struct.unpack("<IIIII", _read(handle, 20, "weight header"))
        class_count, tree_count, dense_count, hashed_count, bucket_count = header
        bucket_ids = struct.unpack(
            f"<{hashed_count}i", _read(handle, 4 * hashed_count, "bucket ids")
        )

        trees: list[Tree] = []
        for index in range(tree_count):
            class_index, internal_count, leaf_count = struct.unpack(
                "<HHH", _read(handle, 6, f"tree {index} header")
            )
            features: list[int] = []
            thresholds: list[float] = []
            left: list[int] = []
            right: list[int] = []
            for _ in range(internal_count):
                feature, threshold, left_child, right_child = struct.unpack(
                    "<Hfhh", _read(handle, 10, f"tree {index} node")
                )
                features.append(feature)
                thresholds.append(threshold)
                left.append(left_child)
                right.append(right_child)
            leaves = struct.unpack(
                f"<{leaf_count}f", _read(handle, 4 * leaf_count, f"tree {index} leaves")
            )
            trees.append(
                Tree(
                    class_index=class_index,
                    features=tuple(features),
                    thresholds=tuple(thresholds),
                    left=tuple(left),
                    right=tuple(right),
                    leaves=tuple(leaves),
                )
            )

    labels = tuple(metadata["labels"])
    if len(labels) != class_count:
        raise NrmError(f"{path} declares {len(labels)} labels but {class_count} classes of trees")
    calibration = metadata["calibration"]
    return NrmModel(
        metadata=metadata,
        labels=labels,
        trees=tuple(trees),
        dense_feature_count=dense_count,
        bucket_ids=bucket_ids,
        bucket_count=bucket_count,
        slope=float(calibration["slope"]),
        intercept=float(calibration["intercept"]),
        thresholds={name: float(value) for name, value in metadata["thresholds"].items()},
    )
