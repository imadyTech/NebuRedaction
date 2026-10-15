"""Turning merged labels into training matrices (NR-0004 §2 item 2).

Reads NR-0003's ``train_v0.jsonl`` and ``val_v0.jsonl``, extracts features with
the normative extractor, and builds the sparse matrices models A, B and C are
fitted on.

**Soft labels are distilled on positives only (Frank's decision, 2026-10-14).**
§2 item 2 originally said "hard labels + teacher soft labels jointly", and
NR-0003 measured why that has to be narrowed: the teacher agrees with the
generator on 0.807 of positives but only 0.342 of negatives, and its negative
errors are not diffuse — ``private_key`` was predicted 33,708 times against 813
true instances, with 25,096 of those on true negatives, spread near-uniformly
across uuids, Windows paths, SSH *public* keys, correlation slugs, sha256
digests, e-mail addresses and IPv4 addresses. Distilling the full distribution
would teach the student that all of those are private keys. So a negative
contributes its construction-guaranteed hard label and nothing else.

**The target is a mixture, not a replacement.** On a positive the training
target is ``(1 - alpha)`` on the hard label plus ``alpha`` times the teacher's
distribution. At ``alpha = 0`` this is ordinary supervised training, which is
the control the report compares distillation against — if distillation does not
beat it, the teacher added nothing and that is a finding rather than a failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from neburedaction.features import (
    DEFAULT_BUCKETS,
    DENSE_FEATURES,
    FEATURE_VERSION,
    apply_mask,
    dense_features,
    hashed_features,
    rare_bucket_mask,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: The seven classes, in the order a model's outputs are in. Taken from
#: datagen so there is one definition, not two that can drift.
from datagen.unit import LABELS  # noqa: E402

TRAIN_LABELS = REPOSITORY_ROOT / "data" / "labels" / "train_v0.jsonl"
VAL_LABELS = REPOSITORY_ROOT / "data" / "labels" / "val_v0.jsonl"

#: Smallest number of training rows a hashed bucket must appear in
#: (plan §7.5 item 4). Two means "seen more than once"; a bucket in exactly one
#: row is a lookup for that row's value.
MIN_BUCKET_OCCURRENCES = 2

#: How much of a positive's target comes from the teacher. 0.0 is the
#: supervised control.
DEFAULT_SOFT_LABEL_WEIGHT = 0.3


@dataclass(frozen=True, slots=True)
class Record:
    """One merged label as the trainer needs it.

    Attributes:
        unit_id: The unit's id, kept so a prediction can be traced back.
        key: Attribute name.
        value: Attribute value.
        template: Message template, if any.
        scope: Scope name, if any.
        label: The hard label.
        teacher: The teacher's distribution, or None when it was never asked.
        source: ``synth`` or ``own``.
        label_source: Provenance from NR-0003's merge.
        family: Format family, for slicing.
    """

    unit_id: str
    key: str
    value: str
    template: str | None
    scope: str | None
    label: str
    teacher: dict[str, float] | None
    source: str
    label_source: str
    family: str | None

    @property
    def is_positive(self) -> bool:
        """Whether this unit should be redacted."""
        return self.label != "none"


def read_records(path: Path) -> Iterator[Record]:
    """Read merged labels.

    Args:
        path: ``train_v0.jsonl`` or ``val_v0.jsonl``.

    Yields:
        One record per line.

    Raises:
        ValueError: If a record has no ``meta.label_source``. NR-0003 writes it
            on every record precisely so the trainer can tell a
            construction-guaranteed label from a model's guess, and training on
            the two as though they were the same is the difference between
            distillation and self-training.
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            if "label_source" not in meta:
                raise ValueError(f"{path}:{number}: record has no meta.label_source")
            yield Record(
                unit_id=row["id"],
                key=row["key"],
                value=row["value"],
                template=row.get("template"),
                scope=row.get("scope"),
                label=row["label"],
                teacher=row.get("teacher"),
                source=row["source"],
                label_source=meta["label_source"],
                family=row.get("family"),
            )


def soft_target(record: Record, weight: float = DEFAULT_SOFT_LABEL_WEIGHT) -> np.ndarray:
    """Build one row's training target.

    Args:
        record: The record.
        weight: How much of a positive's target comes from the teacher, in
            [0, 1]. 0.0 gives pure hard labels.

    Returns:
        A probability vector over :data:`LABELS`.

    Raises:
        ValueError: If the weight is outside [0, 1], or the hard label is not
            one of the seven classes.

    A negative's target is one-hot on ``none`` whatever the teacher said, for
    the reason in the module docstring. A positive with no teacher opinion —
    about 45,000 of the 200,000 training rows, because the teacher's scope
    sampled the rule-high-confidence pool — also gets a one-hot target rather
    than being dropped: its hard label is still guaranteed by construction.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"soft-label weight must be in [0, 1], got {weight}")
    if record.label not in LABELS:
        raise ValueError(f"{record.unit_id}: unknown label {record.label!r}")

    target = np.zeros(len(LABELS), dtype=np.float64)
    target[LABELS.index(record.label)] = 1.0
    if weight == 0.0 or not record.is_positive or record.teacher is None:
        return target

    teacher = np.array([float(record.teacher.get(name, 0.0)) for name in LABELS])
    total = teacher.sum()
    if total <= 0:
        return target
    return (1.0 - weight) * target + weight * (teacher / total)


@dataclass
class Dataset:
    """A featurised split.

    Attributes:
        features: Sparse row matrix, dense features then hashed buckets.
        targets: Soft target matrix, one row per unit.
        hard: Hard class index per unit.
        records: The underlying records, aligned with the rows.
        buckets: Number of hash buckets.
        mask: Buckets suppressed as too rare.
        soft_label_weight: The weight used.
    """

    features: sparse.csr_matrix
    targets: np.ndarray
    hard: np.ndarray
    records: list[Record]
    buckets: int
    mask: frozenset[int]
    soft_label_weight: float

    @property
    def positives(self) -> int:
        """How many rows should be redacted."""
        return int((self.hard != LABELS.index("none")).sum())

    def statistics(self) -> dict[str, Any]:
        """Summarise the split for the report."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.label] = counts.get(record.label, 0) + 1
        with_teacher = sum(1 for record in self.records if record.teacher is not None)
        distilled = sum(
            1 for record in self.records if record.teacher is not None and record.is_positive
        )
        return {
            "rows": self.features.shape[0],
            "columns": self.features.shape[1],
            "nonzeros": int(self.features.nnz),
            "density": round(
                self.features.nnz / (self.features.shape[0] * self.features.shape[1]), 8
            ),
            "positives": self.positives,
            "by_label": dict(sorted(counts.items())),
            "with_teacher_opinion": with_teacher,
            "distilled_rows": distilled,
            "soft_label_weight": self.soft_label_weight,
            "suppressed_buckets": len(self.mask),
            "feature_version": FEATURE_VERSION,
        }


def _rows(
    records: Sequence[Record], buckets: int
) -> tuple[list[tuple[float, ...]], list[dict[int, float]]]:
    """Extract both feature families for every record."""
    dense = [
        dense_features(record.key, record.value, record.template, record.scope)
        for record in records
    ]
    hashed = [
        hashed_features(record.key, record.value, record.template, record.scope, buckets)
        for record in records
    ]
    return dense, hashed


def build(
    records: Sequence[Record],
    buckets: int = DEFAULT_BUCKETS,
    mask: frozenset[int] | None = None,
    soft_label_weight: float = DEFAULT_SOFT_LABEL_WEIGHT,
    min_bucket_occurrences: int = MIN_BUCKET_OCCURRENCES,
) -> Dataset:
    """Featurise records into a sparse matrix.

    Args:
        records: The records to featurise.
        buckets: Number of hash buckets.
        mask: Buckets to suppress. ``None`` computes one from these records,
            which is correct for the training split; the validation split must
            be passed the **training** split's mask.
        soft_label_weight: How much of a positive's target comes from the
            teacher.
        min_bucket_occurrences: Minimum rows a bucket must appear in.

    Returns:
        The featurised split.

    The mask is computed on training data only and then applied unchanged,
    for the same reason thresholds are chosen on validation only: computing it
    from the validation split would let the validation set influence the
    model's feature space, which is a leak even though it looks like
    housekeeping.
    """
    dense, hashed = _rows(records, buckets)

    if mask is None:
        suppressed = frozenset(rare_bucket_mask(hashed, min_bucket_occurrences, buckets))
    else:
        suppressed = mask
    hashed = [apply_mask(row, suppressed) for row in hashed]

    width = len(DENSE_FEATURES) + buckets
    indices: list[int] = []
    pointers: list[int] = [0]
    data: list[float] = []
    for dense_row, hashed_row in zip(dense, hashed, strict=True):
        for column, number in enumerate(dense_row):
            if number != 0.0:
                indices.append(column)
                data.append(number)
        for bucket, weight in sorted(hashed_row.items()):
            indices.append(len(DENSE_FEATURES) + bucket)
            data.append(weight)
        pointers.append(len(indices))

    features = sparse.csr_matrix(
        (np.array(data, dtype=np.float64), np.array(indices), np.array(pointers)),
        shape=(len(records), width),
    )
    targets = np.vstack([soft_target(record, soft_label_weight) for record in records])
    hard = np.array([LABELS.index(record.label) for record in records], dtype=np.int64)

    return Dataset(
        features=features,
        targets=targets,
        hard=hard,
        records=list(records),
        buckets=buckets,
        mask=suppressed,
        soft_label_weight=soft_label_weight,
    )


def load_splits(
    buckets: int = DEFAULT_BUCKETS,
    soft_label_weight: float = DEFAULT_SOFT_LABEL_WEIGHT,
    train_path: Path = TRAIN_LABELS,
    val_path: Path = VAL_LABELS,
    limit: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and featurise both splits.

    Args:
        buckets: Number of hash buckets.
        soft_label_weight: How much of a positive's target comes from the
            teacher.
        train_path: Merged training labels.
        val_path: Merged validation labels.
        limit: Keep only the first N records of each split, for a quick run.

    Returns:
        ``(train, validation)``, the validation split masked with the training
        split's mask.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run `python -m labeling.merge` (NR-0003 §2.5)"
            )
    train_records = list(read_records(train_path))
    val_records = list(read_records(val_path))
    if limit is not None:
        train_records = train_records[:limit]
        val_records = val_records[:limit]

    train = build(train_records, buckets, None, soft_label_weight)
    validation = build(val_records, buckets, train.mask, soft_label_weight)
    return train, validation


def featurise_units(units: Sequence[Any], buckets: int) -> sparse.csr_matrix:
    """Featurise anything unit-shaped into the training matrix layout.

    Args:
        units: Objects with ``key``, ``value``, ``template`` and ``scope``.
        buckets: Number of hash buckets, which must match the model's.

    Returns:
        A sparse row matrix of width ``len(DENSE_FEATURES) + buckets``.

    The inference-side counterpart to :func:`build`, without the targets or the
    mask. It exists so that model A's scoring, model B's scoring and the
    evaluation adapters all construct features the same way: three
    near-identical featurisers would be three chances for one of them to drift
    from the normative extractor.

    No mask is applied. The rare-bucket mask is a training-time device — it
    zeroes buckets before fitting so the learned weight is zero — which is
    exactly what keeps inference independent of which corpus a model was
    trained on.
    """
    indices: list[int] = []
    pointers: list[int] = [0]
    data: list[float] = []
    for unit in units:
        template = getattr(unit, "template", None)
        scope = getattr(unit, "scope", None)
        for column, number in enumerate(dense_features(unit.key, unit.value, template, scope)):
            if number != 0.0:
                indices.append(column)
                data.append(number)
        hashed = hashed_features(unit.key, unit.value, template, scope, buckets)
        for bucket, weight in sorted(hashed.items()):
            indices.append(len(DENSE_FEATURES) + bucket)
            data.append(weight)
        pointers.append(len(indices))

    return sparse.csr_matrix(
        (np.array(data, dtype=np.float64), np.array(indices), np.array(pointers)),
        shape=(len(units), len(DENSE_FEATURES) + buckets),
    )
