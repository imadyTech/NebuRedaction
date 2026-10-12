"""The evaluation itself: thresholds, slices and the assembled result.

The order of operations here is the whole point of the work order, so it is
stated once, plainly:

1. A continuous system's thresholds are chosen on the **selection set**, which
   is the synthetic validation split and nothing else. EXECUTION.md §8 forbids
   choosing thresholds on third-party data, so CredData is never used for this —
   not even indirectly.
2. Those thresholds are then applied, unchanged, to every test set.
3. For each test set, the **measured** false-positive rate is reported next to
   recall. The target rate is what the threshold was chosen for; the measured
   rate is what it actually achieves on different data, and the gap between them
   is a finding, not an inconvenience.
4. A binary system has no threshold to choose, so it is reported at its single
   operating point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from datagen.unit import DetectionUnit

from eval import metrics
from eval.slices import Slice, slices_for
from eval.system import SystemOutput
from eval.transforms import TRANSFORMS, apply_transform

#: False-positive rates the report fixes operating points at (plan §6.2).
TARGET_FALSE_POSITIVE_RATES: tuple[float, ...] = (0.001, 0.01)

#: Bootstrap settings (NR-0002 §2.1 item 3).
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20261013

#: Threshold used for a binary system: anything it flagged at all.
BINARY_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class TestSet:
    """A named set of units to evaluate against.

    Attributes:
        name: Name used in the report.
        units: The units.
        description: One line on what it is and what it can answer.
        robustness: Whether the robustness slices apply to this set. See
            TestSetSpec for why they are synthetic-only.
    """

    name: str
    units: Sequence[DetectionUnit]
    description: str
    robustness: bool = False


@dataclass(slots=True)
class SystemRun:
    """One system's results across every test set.

    Attributes:
        name: System name.
        kind: ``continuous`` or ``binary``.
        version: The system's version.
        notes: Carried through to the report.
        thresholds: Chosen threshold per target false-positive rate; empty for a
            binary system.
        results: Nested results, keyed by test set then slice.
    """

    name: str
    kind: str
    version: str
    notes: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, float | None] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)


def truth_of(units: Sequence[DetectionUnit]) -> list[bool]:
    """Return the main-task ground truth: is each unit a secret.

    Args:
        units: The units.

    Returns:
        One boolean per unit.

    Raises:
        ValueError: If any unit is unlabelled. An unlabelled unit cannot
            contribute to a metric, and silently treating it as a negative would
            inflate every false-positive rate; the caller must exclude it.
    """
    missing = [unit.id for unit in units if unit.label is None]
    if missing:
        raise ValueError(
            f"{len(missing)} unit(s) are unlabelled and cannot be evaluated, first: {missing[0]!r}"
        )
    return [unit.should_redact for unit in units]


def choose_thresholds(
    output: SystemOutput, units: Sequence[DetectionUnit]
) -> dict[str, float | None]:
    """Choose an operating point per target false-positive rate.

    Args:
        output: The system's output on the **selection** set.
        units: The selection set, in the same order.

    Returns:
        A mapping from the target rate (as a string key, so it survives JSON) to
        the chosen threshold, or ``None`` when no threshold achieves that rate.
    """
    scores = output.redaction_scores()
    truth = truth_of(units)
    return {
        f"fpr_{rate}": metrics.threshold_for_false_positive_rate(scores, truth, rate)
        for rate in TARGET_FALSE_POSITIVE_RATES
    }


def _slice_result(
    scores: Sequence[float],
    class_scores: Sequence[dict[str, float]],
    units: Sequence[DetectionUnit],
    indices: Sequence[int],
    threshold: float,
    *,
    with_confidence: bool,
) -> dict[str, Any]:
    """Compute one slice's metrics at one threshold.

    Args:
        scores: Main-task scores for the whole test set.
        class_scores: Per-class scores for the whole test set.
        units: The whole test set.
        indices: Positions belonging to this slice.
        threshold: The operating point.
        with_confidence: Whether to bootstrap intervals. Skipped for the
            robustness slices, where the cost would be 1,000 resamples per
            transform per system for a number nobody reads.

    Returns:
        The slice's metrics.
    """
    sliced_scores = [scores[i] for i in indices]
    sliced_truth = [units[i].should_redact for i in indices]
    sliced_labels = [units[i].label for i in indices]
    sliced_class_scores = [class_scores[i] for i in indices]

    counts = metrics.confusion_at(sliced_scores, sliced_truth, threshold)
    result: dict[str, Any] = {
        "count": len(indices),
        **counts.to_dict(),
        "average_precision": round(metrics.average_precision(sliced_scores, sliced_truth), 6),
        "per_class": metrics.per_class_recall(
            sliced_class_scores, sliced_labels, sliced_scores, threshold
        ),
    }
    if with_confidence:
        result["recall_ci"] = metrics.bootstrap_interval(
            metrics.recall_indicators(sliced_scores, sliced_truth, threshold),
            resamples=BOOTSTRAP_RESAMPLES,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=BOOTSTRAP_SEED,
        )
        result["false_positive_rate_ci"] = metrics.bootstrap_interval(
            metrics.false_positive_indicators(sliced_scores, sliced_truth, threshold),
            resamples=BOOTSTRAP_RESAMPLES,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=BOOTSTRAP_SEED,
        )
    return result


def evaluate_on(
    system: Any,
    test_set: TestSet,
    thresholds: dict[str, float | None],
    *,
    with_transforms: bool = True,
) -> dict[str, Any]:
    """Evaluate one system on one test set, across slices and transforms.

    Args:
        system: The system to evaluate.
        test_set: The test set.
        thresholds: Thresholds chosen on the selection set; ignored for a binary
            system.
        with_transforms: Whether to compute the robustness slices.

    Returns:
        The test set's results: operating points, slices, and robustness.
    """
    units = list(test_set.units)
    output = system.score(units)
    scores = output.redaction_scores()
    class_scores = list(output.scores)
    applicable = slices_for(units)

    operating_points: dict[str, Any] = {}
    if output.kind == "binary":
        points = {"own_operating_point": BINARY_THRESHOLD}
    else:
        points = {name: value for name, value in thresholds.items() if value is not None}
        for name, value in thresholds.items():
            if value is None:
                operating_points[name] = {
                    "threshold": None,
                    "note": (
                        "no threshold on the selection set achieves this false-positive rate; "
                        "the system cannot be operated here"
                    ),
                }

    for point_name, threshold in points.items():
        per_slice: dict[str, Any] = {}
        for name, indices in _slice_indices(applicable, units).items():
            per_slice[name] = _slice_result(
                scores, class_scores, units, indices, threshold, with_confidence=True
            )
        operating_points[point_name] = {"threshold": threshold, "slices": per_slice}

    result: dict[str, Any] = {
        "description": test_set.description,
        "count": len(units),
        "positives": sum(1 for unit in units if unit.should_redact),
        "negatives": sum(1 for unit in units if not unit.should_redact),
        "slice_descriptions": {s.name: s.description for s in applicable},
        "operating_points": operating_points,
    }

    if with_transforms:
        result["robustness"] = _robustness(system, units, points)
    return result


def _slice_indices(
    applicable: Sequence[Slice], units: Sequence[DetectionUnit]
) -> dict[str, list[int]]:
    """Map each slice to the positions it contains.

    Args:
        applicable: The slices to compute.
        units: The test set.

    Returns:
        A mapping from slice name to indices, omitting empty slices.
    """
    result: dict[str, list[int]] = {}
    for definition in applicable:
        indices = [i for i, unit in enumerate(units) if definition.predicate(unit)]
        if indices:
            result[definition.name] = indices
    return result


def _robustness(
    system: Any, units: Sequence[DetectionUnit], points: dict[str, float]
) -> dict[str, Any]:
    """Compute the robustness slices (plan §6.2).

    Args:
        system: The system to evaluate.
        units: The untransformed test set.
        points: Operating points to evaluate at.

    Returns:
        Per-transform results at each operating point, with the recall delta
        against the canonical slice — the delta is what the slice is for, so it
        is computed here rather than left to the reader.
    """
    result: dict[str, Any] = {}
    canonical: dict[str, float] = {}

    for transform in TRANSFORMS:
        transformed = apply_transform(units, transform)
        output = system.score(transformed)
        scores = output.redaction_scores()
        truth = [unit.should_redact for unit in transformed]

        per_point: dict[str, Any] = {}
        for point_name, threshold in points.items():
            counts = metrics.confusion_at(scores, truth, threshold)
            entry = {
                "recall": round(counts.recall, 6),
                "false_positive_rate": round(counts.false_positive_rate, 8),
            }
            if transform.name == "canonical":
                canonical[point_name] = counts.recall
            else:
                entry["recall_delta_vs_canonical"] = round(
                    counts.recall - canonical.get(point_name, 0.0), 6
                )
            per_point[point_name] = entry

        result[transform.name] = {
            "description": transform.description,
            "operating_points": per_point,
        }
    return result


def run_system(
    system: Any,
    selection_set: TestSet,
    test_sets: Sequence[TestSet],
    *,
    with_transforms: bool = True,
) -> SystemRun:
    """Evaluate one system end to end.

    Args:
        system: The system to evaluate.
        selection_set: The set thresholds are chosen on — synthetic validation
            data only.
        test_sets: The sets to report against.
        with_transforms: Whether to compute robustness slices.

    Returns:
        The system's complete results.
    """
    probe = system.score(list(selection_set.units)[:1])
    kind = probe.kind

    thresholds: dict[str, float | None] = {}
    if kind == "continuous":
        selection_output = system.score(list(selection_set.units))
        thresholds = choose_thresholds(selection_output, selection_set.units)

    run = SystemRun(
        name=getattr(system, "name", type(system).__name__),
        kind=kind,
        version=system.version(),
        thresholds=thresholds,
    )

    for test_set in test_sets:
        evaluated = evaluate_on(
            system,
            test_set,
            thresholds,
            with_transforms=with_transforms and test_set.robustness,
        )
        run.results[test_set.name] = evaluated
        if not run.notes:
            # Notes come from the output and are identical across test sets;
            # take them once.
            run.notes = dict(system.score(list(test_set.units)[:1]).notes)

    return run
