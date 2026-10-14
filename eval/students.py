"""Evaluate the student models and G2's candidate (NR-0004 §2 item 4).

Usage:
    python -m eval.students --model-dir runs/NR-0004/<timestamp>

Runs the ordinary framework — the same metrics, thresholds, slices, confidence
intervals and report format NR-0002 fixed before any model existed — over the
students and the "rules + student" combination, against the locked test sets.
It also reports per-unit inference latency, which §2 item 4 asks for alongside
size.

**G2 is judged here and nowhere else.** The grid search in ``training.train_a``
computes a recall at 0.1% to choose a penalty with, on the validation split;
that number is for selection and the module says so. This is the measurement,
on the test sets, under the comparison rule written into ``eval/README.md``
before a model existed: because the pure rule baseline's false-positive rate is
11.17% rather than 0.1%, both operating points are reported and judgement is on
the 0.1% one.

Plan §6.4 made G2 a recording gate on 2026-10-14 — measured and reported, not
go/no-go — so a result below the bar is written down and the project continues
with the best student.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval import testsets
from eval.baselines import build
from eval.baselines.student import RulesPlusStudent, StudentSystem
from eval.evaluate import TestSet, run_system
from eval.report import build_report, build_run_meta, write_report

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: The baseline G2 compares against, and its measured recall on the synthetic
#: test set from NR-0002's own report.
PURE_RULES_RECALL = 0.734
PURE_RULES_FPR = 0.1117

#: G2's bar: ten points of recall above the pure rule baseline, at 0.1%.
G2_RECALL_BAR = PURE_RULES_RECALL + 0.10

#: G2's size limit in bytes.
G2_SIZE_LIMIT = 100 * 1024


def measure_latency(system: Any, units: Sequence[Any], repeats: int = 3) -> dict[str, Any]:
    """Measure per-unit scoring latency.

    Args:
        system: The system to time.
        units: Units to score.
        repeats: How many passes to time.

    Returns:
        Microseconds per unit, best of the repeats.

    Best-of rather than mean, because the thing being measured is the cost of
    the work and a slow pass measures something else that happened on the
    machine. Reported as a Python-side number: plan §6.4's G3 latency target is
    about the .NET runtime, which NR-0005 builds, and this is only the
    order-of-magnitude check §2 item 4 asks for.
    """
    best = None
    for _ in range(repeats):
        started = time.perf_counter()
        system.score(list(units))
        elapsed = time.perf_counter() - started
        if best is None or elapsed < best:
            best = elapsed
    assert best is not None
    return {
        "units": len(units),
        "seconds_total": round(best, 3),
        "microseconds_per_unit": round(best / len(units) * 1e6, 1),
        "note": (
            "Python reference path, best of 3. G3's p99 target applies to the .NET "
            "runtime (NR-0005); this is the order-of-magnitude check §2 item 4 asks for."
        ),
    }


def g2_verdict(runs: Sequence[Any], report: dict[str, Any]) -> dict[str, Any]:
    """Extract the G2 numbers from a finished report.

    Args:
        runs: The system runs.
        report: The assembled report.

    Returns:
        The combination's recall at 0.1% on the synthetic test set, the bar, and
        whether it was met — with every caveat the comparison carries.

    Deliberately reads the report rather than recomputing, so the number in the
    verdict is the number in the report. Two paths to one quantity is how a
    gate ends up with two answers.
    """
    systems = report.get("systems", {})
    verdict: dict[str, Any] = {
        "bar": {
            "recall_at_0.001": G2_RECALL_BAR,
            "derivation": (
                f"pure rules {PURE_RULES_RECALL} on the synthetic test set, plus 10 points"
            ),
            "size_limit_bytes": G2_SIZE_LIMIT,
            "comparison_rule": (
                "eval/README.md, fixed before any model existed. The pure rule baseline "
                f"measures {PURE_RULES_FPR:.2%}, above 0.1%, so both operating points are "
                "reported and judgement is on 0.1%."
            ),
            "gate_kind": (
                "recording gate since 2026-10-14 (plan §6.4): measured and reported, not go/no-go"
            ),
        },
        "candidates": {},
    }

    for name, system in systems.items():
        results = system.get("results", {})
        synthetic = results.get("synth_test", {})
        points = synthetic.get("operating_points", {})
        point = points.get("fpr_0.001", {})
        slices = point.get("slices", {})
        overall = slices.get("all", {})
        size = system.get("notes", {}).get("size_bytes")
        recall = overall.get("recall")
        verdict["candidates"][name] = {
            "attainable_at_0.001": point.get("threshold") is not None,
            "recall_at_0.001": recall,
            "measured_fpr": overall.get("false_positive_rate"),
            "size_bytes": int(size) if size is not None else None,
            "within_size_limit": (int(size) <= G2_SIZE_LIMIT) if size is not None else None,
            "meets_bar": (
                recall is not None
                and recall >= G2_RECALL_BAR
                and size is not None
                and int(size) <= G2_SIZE_LIMIT
            ),
        }
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the students and write the report.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the model is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration JSON; defaults to calibration.json inside --model-dir",
    )
    parser.add_argument("--variant", default="a", choices=["a", "b", "c"])
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="also run the NR-0002 rule baselines, so the comparison is in one report",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY_ROOT / "runs" / "eval" / "students-v0"
    )
    args = parser.parse_args(argv)

    calibration = args.calibration or (args.model_dir / "calibration.json")
    try:
        student = StudentSystem(
            model_dir=args.model_dir,
            name=f"student-{args.variant}",
            calibration_path=calibration if calibration.is_file() else None,
            variant=args.variant,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    if not calibration.is_file():
        print(
            f"WARNING: {calibration} not found; scoring raw. Thresholds are still chosen "
            "on the selection set, so the operating points remain valid, but a reported "
            "probability will not mean what it says (§2 item 3).",
            file=sys.stderr,
        )

    combination = RulesPlusStudent(student=student, name=f"rules+student-{args.variant}")

    try:
        selection_set, test_sets = testsets.load_all()
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    def trim(test_set: TestSet) -> TestSet:
        kept = testsets.labelled_only(test_set.units)
        if args.limit is not None:
            kept = kept[: args.limit]
        return TestSet(
            name=test_set.name,
            units=kept,
            description=test_set.description
            + ("" if args.limit is None else f" [SUBSET: first {args.limit}]"),
            robustness=test_set.robustness,
        )

    selection_set = trim(selection_set)
    test_sets = [trim(test_set) for test_set in test_sets]
    print(f"selection: {selection_set.name}, {len(selection_set.units)} units")
    for test_set in test_sets:
        print(f"test set:  {test_set.name}, {len(test_set.units)} units")

    systems: list[Any] = [student, combination]
    if args.with_baselines:
        systems += [build("rules"), build("rules-values-only"), build("rules+entropy")]

    started = time.perf_counter()
    runs = []
    for system in systems:
        print(f"evaluating {system.name} ...", flush=True)
        runs.append(run_system(system, selection_set, test_sets, with_transforms=False))
    duration = time.perf_counter() - started

    lock = testsets.read_lock()
    report = build_report(
        runs,
        test_sets,
        selection_set,
        schema_version=testsets.SCHEMA_VERSION,
        testsets_lock={**lock, "drift": testsets.verify_lock()},
    )
    report["g2"] = g2_verdict(runs, report)
    report["latency"] = {
        system.name: measure_latency(system, selection_set.units[:2000]) for system in systems[:2]
    }

    paths = write_report(report, build_run_meta(duration, [s.name for s in systems]), args.out)
    print(f"\n{paths['report_json']}\n{paths['report_md']}  ({duration:.1f}s)")

    print("\n=== G2 ===")
    print(f"bar: recall >= {G2_RECALL_BAR:.3f} at 0.1% FPR, size <= {G2_SIZE_LIMIT / 1024:.0f} KB")
    for name, candidate in report["g2"]["candidates"].items():
        recall = candidate["recall_at_0.001"]
        print(
            f"  {name:22} recall={recall if recall is not None else 'n/a':<10} "
            f"size={candidate['size_bytes']} meets_bar={candidate['meets_bar']}"
        )
    print("\nG2 is a recording gate (plan §6.4): this is measured and reported, not go/no-go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
