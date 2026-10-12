"""Evaluation runner (NR-0002 §2.1 item 4).

Usage:
    python -m eval.run --systems rules rules+entropy credsweeper --out runs/eval/<ts>/
    python -m eval.run --write-lock          # refresh eval/testsets.lock
    python -m eval.run --list                # show registered systems
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval import testsets
from eval.baselines import DEFAULT_SYSTEMS, available_systems, build
from eval.evaluate import TestSet, run_system
from eval.report import build_report, build_run_meta, write_report

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _sample(test_set: TestSet, limit: int | None) -> TestSet:
    """Take a deterministic prefix of a test set, for quick runs.

    Args:
        test_set: The set to shorten.
        limit: Maximum units, or ``None`` for all of them.

    Returns:
        The possibly shortened set. A prefix rather than a random sample, so a
        ``--limit`` run is reproducible and comparable between systems — but the
        description records that it is a subset, because a prefix is not a
        representative sample and no number from it belongs in a report.
    """
    if limit is None or limit >= len(test_set.units):
        return test_set
    return TestSet(
        name=test_set.name,
        units=list(test_set.units)[:limit],
        description=f"{test_set.description} [SUBSET: first {limit} units, not representative]",
        robustness=test_set.robustness,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and write the report.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the data or the lock is unusable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help=f"systems to evaluate (registered: {', '.join(available_systems())})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory; defaults to runs/eval/<timestamp>/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N units of each set, for a quick check",
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="skip the robustness slices",
    )
    parser.add_argument("--write-lock", action="store_true", help="refresh the test-set lock")
    parser.add_argument("--list", action="store_true", help="list registered systems and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in available_systems():
            print(name)
        return 0

    if args.write_lock:
        lock = testsets.write_lock()
        print(f"wrote {testsets.LOCK_PATH}")
        for entry in lock["testsets"]:
            digest = (entry["sha256"] or "missing")[:16]
            print(f"  {entry['name']:<12} {entry['role']:<10} {entry['units']} units  {digest}")
        return 0

    try:
        selection_set, test_sets = testsets.load_all()
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    # Unlabelled units cannot contribute to a metric. Dropping them here, loudly,
    # beats letting them count as negatives and quietly inflate every
    # false-positive rate.
    def _labelled(test_set: TestSet) -> TestSet:
        kept = testsets.labelled_only(test_set.units)
        dropped = len(test_set.units) - len(kept)
        description = test_set.description
        if dropped:
            description += f" [{dropped} unlabelled unit(s) excluded]"
            print(f"note: {test_set.name}: excluded {dropped} unlabelled unit(s)")
        return TestSet(
            name=test_set.name,
            units=kept,
            description=description,
            robustness=test_set.robustness,
        )

    selection_set = _sample(_labelled(selection_set), args.limit)
    test_sets = [_sample(_labelled(test_set), args.limit) for test_set in test_sets]

    lock = testsets.read_lock()
    drift = testsets.verify_lock()
    if drift:
        print("WARNING: evaluation data differs from eval/testsets.lock:", file=sys.stderr)
        for problem in drift:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "  EXECUTION.md §8 requires declaring such a change in the report; "
            "it is recorded there under testsets_lock.drift.",
            file=sys.stderr,
        )
    lock = {**lock, "drift": drift}

    out_dir = args.out or (
        REPOSITORY_ROOT / "runs" / "eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )

    started = time.perf_counter()
    runs = []
    for name in args.systems:
        try:
            system = build(name)
        except KeyError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 2
        print(f"evaluating {name} ...")
        runs.append(
            run_system(
                system,
                selection_set,
                test_sets,
                with_transforms=not args.no_transforms,
            )
        )
    duration = time.perf_counter() - started

    report = build_report(
        runs,
        test_sets,
        selection_set,
        schema_version=testsets.SCHEMA_VERSION,
        testsets_lock=lock,
    )
    paths = write_report(report, build_run_meta(duration, args.systems), out_dir)

    print(f"\n{paths['report_json']}")
    print(f"{paths['report_md']}")
    print(f"{paths['run_meta']}  ({duration:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
