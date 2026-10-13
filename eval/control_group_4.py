"""Evaluate control group 4, the Laya teacher zero-shot (NR-0003 §2.2 item 5).

Usage:
    python -m eval.control_group_4

Uses the ordinary framework — the same metrics, thresholds, slices, confidence
intervals and report format as NR-0002's four systems — so control group 4's
numbers sit beside theirs on the same footing. What it does not use is
``eval.run``'s test-set specs, because those load the full corpora and the
teacher was run on stratified samples of them (§2.2 item 5 permits a sample and
requires the ratio to be declared). Each sample becomes its own ``TestSet``
with the ratio in its description, so no number can be read as covering the
whole set.

**Results here must not flow back into any label.** §2.2 item 5 is explicit,
and plan §5.1 means a label derived from CredData must not exist at all. This
module writes a report and nothing else; nothing in ``labeling/`` reads
``runs/eval/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_evaluation
from datagen.unit import DetectionUnit

from eval.baselines.laya_replay import LayaReplay
from eval.evaluate import TestSet, run_system
from eval.report import build_report, build_run_meta, write_report
from eval.testsets import SCHEMA_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: Each corpus, the precomputed teacher probabilities for it, and whether the
#: robustness transforms apply. Transforms are synthetic-only: a transformed
#: CredData value would be a third-party value we had modified, and the teacher
#: was not run on the transformed variants in any case.
CORPORA: tuple[tuple[str, str, str, bool], ...] = (
    (
        "synth_test",
        "data/synth/v0/test.jsonl",
        "data/labels/teacher-test.jsonl",
        False,
    ),
    (
        "creddata",
        "data/third_party/eval_v0.jsonl",
        "data/labels/teacher-creddata.jsonl",
        False,
    ),
)

#: The selection set. Thresholds are chosen here and nowhere else
#: (EXECUTION.md §8), and it is the same split NR-0002 pinned as role
#: ``selection``.
SELECTION = ("synth_val", "data/synth/v0/val.jsonl", "data/labels/teacher-val.jsonl")


def load_sampled(corpus: Path, labels: Path, name: str, robustness: bool) -> TestSet:
    """Build a test set restricted to the units the teacher actually scored.

    Args:
        corpus: The full corpus.
        labels: Precomputed teacher probabilities for a sample of it.
        name: Test-set name.
        robustness: Whether robustness slices apply.

    Returns:
        A test set of the sampled, labelled units, with the sampling ratio in
        its description.

    Raises:
        FileNotFoundError: If either file is missing.

    Restricting the set rather than scoring the whole corpus is the point. An
    unscored unit cannot be given a score of zero: that counts as a confident
    negative and would improve the measured false-positive rate in proportion
    to how much of the set was unlabelled, which is the most flattering
    possible way to be wrong.
    """
    for path in (corpus, labels):
        if not path.is_file():
            raise FileNotFoundError(path)

    scored: set[str] = set()
    with labels.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scored.add(json.loads(line)["unit_id"])

    units: list[DetectionUnit] = []
    total = 0
    for unit in load_for_evaluation([corpus]):
        total += 1
        if unit.id in scored and unit.label is not None:
            units.append(unit)

    ratio = len(units) / total if total else 0.0
    return TestSet(
        name=name,
        units=units,
        description=(
            f"Control group 4 on a stratified {ratio:.1%} sample of {corpus.name} "
            f"({len(units)} of {total} labelled units). §2.2 item 5 permits a sample; "
            "the ratio is declared here because no metric from this set describes the "
            "whole corpus."
        ),
        robustness=robustness,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate control group 4 and write the report.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when a precomputed label file is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPOSITORY_ROOT / "runs" / "eval" / "control-group-4"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "recalibration temperature. The default of 1.0 is deliberate: "
            "recalibration is part of using the teacher as a label source, not part "
            "of what the zero-shot control group is"
        ),
    )
    args = parser.parse_args(argv)

    name, corpus, labels = SELECTION
    try:
        selection_set = load_sampled(
            REPOSITORY_ROOT / corpus, REPOSITORY_ROOT / labels, name, False
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {exc} not found", file=sys.stderr)
        print(
            "        run: python -m labeling.teacher.label_corpus --units "
            f"{corpus} --out {labels} --sample 5000",
            file=sys.stderr,
        )
        return 2

    test_sets: list[TestSet] = []
    for set_name, corpus_path, labels_path, robustness in CORPORA:
        try:
            test_sets.append(
                load_sampled(
                    REPOSITORY_ROOT / corpus_path,
                    REPOSITORY_ROOT / labels_path,
                    set_name,
                    robustness,
                )
            )
        except FileNotFoundError as exc:
            print(f"skipping {set_name}: {exc} not found", file=sys.stderr)

    if not test_sets:
        print("FAILED: no test set has precomputed teacher probabilities", file=sys.stderr)
        return 2

    print(f"selection: {selection_set.name}, {len(selection_set.units)} units")
    for test_set in test_sets:
        print(f"test set:  {test_set.name}, {len(test_set.units)} units")

    system = LayaReplay(labels_path=REPOSITORY_ROOT / SELECTION[2], temperature=args.temperature)
    # One adapter cannot hold every corpus's probabilities, so the scores are
    # merged into a single lookup: the adapter is a file reader, and which file
    # a unit came from does not change what its probability is.
    for _, _, labels_path, _ in CORPORA:
        path = REPOSITORY_ROOT / labels_path
        if path.is_file():
            extra = LayaReplay(labels_path=path, temperature=args.temperature)
            system._probabilities.update(extra._probabilities)

    started = time.perf_counter()
    run = run_system(system, selection_set, test_sets, with_transforms=False)
    duration = time.perf_counter() - started

    report = build_report(
        [run],
        test_sets,
        selection_set,
        schema_version=SCHEMA_VERSION,
        testsets_lock={
            "note": (
                "Control group 4 runs on stratified samples, not the locked full test "
                "sets, so eval/testsets.lock does not describe these inputs. Each set's "
                "description carries its sampling ratio."
            ),
            "testsets": [],
            "drift": [],
        },
    )
    paths = write_report(report, build_run_meta(duration, ["laya"]), args.out)
    print(f"\n{paths['report_json']}\n{paths['report_md']}  ({duration:.1f}s)")
    print(
        "\nThese results must not flow back into any label (§2.2 item 5); "
        "CredData labels must not exist at all (plan §5.1)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
