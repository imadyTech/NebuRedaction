"""Teacher labelling of an arbitrary corpus (NR-0003 §2.2 items 5 and 6).

Usage:
    python -m labeling.teacher.label_corpus --units data/synth/v0/val.jsonl \
        --out data/labels/teacher-val.jsonl --sample 5000

Kept separate from ``python -m labeling.teacher``, which implements §2.2 item
3's specific scope — all own logs, all rule-low-confidence training units, a
stratified tenth of the high-confidence ones — and should not be overloaded
with a general mode that dilutes those semantics.

This exists because §2.2 item 3's scope does not cover everything §2.2 needs.
Item 6 requires temperature recalibration on the synthetic **validation** split,
and item 5 requires control group 4 to be evaluated on the synthetic test set
and on CredData. None of those three corpora are in the labelling scope, and
each needs its own pass.

**The ``--evaluation`` flag is a one-way door.** Without it, units are read
through ``load_for_training``, which refuses third-party data. With it they are
read through ``load_for_evaluation``, which permits CredData — and the output is
marked ``evaluation_only`` so that nothing downstream can mistake it for a label
source. §2.2 item 5 is explicit that control-group results must not flow back
into any label, and plan §5.1 means a label derived from CredData must not exist
at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from datagen.loader import load_for_evaluation, load_for_training
from datagen.unit import DetectionUnit

from labeling.teacher.prompt import DEFAULT_VERSION, load_prompt
from labeling.teacher.run import (
    BATCH_SIZE,
    BLOCK_SIZE,
    MODEL,
    THREADS,
    LayaTeacher,
    RunConfig,
    run_teacher,
)
from labeling.teacher.scope import MEASURED_SAMPLES_PER_SECOND, projected_hours

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


def stratified_sample(units: Sequence[DetectionUnit], size: int, seed: int) -> list[DetectionUnit]:
    """Take a class-stratified sample, in corpus order.

    Args:
        units: The corpus.
        size: Target sample size.
        seed: Fixed seed, so the sample is reproducible and declarable.

    Returns:
        The sample, in corpus order. Each class contributes in proportion to
        its share, computed as an exact integer rather than by independent
        coin flips, so a rare class cannot vanish by luck — the same choice
        NR-0001 made for the split composition and `scope` made for the
        high-confidence sample.

    Unlabelled units are grouped under ``none`` for stratification purposes
        only; CredData units may legitimately have no label.
    """
    if size >= len(units):
        return list(units)
    by_class: dict[str, list[str]] = {}
    for unit in units:
        by_class.setdefault(unit.label or "(unlabelled)", []).append(unit.id)

    rng = random.Random(seed)
    chosen: set[str] = set()
    share = size / len(units)
    for name in sorted(by_class):
        ids = by_class[name]
        count = max(1, round(len(ids) * share))
        chosen.update(rng.sample(sorted(ids), min(count, len(ids))))
    return [unit for unit in units if unit.id in chosen]


def main(argv: Sequence[str] | None = None) -> int:
    """Label a corpus with the teacher.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the corpus is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="class-stratified sample size; the ratio is recorded in the output's sidecar",
    )
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help=(
            "read through load_for_evaluation, permitting third-party data. The output is "
            "marked evaluation_only and must never be used as a label source (§2.2 item 5)"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_VERSION)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0003")
    args = parser.parse_args(argv)

    if args.threads > 10:
        parser.error("§2.2A item 1 caps the thread count at 10")
    if not args.units.is_file():
        print(f"FAILED: {args.units} not found", file=sys.stderr)
        return 2

    loader = load_for_evaluation if args.evaluation else load_for_training
    corpus = list(loader([args.units]))
    units = stratified_sample(corpus, args.sample, args.seed) if args.sample else corpus
    ratio = len(units) / len(corpus) if corpus else 0.0

    prompt = load_prompt(args.prompt)
    digest = hashlib.sha256()
    for unit in units:
        digest.update(unit.id.encode("utf-8"))
        digest.update(b"\n")

    plan = {
        "work_order": "NR-0003",
        "units_path": str(args.units).replace("\\", "/"),
        "corpus_units": len(corpus),
        "sampled_units": len(units),
        "sampling_ratio": round(ratio, 6),
        "seed": args.seed,
        "evaluation_only": bool(args.evaluation),
        "prompt": {"version": prompt.version, "digest": prompt.digest},
        "model": args.model,
        "threads": args.threads,
        "projected_hours": round(projected_hours(len(units), MEASURED_SAMPLES_PER_SECOND), 3),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\ndry run: nothing was labelled and no model was loaded")
        return 0

    stamped = args.runs / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stamped.mkdir(parents=True, exist_ok=True)
    (stamped / f"{args.out.stem}-plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    teacher = LayaTeacher(
        prompt=prompt, model=args.model, threads=args.threads, batch_size=args.batch_size
    )
    print(f"\nloading {args.model} with {args.threads} threads ...")
    load_seconds = teacher.load()
    print(f"loaded in {load_seconds:.1f}s; labelling {len(units)} units")

    summary = run_teacher(
        units,
        teacher,
        prompt,
        args.out,
        config=RunConfig(
            prompt_version=prompt.version,
            prompt_digest=prompt.digest,
            model=args.model,
            scope_digest=digest.hexdigest(),
            threads=args.threads,
            batch_size=args.batch_size,
            block_size=args.block_size,
        ),
        log=stamped / f"{args.out.stem}-blocks.jsonl",
        block_size=args.block_size,
        batch_size=args.batch_size,
        on_block=lambda record: print(
            f"  block {record['block']}: {record['cumulative_labelled']}/{len(units)} "
            f"at {record['samples_per_second']} samples/s",
            flush=True,
        ),
    )
    summary["plan"] = plan
    (stamped / f"{args.out.stem}-run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nlabelled {summary['labelled']} in {summary.get('seconds')}s -> {args.out}")
    if args.evaluation:
        print("evaluation_only: this output must not be used as a label source (§2.2 item 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
