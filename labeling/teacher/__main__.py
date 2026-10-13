"""Run teacher labelling (NR-0003 §2.2, §2.2A).

Usage:
    python -m labeling.teacher --dry-run        # the plan and the projection
    python -m labeling.teacher                  # label, resuming if interrupted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from datagen.loader import load_for_training

from labeling.rules import read_labels
from labeling.teacher.prompt import DEFAULT_VERSION, load_prompt
from labeling.teacher.run import (
    BATCH_SIZE,
    BLOCK_SIZE,
    MODEL,
    THREADS,
    LayaTeacher,
    RunConfig,
    completed_ids,
    run_teacher,
)
from labeling.teacher.scope import (
    MEASURED_SAMPLES_PER_SECOND,
    SPECIFIED_HIGH_CONFIDENCE_RATIO,
    plan_scope,
    projected_hours,
    ratios_within_budget,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

OWN_UNITS = REPOSITORY_ROOT / "data" / "own" / "own_v0.jsonl"
SYNTH_TRAIN = REPOSITORY_ROOT / "data" / "synth" / "v0" / "train.jsonl"
RULE_LABELS = REPOSITORY_ROOT / "data" / "labels" / "rules-synth-train.jsonl"
DEFAULT_OUT = REPOSITORY_ROOT / "data" / "labels" / "teacher_v0.jsonl"

#: Sampling seed. Fixed, as §2.2 item 3 requires.
SEED = 20261013

#: Measured on this machine by the §2.2A item 1 and 2 trials. Defined in
#: `scope` so that the plan's own projection and this report cannot disagree.
MEASURED_RATE = MEASURED_SAMPLES_PER_SECOND


def scope_digest(unit_ids: Sequence[str]) -> str:
    """Digest the selected unit ids.

    Args:
        unit_ids: The ids in scope, in order.

    Returns:
        A SHA-256 over the id list. Recorded in the run configuration so that a
        resume against a different scope is refused rather than producing a
        file that is half one job and half another.
    """
    digest = hashlib.sha256()
    for unit_id in unit_ids:
        digest.update(unit_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Plan and run teacher labelling.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when an input is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the projection without loading the model",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", default=DEFAULT_VERSION)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help="torch thread cap; §2.2A item 1 caps this at 10",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--low-confidence-ratio",
        type=float,
        default=1.0,
        help="fraction of rule-low-confidence units to label; 1.0 is §2.2 item 3 as written",
    )
    parser.add_argument(
        "--high-confidence-ratio",
        type=float,
        default=SPECIFIED_HIGH_CONFIDENCE_RATIO,
        help="fraction of rule-high-confidence units to label; 0.10 is §2.2 item 3 as written",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="if set, reduce the ratios to fit this many hours (§2.2 item 3's cap)",
    )
    parser.add_argument(
        "--runs", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0003", help="run log directory"
    )
    args = parser.parse_args(argv)

    if args.threads > 10:
        parser.error("§2.2A item 1 caps the thread count at 10")

    for path in (OWN_UNITS, SYNTH_TRAIN, RULE_LABELS):
        if not path.is_file():
            print(f"FAILED: {path} not found", file=sys.stderr)
            if path == RULE_LABELS:
                print("        run `python -m labeling.rules` first", file=sys.stderr)
            return 2

    prompt = load_prompt(args.prompt)
    own_units = list(load_for_training([OWN_UNITS]))
    synth_units = list(load_for_training([SYNTH_TRAIN]))
    rule_labels = read_labels(RULE_LABELS)

    low_ratio, high_ratio = args.low_confidence_ratio, args.high_confidence_ratio
    if args.budget_hours is not None:
        high_total = sum(1 for label in rule_labels if label.is_high_confidence)
        low_total = len(rule_labels) - high_total
        low_ratio, high_ratio = ratios_within_budget(
            len(own_units),
            low_total,
            high_total,
            budget_hours=args.budget_hours,
            high_confidence_ratio=high_ratio,
            rate=MEASURED_RATE,
        )

    plan = plan_scope(
        synth_units,
        rule_labels,
        own_units,
        low_confidence_ratio=low_ratio,
        high_confidence_ratio=high_ratio,
        seed=args.seed,
    )

    already = completed_ids(args.out)
    remaining = plan.calls - len(already & set(plan.unit_ids))
    report = {
        "work_order": "NR-0003",
        "section": "2.2",
        "prompt": {"version": prompt.version, "digest": prompt.digest},
        "model": args.model,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "scope": plan.to_dict(),
        "measured_rate_samples_per_second": MEASURED_RATE,
        "measured_rate_source": "NR-0003 §2.2A trial: 10 threads, batch 8, length-bucketed",
        "projected_hours_at_measured_rate": round(projected_hours(plan.calls, MEASURED_RATE), 2),
        "already_labelled": len(already),
        "remaining": remaining,
        "projected_hours_remaining": round(projected_hours(remaining, MEASURED_RATE), 2),
        "scope_digest": scope_digest(plan.unit_ids),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("\ndry run: nothing was labelled and no model was loaded")
        return 0

    stamped = args.runs / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stamped.mkdir(parents=True, exist_ok=True)
    (stamped / "teacher-plan.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    by_id = {unit.id: unit for unit in own_units}
    by_id.update({unit.id: unit for unit in synth_units})
    units = [by_id[unit_id] for unit_id in plan.unit_ids]

    teacher = LayaTeacher(
        prompt=prompt, model=args.model, threads=args.threads, batch_size=args.batch_size
    )
    print(f"\nloading {args.model} with {args.threads} threads ...")
    load_seconds = teacher.load()
    print(f"loaded in {load_seconds:.1f}s; labelling {remaining} units")

    def progress(record: dict[str, object]) -> None:
        """Print one line per block, so an overnight run can be checked on."""
        done = record["cumulative_labelled"]
        assert isinstance(done, int)
        rate = record["samples_per_second"]
        left = (remaining - done) / MEASURED_RATE / 3600 if rate else 0.0
        print(
            f"  block {record['block']}: {done}/{remaining} at {rate} samples/s, "
            f"~{left:.1f} h left",
            flush=True,
        )

    summary = run_teacher(
        units,
        teacher,
        prompt,
        args.out,
        config=RunConfig(
            prompt_version=prompt.version,
            prompt_digest=prompt.digest,
            model=args.model,
            scope_digest=report["scope_digest"],  # type: ignore[arg-type]
            threads=args.threads,
            batch_size=args.batch_size,
            block_size=args.block_size,
        ),
        log=stamped / "teacher-blocks.jsonl",
        block_size=args.block_size,
        batch_size=args.batch_size,
        on_block=progress,
    )
    summary["load_seconds"] = round(load_seconds, 2)
    (stamped / "teacher-run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nlabelled {summary['labelled']} in {summary.get('seconds')}s")
    print(f"{args.out}\n{stamped / 'teacher-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
