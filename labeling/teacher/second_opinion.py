"""CredSweeper's second opinion on the own logs (NR-0003 §2.2 item 2).

Usage:
    python -m labeling.teacher.second_opinion

§2.3 will not auto-adopt an own-log label on the teacher's word alone: two
independent systems must agree. This produces the second system's opinion by
reusing NR-0002's adapter unmodified, so the control group and the second
opinion are the same code and cannot drift apart.

**Only the own logs.** On synthetic data the generator already supplies ground
truth, so a second opinion would cost CPU and change nothing. On CredData a
second opinion must never be produced at all: labels derived from third-party
data must not exist (plan §5.1), which the loader enforces.

The opinion is binary — "a credential" or "not" — because CredSweeper applies
its own ML threshold internally and reports a decision rather than one of our
seven classes. Comparing it with the teacher on redaction rather than class is
therefore not a simplification: it is the only comparison the two systems can
both express.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

OWN_UNITS = REPOSITORY_ROOT / "data" / "own" / "own_v0.jsonl"
DEFAULT_OUT = REPOSITORY_ROOT / "data" / "labels" / "credsweeper-own.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Run CredSweeper over the own logs and write its verdicts.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when the corpus is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=OWN_UNITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.units.is_file():
        print(f"FAILED: {args.units} not found", file=sys.stderr)
        return 2

    from eval.baselines.credsweeper_adapter import CredSweeperBaseline

    units = list(load_for_training([args.units]))
    print(f"running CredSweeper over {len(units)} units ...")
    baseline = CredSweeperBaseline()
    output = baseline.score(units)
    scores = output.redaction_scores()

    verdicts = {unit.id: bool(score >= 1.0) for unit, score in zip(units, scores, strict=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    flagged = sum(verdicts.values())
    print(f"{args.out}")
    print(f"{flagged} of {len(units)} flagged as a credential ({flagged / len(units):.2%})")
    print(f"version: {output.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
