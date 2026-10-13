"""Run the rule labeller over a corpus (NR-0003 §2.1).

Usage:
    python -m labeling.rules                      # own + synthetic train and val
    python -m labeling.rules --data data/own/own_v0.jsonl --out data/labels/rules-own.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training

from labeling.rules import VERSION, label_units, statistics, write_labels

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

#: The corpora the labeller runs over by default, and where each one's labels go.
#: `data/third_party/` is absent and cannot be added by accident: the units are
#: read through `load_for_training`, which refuses third-party data by path and
#: again per record.
DEFAULT_CORPORA: tuple[tuple[str, str], ...] = (
    ("data/own/own_v0.jsonl", "data/labels/rules-own.jsonl"),
    ("data/synth/v0/train.jsonl", "data/labels/rules-synth-train.jsonl"),
    ("data/synth/v0/val.jsonl", "data/labels/rules-synth-val.jsonl"),
)


def run(data: Path, out: Path) -> dict[str, object]:
    """Label one corpus and write the labels.

    Args:
        data: JSON Lines file of detection units.
        out: Where to write the labels.

    Returns:
        The statistics for this corpus, with the paths it used.

    Raises:
        datagen.loader.ThirdPartyDataError: If the corpus is third-party data.
            Labels derived from CredData must not exist at all (plan §5.1), so
            this is a refusal rather than a filter.
    """
    labels = list(label_units(load_for_training([data])))
    write_labels(labels, out)
    return {
        "data": str(data.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "labels": str(out.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        **statistics(labels),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Label the corpora and write a summary.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when a corpus is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None, help="one corpus to label")
    parser.add_argument("--out", type=Path, default=None, help="where to write its labels")
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "labels" / "rules-summary.json",
        help="where to write the statistics",
    )
    args = parser.parse_args(argv)

    if (args.data is None) != (args.out is None):
        parser.error("--data and --out go together")

    if args.data is not None:
        pairs = [(args.data, args.out)]
    else:
        pairs = [(REPOSITORY_ROOT / data, REPOSITORY_ROOT / out) for data, out in DEFAULT_CORPORA]

    results = []
    for data, out in pairs:
        if not data.is_file():
            print(f"FAILED: {data} not found")
            return 2
        print(f"labelling {data.name} ...")
        result = run(data, out)
        results.append(result)
        confidence = result["by_confidence"]
        assert isinstance(confidence, dict)
        print(
            f"  {result['units']} units: "
            + ", ".join(f"{tier}={count}" for tier, count in confidence.items())
        )

    summary = {"labeller_version": VERSION, "corpora": results}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
