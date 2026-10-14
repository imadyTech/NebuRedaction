"""Export the selected student for NR-0005 (NR-0004 §11.2, §11.3).

Usage:
    python -m training.export_for_runtime --model-dir runs/NR-0004/<ts> --variant b

Collects everything the .NET and Python runtimes need into one directory: the
model, its calibration, the two threshold presets, the feature definition's
version and the class order. NR-0005 defines the shipped ``.nrm`` container, so
this is a handover rather than that format.

**The point of a single export is that nothing about the decision is implicit.**
A runtime that has to be told the bucket count, the class order and the
calibration separately is a runtime that can be told three consistent things
and one stale one. Everything here travels together and carries the feature
version it was fitted under, which the loader checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datagen.unit import LABELS

from neburedaction.features import DEFAULT_BUCKETS, DENSE_FEATURES, FEATURE_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: Files each variant needs carried over.
ARTEFACTS: dict[str, tuple[str, ...]] = {
    "a": ("model-a.npz", "model-a.json"),
    "b": ("model-b.txt", "model-b-columns.npy", "model-b.json"),
    "c": ("model-c.npz", "model-c.json"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Export the chosen model.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` when something the runtime needs is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=["a", "b", "c"])
    parser.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0004" / "export")
    parser.add_argument("--note", default="", help="why this model was chosen")
    args = parser.parse_args(argv)

    calibration_path = args.model_dir / "calibration.json"
    needed = [args.model_dir / name for name in ARTEFACTS[args.variant]] + [calibration_path]
    missing = [path for path in needed if not path.is_file()]
    if missing:
        for path in missing:
            print(f"FAILED: {path} not found", file=sys.stderr)
        if calibration_path in missing:
            print(
                "        fit it with `python -m training.fit_student_calibration "
                f"--model-dir {args.model_dir} --variant {args.variant}`",
                file=sys.stderr,
            )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for path in needed:
        shutil.copy2(path, args.out / path.name)

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(
        (args.model_dir / f"model-{args.variant}.json").read_text(encoding="utf-8")
    )

    manifest: dict[str, Any] = {
        "work_order": "NR-0004",
        "section": "11.2",
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "variant": args.variant,
        "chosen_because": args.note,
        "model": {
            key: model_metadata.get(key)
            for key in ("kind", "size_bytes", "size_kib", "buckets", "trees", "leaves")
            if key in model_metadata
        },
        "calibration": {
            "method": calibration["method"],
            "parameters": calibration["parameters"],
            "thresholds": calibration["thresholds"],
            "preset_meaning": {
                "conservative": "target false-positive rate 0.01%; redacts less",
                "balanced": "target false-positive rate 0.1%; plan §4.5's default",
            },
        },
        "features": {
            "version": FEATURE_VERSION,
            "buckets": model_metadata.get("buckets", DEFAULT_BUCKETS),
            "dense_feature_order": list(DENSE_FEATURES),
            "normative_implementation": "python/src/neburedaction/features.py",
            "test_vectors": "tests/fixtures/feature-vectors-v1.json",
        },
        "labels": list(LABELS),
        "caveats": [
            "G2 was measured and not met: the best candidate reached 0.751-0.804 "
            "recall at a 0.1% false-positive rate against a bar of 0.834. Plan §6.4 "
            "makes G2 a recording gate, so this ships with the shortfall documented.",
            "The threshold presets are promises about the synthetic validation split. "
            "Measured on the synthetic test set the balanced preset's false-positive "
            "rate is several times its target, because held-out format families "
            "behave quite differently from seen ones.",
            "On NebuLog's own 2,692 real log units the model flagged none at either "
            "preset, which is the only in-distribution reading on real data.",
        ],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"exported variant {args.variant} to {args.out}")
    for path in sorted(args.out.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
