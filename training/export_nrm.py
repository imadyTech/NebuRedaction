"""Export the selected student as a `.nrm` v1 file (NR-0005 §2.1 item 5).

Usage:
    python -m training.export_nrm --model-dir runs/NR-0004/<ts> --variant b

The format is specified in ``docs/spec/nrm-v1.md``; this is the only thing that
writes it, which is why it is verified against the trainer's own predictions
before it returns. A converter that silently reorders a tree would otherwise
produce a file that loads cleanly and classifies differently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS

from neburedaction.features import DENSE_FEATURES, FEATURE_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

MAGIC = b"NRM1"
FORMAT_VERSION = 1

#: Where the default model is written. `.gitignore` excludes `*.nrm`, and
#: NR-0005 §2.1 records Frank's decision of 2026-10-14 not to commit it.
DEFAULT_OUT = REPOSITORY_ROOT / "models" / "neburedaction-v0.1.0.nrm"


def flatten_tree(structure: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:
    """Flatten one dumped LightGBM tree into internal nodes and leaves.

    Args:
        structure: A ``tree_structure`` from ``Booster.dump_model()``.

    Returns:
        ``(internal_nodes, leaf_values)``. Each internal node is a dict with
        ``feature``, ``threshold``, ``left`` and ``right``, where a negative
        child is ``~leaf_index``.

    Raises:
        ValueError: If a node uses a categorical split. The format has no
            encoding for one, and silently treating it as numerical would
            change the model's decisions.

    Depth-first, assigning internal indices in visit order, which fixes the
    traversal the runtimes perform. Recursion is avoided — a boosted tree is
    shallow, but an explicit stack keeps the index assignment obvious, and the
    index assignment is the part a reader has to trust.
    """
    internals: list[dict[str, Any]] = []
    leaves: list[float] = []

    def visit(node: dict[str, Any]) -> int:
        """Return this node's child reference: index, or ~leaf for a leaf."""
        if "leaf_value" in node and "split_feature" not in node:
            leaves.append(float(node["leaf_value"]))
            return ~(len(leaves) - 1)
        if node.get("decision_type") not in (None, "<=", "no_greater"):
            raise ValueError(
                f"unsupported split decision {node.get('decision_type')!r}; the .nrm "
                "format encodes numerical `value <= threshold` splits only"
            )
        position = len(internals)
        internals.append(
            {
                "feature": int(node["split_feature"]),
                "threshold": float(node["threshold"]),
                "left": 0,
                "right": 0,
            }
        )
        internals[position]["left"] = visit(node["left_child"])
        internals[position]["right"] = visit(node["right_child"])
        return position

    visit(structure)
    return internals, leaves


def build_weight_section(
    dumped: dict[str, Any], columns: np.ndarray, buckets: int, class_count: int
) -> bytes:
    """Serialise the trees into the v1 weight section.

    Args:
        dumped: ``Booster.dump_model()`` output.
        columns: The hashed buckets the model reads, as absolute feature
            columns.
        buckets: The hashing space size.
        class_count: Number of classes.

    Returns:
        The weight section's bytes.

    Raises:
        ValueError: If a tree exceeds what the compact encoding can express.
            The limits are generous for a boosted ensemble — 65,535 nodes per
            tree — and exceeding one means the model is not the kind this
            format was designed for, which is worth saying rather than
            truncating.
    """
    bucket_ids = [int(column) - len(DENSE_FEATURES) for column in columns]
    trees = dumped.get("tree_info", [])

    parts: list[bytes] = [
        struct.pack(
            "<IIIII",
            class_count,
            len(trees),
            len(DENSE_FEATURES),
            len(bucket_ids),
            buckets,
        ),
        struct.pack(f"<{len(bucket_ids)}i", *bucket_ids),
    ]

    for index, tree in enumerate(trees):
        internals, leaves = flatten_tree(tree["tree_structure"])
        if len(internals) > 0xFFFF or len(leaves) > 0xFFFF:
            raise ValueError(f"tree {index} has too many nodes for the v1 encoding")
        for node in internals:
            if not -0x8000 <= node["left"] <= 0x7FFF or not -0x8000 <= node["right"] <= 0x7FFF:
                raise ValueError(f"tree {index} has a child index outside int16")
        # LightGBM interleaves classes: tree i belongs to class i % class_count.
        parts.append(struct.pack("<HHH", index % class_count, len(internals), len(leaves)))
        for node in internals:
            parts.append(
                struct.pack(
                    "<Hfhh",
                    node["feature"],
                    node["threshold"],
                    node["left"],
                    node["right"],
                )
            )
        parts.append(struct.pack(f"<{len(leaves)}f", *leaves))

    return b"".join(parts)


def training_data_digest(paths: Sequence[Path]) -> str:
    """Digest the training inputs, for the metadata.

    Args:
        paths: Files the model was trained on.

    Returns:
        A SHA-256 over each file's own digest, so the order is fixed and a
        missing file is visible rather than silently skipped.
    """
    outer = hashlib.sha256()
    for path in sorted(paths):
        outer.update(path.name.encode("utf-8"))
        if path.is_file():
            inner = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    inner.update(chunk)
            outer.update(inner.digest())
        else:
            outer.update(b"<absent>")
    return outer.hexdigest()


def write_nrm(path: Path, metadata: dict[str, Any], weights: bytes) -> Path:
    """Write a `.nrm` file.

    Args:
        path: Output file.
        metadata: The metadata document.
        weights: The weight section.

    Returns:
        The path written.
    """
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<H", FORMAT_VERSION))
        handle.write(struct.pack("<I", len(encoded)))
        handle.write(encoded)
        handle.write(weights)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Export the model and verify the conversion.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` on a missing input; ``3`` if the exported file
        does not reproduce the trainer's predictions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--variant", default="b", choices=["b"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-units", type=int, default=2000)
    args = parser.parse_args(argv)

    booster_path = args.model_dir / "model-b.txt"
    columns_path = args.model_dir / "model-b-columns.npy"
    metadata_path = args.model_dir / "model-b.json"
    calibration_path = args.model_dir / "calibration.json"
    for path in (booster_path, columns_path, metadata_path, calibration_path):
        if not path.is_file():
            print(f"FAILED: {path} not found", file=sys.stderr)
            return 2

    import lightgbm

    booster = lightgbm.Booster(model_str=booster_path.read_text(encoding="utf-8"))
    columns = np.load(columns_path)
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration["method"] != "platt":
        print(
            f"FAILED: .nrm v1 stores Platt parameters; this model's calibration is "
            f"{calibration['method']!r}",
            file=sys.stderr,
        )
        return 2

    weights = build_weight_section(
        booster.dump_model(), columns, int(saved["buckets"]), len(LABELS)
    )

    metadata = {
        "format_version": FORMAT_VERSION,
        "model_type": "gbdt",
        "feature_version": FEATURE_VERSION,
        "base_model_version": "0.1.0",
        "training_data_sha256": training_data_digest(
            [
                REPOSITORY_ROOT / "data" / "labels" / "train_v0.jsonl",
                REPOSITORY_ROOT / "data" / "labels" / "val_v0.jsonl",
            ]
        ),
        "labels": list(LABELS),
        "calibration": {
            "method": "platt",
            "slope": calibration["parameters"]["slope"],
            "intercept": calibration["parameters"]["intercept"],
        },
        "thresholds": calibration["thresholds"],
        "evaluation": {
            "g2_met": False,
            "g2_bar_recall_at_0.001": 0.834,
            "rules_plus_student_recall_at_0.001": 0.804,
            "student_recall_at_0.001": 0.783,
            "note": (
                "G2 measured and not met (NR-0004 §12.6). Thresholds are promises "
                "about the synthetic validation split; on the synthetic test set the "
                "balanced preset measures several times its target because held-out "
                "format families behave differently. On NebuLog's own 2,692 real log "
                "units the model flagged none."
            ),
        },
        "license": "MIT",
        "experimental": True,
    }

    path = write_nrm(args.out, metadata, weights)
    size = path.stat().st_size
    print(f"{path}  ({size} bytes, {size / 1024:.1f} KB)")

    # Verify the conversion against the trainer's own predictions. This is the
    # only writer of the format, so a transcription error here would be
    # invisible: the file would load and classify, just differently.
    from training.dataset import featurise_units, read_records
    from training.model_b import project

    from neburedaction.nrm import load_nrm

    records = list(read_records(REPOSITORY_ROOT / "data" / "labels" / "val_v0.jsonl"))[
        : args.check_units
    ]
    features = featurise_units(records, int(saved["buckets"]))
    reference = np.asarray(booster.predict(project(features, columns)), dtype=np.float64)

    model = load_nrm(path)
    produced = np.vstack([model.class_probabilities(record) for record in records])
    difference = float(np.abs(reference - produced).max())
    print(f"conversion check on {len(records)} units: max |Δp| = {difference:.3e}")
    if difference >= 1e-6:
        print(
            "FAILED: the exported file does not reproduce the trainer's predictions.",
            file=sys.stderr,
        )
        return 3
    print("conversion verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
