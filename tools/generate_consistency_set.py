"""Export the G3 consistency set the .NET library is checked against.

Usage:
    uv run python tools/generate_consistency_set.py

NR-0005 §2.4 item 1 asks for 1,000 synthetic units, 1,000 own-log units, 200
NR-0002 adversarial negatives and 200 robustness-transformed samples, drawn
under a fixed seed, each carrying the Python reference implementation's
per-class probabilities. Plan §6.4's G3 then requires the .NET port to agree to
within 1e-6 per sample with identical redaction decisions.

**The output is not committed.** §2.4 item 1 allows committing it only if it
passes the synthetic-sample manifest check, and it cannot: the synthetic
generator emits format-correct vendor credentials on purpose, so the 1,000
synthetic units carry vendor-shaped strings and a committed file would be
indistinguishable from a leak to any scanner matching vendor patterns — the
same conclusion NR-0002 reached about fixtures and the reason `.gitignore`
excludes `data/`. The set is therefore generated on demand, deterministically
from the seed, and written under `data/`. CI generates it before running the
.NET tests.

The sample is drawn with `random.Random(SEED)` rather than NumPy so that the
file can be reproduced by anything with a Python interpreter, including a
checkout with no scientific stack installed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "src"))

from datagen.loader import iter_units  # noqa: E402
from datagen.unit import DetectionUnit  # noqa: E402
from eval.transforms import TRANSFORMS, apply_transform  # noqa: E402
from labeling.teacher.adversarial import is_adversarial  # noqa: E402

from neburedaction.detector import (  # noqa: E402
    BALANCED,
    CONSERVATIVE,
    SecretDetector,
    SecretDetectorOptions,
)
from neburedaction.nrm import load_nrm  # noqa: E402

#: Fixed so the set is reproducible. Changing it changes the set, which §2.4
#: item 3 forbids doing in response to a failure.
SEED = 20261015

#: The counts §2.4 item 1 specifies.
SYNTHETIC_SAMPLES = 1_000
OWN_SAMPLES = 1_000
ADVERSARIAL_SAMPLES = 200
ROBUSTNESS_SAMPLES = 200

SYNTHETIC_PATH = REPOSITORY_ROOT / "data" / "synth" / "v0" / "test.jsonl"
OWN_PATH = REPOSITORY_ROOT / "data" / "own" / "own_v0.jsonl"
MODEL_PATH = REPOSITORY_ROOT / "models" / "neburedaction-v0.1.0.nrm"
OUT = REPOSITORY_ROOT / "data" / "consistency" / "consistency-set-v1.json"

#: Schema version of the exported file, so a .NET test reading an older export
#: fails with a clear message instead of a missing field.
SCHEMA_VERSION = "1.0"


def _sample(units: list[DetectionUnit], count: int, seed: int) -> list[DetectionUnit]:
    """Draw up to ``count`` units without replacement, reproducibly.

    Args:
        units: The population, in file order.
        count: How many to draw.
        seed: The seed for this draw.

    Returns:
        The drawn units.

    Each draw gets its own ``Random`` seeded from :data:`SEED` and an offset, so
    that adding a later section cannot shift an earlier one's sample.
    """
    if len(units) <= count:
        return list(units)
    return random.Random(seed).sample(units, count)


def _robustness_units(units: list[DetectionUnit], count: int, seed: int) -> list[DetectionUnit]:
    """Apply the NR-0002 robustness transforms to a sample of units.

    Args:
        units: The population to draw from.
        count: How many transformed units to produce.
        seed: The seed for the draw.

    Returns:
        The transformed units, spread evenly over the non-identity transforms.

    ``canonical`` is excluded: it leaves the value untouched, so including it
    would spend part of the budget re-testing units the synthetic section
    already covers.
    """
    transforms = [transform for transform in TRANSFORMS if transform.name != "canonical"]
    if not transforms:
        raise SystemExit("eval.transforms exposes no non-canonical transform")

    per_transform = max(1, count // len(transforms))
    produced: list[DetectionUnit] = []
    for index, transform in enumerate(transforms):
        drawn = _sample(units, per_transform, seed + index)
        for unit in apply_transform(drawn, transform):
            produced.append(unit)
            if len(produced) == count:
                return produced
    return produced


def _record(
    detector_balanced: SecretDetector,
    detector_conservative: SecretDetector,
    unit: DetectionUnit,
    section: str,
) -> dict[str, Any]:
    """Score one unit under both profiles and record the result.

    Args:
        detector_balanced: A detector on the balanced profile.
        detector_conservative: The same model on the conservative profile.
        unit: The unit.
        section: Which section of the set it belongs to.

    Returns:
        The record a .NET test compares against.

    Both profiles are recorded because they differ only in a threshold, so the
    second costs one comparison and catches a .NET build that reads the wrong
    threshold out of the file — which no amount of probability agreement would
    reveal.
    """
    balanced = detector_balanced.classify(unit.key, unit.value, unit.template, unit.scope)
    conservative = detector_conservative.classify(unit.key, unit.value, unit.template, unit.scope)
    return {
        "id": unit.id,
        "section": section,
        "key": unit.key,
        "value": unit.value,
        "template": unit.template,
        "scope": unit.scope,
        "probabilities": balanced.probabilities,
        "category": balanced.category,
        "score": balanced.score,
        "should_redact_balanced": balanced.should_redact,
        "should_redact_conservative": conservative.should_redact,
    }


def main() -> int:
    """Write the consistency set.

    Returns:
        0 on success, 2 when a required input is missing.
    """
    for path in (SYNTHETIC_PATH, OWN_PATH, MODEL_PATH):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    model = load_nrm(MODEL_PATH)
    detector_balanced = SecretDetector.load(MODEL_PATH, SecretDetectorOptions(profile=BALANCED))
    detector_conservative = SecretDetector.load(
        MODEL_PATH, SecretDetectorOptions(profile=CONSERVATIVE)
    )

    synthetic = list(iter_units(SYNTHETIC_PATH))
    own = list(iter_units(OWN_PATH))
    adversarial = [unit for unit in synthetic if is_adversarial(unit)]

    sections: list[tuple[str, list[DetectionUnit]]] = [
        ("synthetic", _sample(synthetic, SYNTHETIC_SAMPLES, SEED)),
        ("own", _sample(own, OWN_SAMPLES, SEED + 1)),
        ("adversarial", _sample(adversarial, ADVERSARIAL_SAMPLES, SEED + 2)),
        ("robustness", _robustness_units(synthetic, ROBUSTNESS_SAMPLES, SEED + 100)),
    ]

    samples: list[dict[str, Any]] = []
    for name, units in sections:
        for unit in units:
            samples.append(_record(detector_balanced, detector_conservative, unit, name))
        print(f"{name}: {len(units)} units")

    document = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "model": MODEL_PATH.name,
        "feature_version": model.metadata["feature_version"],
        "labels": list(model.labels),
        "thresholds": {
            BALANCED: detector_balanced.threshold,
            CONSERVATIVE: detector_conservative.threshold,
        },
        "counts": {name: len(units) for name, units in sections},
        "samples": samples,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Written with repr-exact floats so the comparison is against what Python
    # computed, not against a rounded transcription of it.
    OUT.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
