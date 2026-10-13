"""The fixed adversarial spot-check set (NR-0003 §2.2 item 6).

Usage:
    python -m labeling.teacher.adversarial --build     # select and freeze the set
    python -m labeling.teacher.adversarial --measure   # the teacher's rate on it

§2.2 item 6 asks for a **fixed** set of at least 200 units — credential-
suggesting key names over non-credential values — and for the teacher's
false-positive rate on it. The reason it is fixed rather than resampled is that
a number measured on a fresh sample each time is not comparable across prompt
versions or across work orders, and comparing v1 with a future v2 is the whole
use of the measurement.

The set is frozen by **unit id and value digest**, written to
``labeling/teacher/adversarial-v1.json`` and committed. Ids alone would not
notice a regenerated corpus reassigning them; digests of the values catch that,
and no value is written to the file, so it can be committed without putting
credential-shaped strings in the repository. Tests assert that the frozen set
still resolves against the corpus.

**What counts as adversarial here.** NR-0001 §2.4 constructs three carriers for
a negative: a credential-suggesting key name, a misleading key name, and a
credential word in the message template. All three are included, because
NR-0002 measured the *template* as the single largest source of rule false
positives (416 of 1,296) and NR-0000's defining failure was a template hint.
Restricting the set to key names would have measured the easier half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training
from datagen.unit import DetectionUnit

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

#: Where the frozen set lives. Committed, unlike everything under `data/`.
FROZEN = Path(__file__).resolve().parent / "adversarial-v1.json"

#: §2.2 item 6's floor.
MINIMUM_UNITS = 200

#: Target size. Comfortably above the floor, so that a few units going missing
#: to a corpus regeneration does not drop the set below what the work order
#: requires, and small enough to label in under a minute.
TARGET_UNITS = 400

#: The adversarial carriers NR-0001 builds, recorded in each unit's meta.
CARRIERS = ("sensitive_key", "misleading_key", "template")


def digest_value(value: str) -> str:
    """Digest a unit's value.

    Args:
        value: The value.

    Returns:
        The first sixteen hex characters of its SHA-256. Enough to detect a
        corpus regeneration that reassigned ids, and not a value.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def carrier_of(unit: DetectionUnit) -> str | None:
    """Identify which adversarial carrier a negative uses.

    Args:
        unit: The unit.

    Returns:
        One of :data:`CARRIERS`, or ``None`` if the unit is not an adversarial
        negative.

    **The carrier has to be derived, not read, and that is an NR-0001 metadata
    gap rather than a choice here.** The generator records
    ``meta.adversarial_carrier`` only for the template carrier, which
    ``assemble_template_adversarial`` sets explicitly. The other two are
    distinguishable only by the forced context: of the 27,840 adversarial
    negatives in the training split, 8,352 carry ``adversarial_carrier:
    template``, 11,136 have ``context: hinting`` and no carrier, and 8,352 have
    ``context: misleading`` and no carrier. Those correspond to the
    sensitive-key and misleading-key carriers that NR-0001 §2.4's plan names.

    Deriving it here keeps this work order moving, and the gap is reported as a
    defect for NR-0001 to close by writing the carrier for all three. It is a
    metadata gap and not a labelling error: no label is wrong because of it, so
    nothing needs regenerating.
    """
    meta = unit.meta or {}
    if not meta.get("adversarial"):
        return None
    explicit = meta.get("adversarial_carrier")
    if explicit in CARRIERS:
        return str(explicit)
    context = meta.get("context")
    if context == "hinting":
        return "sensitive_key"
    if context == "misleading":
        return "misleading_key"
    return None


def is_adversarial(unit: DetectionUnit) -> bool:
    """Whether a unit is an adversarial negative.

    Args:
        unit: The unit.

    Returns:
        True for a unit labelled ``none`` that carries a credential-suggesting
        key name, a misleading key name, or a credential word in its template.
    """
    return unit.label == "none" and carrier_of(unit) is not None


def select(units: Sequence[DetectionUnit], size: int = TARGET_UNITS) -> list[DetectionUnit]:
    """Choose the spot-check set, balanced across carriers.

    Args:
        units: The corpus to draw from.
        size: Target size.

    Returns:
        Up to ``size`` units, as evenly spread across the three carriers as the
        corpus allows, in corpus order.

    Balanced across carriers rather than drawn at random, because the three are
    not equally hard and an unbalanced set would make the reported rate depend
    on the corpus's carrier mix rather than on the teacher. Selection is
    deterministic — the first N of each carrier in corpus order — since the set
    is frozen immediately and never resampled, so a seed would add a parameter
    without adding anything.
    """
    by_carrier: dict[str, list[DetectionUnit]] = {carrier: [] for carrier in CARRIERS}
    for unit in units:
        carrier = carrier_of(unit) if unit.label == "none" else None
        if carrier is not None:
            by_carrier[carrier].append(unit)

    per_carrier = size // len(CARRIERS)
    chosen: set[str] = set()
    for carrier in CARRIERS:
        chosen.update(unit.id for unit in by_carrier[carrier][:per_carrier])
    return [unit for unit in units if unit.id in chosen]


def load_frozen(path: Path = FROZEN) -> dict[str, str]:
    """Read the frozen set.

    Args:
        path: The frozen-set file.

    Returns:
        Unit id to value digest.

    Raises:
        FileNotFoundError: If the set has not been built.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run with --build")
    return dict(json.loads(path.read_text(encoding="utf-8"))["units"])


def resolve(
    frozen: dict[str, str], units: Sequence[DetectionUnit]
) -> tuple[list[DetectionUnit], list[str]]:
    """Match the frozen set against a corpus.

    Args:
        frozen: Unit id to value digest.
        units: The corpus.

    Returns:
        ``(resolved, problems)``. A problem is an id that is missing from the
        corpus or whose value no longer matches its digest — either of which
        means the corpus was regenerated and the measurement is no longer
        comparable with earlier ones.
    """
    by_id = {unit.id: unit for unit in units}
    resolved: list[DetectionUnit] = []
    problems: list[str] = []
    for unit_id, expected in frozen.items():
        unit = by_id.get(unit_id)
        if unit is None:
            problems.append(f"{unit_id}: not in the corpus")
        elif digest_value(unit.value) != expected:
            problems.append(f"{unit_id}: value digest changed")
        else:
            resolved.append(unit)
    return resolved, problems


def measure(
    units: Sequence[DetectionUnit], teacher: dict[str, dict[str, float]], temperature: float = 1.0
) -> dict[str, object]:
    """Measure the teacher's false-positive rate on the set.

    Args:
        units: The resolved adversarial units, all of which are negatives.
        teacher: Distributions by unit id.
        temperature: Recalibration temperature to apply first.

    Returns:
        The overall rate, the rate per carrier, the distribution of wrong
        classes, and the confident subset — the units it got wrong at or above
        0.9 after recalibration, which is the number §2.3's auto-adopt
        threshold actually depends on.
    """
    from labeling.teacher.calibration import apply_temperature

    total = 0
    wrong = 0
    confident_wrong = 0
    by_carrier: dict[str, dict[str, int]] = {}
    wrong_classes: dict[str, int] = {}
    worst: list[dict[str, object]] = []

    for unit in units:
        distribution = teacher.get(unit.id)
        if distribution is None:
            continue
        scaled = apply_temperature(distribution, temperature)
        choice = max(scaled, key=lambda label: scaled[label])
        confidence = scaled[choice]
        carrier = carrier_of(unit) or "(none)"
        bucket = by_carrier.setdefault(carrier, {"units": 0, "wrong": 0})
        bucket["units"] += 1
        total += 1
        if choice != "none":
            wrong += 1
            bucket["wrong"] += 1
            wrong_classes[choice] = wrong_classes.get(choice, 0) + 1
            if confidence >= 0.9:
                confident_wrong += 1
            worst.append(
                {
                    "unit_id": unit.id,
                    "key": unit.key,
                    "template": unit.template,
                    "carrier": carrier,
                    "called": choice,
                    "probability": round(confidence, 4),
                }
            )

    for bucket in by_carrier.values():
        bucket["rate"] = (  # type: ignore[assignment]
            round(bucket["wrong"] / bucket["units"], 4) if bucket["units"] else None
        )
    worst.sort(key=lambda item: -float(item["probability"]))  # type: ignore[arg-type]
    return {
        "units": total,
        "false_positives": wrong,
        "false_positive_rate": round(wrong / total, 4) if total else None,
        "confident_false_positives_at_0.9": confident_wrong,
        "confident_rate": round(confident_wrong / total, 4) if total else None,
        "temperature": temperature,
        "by_carrier": dict(sorted(by_carrier.items())),
        "called_instead": dict(sorted(wrong_classes.items(), key=lambda pair: -pair[1])),
        "worst_20": worst[:20],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build or measure the adversarial spot-check set.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success; ``2`` on a missing input or a set below the floor.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument(
        "--units",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synth" / "v0" / "train.jsonl",
    )
    parser.add_argument(
        "--teacher", type=Path, default=REPOSITORY_ROOT / "data" / "labels" / "teacher_v0.jsonl"
    )
    parser.add_argument(
        "--calibration", type=Path, default=REPOSITORY_ROOT / "data" / "labels" / "calibration.json"
    )
    parser.add_argument("--size", type=int, default=TARGET_UNITS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.units.is_file():
        print(f"FAILED: {args.units} not found", file=sys.stderr)
        return 2
    corpus = list(load_for_training([args.units]))

    if args.build:
        chosen = select(corpus, args.size)
        if len(chosen) < MINIMUM_UNITS:
            print(
                f"FAILED: only {len(chosen)} adversarial units found; §2.2 item 6 requires "
                f"at least {MINIMUM_UNITS}",
                file=sys.stderr,
            )
            return 2
        FROZEN.write_text(
            json.dumps(
                {
                    "work_order": "NR-0003",
                    "section": "2.2 item 6",
                    "note": (
                        "Fixed adversarial spot-check set: negatives carrying a "
                        "credential-suggesting key name, a misleading key name, or a "
                        "credential word in the template. Frozen by unit id and value "
                        "digest so the measurement is comparable across prompt versions. "
                        "No value is stored."
                    ),
                    "source": str(args.units).replace("\\", "/"),
                    "units": {unit.id: digest_value(unit.value) for unit in chosen},
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        carriers: dict[str, int] = {}
        for unit in chosen:
            carrier = carrier_of(unit) or "(none)"
            carriers[carrier] = carriers.get(carrier, 0) + 1
        print(f"{FROZEN}: {len(chosen)} units")
        for carrier, count in sorted(carriers.items()):
            print(f"  {count:>4}  {carrier}")
        return 0

    if not args.measure:
        parser.error("pass --build or --measure")

    frozen = load_frozen()
    resolved, problems = resolve(frozen, corpus)
    if problems:
        print(f"WARNING: {len(problems)} frozen unit(s) no longer resolve:", file=sys.stderr)
        for problem in problems[:10]:
            print(f"  - {problem}", file=sys.stderr)

    if not args.teacher.is_file():
        print(f"FAILED: {args.teacher} not found", file=sys.stderr)
        return 2
    teacher: dict[str, dict[str, float]] = {}
    with args.teacher.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                teacher[record["unit_id"]] = record["probabilities"]

    temperature = 1.0
    if args.calibration.is_file():
        temperature = float(json.loads(args.calibration.read_text(encoding="utf-8"))["temperature"])

    result = measure(resolved, teacher, temperature)
    result["frozen_units"] = len(frozen)
    result["unresolved"] = problems
    out = args.out or (REPOSITORY_ROOT / "runs" / "NR-0003" / "adversarial-spot-check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "worst_20"}, indent=2))
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
