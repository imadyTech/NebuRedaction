r"""Generate the synthetic corpus.

Usage:
    python -m datagen.generate --config datagen/configs/v0.yaml \\
        --seed 20261011 --out data/synth/v0/

Determinism is a requirement, not a nicety: NR-0001 §2.2 item 7 asks that the
same config and seed produce byte-identical output. Two decisions deliver it.

First, every record draws from its own :class:`random.Random`, seeded from
``(master seed, split, index)``. No record's value depends on how many records
came before it, so adding a split or reordering a loop cannot shift the stream.

Second, the *composition* of each split is computed as exact integer counts and
then shuffled, rather than sampled. Drawing each record's kind at random would
give approximately the configured proportions; computing them gives precisely
the proportions, which is what lets the adversarial-negative floor in
NR-0001 §2.2 item 4 be a guarantee rather than a statistical expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datagen import GENERATOR_VERSION
from datagen.assemble import (
    ContextMix,
    assemble_template_adversarial,
    assemble_unit,
)
from datagen.families import REGISTRY, Family, negative_families, positive_families
from datagen.lexicon import (
    CONTEXT_HINTING,
    CONTEXT_MISLEADING,
    CONTEXT_NEUTRAL,
    Lexicon,
    load_lexicon,
)
from datagen.unit import DetectionUnit

SPLITS = ("train", "val", "test")

#: Kinds of record in a split plan.
#:
#: Positives draw their key-name context from the configured mix. Negatives do
#: not: their context is dictated by the kind, so the adversarial share of the
#: negatives is exactly what the config says rather than whatever the draw
#: produced. That matters because real logs are overwhelmingly negatives under
#: harmless names, and letting the mix apply to negatives too would have put
#: most of them under credential-suggesting names — inflating the adversarial
#: count while making the distribution unlike anything in production.
KIND_POSITIVE = "positive"
KIND_NEGATIVE_PLAIN = "negative_plain"
KIND_NEGATIVE_SENSITIVE_KEY = "negative_sensitive_key"
KIND_NEGATIVE_MISLEADING_KEY = "negative_misleading_key"
KIND_NEGATIVE_TEMPLATE = "negative_template"

#: The context each negative kind forces.
_NEGATIVE_CONTEXT: dict[str, str] = {
    KIND_NEGATIVE_PLAIN: CONTEXT_NEUTRAL,
    KIND_NEGATIVE_SENSITIVE_KEY: CONTEXT_HINTING,
    KIND_NEGATIVE_MISLEADING_KEY: CONTEXT_MISLEADING,
}


@dataclass(frozen=True, slots=True)
class Config:
    """A parsed generator configuration.

    Attributes:
        name: Corpus name, used in unit ids and in the output directory.
        sizes: Number of units per split.
        positive_ratio: Share of each split that is a secret.
        adversarial_negative_ratio: Share of negatives that are adversarial.
        adversarial_carrier_split: How adversarial negatives divide between the
            key-name and template carriers.
        context_mix: Key-name context weights per split.
        held_out_positive: Positive families excluded from train and val.
        held_out_negative: Negative families excluded from train and val.
        template_adversarial_families: Negative families suitable as fillers in
            the template-adversarial slots.
    """

    name: str
    sizes: dict[str, int]
    positive_ratio: float
    adversarial_negative_ratio: float
    adversarial_carrier_split: dict[str, float]
    context_mix: dict[str, ContextMix]
    held_out_positive: tuple[str, ...]
    held_out_negative: tuple[str, ...]
    template_adversarial_families: tuple[str, ...]

    @property
    def held_out(self) -> frozenset[str]:
        """Every held-out family name.

        Returns:
            The union of the positive and negative held-out sets.
        """
        return frozenset(self.held_out_positive) | frozenset(self.held_out_negative)


def load_config(path: Path) -> Config:
    """Read and validate a generator configuration.

    Args:
        path: Configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If a field is missing or inconsistent, if the configuration
            requires a different generator version, or if it names a family that
            does not exist. A typo in a held-out family name would otherwise
            leave that family in training and quietly invalidate the
            generalisation measurement, so unknown names are fatal.
    """
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    required = document.get("generator_version_required")
    if required is not None and required != GENERATOR_VERSION:
        raise ValueError(
            f"{path}: config requires generator version {required!r}, this is {GENERATOR_VERSION!r}"
        )

    splits = document.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError(f"{path}: 'splits' must give a size for exactly {list(SPLITS)}")

    raw_mix = document.get("context_mix")
    if not isinstance(raw_mix, dict) or set(raw_mix) != set(SPLITS):
        raise ValueError(f"{path}: 'context_mix' must have an entry for each of {list(SPLITS)}")
    context_mix = {
        split: ContextMix(
            hinting=float(weights["hinting"]),
            neutral=float(weights["neutral"]),
            misleading=float(weights["misleading"]),
        )
        for split, weights in raw_mix.items()
    }

    held_out = document.get("held_out_families") or {}
    held_out_positive = tuple(held_out.get("positive") or ())
    held_out_negative = tuple(held_out.get("negative") or ())
    template_families = tuple(document.get("template_adversarial_families") or ())

    config = Config(
        name=str(document.get("name", path.stem)),
        sizes={split: int(size) for split, size in splits.items()},
        positive_ratio=float(document["positive_ratio"]),
        adversarial_negative_ratio=float(document["adversarial_negative_ratio"]),
        adversarial_carrier_split={
            str(key): float(value) for key, value in document["adversarial_carrier_split"].items()
        },
        context_mix=context_mix,
        held_out_positive=held_out_positive,
        held_out_negative=held_out_negative,
        template_adversarial_families=template_families,
    )

    unknown = sorted(
        name
        for name in (*held_out_positive, *held_out_negative, *template_families)
        if name not in REGISTRY
    )
    if unknown:
        raise ValueError(f"{path}: unknown families: {', '.join(unknown)}")

    for name in held_out_positive:
        if REGISTRY[name].is_negative:
            raise ValueError(f"{path}: {name!r} is listed as a held-out positive but is negative")
    for name in held_out_negative:
        if not REGISTRY[name].is_negative:
            raise ValueError(f"{path}: {name!r} is listed as a held-out negative but is positive")
    for name in template_families:
        if not REGISTRY[name].is_negative:
            raise ValueError(
                f"{path}: {name!r} is a template-adversarial filler but is not a negative family"
            )

    if not 0.0 < config.positive_ratio < 1.0:
        raise ValueError(f"{path}: 'positive_ratio' must be strictly between 0 and 1")
    expected_carriers = {"sensitive_key", "misleading_key", "template"}
    if set(config.adversarial_carrier_split) != expected_carriers:
        raise ValueError(
            f"{path}: 'adversarial_carrier_split' must have exactly the keys "
            f"{sorted(expected_carriers)}"
        )
    carrier_total = sum(config.adversarial_carrier_split.values())
    if abs(carrier_total - 1.0) > 1e-9:
        raise ValueError(
            f"{path}: 'adversarial_carrier_split' values sum to {carrier_total}, expected 1.0"
        )
    if not 0.10 <= config.adversarial_negative_ratio <= 1.0:
        raise ValueError(
            f"{path}: 'adversarial_negative_ratio' must be at least 0.10 "
            "(NR-0001 §2.2 item 4 sets that floor)"
        )
    return config


def derive_seed(master_seed: int, split: str, index: int) -> int:
    """Derive a per-record seed from the master seed.

    Hashing rather than arithmetic keeps nearby records from sharing correlated
    streams, which matters because many records draw only a handful of values
    and a linear seed sequence would make neighbouring records visibly similar.

    Args:
        master_seed: The seed given on the command line.
        split: The split being generated.
        index: The record's index within the split.

    Returns:
        A 64-bit seed.
    """
    material = f"{master_seed}:{split}:{index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _allowed_families(config: Config, split: str) -> tuple[list[Family], list[Family]]:
    """Return the positive and negative families permitted in a split.

    Args:
        config: The configuration.
        split: The split being generated.

    Returns:
        ``(positives, negatives)``. Train and val exclude the held-out families;
        test includes everything, so it measures both seen and unseen formats.
    """
    if split == "test":
        return positive_families(), negative_families()
    held_out = config.held_out
    return (
        [family for family in positive_families() if family.name not in held_out],
        [family for family in negative_families() if family.name not in held_out],
    )


def build_plan(config: Config, split: str, rng: random.Random) -> list[tuple[str, Family]]:
    """Compute the exact composition of a split and shuffle it.

    Args:
        config: The configuration.
        split: The split being generated.
        rng: A seeded source of randomness, used only for the shuffle.

    Returns:
        One ``(kind, family)`` pair per record, in generation order. Families are
        assigned round-robin before shuffling, so every permitted family appears
        and the per-family counts differ by at most one.

    Raises:
        ValueError: If the split is too small to hold at least one record of
            each kind.
    """
    total = config.sizes[split]
    positives, negatives = _allowed_families(config, split)
    template_fillers = [
        REGISTRY[name] for name in config.template_adversarial_families
    ] or negatives

    if split != "test":
        template_fillers = [
            family for family in template_fillers if family.name not in config.held_out
        ] or negatives

    positive_count = round(total * config.positive_ratio)
    negative_count = total - positive_count
    adversarial_count = round(negative_count * config.adversarial_negative_ratio)

    # Divide the adversarial quota between its three carriers. The last share is
    # the remainder so the three always sum to the quota exactly, whatever
    # rounding does to the first two.
    sensitive_key_count = round(
        adversarial_count * config.adversarial_carrier_split.get("sensitive_key", 0.0)
    )
    misleading_key_count = round(
        adversarial_count * config.adversarial_carrier_split.get("misleading_key", 0.0)
    )
    template_count = adversarial_count - sensitive_key_count - misleading_key_count
    plain_negative_count = negative_count - adversarial_count

    counts = {
        KIND_POSITIVE: positive_count,
        KIND_NEGATIVE_PLAIN: plain_negative_count,
        KIND_NEGATIVE_SENSITIVE_KEY: sensitive_key_count,
        KIND_NEGATIVE_MISLEADING_KEY: misleading_key_count,
        KIND_NEGATIVE_TEMPLATE: template_count,
    }
    too_small = sorted(kind for kind, count in counts.items() if count < 1)
    if too_small:
        raise ValueError(
            f"split {split!r} of size {total} is too small to contain every record kind; "
            f"these would be empty: {', '.join(too_small)}"
        )

    pools = {
        KIND_POSITIVE: positives,
        KIND_NEGATIVE_PLAIN: negatives,
        KIND_NEGATIVE_SENSITIVE_KEY: negatives,
        KIND_NEGATIVE_MISLEADING_KEY: negatives,
        KIND_NEGATIVE_TEMPLATE: template_fillers,
    }

    plan: list[tuple[str, Family]] = []
    for kind, count in counts.items():
        pool = pools[kind]
        for index in range(count):
            plan.append((kind, pool[index % len(pool)]))

    rng.shuffle(plan)
    return plan


def generate_split(
    config: Config,
    split: str,
    master_seed: int,
    lexicon: Lexicon,
) -> list[DetectionUnit]:
    """Generate one split.

    Args:
        config: The configuration.
        split: ``train``, ``val`` or ``test``.
        master_seed: The seed given on the command line.
        lexicon: The key-name lexicon.

    Returns:
        The generated units, in file order.
    """
    plan_rng = random.Random(derive_seed(master_seed, split, -1))
    plan = build_plan(config, split, plan_rng)
    held_out = config.held_out
    context_mix = config.context_mix[split]

    units: list[DetectionUnit] = []
    for index, (kind, family) in enumerate(plan):
        seed = derive_seed(master_seed, split, index)
        rng = random.Random(seed)
        unit_id = f"{config.name}-{split}-{index:07d}"
        is_held_out = family.name in held_out

        if kind == KIND_NEGATIVE_TEMPLATE:
            unit = assemble_template_adversarial(
                rng,
                unit_id=unit_id,
                family=family,
                split=split,
                seed=seed,
                held_out_family=is_held_out,
            )
        else:
            unit = assemble_unit(
                rng,
                unit_id=unit_id,
                family=family,
                lexicon=lexicon,
                context_mix=context_mix,
                split=split,
                seed=seed,
                held_out_family=is_held_out,
                force_context=_NEGATIVE_CONTEXT.get(kind),
            )
        units.append(unit)
    return units


def write_split(units: list[DetectionUnit], path: Path) -> str:
    """Write a split to a JSON Lines file and return its digest.

    Args:
        units: The units to write.
        path: Destination file.

    Returns:
        The SHA-256 hex digest of the bytes written, so the caller can record it
        and later prove the file was not regenerated with different code.
    """
    payload = "".join(unit.to_json() + "\n" for unit in units).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def summarise(units: list[DetectionUnit]) -> dict[str, Any]:
    """Compute the statistics recorded in the dataset metadata.

    Args:
        units: The units of one split.

    Returns:
        Counts by label, by family, by context, and the adversarial, ambiguous
        and held-out shares.
    """
    labels = Counter(unit.label for unit in units)
    families = Counter(unit.family for unit in units)
    contexts = Counter(unit.meta.get("context") for unit in units)
    negatives = sum(1 for unit in units if unit.label == "none")
    adversarial = sum(1 for unit in units if unit.meta.get("adversarial"))
    return {
        "count": len(units),
        "by_label": dict(sorted(labels.items(), key=lambda item: str(item[0]))),
        "by_context": dict(sorted(contexts.items(), key=lambda item: str(item[0]))),
        "family_count": len(families),
        "by_family": dict(sorted(families.items(), key=lambda item: str(item[0]))),
        "negatives": negatives,
        "adversarial": adversarial,
        "adversarial_share_of_negatives": round(adversarial / negatives, 4) if negatives else 0.0,
        "ambiguous": sum(1 for unit in units if unit.ambiguous),
        "held_out_family_units": sum(1 for unit in units if unit.meta.get("held_out_family")),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate every split and write the corpus with its metadata.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("datagen/configs/v0.yaml"), help="generator config"
    )
    parser.add_argument("--seed", type=int, required=True, help="master seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="override every split size, for quick smoke runs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="also write a synthetic-sample manifest here (NR-0001 §2.3)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.size is not None:
        config = replace(config, sizes=dict.fromkeys(SPLITS, args.size))
    lexicon = load_lexicon()

    metadata: dict[str, Any] = {
        "name": config.name,
        "generator_version": GENERATOR_VERSION,
        "config": str(args.config).replace("\\", "/"),
        "seed": args.seed,
        "families_total": len(REGISTRY),
        "families_positive": len(positive_families()),
        "families_negative": len(negative_families()),
        "held_out_families": sorted(config.held_out),
        "splits": {},
    }

    all_units: list[DetectionUnit] = []
    for split in SPLITS:
        units = generate_split(config, split, args.seed, lexicon)
        path = args.out / f"{split}.jsonl"
        digest = write_split(units, path)
        metadata["splits"][split] = {
            "file": path.name,
            "sha256": digest,
            **summarise(units),
        }
        all_units.extend(units)
        print(f"{path}: {len(units)} units, sha256 {digest[:16]}...")

    metadata_path = args.out / "dataset.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{metadata_path}: metadata written")

    if args.manifest is not None:
        from datagen.manifest import write_manifest

        count = write_manifest(all_units, args.manifest, seed=args.seed, config_name=config.name)
        print(f"{args.manifest}: {count} secret value digests")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
