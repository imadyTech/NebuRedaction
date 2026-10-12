"""Shared fixtures: a generated corpus and its manifest, built on demand.

Why these are generated rather than committed. The generator's whole purpose is
to produce format-correct vendor credentials, so a committed fixture corpus is
indistinguishable from a leak to any scanner that matches vendor patterns —
including GitHub's push protection, which blocked a push of exactly these files
with 33 findings (Stripe, SendGrid and Slack keys among them).

Push protection is a control this project wants switched on. The options were to
bypass it per secret, to exclude the path from scanning, or to stop shipping
vendor-pattern secrets. The third costs nothing real: generation is
deterministic from the seed, so a corpus built in a temporary directory is
byte-identical to one that had been committed, and the tests are just as stable.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from datagen.generate import SPLITS, generate_split, load_config, write_split
from datagen.lexicon import load_lexicon
from datagen.manifest import write_manifest

CONFIG_PATH = Path("datagen/configs/v0.yaml")

#: Seed and size for the generated fixture corpus. The seed matches the one the
#: project's real corpus uses, so a fixture record is a genuine sample of it.
FIXTURE_SEED = 20261011
FIXTURE_SIZE = 120


@pytest.fixture(scope="session")
def fixture_corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate a small corpus and its manifest in a temporary directory.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        A mapping with a ``Path`` per split, plus ``manifest`` and ``directory``.
    """
    directory = tmp_path_factory.mktemp("synthetic-corpus")
    config = replace(load_config(CONFIG_PATH), sizes=dict.fromkeys(SPLITS, FIXTURE_SIZE))
    lexicon = load_lexicon()

    paths: dict[str, Path] = {"directory": directory}
    everything = []
    for split in SPLITS:
        units = generate_split(config, split, FIXTURE_SEED, lexicon)
        path = directory / f"{split}.jsonl"
        write_split(units, path)
        paths[split] = path
        everything.extend(units)

    manifest = directory.parent / "synthetic-manifest.json"
    write_manifest(everything, manifest, seed=FIXTURE_SEED, config_name=config.name)
    paths["manifest"] = manifest
    return paths


@pytest.fixture(scope="session")
def fixture_manifest_digests(fixture_corpus: dict[str, Path]) -> set[str]:
    """Return the digest set from the generated manifest.

    Args:
        fixture_corpus: The generated corpus.

    Returns:
        The registered digests.
    """
    return set(json.loads(fixture_corpus["manifest"].read_text(encoding="utf-8"))["digests"])
