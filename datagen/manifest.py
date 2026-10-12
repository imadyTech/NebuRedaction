"""The synthetic-sample manifest (NR-0001 §2.3).

The problem this solves: a credential detector cannot be tested without
credential-shaped strings in its fixtures, so ``tests/fixtures/synthetic/`` is
exempted from the gitleaks scan. An exemption with nothing behind it is a hole —
anyone could drop a real key there and the scanner would stay quiet.

The manifest closes it. The generator records the SHA-256 of every secret value
it produced; the checker (``tools/check_synthetic_manifest.py``) requires every
credential-shaped string under that path to appear in the manifest. A real
credential cannot be in the manifest, because the generator never saw it.

The manifest stores digests, never values, so the manifest itself is safe to
commit and says nothing about what the secrets were.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datagen import GENERATOR_VERSION
from datagen.unit import DetectionUnit

#: Manifest format version, so the checker can reject a format it predates.
MANIFEST_VERSION = 1


def value_digest(value: str) -> str:
    """Return the manifest digest of a value.

    Args:
        value: The raw value, exactly as it appears in the data.

    Returns:
        The SHA-256 hex digest of the UTF-8 encoding. Values are hashed verbatim
        — no trimming or case folding — because a value that differs by a
        character is a different value, and the checker must not be talked into
        accepting a near miss.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collect_digests(units: Iterable[DetectionUnit]) -> set[str]:
    """Collect the digests of every secret value among some units.

    Multi-line values are registered line by line as well as whole. A fixture
    may legitimately contain one line of a PEM block, and the checker works on
    whatever string the scanner flagged, which for a PEM block is usually the
    body rather than the full envelope.

    Args:
        units: The units to scan.

    Returns:
        The digests to record.
    """
    digests: set[str] = set()
    for unit in units:
        if not unit.should_redact:
            continue
        digests.add(value_digest(unit.value))
        if "\n" in unit.value:
            for line in unit.value.splitlines():
                stripped = line.strip()
                if stripped:
                    digests.add(value_digest(stripped))
    return digests


def write_manifest(
    units: Iterable[DetectionUnit],
    path: Path,
    *,
    seed: int,
    config_name: str,
) -> int:
    """Write a manifest for the secret values in some units.

    Args:
        units: The units whose secret values should be registered.
        path: Destination file.
        seed: The generator seed, recorded so the manifest can be regenerated.
        config_name: The corpus name, recorded for the same reason.

    Returns:
        How many digests were written.
    """
    digests = sorted(collect_digests(units))
    document: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "generator_version": GENERATOR_VERSION,
        "config_name": config_name,
        "seed": seed,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "sha256",
        "note": (
            "SHA-256 digests of synthetic secret values. Values themselves are never "
            "stored. Every credential-shaped string under tests/fixtures/synthetic/ must "
            "appear here; see tools/check_synthetic_manifest.py."
        ),
        "count": len(digests),
        "digests": digests,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(digests)


def load_manifest(path: Path) -> set[str]:
    """Read the digest set from a manifest file.

    Args:
        path: Manifest file.

    Returns:
        The registered digests.

    Raises:
        ValueError: If the manifest format version is newer than this code
            understands, or the file is not a manifest.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "digests" not in document:
        raise ValueError(f"{path}: not a synthetic-sample manifest")
    version = document.get("manifest_version")
    if not isinstance(version, int) or version > MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest version {version!r} is newer than this checker "
            f"understands ({MANIFEST_VERSION})"
        )
    return set(document["digests"])
