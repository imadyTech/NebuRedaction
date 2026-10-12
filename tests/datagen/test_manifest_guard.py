"""End-to-end test of the synthetic-fixture guard.

`tools/check_synthetic_manifest.py` is what stands behind the gitleaks allowlist
for `tests/fixtures/synthetic/`. Since no fixture corpus is committed any more
(see `conftest.py` for why), this is what proves the guard still works — and it
proves it in both directions, which is the only way a guard is worth anything:

* a generated corpus and its own manifest pass;
* a credential-shaped string that the generator never produced fails.

The test needs a real gitleaks binary, because the guard is defined in terms of
what gitleaks flags. It skips when one is unavailable rather than passing
vacuously; CI's `synthetic-manifest` job installs one, so the guard is always
exercised there.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPOSITORY_ROOT / "tools" / "check_synthetic_manifest.py"


def _unregistered_credential_line(family_name: str, registered: set[str], unit_id: str) -> str:
    """Build a JSON Lines record holding a credential-shaped, unregistered value.

    The value is produced by the generator itself from a seed outside the one the
    fixture corpus used, so it is genuinely the right shape while provably absent
    from the manifest — which is exactly the situation the guard exists for.

    Generating it beats writing a literal into this file. A vendor-pattern
    literal in source is itself a secret-scanner finding: gitleaks flags it and
    GitHub's push protection blocks the push, which is the very problem that
    stopped the fixture corpus being committed in the first place.

    Args:
        family_name: The format family to draw from.
        registered: Digests from the manifest, to confirm absence.
        unit_id: Identifier for the record.

    Returns:
        One JSON Lines record, newline-terminated.

    Raises:
        AssertionError: If no unregistered value turns up, which would mean the
            manifest and the seed space overlap far more than they should.
    """
    import json
    import random

    from datagen.families import REGISTRY
    from datagen.manifest import value_digest

    family = REGISTRY[family_name]
    for offset in range(100):
        value = family.generate(random.Random(0xC0FFEE + offset))
        if value_digest(value) not in registered:
            return (
                json.dumps(
                    {
                        "id": unit_id,
                        "key": "ApiKey",
                        "value": value,
                        "source": "synth",
                        "label": family.label,
                    }
                )
                + "\n"
            )
    raise AssertionError(f"every candidate from {family_name} was already registered")


def _gitleaks_available() -> bool:
    """Report whether a gitleaks binary can be found.

    Returns:
        True when the checker will be able to scan.
    """
    sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))
    from check_synthetic_manifest import _gitleaks_executable

    return _gitleaks_executable() is not None


requires_gitleaks = pytest.mark.skipif(
    not _gitleaks_available(),
    reason="no gitleaks binary on PATH or in the pre-commit cache",
)


def _run_checker(fixtures: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    """Run the fixture guard over a directory.

    Args:
        fixtures: Directory to scan.
        manifest: Manifest to check against.

    Returns:
        The completed process, with output captured.
    """
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--fixtures",
            str(fixtures),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )


@requires_gitleaks
def test_a_generated_corpus_passes_its_own_manifest(fixture_corpus):
    """The ordinary case: every credential-shaped value is registered."""
    result = _run_checker(fixture_corpus["directory"], fixture_corpus["manifest"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all registered" in result.stdout


@requires_gitleaks
def test_the_corpus_really_contains_things_gitleaks_flags(fixture_corpus):
    """Guard against the previous test passing because there was nothing to find.

    A scan that flags nothing would make the guard look healthy while proving
    nothing at all, so the count is asserted rather than assumed.
    """
    result = _run_checker(fixture_corpus["directory"], fixture_corpus["manifest"])
    assert result.returncode == 0
    # "ok: N credential-shaped string(s) in ..."
    count = int(result.stdout.split("ok: ", 1)[1].split(" ", 1)[0])
    assert count > 0, result.stdout


@requires_gitleaks
def test_an_unregistered_credential_shaped_string_is_rejected(fixture_corpus, tmp_path):
    """The case the guard exists for: something the generator never produced."""
    scratch = tmp_path / "fixtures"
    scratch.mkdir()
    for split in ("train", "val", "test"):
        shutil.copy(fixture_corpus[split], scratch / fixture_corpus[split].name)

    registered = set(json.loads(fixture_corpus["manifest"].read_text(encoding="utf-8"))["digests"])
    planted = "".join(
        _unregistered_credential_line(name, registered, f"probe-{index}")
        for index, name in enumerate(("aws.access_key_id", "github.pat_classic"))
    )
    (scratch / "unregistered.jsonl").write_text(planted, encoding="utf-8")

    result = _run_checker(scratch, fixture_corpus["manifest"])
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not in" in result.stderr
    assert "unregistered.jsonl" in result.stderr


@requires_gitleaks
def test_a_missing_manifest_is_a_failure_not_a_pass(fixture_corpus, tmp_path):
    """A check that cannot run has not succeeded."""
    result = _run_checker(fixture_corpus["directory"], tmp_path / "absent.json")
    assert result.returncode == 2, result.stdout + result.stderr


def test_an_empty_fixture_directory_passes(tmp_path):
    """No fixtures is a valid state, and needs no gitleaks to establish."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"manifest_version": 1, "digests": []}', encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _run_checker(empty, manifest)
    assert result.returncode == 0, result.stdout + result.stderr
