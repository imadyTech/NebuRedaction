"""Verify that every credential-shaped string in the fixtures is a known synthetic sample.

Usage:
    python tools/check_synthetic_manifest.py [paths ...]

Why this exists. ``.gitleaks.toml`` exempts ``tests/fixtures/synthetic/`` from the
secret scan, because a credential detector cannot be tested without
credential-shaped fixtures. That exemption is a hole unless something else
guards the path. This is that something: it runs gitleaks over the fixtures with
the allowlist removed, takes every finding, and requires each finding's secret to
appear in the synthetic-sample manifest. The generator wrote that manifest from
values it produced itself, so a real credential cannot be in it.

Exit codes:
    0  every credential-shaped string is registered (or there are none)
    1  an unregistered credential-shaped string was found
    2  the check could not be performed — no manifest, or gitleaks missing

Note that 2 is a failure, not a pass. A check that cannot run has not
succeeded, and treating "gitleaks is not installed" as a green tick is how this
kind of guard quietly stops working.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic-manifest.json"

#: A gitleaks config that keeps the default rules but drops our allowlist. The
#: committed `.gitleaks.toml` exempts the fixture path, which is exactly the
#: path this tool has to look at, so it cannot reuse that file.
_SCAN_CONFIG = """\
title = "nebuRedaction synthetic-fixture audit"

[extend]
useDefault = true
"""


def _gitleaks_executable() -> str | None:
    """Locate a gitleaks binary.

    Returns:
        The executable path, or ``None`` if gitleaks is not available. The
        pre-commit hook installs its own copy, so this also looks inside the
        pre-commit cache before giving up.
    """
    found = shutil.which("gitleaks")
    if found:
        return found
    cache = Path.home() / ".cache" / "pre-commit"
    if cache.is_dir():
        for candidate in sorted(cache.glob("repo*/gitleaks*")):
            if candidate.is_file():
                return str(candidate)
        for candidate in sorted(cache.glob("repo*/**/gitleaks.exe")):
            return str(candidate)
        for candidate in sorted(cache.glob("repo*/**/gitleaks")):
            if candidate.is_file():
                return str(candidate)
    return None


def scan_for_secrets(target: Path, gitleaks: str) -> list[dict[str, object]]:
    """Run gitleaks over a directory and return its findings.

    Args:
        target: Directory to scan.
        gitleaks: Path to the gitleaks executable.

    Returns:
        The findings as gitleaks reports them, each a mapping with at least
        ``Secret``, ``File``, ``StartLine`` and ``RuleID``.

    Raises:
        RuntimeError: If gitleaks fails for a reason other than finding
            secrets. Finding secrets is the expected case here — they are
            supposed to be there — so a non-zero exit is not by itself an error.
    """
    with tempfile.TemporaryDirectory() as workspace:
        config_path = Path(workspace) / "gitleaks-audit.toml"
        config_path.write_text(_SCAN_CONFIG, encoding="utf-8")
        report_path = Path(workspace) / "report.json"

        completed = subprocess.run(
            [
                gitleaks,
                "dir",
                str(target),
                "--config",
                str(config_path),
                "--report-format",
                "json",
                "--report-path",
                str(report_path),
                "--no-banner",
                "--exit-code",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"gitleaks exited {completed.returncode}: {completed.stderr.strip()}"
            )
        if not report_path.is_file():
            return []
        text = report_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        findings = json.loads(text)
        return findings if isinstance(findings, list) else []


def _values_on_line(path: Path, line_number: int) -> set[str]:
    """Return the dataset values recorded on one line of a fixture file.

    gitleaks reports the substring *its* rule matched, which is rarely the whole
    value: for a Slack webhook it reports the entire URL, and for a generic
    high-entropy key it reports the token without the surrounding JSON. So a
    finding's secret cannot be digested and looked up directly. What can be
    looked up is the value the dataset record actually holds, which is on the
    line the finding points at.

    Args:
        path: The fixture file.
        line_number: 1-based line number from the finding.

    Returns:
        The ``value`` field of the record on that line, plus its individual
        lines for multi-line values. Empty if the file is not JSON Lines or the
        line does not parse — in which case the caller falls back to digesting
        the reported secret verbatim.
    """
    if path.suffix != ".jsonl" or line_number < 1:
        return set()
    try:
        with path.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                if number != line_number:
                    continue
                record = json.loads(raw)
                if not isinstance(record, dict):
                    return set()
                value = record.get("value")
                if not isinstance(value, str):
                    return set()
                values = {value}
                values.update(part.strip() for part in value.splitlines() if part.strip())
                return values
    except (OSError, json.JSONDecodeError):
        return set()
    return set()


def _is_registered(
    finding: dict[str, object],
    registered: set[str],
    digest: Callable[[str], str],
) -> bool:
    """Decide whether one gitleaks finding is accounted for by the manifest.

    A finding is accounted for when the secret it reported is itself registered,
    or when it falls inside a registered dataset value on the same line. The
    second case is the normal one; the first covers fixtures that are not
    dataset files.

    Args:
        finding: One gitleaks finding.
        registered: The digests from the manifest.
        digest: The manifest's digest function.

    Returns:
        True when the finding is accounted for.
    """
    secret = finding.get("Secret")
    if not isinstance(secret, str) or not secret:
        # Nothing to check: a rule fired without capturing a secret.
        return True

    if digest(secret) in registered or digest(secret.strip()) in registered:
        return True

    raw_file = finding.get("File")
    raw_line = finding.get("StartLine")
    if not isinstance(raw_file, str) or not isinstance(raw_line, int):
        return False

    for value in _values_on_line(Path(raw_file), raw_line):
        if digest(value) not in registered:
            continue
        # The value is a known synthetic sample. Accept the finding only if what
        # gitleaks matched really came from inside it, so a real credential
        # sitting on the same line as a registered one is still caught.
        if secret.strip() in value or value in secret:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Check the fixtures against the manifest.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code as documented in the module docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "files being committed, as pre-commit passes them. Only those under the "
            "fixture directory matter; the scan itself always covers the whole directory."
        ),
    )
    parser.add_argument(
        "--fixtures", type=Path, default=DEFAULT_FIXTURE_DIR, help="fixture directory to scan"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="synthetic-sample manifest"
    )
    args = parser.parse_args(argv)

    if args.paths and not any(
        args.fixtures.resolve() in path.resolve().parents for path in args.paths if path.exists()
    ):
        # Nothing being committed touches the guarded path.
        return 0

    if not args.fixtures.is_dir():
        print(f"ok: no fixture directory at {args.fixtures}, nothing to check")
        return 0

    fixture_files = [path for path in args.fixtures.rglob("*") if path.is_file()]
    if not fixture_files:
        print(f"ok: {args.fixtures} is empty, nothing to check")
        return 0

    if not args.manifest.is_file():
        print(
            f"FAILED: {args.fixtures} contains {len(fixture_files)} file(s) but there is no "
            f"manifest at {args.manifest}.\n"
            "Generate one with:\n"
            "  uv run --group datagen python -m datagen.generate --config "
            "datagen/configs/v0.yaml --seed <seed> --out data/synth/v0 "
            f"--manifest {args.manifest}",
            file=sys.stderr,
        )
        return 2

    gitleaks = _gitleaks_executable()
    if gitleaks is None:
        print(
            "FAILED: gitleaks is not on PATH and was not found in the pre-commit cache, so "
            "the fixtures could not be checked. Install gitleaks, or run this through "
            "`pre-commit run check-synthetic-manifest`.",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from datagen.manifest import load_manifest, value_digest

    try:
        registered = load_manifest(args.manifest)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    try:
        findings = scan_for_secrets(args.fixtures, gitleaks)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    unregistered = []
    for finding in findings:
        if not _is_registered(finding, registered, value_digest):
            unregistered.append(finding)

    if unregistered:
        print(
            f"FAILED: {len(unregistered)} credential-shaped string(s) under {args.fixtures} "
            f"are not in {args.manifest.name}.",
            file=sys.stderr,
        )
        print(
            "Every such string must be provably generator-produced. If these are real "
            "credentials, remove them and rotate them. If they are synthetic, regenerate "
            "the fixtures and the manifest together so the digests match.",
            file=sys.stderr,
        )
        for finding in unregistered[:20]:
            print(
                f"  {finding.get('File')}:{finding.get('StartLine')}: "
                f"rule {finding.get('RuleID')} (digest "
                f"{value_digest(str(finding.get('Secret')))[:16]}...)",
                file=sys.stderr,
            )
        if len(unregistered) > 20:
            print(f"  ... and {len(unregistered) - 20} more", file=sys.stderr)
        return 1

    print(
        f"ok: {len(findings)} credential-shaped string(s) in {len(fixture_files)} fixture "
        f"file(s), all registered in {args.manifest.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
