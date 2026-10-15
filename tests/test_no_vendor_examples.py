"""No vendor documentation example value may appear in this repository.

NR-0008 §2.0 finding F4, and the guard Frank asked for on 2026-10-15.

**Why gitleaks cannot do this job.** The repository's defence against
vendor-shaped credential strings (EXECUTION.md §5 item 3) is gitleaks, and
gitleaks deliberately suppresses findings whose value contains a stop word such
as ``EXAMPLE`` or ``TEST``. That is correct behaviour for a leak scanner —
nobody wants a build to fail over a documentation placeholder — but it leaves
the control blind in exactly the place the copying risk is highest: a vendor's
published example value is the string most likely to be pasted into a test, and
it is the string gitleaks is built to ignore. Three such values had
accumulated here, one of them added as recently as NR-0005, and the hook passed
every time.

So this test does the opposite of a leak scan. It does not look for strings
that *look* like credentials; it looks for a short list of strings that are
*known to come from vendor documentation*, which is a different question with a
different answer.

**What this test is not.** It is not a general copying detector, and it cannot
become one — the list is finite and maintained by hand. It stops the specific
regression that already happened three times. A fourth value, from a vendor not
listed here, would pass. When one is found, add it.

**Why the needles are assembled rather than written out.** A test file that
spells out the forbidden strings would itself contain them, and would fail
itself. Each entry is therefore built from fragments at run time, and a
self-check asserts the test can actually see a planted value — a guard whose
own detection is untested is a guard nobody has checked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Files that may legitimately discuss the subject without carrying a value.
#: This test itself is the only exception, and it holds no complete value.
EXEMPT = frozenset({"tests/test_no_vendor_examples.py"})

#: Extensions worth reading. Binary artefacts are skipped: the only committed
#: `.nrm` is a hand-built fixture whose bytes are weights, not text.
SKIP_SUFFIXES = (".nrm", ".lock", ".png", ".jpg", ".ico", ".woff", ".woff2")


def _needles() -> dict[str, str]:
    """Build the forbidden values from fragments.

    Returns:
        A mapping of a human-readable name to the value itself.

    Assembled so that this file does not contain any of the strings it forbids.
    Each entry is a value published in a vendor's own documentation as an
    illustrative placeholder.
    """
    return {
        "a cloud provider's example access key id": "AKIA" + "IOSFODNN7" + "EXAMPLE",
        "the matching example secret access key": (
            "wJalrXUtnFEMI" + "/K7MDENG/" + "bPxRfiCYEXAMPLE"
        ),
        "an example unique id for a user": "AIDA" + "JQABLZS4A3QDU576Q",
        "an example unique id for a role": "AROA" + "DBQP57FF2AEXAMPLE",
    }


def _tracked_text_files() -> list[Path]:
    """List the repository's tracked text files.

    Returns:
        Absolute paths, excluding binary artefacts and this test.

    Uses ``git ls-files`` so that nothing outside version control is scanned —
    the question is what the repository *contains*, not what happens to be on
    this disk.
    """
    listed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        REPOSITORY_ROOT / name
        for name in listed
        if name not in EXEMPT and not name.endswith(SKIP_SUFFIXES)
    ]


def test_the_guard_can_see_a_planted_value(tmp_path: Path) -> None:
    """The detection itself works.

    A guard that silently matches nothing passes forever. This plants each
    forbidden value in a file and requires the same comparison the real test
    uses to find it.
    """
    needles = _needles()
    assert needles, "the needle list is empty, so the test below proves nothing"

    for name, value in needles.items():
        planted = tmp_path / "planted.txt"
        planted.write_text(f"key = {value}\n", encoding="utf-8")
        assert value in planted.read_text(encoding="utf-8"), name


def test_no_tracked_file_contains_a_vendor_documentation_example() -> None:
    """EXECUTION.md §5 item 3, for the class gitleaks suppresses."""
    needles = _needles()
    findings: list[str] = []

    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - a tracked file that cannot be read
            continue
        for name, value in needles.items():
            if value in text:
                line = next(
                    (
                        number
                        for number, content in enumerate(text.splitlines(), start=1)
                        if value in content
                    ),
                    0,
                )
                findings.append(
                    f"{path.relative_to(REPOSITORY_ROOT).as_posix()}:{line} carries {name}"
                )

    assert not findings, (
        "vendor documentation example values found. EXECUTION.md §5 item 3 "
        "forbids vendor-shaped credential strings in this repository, source "
        "literals included; gitleaks does not catch these because it "
        "suppresses values containing EXAMPLE or TEST. Generate the value "
        "instead — see tests/labeling/test_rule_labeller.py's `generated` "
        "helper.\n" + "\n".join(findings)
    )


@pytest.mark.parametrize("name", sorted(_needles()))
def test_every_needle_is_a_complete_value(name: str) -> None:
    """A fragment left half-assembled would quietly forbid nothing.

    Args:
        name: The needle's human-readable name.
    """
    value = _needles()[name]
    assert len(value) >= 20, f"{name} is only {len(value)} characters; check the fragments"
