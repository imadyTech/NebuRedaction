"""Test-set definition and the lock file that pins them (NR-0002 §2.3 item 2).

The lock file records each test set's path, unit count and SHA-256. It exists so
that a later report cannot silently be computed against different data:
EXECUTION.md §8 forbids changing the test set after G1 without declaring it, and
a declaration is only meaningful if there is something to compare against.

Which sets exist, and why each one is or is not here:

* ``synth_test`` — the synthetic test split, including the held-out families.
  The main set.
* ``creddata`` — the external set. Used for evaluation only; never for choosing
  a threshold (EXECUTION.md §8).
* ``synth_val`` — the **selection** set. Listed in the lock because what the
  thresholds were chosen on is part of the method, not an implementation detail.
* own logs — **absent**. They are unlabelled until NR-0003, and NR-0002 §2.3A
  item 4 says so explicitly; including them would mean guessing labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datagen.loader import iter_units
from datagen.unit import DetectionUnit

from eval.evaluate import TestSet

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = Path(__file__).parent / "testsets.lock"

#: The detection-unit schema version frozen by NR-0002 §2.3 item 1.
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class TestSetSpec:
    """Where a test set comes from and what it is for.

    Attributes:
        name: Name used in reports and the lock file.
        path: Path relative to the repository root.
        description: One line on what it can answer.
        role: ``selection`` for the set thresholds are chosen on, ``test``
            otherwise. The distinction is load-bearing, so it is data rather
            than a convention about names.
        robustness: Whether to compute the robustness slices for this set. True
            only for synthetic data: the transforms are synthetic, and applying
            them to CredData's already-real values would produce
            doubly-synthetic inputs whose numbers answer no question the project
            has — while quadrupling the slowest system's work.
    """

    name: str
    path: str
    description: str
    role: str = "test"
    robustness: bool = False


SPECS: tuple[TestSetSpec, ...] = (
    TestSetSpec(
        "synth_val",
        "data/synth/v0/val.jsonl",
        "Synthetic validation split. Operating points are chosen here and nowhere else.",
        role="selection",
    ),
    TestSetSpec(
        "synth_test",
        "data/synth/v0/test.jsonl",
        "Synthetic test split, including the ten held-out format families.",
        robustness=True,
    ),
    TestSetSpec(
        "creddata",
        "data/third_party/eval_v0.jsonl",
        "CredData, converted to detection units. Independent of the training distribution; "
        "evaluation only. 18.2% of CredData's labelled rows could not be acquired on this "
        "platform (NR-0001 §9.5), and it contains no cookie-class units.",
    ),
)


def file_digest(path: Path) -> str:
    """Return a file's SHA-256.

    Args:
        path: The file.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(spec: TestSetSpec, root: Path = REPOSITORY_ROOT) -> TestSet:
    """Load one test set from disk.

    Args:
        spec: Which set to load.
        root: Repository root.

    Returns:
        The loaded set.

    Raises:
        FileNotFoundError: If the file is absent, naming the work order that
            produces it.
    """
    path = root / spec.path
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The evaluation sets are produced by NR-0001; "
            "regenerate with datagen.generate and corpus/convert_creddata.py."
        )
    return TestSet(
        name=spec.name,
        units=list(iter_units(path)),
        description=spec.description,
        robustness=spec.robustness,
    )


def load_all(root: Path = REPOSITORY_ROOT) -> tuple[TestSet, list[TestSet]]:
    """Load the selection set and the test sets.

    Args:
        root: Repository root.

    Returns:
        ``(selection_set, test_sets)``.

    Raises:
        ValueError: If no selection set is defined, which would leave continuous
            systems with nowhere legitimate to choose a threshold.
    """
    loaded = {spec.name: load_spec(spec, root) for spec in SPECS}
    selection = [spec for spec in SPECS if spec.role == "selection"]
    if not selection:
        raise ValueError("no selection set is defined; thresholds would have nowhere to come from")
    selection_set = loaded[selection[0].name]
    test_sets = [loaded[spec.name] for spec in SPECS if spec.role == "test"]
    return selection_set, test_sets


def build_lock(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Compute the lock contents from the files on disk.

    Args:
        root: Repository root.

    Returns:
        The lock document.
    """
    entries = []
    for spec in SPECS:
        path = root / spec.path
        entries.append(
            {
                "name": spec.name,
                "path": spec.path,
                "role": spec.role,
                "robustness_slices": spec.robustness,
                "sha256": file_digest(path) if path.is_file() else None,
                "units": sum(1 for _ in path.open(encoding="utf-8") if _.strip())
                if path.is_file()
                else None,
                "description": spec.description,
            }
        )
    return {
        "lock_version": 1,
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Pins the evaluation sets. EXECUTION.md §8 forbids changing a test set or a metric "
            "definition after G1 without declaring it in the report; this file is what such a "
            "declaration is checked against. Own-log data is deliberately absent: it is "
            "unlabelled until NR-0003 (NR-0002 §2.3A item 4)."
        ),
        "testsets": entries,
    }


def write_lock(path: Path = LOCK_PATH, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Write the lock file.

    Args:
        path: Where to write.
        root: Repository root.

    Returns:
        The lock document that was written.
    """
    lock = build_lock(root)
    path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return lock


def read_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    """Read the lock file.

    Args:
        path: The lock file.

    Returns:
        The parsed lock, or an empty structure when absent.
    """
    if not path.is_file():
        return {"lock_version": 1, "schema_version": SCHEMA_VERSION, "testsets": []}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock(path: Path = LOCK_PATH, root: Path = REPOSITORY_ROOT) -> list[str]:
    """Check the data on disk against the lock.

    Args:
        path: The lock file.
        root: Repository root.

    Returns:
        A list of mismatches, empty when everything agrees. Returned rather than
        raised so the runner can record the drift in the report — which is what
        "declare it" means in practice — instead of refusing to run.
    """
    lock = read_lock(path)
    problems: list[str] = []
    by_name = {entry["name"]: entry for entry in lock.get("testsets", [])}
    for spec in SPECS:
        entry = by_name.get(spec.name)
        if entry is None:
            problems.append(f"{spec.name}: not in the lock file")
            continue
        file_path = root / spec.path
        if not file_path.is_file():
            problems.append(f"{spec.name}: {spec.path} is missing")
            continue
        actual = file_digest(file_path)
        if entry.get("sha256") and entry["sha256"] != actual:
            problems.append(
                f"{spec.name}: digest changed since the lock was written "
                f"(locked {entry['sha256'][:12]}…, actual {actual[:12]}…)"
            )
    return problems


def labelled_only(units: Sequence[DetectionUnit]) -> list[DetectionUnit]:
    """Drop unlabelled units.

    Args:
        units: The units to filter.

    Returns:
        Only the labelled units. An unlabelled unit cannot contribute to a
        metric, and counting it as a negative would inflate false-positive
        rates.
    """
    return [unit for unit in units if unit.label is not None]
