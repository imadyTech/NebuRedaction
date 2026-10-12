"""The unified data loader, and the source restriction it enforces.

EXECUTION.md §5 item 2 requires that training and labeling read data only
through one loader, and that the loader refuse third-party data. The point is to
make the rule structural rather than advisory: a rule written in a document is
followed by whoever has read the document, while a loader that raises is followed
by everybody.

Two independent checks, because either alone can be evaded by accident:

* **Path** — anything under a ``third_party`` directory is refused before the
  file is opened, so a mislabelled file cannot get in on its ``source`` field.
* **Record** — every record's ``source`` is checked as it is read, so a
  third-party record copied into a ``synth`` file is refused too.

The evaluation path uses :func:`load_for_evaluation`, which permits every
source. That asymmetry is the whole design: evaluation is allowed to see
CredData, training is not, and the two entry points make which is which
impossible to confuse.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from datagen.unit import SOURCE_OWN, SOURCE_SYNTH, DetectionUnit, is_third_party

#: Sources the training and labeling paths may read (plan §5.1 items 1 and 2).
TRAINING_SOURCES: frozenset[str] = frozenset({SOURCE_SYNTH, SOURCE_OWN})

#: Directory name that marks third-party data on disk.
THIRD_PARTY_DIRECTORY = "third_party"


class ThirdPartyDataError(RuntimeError):
    """Raised when third-party data reaches a path that must not see it.

    Deliberately not a subclass of ``ValueError``: this is a policy violation,
    not a malformed input, and a caller catching ``ValueError`` to skip bad
    records must not swallow it.
    """


def _refuse_third_party_path(path: Path) -> None:
    """Refuse a path that lies under a third-party data directory.

    Args:
        path: The path about to be read.

    Raises:
        ThirdPartyDataError: If any component of the path is the third-party
            directory.
    """
    parts = {part.lower() for part in path.resolve().parts}
    if THIRD_PARTY_DIRECTORY in parts:
        raise ThirdPartyDataError(
            f"refusing to read {path} on a training or labeling path: it is under "
            f"{THIRD_PARTY_DIRECTORY!r}. Third-party data is for evaluation only "
            "(plan §5.1, EXECUTION.md §5 item 2). Use load_for_evaluation() if this "
            "really is the evaluation path."
        )


def _parse_line(path: Path, number: int, text: str) -> DetectionUnit:
    """Parse one JSON Lines record into a unit.

    Args:
        path: File being read, for the error message.
        number: 1-based line number, for the error message.
        text: The line's text.

    Returns:
        The parsed unit.

    Raises:
        ValueError: If the line is not a valid unit.
    """
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{number}: not valid JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{number}: expected a JSON object")
    try:
        return DetectionUnit.from_dict(record)
    except KeyError as exc:
        raise ValueError(f"{path}:{number}: missing required field {exc.args[0]!r}") from exc


def iter_units(path: Path) -> Iterator[DetectionUnit]:
    """Read units from a JSON Lines file with no source restriction at all.

    This is the low-level reader. Callers in training or labeling must use
    :func:`load_for_training` instead; this one exists for evaluation, for
    conversion tools and for tests.

    Args:
        path: File to read.

    Yields:
        Each unit in file order.
    """
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text:
                yield _parse_line(path, number, text)


def load_for_training(paths: Iterable[Path]) -> Iterator[DetectionUnit]:
    """Read units for training or labeling, refusing third-party data.

    Args:
        paths: Files to read, in order.

    Yields:
        Each unit, in file order.

    Raises:
        ThirdPartyDataError: If a path is under a third-party directory, or if
            any record's ``source`` is a third-party source.
    """
    for path in paths:
        _refuse_third_party_path(path)
        for unit in iter_units(path):
            if is_third_party(unit.source) or unit.source not in TRAINING_SOURCES:
                raise ThirdPartyDataError(
                    f"{path}: unit {unit.id!r} has source {unit.source!r}, which is not "
                    f"permitted for training or labeling (allowed: {sorted(TRAINING_SOURCES)}). "
                    "Third-party data must not enter the training set (plan §5.1)."
                )
            yield unit


def load_for_evaluation(paths: Iterable[Path]) -> Iterator[DetectionUnit]:
    """Read units for evaluation, where every source is permitted.

    Args:
        paths: Files to read, in order.

    Yields:
        Each unit, in file order.
    """
    for path in paths:
        yield from iter_units(path)
