"""Validate a detection-unit JSON Lines file against the unit schema.

Usage:
    python -m datagen.validate data/synth/v0/train.jsonl [more.jsonl ...]

Beyond the schema, this checks the invariants a schema cannot express: ids are
unique within a file, and a unit's label must be consistent with its family's
polarity (a ``neg.*`` family cannot carry a secret label, and a positive family
cannot be labelled ``none``). Those two rules catch the mistakes that actually
happen when a family is added — a miswired label is invisible to the schema but
poisons the training set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datagen.unit import SECRET_LABELS

#: Path of the schema, resolved relative to this module so the CLI works from
#: any working directory.
SCHEMA_PATH = Path(__file__).parent / "schema" / "unit.schema.json"

#: Prefix marking a negative (non-secret) format family.
NEGATIVE_FAMILY_PREFIX = "neg."


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    """Read the detection-unit JSON Schema.

    Args:
        path: Schema location. Defaults to the schema shipped with this package.

    Returns:
        The parsed schema.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield the non-blank lines of a file with their 1-based line numbers.

    Args:
        path: File to read.

    Yields:
        ``(line_number, text)`` for each line that is not blank.
    """
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text:
                yield number, text


def check_label_polarity(record: dict[str, Any]) -> str | None:
    """Check that a unit's label agrees with its family's polarity.

    Args:
        record: A parsed unit.

    Returns:
        An error message, or ``None`` if the unit is consistent. Units without
        a family or without a label are consistent by default: unlabelled data
        is legitimate, and third-party units need not carry a family.
    """
    family = record.get("family")
    label = record.get("label")
    if family is None or label is None:
        return None
    if family.startswith(NEGATIVE_FAMILY_PREFIX) and label in SECRET_LABELS:
        return f"negative family {family!r} carries secret label {label!r}"
    if not family.startswith(NEGATIVE_FAMILY_PREFIX) and label == "none":
        return f"positive family {family!r} is labelled 'none'"
    return None


def validate_file(path: Path, validator: Any) -> list[str]:
    """Validate one JSON Lines file.

    Args:
        path: File to validate.
        validator: A ``jsonschema`` validator built from the unit schema.

    Returns:
        A list of human-readable problems, empty when the file is valid.
    """
    problems: list[str] = []
    seen_ids: dict[str, int] = {}

    for number, text in iter_lines(path):
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"{path}:{number}: not valid JSON: {exc.msg}")
            continue

        if not isinstance(record, dict):
            problems.append(f"{path}:{number}: expected a JSON object")
            continue

        for error in validator.iter_errors(record):
            location = "/".join(str(part) for part in error.absolute_path) or "(root)"
            problems.append(f"{path}:{number}: {location}: {error.message}")

        unit_id = record.get("id")
        if isinstance(unit_id, str):
            previous = seen_ids.get(unit_id)
            if previous is not None:
                problems.append(
                    f"{path}:{number}: duplicate id {unit_id!r} (first seen on line {previous})"
                )
            else:
                seen_ids[unit_id] = number

        polarity = check_label_polarity(record)
        if polarity is not None:
            problems.append(f"{path}:{number}: {polarity}")

    return problems


def main(argv: list[str] | None = None) -> int:
    """Validate every file named on the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` when every file is valid, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="JSON Lines files to validate")
    parser.add_argument(
        "--max-problems",
        type=int,
        default=25,
        help="stop printing after this many problems (0 for no limit)",
    )
    args = parser.parse_args(argv)

    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_schema())

    problems: list[str] = []
    checked = 0
    for path in args.files:
        if not path.is_file():
            problems.append(f"{path}: not a file")
            continue
        problems.extend(validate_file(path, validator))
        checked += 1

    if not problems:
        print(f"ok: {checked} file(s) valid")
        return 0

    shown = problems if args.max_problems == 0 else problems[: args.max_problems]
    for problem in shown:
        print(problem, file=sys.stderr)
    if len(shown) < len(problems):
        print(f"... and {len(problems) - len(shown)} more", file=sys.stderr)
    print(f"FAILED: {len(problems)} problem(s) in {checked} file(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
