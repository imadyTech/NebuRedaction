"""Convert CredData into detection units for evaluation (NR-0001 §2.4 items 2 and 3).

Reads CredData's ``meta`` CSVs plus the files its pipeline materialised, and
writes ``data/third_party/eval_v0.jsonl`` with ``source: third_party:creddata``.

Three things make this more than a format change.

**The key name has to be recovered.** CredData marks where the credential value
sits on a line but not what it was called, and plan §3.3 is explicit that the
key name is a necessary feature. So the text to the left of the value is parsed
for the nearest identifier — the ``KEY=`` or ``"key":`` or ``--key`` it was
assigned to. That is a heuristic, it fails on some lines, and when it fails the
unit records ``<slot:value>`` rather than inventing a name.

**Labels have to be mapped, not translated.** CredData's categories are a
colon-joined set and its taxonomy is finer than the seven classes; the mapping
file beside this one resolves them and records what it drops.

**Nothing here may ever reach training.** The output carries a third-party
``source`` and lives under ``data/third_party/``, so both halves of the loader
guard (:mod:`datagen.loader`) refuse it on a training path.

Usage:
    python corpus/convert_creddata.py
    python corpus/convert_creddata.py --limit 2000   # quick check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from datagen.unit import DetectionUnit  # noqa: E402

CREDDATA_ROOT = REPOSITORY_ROOT / "data" / "third_party" / "creddata"
META_DIR = CREDDATA_ROOT / "CredData" / "meta"
DATASET_DIR = CREDDATA_ROOT / "data"
MAPPING_PATH = Path(__file__).parent / "mappings" / "creddata.yaml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "third_party" / "eval_v0.jsonl"

SOURCE = "third_party:creddata"

#: Matches an identifier immediately to the left of a value, across the
#: assignment syntaxes that appear in configuration and source files:
#: ``KEY=``, ``key: ``, ``"key" =>``, ``--key ``, ``key": "``.
_KEY_BEFORE_VALUE = re.compile(
    r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_.:\-\[\]]{0,80})
    ["']?
    \s*
    (?: = | : | => | :: )
    \s*
    ["'(\[]?
    \s*
    $
    """,
    re.VERBOSE,
)

#: Matches a long-option style key: ``--password <value>`` or ``-p <value>``.
_OPTION_BEFORE_VALUE = re.compile(r"(?:^|\s)-{1,2}(?P<key>[A-Za-z][A-Za-z0-9_-]{0,40})\s+$")


@dataclass
class Stats:
    """Counters describing one conversion run.

    Attributes:
        meta_rows: Rows read from the meta CSVs.
        by_ground_truth: Rows per CredData ground-truth value.
        emitted: Units written, by label.
        dropped: Rows dropped, by reason.
        unmappable_categories: Categories that produced no class.
        excluded_categories: Categories excluded as non-secrets.
        missing_files: Referenced files that were not on disk.
        key_recovered: How often a key name was recovered versus defaulted.
    """

    meta_rows: int = 0
    by_ground_truth: Counter[str] = field(default_factory=Counter)
    emitted: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    unmappable_categories: Counter[str] = field(default_factory=Counter)
    excluded_categories: Counter[str] = field(default_factory=Counter)
    missing_files: Counter[str] = field(default_factory=Counter)
    key_recovered: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class Mapping:
    """The parsed category mapping.

    Attributes:
        priority: Classes in resolution order, most specific first.
        token_to_label: Each CredData category token's class.
        ignored: Tokens that contribute nothing.
        excluded: Tokens whose values are not secrets under plan §3.2.
        ambiguity: Tokens that mark a unit ambiguous.
        ground_truth: What to do with each ground-truth value.
    """

    priority: tuple[str, ...]
    token_to_label: dict[str, str]
    ignored: frozenset[str]
    excluded: frozenset[str]
    ambiguity: frozenset[str]
    ground_truth: dict[str, str]


def load_mapping(path: Path = MAPPING_PATH) -> Mapping:
    """Read the category mapping.

    Args:
        path: Mapping file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If a class appears in ``tokens`` but not in ``priority``, or
            a token is assigned to two classes. Either mistake would make the
            resolution order undefined and the labels arbitrary.
    """
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    priority = tuple(document["priority"])

    token_to_label: dict[str, str] = {}
    for label, tokens in (document.get("tokens") or {}).items():
        if label not in priority:
            raise ValueError(f"{path}: class {label!r} is not listed in 'priority'")
        for token in tokens or ():
            if token in token_to_label and token_to_label[token] != label:
                raise ValueError(
                    f"{path}: token {token!r} is mapped to both "
                    f"{token_to_label[token]!r} and {label!r}"
                )
            token_to_label[token] = label

    return Mapping(
        priority=priority,
        token_to_label=token_to_label,
        ignored=frozenset(document.get("ignored_tokens") or ()),
        excluded=frozenset(document.get("excluded_tokens") or ()),
        ambiguity=frozenset(document.get("ambiguity_tokens") or ()),
        ground_truth=dict(document.get("ground_truth") or {}),
    )


def resolve_label(category: str, mapping: Mapping) -> tuple[str | None, bool, str]:
    """Resolve a CredData category string to one of the seven classes.

    Args:
        category: The raw ``Category`` field, colon-joined.
        mapping: The parsed mapping.

    Returns:
        ``(label, ambiguous, reason)``. ``label`` is ``None`` when the row
        cannot be used, and ``reason`` then says whether it was excluded as a
        non-secret or was unmappable.
    """
    tokens = [token.strip() for token in category.split(":") if token.strip()]
    if not tokens:
        return None, False, "empty_category"

    labels = {mapping.token_to_label[token] for token in tokens if token in mapping.token_to_label}
    ambiguous = any(token in mapping.ambiguity for token in tokens)

    if labels:
        for candidate in mapping.priority:
            if candidate in labels:
                return candidate, ambiguous, "mapped"

    unknown = [
        token
        for token in tokens
        if token not in mapping.token_to_label
        and token not in mapping.ignored
        and token not in mapping.excluded
    ]
    if unknown:
        return None, ambiguous, "unmappable"
    if any(token in mapping.excluded for token in tokens):
        return None, ambiguous, "excluded_not_a_secret"
    return None, ambiguous, "ignored_only"


def recover_key(prefix: str) -> tuple[str, str]:
    """Recover the attribute name a value was assigned to.

    Args:
        prefix: The text on the line to the left of the value.

    Returns:
        ``(key, how)`` where ``how`` is ``assignment``, ``option`` or
        ``defaulted``. A defaulted key is reported as ``<slot:value>``: the
        body-mode form from plan §3.1, which is honest about there being no
        attribute name rather than guessing one.
    """
    trimmed = prefix.rstrip()
    match = _KEY_BEFORE_VALUE.search(trimmed)
    if match:
        return match.group("key"), "assignment"
    match = _OPTION_BEFORE_VALUE.search(prefix)
    if match:
        return match.group("key"), "option"
    return "<slot:value>", "defaulted"


def _read_line(path: Path, line_number: int) -> str | None:
    """Read one line of a file, tolerating undecodable bytes.

    Args:
        path: File to read.
        line_number: 1-based line number.

    Returns:
        The line without its terminator, or ``None`` if it does not exist.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if number == line_number:
                    return line.rstrip("\r\n")
    except OSError:
        return None
    return None


def _resolve_data_file(file_id: str, dataset_dir: Path) -> Path | None:
    """Find the materialised file for a meta row.

    CredData lays its data out as ``data/<scope>/<ext>/<FileID>`` where the
    scope and extension directories come from the original path. Searching by
    file id avoids having to reimplement that layout.

    Args:
        file_id: The row's ``FileID``.
        dataset_dir: CredData's materialised data directory.

    Returns:
        The file, or ``None`` if it is absent.
    """
    matches = list(dataset_dir.rglob(file_id))
    for match in matches:
        if match.is_file():
            return match
    return None


def build_file_index(dataset_dir: Path) -> dict[str, Path]:
    """Index the materialised data files by file id, once.

    CredData lays the extracted data out as
    ``data/<repo>/<scope>/<file_id>.<ext>``, so the name on disk is the file id
    *plus* its extension while the metadata's ``FileID`` column is the bare id.
    Indexing by stem as well as by name makes both forms resolve.

    This is only the fallback. The primary lookup uses the metadata's own
    ``FilePath`` column, which gives the exact relative path and so cannot
    collide between repositories that happen to share a file id.

    Args:
        dataset_dir: CredData's materialised data directory.

    Returns:
        A mapping of file name and stem to path. Built up front because a
        per-row ``rglob`` over tens of thousands of rows is quadratic and turns
        a one-minute conversion into an afternoon.
    """
    index: dict[str, Path] = {}
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def _offset(row: dict[str, str], field: str) -> int:
    """Read an offset column from a meta row, treating an empty cell as absent.

    Args:
        row: The meta row.
        field: Column name.

    Returns:
        The integer value, or ``-1`` when the cell is empty or unparseable.
        CredData leaves the value-span columns blank for rows that mark no
        credential, so "absent" is ordinary here rather than an error.
    """
    raw = (row.get(field) or "").strip()
    try:
        return int(raw)
    except ValueError:
        return -1


def convert(
    meta_dir: Path,
    dataset_dir: Path,
    mapping: Mapping,
    *,
    limit: int | None = None,
) -> tuple[list[DetectionUnit], Stats]:
    """Convert CredData's meta rows into detection units.

    Args:
        meta_dir: Directory of meta CSVs.
        dataset_dir: CredData's materialised data directory.
        mapping: The parsed category mapping.
        limit: Stop after this many meta rows, for a quick check.

    Returns:
        ``(units, stats)``.

    Raises:
        FileNotFoundError: If either directory is missing.
    """
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"{meta_dir} not found; fetch CredData first")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_dir} not found; run corpus/fetch_creddata.py to materialise the data"
        )

    # CredData's FilePath column is relative to the repository root and starts
    # with "data/", so paths resolve against the directory containing the
    # dataset rather than the dataset itself.
    dataset_root = dataset_dir.parent

    print(f"indexing {dataset_dir} ...")
    index = build_file_index(dataset_dir)
    print(f"  {len(index)} index entries")

    stats = Stats()
    units: list[DetectionUnit] = []

    for meta_path in sorted(meta_dir.glob("*.csv")):
        with meta_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if limit is not None and stats.meta_rows >= limit:
                    return units, stats
                stats.meta_rows += 1

                ground_truth = (row.get("GroundTruth") or "").strip()
                stats.by_ground_truth[ground_truth] += 1
                action = mapping.ground_truth.get(ground_truth, "drop")
                if action == "drop":
                    stats.dropped[f"ground_truth_{ground_truth or 'blank'}"] += 1
                    continue

                category = (row.get("Category") or "").strip()
                if action == "negative":
                    label: str | None = "none"
                    ambiguous = False
                else:
                    label, ambiguous, reason = resolve_label(category, mapping)
                    if label is None:
                        stats.dropped[reason] += 1
                        if reason == "unmappable":
                            stats.unmappable_categories[category] += 1
                        elif reason in ("excluded_not_a_secret", "ignored_only"):
                            stats.excluded_categories[category] += 1
                        continue

                file_id = (row.get("FileID") or "").strip()
                relative = (row.get("FilePath") or "").strip().replace("\\", "/")
                path = None
                if relative:
                    candidate = dataset_root / relative
                    if candidate.is_file():
                        path = candidate
                if path is None:
                    path = index.get(file_id)
                if path is None:
                    stats.dropped["file_not_on_disk"] += 1
                    stats.missing_files[relative or file_id] += 1
                    continue

                line_start = _offset(row, "LineStart")
                if line_start < 1:
                    stats.dropped["no_line_number"] += 1
                    continue
                # A negative row marks no credential, so it usually carries no
                # value span at all. That is not malformed data — the unit's
                # value is simply the whole line, which is what a logger would
                # have emitted. Dropping these would throw away most of the
                # negatives, and the negatives are what the external test set
                # is mostly for.
                value_start = _offset(row, "ValueStart")
                value_end = _offset(row, "ValueEnd")

                line = _read_line(path, line_start)
                if line is None:
                    stats.dropped["line_not_found"] += 1
                    continue

                if value_start < 0 or value_end <= value_start or value_end > len(line):
                    # Multi-line credentials (PEM blocks, AWS Multi) carry
                    # offsets that do not describe a single line. Fall back to
                    # the whole line, which is what a logger would have emitted.
                    value = line.strip()
                    prefix = ""
                    stats.key_recovered["multiline_fallback"] += 1
                    key, how = "<slot:value>", "defaulted"
                else:
                    value = line[value_start:value_end]
                    prefix = line[:value_start]
                    key, how = recover_key(prefix)
                    stats.key_recovered[how] += 1

                if not value.strip():
                    stats.dropped["empty_value"] += 1
                    continue

                unit = DetectionUnit(
                    id=f"creddata-{file_id}-{row.get('Id', stats.meta_rows)}",
                    key=key,
                    value=value,
                    source=SOURCE,
                    label=label,
                    template=None,
                    scope="third_party.file_line",
                    family=None,
                    ambiguous=True if ambiguous else None,
                    meta={
                        "source_file": file_id,
                        "source_line": line_start,
                        "original_label": category or ground_truth,
                        "ground_truth": ground_truth,
                        "predefined_pattern": (row.get("PredefinedPattern") or "").strip() or None,
                        "key_recovery": how,
                    },
                )
                units.append(unit)
                stats.emitted[label] += 1

    return units, stats


def main(argv: list[str] | None = None) -> int:
    """Convert CredData and write the evaluation corpus with its statistics.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` if the data has not been fetched.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=META_DIR, help="CredData meta directory")
    parser.add_argument("--data", type=Path, default=DATASET_DIR, help="CredData data directory")
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH, help="category mapping")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output JSON Lines file")
    parser.add_argument("--limit", type=int, default=None, help="stop after N meta rows")
    args = parser.parse_args(argv)

    mapping = load_mapping(args.mapping)
    try:
        units, stats = convert(args.meta, args.data, mapping, limit=args.limit)
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(unit.to_json() + "\n" for unit in units)
    args.out.write_text(payload, encoding="utf-8")

    report: dict[str, Any] = {
        "work_order": "NR-0001",
        "section": "2.4",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": SOURCE,
        "output": str(args.out).replace("\\", "/"),
        "meta_rows_read": stats.meta_rows,
        "by_ground_truth": dict(stats.by_ground_truth),
        "units_written": len(units),
        "by_label": dict(sorted(stats.emitted.items())),
        "ambiguous_units": sum(1 for unit in units if unit.ambiguous),
        "dropped": dict(sorted(stats.dropped.items())),
        "key_recovery": dict(sorted(stats.key_recovered.items())),
        "top_unmappable_categories": dict(stats.unmappable_categories.most_common(25)),
        "top_excluded_categories": dict(stats.excluded_categories.most_common(25)),
        "missing_file_count": len(stats.missing_files),
    }
    stats_path = args.out.with_name(args.out.stem + "-stats.json")
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{args.out}: {len(units)} units")
    print(f"{stats_path}: statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
