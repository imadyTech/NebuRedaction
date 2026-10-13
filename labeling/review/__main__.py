"""The review tool's terminal interface (NR-0003 §2.4).

Usage:
    python -m labeling.review --build      # build the queue, then stop
    python -m labeling.review              # review the queue
    python -m labeling.review --summary    # what has been decided so far

One page of instructions is in ``labeling/review/README.md``, as §2.4 item 4
requires.

The queue logic is in the package's ``__init__``; this module is only keyboard
and screen, so that the ordering, the cap and the record format can be tested
without a terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datagen.loader import load_for_training

from labeling.review import (
    OWN_SHARE,
    QUEUE_CAP,
    QueueItem,
    append_decision,
    build_queue,
    decide,
    latest_decisions,
    read_decisions,
    statistics,
)
from labeling.rules import read_labels

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

OWN_UNITS = REPOSITORY_ROOT / "data" / "own" / "own_v0.jsonl"
SYNTH_TRAIN = REPOSITORY_ROOT / "data" / "synth" / "v0" / "train.jsonl"
RULE_OWN = REPOSITORY_ROOT / "data" / "labels" / "rules-own.jsonl"
RULE_SYNTH = REPOSITORY_ROOT / "data" / "labels" / "rules-synth-train.jsonl"
TEACHER = REPOSITORY_ROOT / "data" / "labels" / "teacher_v0.jsonl"
CREDSWEEPER = REPOSITORY_ROOT / "data" / "labels" / "credsweeper-own.json"
CALIBRATION = REPOSITORY_ROOT / "data" / "labels" / "calibration.json"
QUEUE = REPOSITORY_ROOT / "data" / "labels" / "review-queue.jsonl"
DECISIONS = REPOSITORY_ROOT / "data" / "labels" / "review_v0.jsonl"

#: How much of a value to show before requiring an explicit expand. A PEM block
#: is about 1,700 characters and would scroll the key name and the template off
#: the screen, which are the two things the decision usually turns on.
VALUE_PREVIEW = 240

#: Single-key shortcuts for the seven classes. `n` for none is the common case
#: and sits under the index finger; the rest are first letters, with `c` taken
#: by cookie and connection_string distinguished as `s` for "string".
KEYS: dict[str, str] = {
    "n": "none",
    "p": "password",
    "t": "token",
    "a": "api_key",
    "k": "private_key",
    "s": "connection_string",
    "c": "cookie",
}


def read_key() -> str:
    """Read one keypress without waiting for Enter.

    Returns:
        The character pressed, lower-cased, or the whole line's first character
        where a raw read is unavailable.

    Falls back to line input when there is no console — in a pipe, or under
    pytest — so the tool degrades to Enter-per-decision rather than crashing.
    At eight to fourteen seconds per unit a raw read halves the keystrokes,
    which is why it is worth the platform branch.
    """
    try:
        import msvcrt

        return str(msvcrt.getwch()).lower()
    except (ImportError, AttributeError):
        pass
    try:
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        saved = termios.tcgetattr(descriptor)
        try:
            tty.setraw(descriptor)
            return sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    except Exception:
        line = sys.stdin.readline()
        return line[:1].lower() if line else "q"


def render(item: QueueItem, position: int, total: int, expanded: bool = False) -> str:
    """Render one queue item for the screen.

    Args:
        item: The item to show.
        position: One-based position in the queue.
        total: Queue length.
        expanded: Whether to show the whole value.

    Returns:
        The text to print.
    """
    value = item.unit.value
    if not expanded and len(value) > VALUE_PREVIEW:
        value = value[:VALUE_PREVIEW] + f"… [{len(value)} chars; press v for all]"
    value = value.replace("\n", "\\n") if not expanded else value

    ranked = sorted(item.teacher.items(), key=lambda pair: -pair[1])[:3]
    teacher_line = "  ".join(f"{label}={probability:.3f}" for label, probability in ranked)

    lines = [
        "",
        "=" * 78,
        f"[{position}/{total}]  {item.unit.id}   source={item.unit.source}"
        + (f"   family={item.unit.family}" if item.unit.family else ""),
        f"reason: {item.reason}" + ("   (DISAGREEMENT)" if item.disagreement else ""),
        "-" * 78,
        f"  key      : {item.unit.key}",
        f"  value    : {value}",
        f"  template : {item.unit.template or '(none)'}",
        f"  scope    : {item.unit.scope or '(none)'}",
        "-" * 78,
        f"  generator: {item.unit.label or '(unlabelled)'}",
        f"  rules    : {item.rule.label}  ({item.rule.confidence}, {item.rule.evidence})",
        f"  teacher  : {teacher_line}   entropy={item.entropy:.3f}",
    ]
    if item.credsweeper is not None:
        lines.append(f"  sweeper  : {'secret' if item.credsweeper else 'not a secret'}")
    lines += [
        "-" * 78,
        f"  current  : {item.current_label}",
        "",
        "  [Enter] confirm    " + "  ".join(f"[{key}] {label}" for key, label in KEYS.items()),
        "  [v] full value   [u] undo   [s] skip   [q] save and quit",
    ]
    return "\n".join(lines)


def load_queue(path: Path) -> list[dict[str, object]]:
    """Read a built queue file.

    Args:
        path: The queue file.

    Returns:
        The queue entries, in review order.

    Raises:
        FileNotFoundError: If the queue has not been built.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m labeling.review --build` first")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build(cap: int, own_share: float | None) -> int:
    """Build the review queue from the labelling outputs.

    Args:
        cap: Maximum queue length.
        own_share: Fraction of the cap reserved for own-log units, or None for
            §2.4 item 2's ordering applied literally. See `build_queue`.

    Returns:
        ``0`` on success, ``2`` when an input is missing.
    """
    missing = [
        path
        for path in (OWN_UNITS, SYNTH_TRAIN, RULE_OWN, RULE_SYNTH, TEACHER)
        if not path.is_file()
    ]
    if missing:
        for path in missing:
            print(f"FAILED: {path} not found", file=sys.stderr)
        return 2

    from labeling.teacher.calibration import apply_temperature

    temperature = 1.0
    if CALIBRATION.is_file():
        temperature = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["temperature"])
        print(f"applying fitted temperature {temperature:.4f}")
    else:
        print(
            "WARNING: no calibration.json; using raw probabilities. §2.2 item 6 requires "
            "recalibration before auto-adoption, so this queue is provisional.",
            file=sys.stderr,
        )

    teacher: dict[str, dict[str, float]] = {}
    with TEACHER.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            teacher[record["unit_id"]] = apply_temperature(record["probabilities"], temperature)

    credsweeper: dict[str, bool] = {}
    if CREDSWEEPER.is_file():
        credsweeper = {
            unit_id: bool(secret)
            for unit_id, secret in json.loads(CREDSWEEPER.read_text(encoding="utf-8")).items()
        }

    rules = {label.unit_id: label for label in read_labels(RULE_OWN)}
    rules.update({label.unit_id: label for label in read_labels(RULE_SYNTH)})
    units = list(load_for_training([OWN_UNITS, SYNTH_TRAIN]))

    queue = build_queue(units, rules, teacher, credsweeper, cap=cap, own_share=own_share)
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8", newline="\n") as handle:
        for item in queue:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    disagreements = sum(1 for item in queue if item.disagreement)
    reasons: dict[str, int] = {}
    for item in queue:
        reasons[item.reason] = reasons.get(item.reason, 0) + 1
    print(f"\n{QUEUE}: {len(queue)} items (cap {cap}), {disagreements} disagreements")
    for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]):
        print(f"  {count:>5}  {reason}")
    return 0


def review(reviewer: str) -> int:
    """Run the interactive review.

    Args:
        reviewer: Recorded against every decision.

    Returns:
        ``0`` on success, ``2`` when the queue is missing.
    """
    try:
        entries = load_queue(QUEUE)
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    from datagen.unit import DetectionUnit

    from labeling.rules import RuleLabel

    items = [
        QueueItem(
            unit=DetectionUnit(
                id=entry["unit_id"],
                key=entry["key"],
                value=entry["value"],
                source=entry["source"],
                label=entry["generator_label"],
                template=entry["template"],
                scope=entry["scope"],
                family=entry["family"],
            ),
            rule=RuleLabel(
                entry["unit_id"],
                entry["rule"]["label"],
                entry["rule"]["confidence"],
                entry["rule"]["evidence"],
                0.0,
            ),
            teacher=entry["teacher"]["probabilities"],
            teacher_choice=entry["teacher"]["choice"],
            credsweeper=entry["credsweeper_secret"],
            reason=entry["reason"],
            disagreement=entry["disagreement"],
            entropy=entry["teacher"]["entropy"],
        )
        for entry in entries
    ]

    done = latest_decisions(read_decisions(DECISIONS))
    pending = [index for index, item in enumerate(items) if item.unit.id not in done]
    if not pending:
        print(f"all {len(items)} items reviewed; see --summary")
        return 0

    print(
        f"\n{len(items)} in the queue, {len(done)} already decided, {len(pending)} to go.\n"
        "Enter confirms the current label. Keys are shown under each unit."
    )

    cursor = 0
    history: list[str] = []
    while cursor < len(pending):
        item = items[pending[cursor]]
        expanded = False
        while True:
            print(render(item, cursor + 1, len(pending), expanded))
            key = read_key()
            if key == "v":
                expanded = True
                continue
            break

        if key == "q":
            break
        if key == "u":
            if not history:
                print("  nothing to undo")
                continue
            previous = history.pop()
            cursor = next(
                index
                for index, position in enumerate(pending)
                if items[position].unit.id == previous
            )
            print(f"  back to {previous}")
            continue

        if key in ("\r", "\n"):
            decision = decide(item, reviewer, item.current_label)
        elif key == "s":
            decision = decide(item, reviewer, None)
        elif key in KEYS:
            decision = decide(item, reviewer, KEYS[key])
        else:
            print(f"  unknown key {key!r}")
            continue

        append_decision(decision, DECISIONS)
        history.append(item.unit.id)
        print(f"  -> {decision.verdict}: {decision.new_label}")
        cursor += 1

    remaining = len(pending) - cursor
    print(f"\nsaved to {DECISIONS}. {remaining} left; re-run to continue.")
    return 0


def summarise() -> int:
    """Print the review statistics.

    Returns:
        ``0`` always.
    """
    resolved = latest_decisions(read_decisions(DECISIONS))
    print(json.dumps(statistics(resolved.values()), indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build, review or summarise.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build the queue and stop")
    parser.add_argument("--summary", action="store_true", help="show what has been decided")
    parser.add_argument("--cap", type=int, default=QUEUE_CAP, help="maximum queue length")
    parser.add_argument(
        "--own-share",
        type=float,
        default=OWN_SHARE,
        help="fraction of the queue reserved for own-log units",
    )
    parser.add_argument(
        "--no-own-share",
        action="store_true",
        help="apply §2.4 item 2's ordering literally across both sources",
    )
    parser.add_argument("--reviewer", default="frank", help="recorded against every decision")
    args = parser.parse_args(argv)

    if args.build:
        return build(args.cap, None if args.no_own_share else args.own_share)
    if args.summary:
        return summarise()
    return review(args.reviewer)


if __name__ == "__main__":
    raise SystemExit(main())
