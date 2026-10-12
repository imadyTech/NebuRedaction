"""Slice definitions (NR-0002 §2.1 item 2 and §2.3A item 2).

A single headline number hides everything that matters about a detector. These
slices are the questions the project actually needs answered:

* **held-out families** — does it generalise to formats it has never seen? This
  is the question plan §5.3 held families out to ask.
* **ambiguous** — how does it do where the value genuinely cannot settle the
  label (plan §3.3)? Reported separately because including it in the headline
  mixes "wrong" with "undecidable".
* **adversarial** — the non-secrets sitting under credential-suggesting names.
  This is where NR-0000's teacher scored a 0.972-confidence false positive, so
  it is the slice the project exists to get right.
* **key_recovery** — for CredData, 59% of units carry the whole line as their
  value and 7% have no recovered key name. Those are different input shapes from
  attribute mode, and averaging over them hides which shape a system handles.
* **source** — synthetic, own and third-party data measured apart, because a
  system tuned on one should not be credited for the other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from datagen.unit import DetectionUnit


@dataclass(frozen=True, slots=True)
class Slice:
    """A named subset of a test set.

    Attributes:
        name: Name used in the report.
        predicate: True for units belonging to the slice.
        description: One line on what the slice asks.
        needs_negatives: True when a false-positive rate is meaningful for this
            slice. A slice of only secrets has no negatives, so its FPR would be
            0 by construction and reporting it would be misleading.
    """

    name: str
    predicate: Callable[[DetectionUnit], bool]
    description: str
    needs_negatives: bool = True


def _is_held_out(unit: DetectionUnit) -> bool:
    """True when the unit's family was held out of training."""
    return bool(unit.meta.get("held_out_family"))


def _is_adversarial(unit: DetectionUnit) -> bool:
    """True when the unit is a non-secret under a credential-suggesting name."""
    return bool(unit.meta.get("adversarial"))


def _is_ambiguous(unit: DetectionUnit) -> bool:
    """True when the value alone cannot settle the label."""
    return bool(unit.ambiguous)


def _is_multiline_value(unit: DetectionUnit) -> bool:
    """True when the unit's value spans several lines."""
    return "\n" in unit.value


#: Slices computed for every test set where the relevant metadata exists.
STANDARD_SLICES: tuple[Slice, ...] = (
    Slice("all", lambda unit: True, "The whole test set."),
    Slice(
        "held_out_families",
        _is_held_out,
        "Units from format families excluded from training: generalisation to unseen formats "
        "(plan §5.3).",
        needs_negatives=True,
    ),
    Slice(
        "seen_families",
        lambda unit: unit.family is not None and not _is_held_out(unit),
        "Units from families training has seen, for comparison with the held-out slice.",
    ),
    Slice(
        "ambiguous",
        _is_ambiguous,
        "Units where the value alone cannot settle the label (plan §3.3); labelled by the "
        "bias-toward-redaction convention.",
        needs_negatives=False,
    ),
    Slice(
        "adversarial_negatives",
        _is_adversarial,
        "Non-secrets under a credential-suggesting key name or template. The slice NR-0000's "
        "teacher probe failed.",
        needs_negatives=True,
    ),
    Slice(
        "multiline_values",
        _is_multiline_value,
        "Values spanning several lines, such as PEM blocks.",
    ),
)


def key_recovery_slices(units: Sequence[DetectionUnit]) -> list[Slice]:
    """Build one slice per key-recovery mode present in a test set.

    Args:
        units: The test set.

    Returns:
        A slice per distinct ``meta.key_recovery`` value, or an empty list when
        the test set does not carry that metadata (the synthetic corpus does
        not — every synthetic unit has a real key name by construction).
    """
    modes = sorted(
        {str(unit.meta.get("key_recovery")) for unit in units if unit.meta.get("key_recovery")}
    )
    explanations = {
        "assignment": 'Key name recovered from an assignment (KEY=, "key":) — closest to '
        "attribute mode.",
        "option": "Key name recovered from a command-line option (--key).",
        "multiline_fallback": "No usable value span, so the whole line is the value. A different "
        "input shape from attribute mode.",
        "defaulted": "No key name could be recovered; the unit carries <slot:value>.",
    }
    return [
        Slice(
            f"key_recovery_{mode}",
            lambda unit, mode=mode: unit.meta.get("key_recovery") == mode,
            explanations.get(mode, f"Units whose key name was recovered by {mode}."),
        )
        for mode in modes
    ]


def source_slices(units: Sequence[DetectionUnit]) -> list[Slice]:
    """Build one slice per ``source`` present in a test set.

    Args:
        units: The test set.

    Returns:
        A slice per distinct source, or an empty list when the test set is from a
        single source (in which case the ``all`` slice already answers it).
    """
    sources = sorted({unit.source for unit in units})
    if len(sources) < 2:
        return []
    return [
        Slice(
            f"source_{source.replace(':', '_')}",
            lambda unit, source=source: unit.source == source,
            f"Units whose source is {source}.",
        )
        for source in sources
    ]


def slices_for(units: Sequence[DetectionUnit]) -> list[Slice]:
    """Return every applicable slice for a test set, dropping the empty ones.

    An empty slice is reported as absent rather than as a row of zeros: zeros
    read as "it scored nothing", which is a different statement from "there was
    nothing to score".

    Args:
        units: The test set.

    Returns:
        The applicable slices, in a stable order.
    """
    candidates = [*STANDARD_SLICES, *key_recovery_slices(units), *source_slices(units)]
    return [s for s in candidates if any(s.predicate(unit) for unit in units)]
