"""Deterministic robustness transforms (NR-0002 §2.1 item 2, plan §6.2).

A detector that only recognises a credential in its canonical form is of limited
use: real logs wrap values in base64, change their case, glue punctuation to
them and truncate them at a field limit. These transforms produce those variants
from the test set so each can be reported as its own slice.

Two rules govern every transform here.

**The label is preserved.** A base64-wrapped AWS key is still an AWS key and
still must be redacted; a truncated private key still leaks key material. If a
transform could turn a secret into a non-secret it does not belong here, and
none of these do.

**The transform is a pure function of the value.** No randomness, so a slice is
reproducible and ``report.json`` is stable across runs (NR-0002 §2.1 item 4).
Where a transform needs to pick between options it derives the choice from a
hash of the value rather than from a random source.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from datagen.unit import DetectionUnit

#: Punctuation and filler that real log lines put next to a value.
_PREFIXES = ('"', "'", "(", "[", "=", ": ", "-> ", "value=")
_SUFFIXES = ('"', "'", ")", "]", ",", ";", " ...", "\\n")


def _stable_choice(value: str, options: Sequence[str]) -> str:
    """Pick an option deterministically from a value.

    Args:
        value: The value being transformed.
        options: Options to choose from.

    Returns:
        One option, chosen by hashing the value so the same value always gets
        the same decoration.
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def identity(value: str) -> str:
    """Return the value unchanged, for the baseline slice."""
    return value


def base64_wrap(value: str) -> str:
    """Base64-encode the value.

    A real case: configuration systems and Kubernetes secrets store values
    base64-encoded, so the credential reaches the log already wrapped. Every
    vendor prefix a rule keys on is destroyed by the encoding, which is exactly
    what makes this slice informative.

    Args:
        value: The value to wrap.

    Returns:
        The standard-alphabet base64 encoding.
    """
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def case_flip(value: str) -> str:
    """Swap the case of every letter.

    Args:
        value: The value to alter.

    Returns:
        The value with upper and lower case exchanged. This breaks
        case-sensitive vendor prefixes (``AKIA``, ``ghp_``, ``SG.``) while
        leaving length, entropy and character classes almost untouched — so it
        separates rules keyed on a literal prefix from heuristics keyed on shape.
    """
    return value.swapcase()


def affix_noise(value: str) -> str:
    r"""Attach plausible punctuation to both ends.

    Args:
        value: The value to decorate.

    Returns:
        The value with a prefix and suffix drawn deterministically from the
        punctuation that surrounds values in real log lines. A rule anchored
        with ``\\b`` survives this; one anchored to start-of-field does not.
    """
    return _stable_choice(value, _PREFIXES) + value + _stable_choice(value, _SUFFIXES)


def truncate(value: str) -> str:
    """Cut the value to 80% of its length, to a minimum of eight characters.

    Args:
        value: The value to truncate.

    Returns:
        The shortened value. Logging frameworks and storage layers impose field
        limits — NebuLog's own server caps an attribute at 8,192 characters — so
        a long credential can arrive cut off. The prefix survives, the length and
        any checksum do not.
    """
    if len(value) <= 8:
        return value
    return value[: max(8, int(len(value) * 0.8))]


@dataclass(frozen=True, slots=True)
class Transform:
    """A named robustness transform.

    Attributes:
        name: Slice name used in the report.
        apply: The pure function applied to each value.
        description: One line on what it simulates and why it is hard.
    """

    name: str
    apply: Callable[[str], str]
    description: str


#: The transforms the report covers. ``canonical`` is included so every slice is
#: compared against the untransformed set computed by the identical code path.
TRANSFORMS: tuple[Transform, ...] = (
    Transform("canonical", identity, "The test set as generated, for comparison."),
    Transform(
        "base64_wrapped",
        base64_wrap,
        "Value base64-encoded, as configuration systems and Kubernetes secrets store it. "
        "Destroys every vendor prefix.",
    ),
    Transform(
        "case_flipped",
        case_flip,
        "Case swapped. Breaks case-sensitive prefixes while preserving length, entropy and "
        "character classes.",
    ),
    Transform(
        "affix_noise",
        affix_noise,
        "Quotes, brackets and separators attached to both ends, as in a real log line.",
    ),
    Transform(
        "truncated",
        truncate,
        "Cut to 80% of its length, as a field limit would. Prefix survives; length and any "
        "checksum do not.",
    ),
)


def apply_transform(units: Sequence[DetectionUnit], transform: Transform) -> list[DetectionUnit]:
    """Apply a transform to every unit's value, preserving labels.

    Args:
        units: The units to transform.
        transform: The transform to apply.

    Returns:
        New units with transformed values. Ids are suffixed with the transform
        name so a finding can be traced back, and ``meta.transform`` records
        which slice the unit belongs to.
    """
    from dataclasses import replace

    if transform.name == "canonical":
        return list(units)
    return [
        replace(
            unit,
            id=f"{unit.id}#{transform.name}",
            value=transform.apply(unit.value),
            meta={**unit.meta, "transform": transform.name},
        )
        for unit in units
    ]
