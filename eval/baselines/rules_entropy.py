"""Control group 2: rules plus entropy, emitting a continuous score (NR-0002 §2.2).

Modelled on detect-secrets' approach (Apache-2.0, attributed in
``THIRD_PARTY_NOTICES.md``): where a value has no recognisable vendor format,
judge it by how random it looks. detect-secrets applies a Shannon-entropy
threshold to base64 and hexadecimal strings; this does the same, but emits a
score rather than a decision so the framework can choose an operating point for
a target false-positive rate.

What the score is built from, and why each part is there:

* a **rule hit** scores 1.0 — a matched vendor format is the strongest evidence
  available and nothing should outrank it;
* otherwise the score combines **entropy relative to the alphabet's maximum**,
  **length**, and a **key-name hint**, each contributing a bounded amount.

The key-name hint is deliberately small (0.25). It has to be: a hint large
enough to carry a value over the threshold on its own would make the system fail
the adversarial slice by construction, which is the mistake this whole project
is about. It is included at all because plan §3.3 is right that the key name is
necessary information — just not sufficient.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from datagen.unit import SECRET_LABELS, DetectionUnit

from eval.baselines.rules import VALUE_RULES, classify_key
from eval.system import SystemOutput

VERSION = "0.1.0"

#: Character classes whose entropy is worth measuring, with the alphabet size
#: used to normalise. detect-secrets measures base64 and hex strings; hex is
#: separated out because its 4 bits per character ceiling makes a raw entropy
#: threshold tuned for base64 reject every digest and every key alike.
_BASE64_CHARS = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_HEX_CHARS = re.compile(r"^[0-9a-fA-F]+$")

#: Minimum length before entropy is considered at all. Short strings have
#: unstable entropy estimates — "a1b2" scores high and means nothing.
_MIN_LENGTH = 16


def shannon_entropy(value: str) -> float:
    """Compute Shannon entropy in bits per character.

    Args:
        value: The string to measure.

    Returns:
        Entropy in bits per character, 0.0 for the empty string.
    """
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for character in value:
        counts[character] = counts.get(character, 0) + 1
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def normalised_entropy(value: str) -> float:
    """Return entropy as a fraction of the maximum for the value's alphabet.

    Normalising is what makes one threshold work across alphabets. A 64-character
    hex digest has about 4 bits per character and a 40-character base64 secret
    about 6, so an absolute threshold either lets every hex string through or
    rejects every one of them — and plan §3.3 names exactly that case as the
    hard one.

    Args:
        value: The string to measure.

    Returns:
        A value in [0, 1], or 0.0 when the string is too short or is not drawn
        from a measurable alphabet.
    """
    stripped = value.strip()
    if len(stripped) < _MIN_LENGTH:
        return 0.0
    if _HEX_CHARS.match(stripped):
        ceiling = math.log2(16)
    elif _BASE64_CHARS.match(stripped):
        ceiling = math.log2(64)
    else:
        return 0.0
    return min(1.0, shannon_entropy(stripped) / ceiling)


def length_factor(value: str) -> float:
    """Score a value's length on a bounded ramp.

    Args:
        value: The string to measure.

    Returns:
        0.0 below 16 characters, rising to 1.0 at 64 and flat above. Credentials
        are long, but past about 64 characters extra length says nothing more —
        and an unbounded term would make every PEM body outrank every API key.
    """
    length = len(value.strip())
    if length < _MIN_LENGTH:
        return 0.0
    return min(1.0, (length - _MIN_LENGTH) / (64 - _MIN_LENGTH))


@dataclass(slots=True)
class RulesPlusEntropy:
    """Control group 2: rule hits, falling back to an entropy heuristic.

    Attributes:
        name: System name in reports.
        kind: ``continuous`` — the framework picks the operating point.
        entropy_weight: Contribution of normalised entropy.
        length_weight: Contribution of the length ramp.
        key_hint_weight: Contribution of a sensitive key name. Small on purpose;
            see the module docstring.
    """

    name: str = "rules+entropy"
    kind: str = "continuous"
    entropy_weight: float = 0.55
    length_weight: float = 0.20
    key_hint_weight: float = 0.25

    def version(self) -> str:
        """Return the baseline's version."""
        return VERSION

    def score_unit(self, unit: DetectionUnit) -> dict[str, float]:
        """Score one unit across the classes.

        Args:
            unit: The unit to score.

        Returns:
            A mapping from class to score. A rule hit yields 1.0 on that class;
            otherwise the heuristic score is attached to the class the key name
            suggests, or to ``api_key`` as the generic high-entropy secret when
            the key name suggests nothing.
        """
        for rule in VALUE_RULES:
            if rule.pattern.search(unit.value):
                return {rule.label: 1.0}

        entropy = normalised_entropy(unit.value)
        length = length_factor(unit.value)
        suggested = classify_key(unit.key)
        hint = 1.0 if suggested is not None else 0.0

        score = (
            self.entropy_weight * entropy
            + self.length_weight * length
            + self.key_hint_weight * hint
        )
        if score <= 0.0:
            return {}

        label = suggested if suggested in SECRET_LABELS else "api_key"
        return {label: round(min(1.0, score), 6)}

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score a sequence of units.

        Args:
            units: The units to score.

        Returns:
            Continuous output.
        """
        return SystemOutput(
            name=self.name,
            kind="continuous",
            scores=[self.score_unit(unit) for unit in units],
            version=self.version(),
            notes={
                "entropy_weight": str(self.entropy_weight),
                "length_weight": str(self.length_weight),
                "key_hint_weight": str(self.key_hint_weight),
                "min_length": str(_MIN_LENGTH),
            },
        )
