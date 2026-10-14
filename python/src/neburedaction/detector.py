"""The inference API (NR-0005 §2.3, plan §7.4 and §8.3).

The Python reference implementation. Its semantics are the ones the .NET
library reproduces, and plan §6.4's G3 requires the two to agree to within 1e-6
per sample, so anything a reimplementation could get subtly wrong is stated
here rather than left to the reader: the key-name comparison, the precedence
between the two override lists, the mask's digest, and what happens to a
message body.

**Experimental.** Plan §1.2 makes this release about being correct and
runnable, not about being good. NR-0004 measured G2 as not met, and the model
card says so; this library should not be relied on as a sole control.

Standard library only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from neburedaction.body import body_candidates
from neburedaction.nrm import NrmModel, load_nrm

#: The two profiles plan §4.5 asks for, matching the thresholds a `.nrm` file
#: carries. ``Balanced`` is the default.
BALANCED = "balanced"
CONSERVATIVE = "conservative"

#: Length of the mask's digest, in hex characters.
MASK_DIGEST_LENGTH = 4


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """What the detector concluded about one unit.

    Attributes:
        probabilities: One probability per class, in the model's label order.
        category: The most likely class.
        should_redact: Whether the calibrated redaction score reached the
            profile's threshold.
        score: The calibrated redaction probability.
        reason: ``model``, ``always`` or ``never`` — which rule decided. An
            override does not change the probabilities, so without this a
            caller cannot tell a model decision from a configured one.
    """

    probabilities: dict[str, float]
    category: str
    should_redact: bool
    score: float
    reason: str = "model"


@dataclass
class SecretDetectorOptions:
    """Runtime options (plan §7.4's L1).

    Attributes:
        profile: ``balanced`` or ``conservative``.
        always_redact_keys: Attribute names always treated as secret.
        never_redact_keys: Attribute names never treated as secret.

    **Key names are compared case-insensitively, and ``never`` loses to
    ``always``.** Both are stated because both are arbitrary: a caller listing
    a name in each is contradicting themselves, and the library has to pick.
    It picks the safer reading — a name the caller has marked as a secret stays
    a secret — and says so rather than depending on which list is consulted
    first.
    """

    profile: str = BALANCED
    always_redact_keys: frozenset[str] = field(default_factory=frozenset)
    never_redact_keys: frozenset[str] = field(default_factory=frozenset)

    def normalised(self) -> SecretDetectorOptions:
        """Return a copy with the key lists lower-cased."""
        return SecretDetectorOptions(
            profile=self.profile,
            always_redact_keys=frozenset(name.lower() for name in self.always_redact_keys),
            never_redact_keys=frozenset(name.lower() for name in self.never_redact_keys),
        )


@dataclass(frozen=True, slots=True)
class _Unit:
    """A detection unit, as the model's scorer expects it."""

    key: str
    value: str
    template: str | None = None
    scope: str | None = None


def mask(value: str) -> str:
    """Mask a value as ``***#<fingerprint>``.

    Args:
        value: The value to mask.

    Returns:
        ``***#`` followed by the first four hex characters of the value's
        SHA-256.

    The fingerprint lets the same value be correlated across log lines without
    being exposed. Four hex characters is 16 bits, which is deliberately weak:
    it is enough to spot "this is the same token as the one in the previous
    line" and far too little to confirm a guess at the value, since roughly one
    value in 65,536 shares any fingerprint.
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"***#{digest[:MASK_DIGEST_LENGTH]}"


class SecretDetector:
    """Classifies log attributes as secret credentials or not.

    Load one with :meth:`load`, then call :meth:`classify`. An instance is
    read-only after construction and safe to share between threads.
    """

    def __init__(self, model: NrmModel, options: SecretDetectorOptions | None = None) -> None:
        """Build a detector around a loaded model.

        Args:
            model: A model from :func:`neburedaction.nrm.load_nrm`.
            options: Runtime options; the defaults are the balanced profile and
                no overrides.

        Raises:
            ValueError: If the profile names a threshold the model does not
                carry. A profile that silently fell back to another one would
                redact at a rate the caller did not ask for.
        """
        self._model = model
        self._options = (options or SecretDetectorOptions()).normalised()
        if self._options.profile not in model.thresholds:
            raise ValueError(
                f"profile {self._options.profile!r} is not in this model's thresholds "
                f"{sorted(model.thresholds)}"
            )
        self._threshold = model.thresholds[self._options.profile]

    @classmethod
    def load(
        cls, source: str | Path | BinaryIO, options: SecretDetectorOptions | None = None
    ) -> SecretDetector:
        """Load a detector from a `.nrm` file.

        Args:
            source: A path, or a readable binary stream.
            options: Runtime options.

        Returns:
            The detector.

        Raises:
            neburedaction.nrm.NrmError: If the file is not a `.nrm` this build
                can read.
        """
        if isinstance(source, (str, Path)):
            return cls(load_nrm(Path(source)), options)
        raise TypeError("loading from a stream needs a seekable path in this build; pass a path")

    @property
    def labels(self) -> tuple[str, ...]:
        """The class order this model reports probabilities in."""
        return self._model.labels

    @property
    def threshold(self) -> float:
        """The calibrated-probability threshold the active profile uses."""
        return self._threshold

    def classify(
        self,
        key: str,
        value: str,
        template: str | None = None,
        scope: str | None = None,
    ) -> DetectionResult:
        """Classify one log attribute.

        Args:
            key: The attribute name.
            value: The attribute value.
            template: The message template the attribute appeared in, if any.
            scope: The scope name, if any.

        Returns:
            The result.

        The override lists are consulted **after** scoring rather than instead
        of it, so an overridden unit still carries real probabilities and a
        caller can see what the model thought. The alternative — skipping the
        model — would make an override invisible in any diagnostic.
        """
        unit = _Unit(key=key, value=value, template=template, scope=scope)
        probabilities = self._model.class_probabilities(unit)
        score = self._model.redaction_score(probabilities)
        best = max(range(len(probabilities)), key=lambda index: probabilities[index])

        lowered = key.lower()
        if lowered in self._options.always_redact_keys:
            decision, reason = True, "always"
        elif lowered in self._options.never_redact_keys:
            decision, reason = False, "never"
        else:
            decision, reason = score >= self._threshold, "model"

        return DetectionResult(
            probabilities=dict(zip(self._model.labels, probabilities, strict=True)),
            category=self._model.labels[best],
            should_redact=decision,
            score=score,
            reason=reason,
        )

    def classify_message(self, message: str) -> list[tuple[str, DetectionResult]]:
        """Classify the candidate fragments of a rendered message body.

        Args:
            message: The rendered message.

        Returns:
            ``(fragment, result)`` for each candidate, in the order they appear.

        Body mode is attribute mode applied to fragments: the tokenizer decides
        what is a candidate, and each candidate is classified exactly as an
        attribute would be, with the whole message as its template. The
        tokenizer is NR-0003 §2.0's, which rejects a fragment with no key name,
        no delimiter and no value-like shape — 1,510 of NebuLog's 1,545
        original body units were prose words.
        """
        results: list[tuple[str, DetectionResult]] = []
        for candidate in body_candidates(message):
            results.append(
                (
                    candidate.value,
                    self.classify(
                        candidate.key, candidate.value, template=message, scope=candidate.scope
                    ),
                )
            )
        return results
