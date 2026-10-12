"""The detection unit: the single record format shared by every dataset.

The format is defined by work order NR-0001 §2.1 and follows plan §7.3, so the
same shape serves the synthetic corpus, the CredData evaluation corpus, the
NebuLog own-log corpus and the data a user supplies for self-training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

#: The seven v0.1 classes, from plan §3.2. ``none`` means "not a secret".
LABELS: tuple[str, ...] = (
    "none",
    "password",
    "token",
    "api_key",
    "private_key",
    "connection_string",
    "cookie",
)

#: Labels that mean the value must be redacted.
SECRET_LABELS: frozenset[str] = frozenset(LABELS) - {"none"}

#: Permitted ``source`` values. Third-party sources are suffixed with their
#: dataset name, e.g. ``third_party:creddata``.
SOURCE_SYNTH = "synth"
SOURCE_OWN = "own"
THIRD_PARTY_PREFIX = "third_party:"

Label = Literal[
    "none", "password", "token", "api_key", "private_key", "connection_string", "cookie"
]


@dataclass(slots=True)
class DetectionUnit:
    """One attribute of one log record, with the context needed to classify it.

    Attributes:
        id: Identifier, unique within a dataset.
        key: The attribute name. Fragments cut out of a message body use the
            ``<slot:...>`` form, since they have no attribute name of their own.
        value: The attribute value, always a string.
        source: ``synth``, ``own``, or ``third_party:<name>``.
        label: One of :data:`LABELS`, or ``None`` for unlabelled data.
        template: The message template the attribute appeared in, if any.
        scope: The scope or origin type, such as ``otel.attribute`` or
            ``config.env``.
        family: The format family, such as ``aws.access_key_id`` or
            ``neg.uuid_v4``. Splits are made by family (plan §5.3).
        ambiguous: True when the value alone cannot settle the label
            (plan §3.3).
        meta: Provenance — generator version, seed, original line number.
    """

    id: str
    key: str
    value: str
    source: str
    label: str | None = None
    template: str | None = None
    scope: str | None = None
    family: str | None = None
    ambiguous: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the on-disk mapping, omitting absent optional fields.

        Returns:
            A mapping with the required fields always present and each optional
            field present only when it carries information. Omitting empty
            optionals keeps the JSON Lines files readable and keeps byte-for-byte
            reproducibility from depending on default values.
        """
        record: dict[str, Any] = {"id": self.id, "key": self.key, "value": self.value}
        if self.label is not None:
            record["label"] = self.label
        if self.template is not None:
            record["template"] = self.template
        if self.scope is not None:
            record["scope"] = self.scope
        if self.family is not None:
            record["family"] = self.family
        if self.ambiguous is not None:
            record["ambiguous"] = self.ambiguous
        record["source"] = self.source
        if self.meta:
            record["meta"] = self.meta
        return record

    def to_json(self) -> str:
        """Serialise to one JSON Lines line.

        Returns:
            A single-line JSON object with sorted-free, insertion-ordered keys
            and no ASCII escaping, so non-ASCII sample values stay legible.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> DetectionUnit:
        """Build a unit from a parsed JSON object.

        Args:
            record: A mapping in the on-disk format.

        Returns:
            The corresponding unit.

        Raises:
            KeyError: If a required field is missing.
        """
        return cls(
            id=record["id"],
            key=record["key"],
            value=record["value"],
            source=record["source"],
            label=record.get("label"),
            template=record.get("template"),
            scope=record.get("scope"),
            family=record.get("family"),
            ambiguous=record.get("ambiguous"),
            meta=record.get("meta", {}),
        )

    @property
    def should_redact(self) -> bool:
        """Whether this unit's label calls for redaction.

        Returns:
            True when the label is a secret class. Unlabelled units return
            False; absence of a label is not evidence either way, so callers
            that care must check :attr:`label` for ``None`` themselves.
        """
        return self.label in SECRET_LABELS


def is_third_party(source: str) -> bool:
    """Report whether a ``source`` value names a third-party dataset.

    Args:
        source: A unit's ``source`` field.

    Returns:
        True if the source is third-party, and therefore barred from training
        and labeling paths (EXECUTION.md §5 item 2).
    """
    return source.startswith(THIRD_PARTY_PREFIX)
