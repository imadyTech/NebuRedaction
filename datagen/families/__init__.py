"""The format-family registry.

A *format family* is one recognisable shape a value can take — one vendor's API
key layout, one flavour of connection string, one kind of identifier that merely
looks like a secret. Families are the unit of dataset splitting: plan §5.3 holds
whole families out of training so the evaluation measures generalisation to
formats the model has never seen, rather than memorisation.

Each family declares the public documentation its format comes from
(NR-0001 §2.2 item 1). Where no vendor documentation exists — generic
high-entropy keys, hand-rolled session identifiers — the family says so rather
than citing something that does not describe it.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["REGISTRY", "Family", "families_for_label", "negative_families", "positive_families"]


@dataclass(frozen=True, slots=True)
class Family:
    """One format family and the generator that produces its values.

    Attributes:
        name: Dotted identifier, lower case, such as ``aws.access_key_id``.
            Negative families are prefixed ``neg.``.
        label: The class a value of this family belongs to (plan §3.2).
        generate: Produces one value from a seeded random source.
        doc_url: Where the format is documented publicly, or a short statement
            that it has no vendor documentation.
        note: One line on what the family covers.
        ambiguous: True when values of this family cannot be classified from the
            value alone (plan §3.3), so the key name decides and the label
            follows the bias-toward-redaction convention.
        multiline: True when values span several lines, as PEM blocks do.
        tags: Free-form markers used by the assembler, such as ``body`` for
            families that belong in message-body fragments.
    """

    name: str
    label: str
    generate: Callable[[random.Random], str]
    doc_url: str
    note: str = ""
    ambiguous: bool = False
    multiline: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_negative(self) -> bool:
        """Whether this family produces non-secret values.

        Returns:
            True for families whose label is ``none``.
        """
        return self.label == "none"


#: Every registered family, keyed by name. Populated by the modules imported at
#: the bottom of this file.
REGISTRY: dict[str, Family] = {}


def register(family: Family) -> Family:
    """Add a family to the registry.

    Args:
        family: The family to register.

    Returns:
        The same family, so this can be used as a declaration expression.

    Raises:
        ValueError: If the name is already taken, or if the ``neg.`` prefix
            disagrees with the label. A miswired family is invisible in the
            generated data until the model trains on it, so it is rejected here.
    """
    if family.name in REGISTRY:
        raise ValueError(f"duplicate family name: {family.name}")
    if family.name.startswith("neg.") != (family.label == "none"):
        raise ValueError(
            f"family {family.name!r} has label {family.label!r}: "
            "the 'neg.' prefix and the 'none' label must agree"
        )
    REGISTRY[family.name] = family
    return family


def positive_families() -> list[Family]:
    """Return every family that produces a secret value.

    Returns:
        The families whose label is not ``none``, in registration order.
    """
    return [family for family in REGISTRY.values() if not family.is_negative]


def negative_families() -> list[Family]:
    """Return every family that produces a non-secret value.

    Returns:
        The families whose label is ``none``, in registration order.
    """
    return [family for family in REGISTRY.values() if family.is_negative]


def families_for_label(label: str) -> list[Family]:
    """Return the families belonging to one class.

    Args:
        label: One of the seven classes.

    Returns:
        The matching families, in registration order.
    """
    return [family for family in REGISTRY.values() if family.label == label]


# Importing these modules is what populates REGISTRY. The imports sit at the
# bottom because each module imports `register` from here.
from datagen.families import (  # noqa: E402  (circular-import ordering is deliberate)
    ambiguous,
    api_keys,
    connection_strings,
    cookies,
    negatives,
    passwords,
    private_keys,
    tokens,
)

#: The family modules, exposed so tests can assert that every one is loaded.
MODULES = (
    api_keys,
    tokens,
    connection_strings,
    passwords,
    private_keys,
    cookies,
    negatives,
    ambiguous,
)
