"""The key-name lexicon: loading and querying.

The data lives in ``keys.yaml`` beside this module, and that file documents
where every name came from and why the three groups exist. This package holds
both so that the data path stays the one NR-0001 §2.2 item 4 specifies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LEXICON_PATH = Path(__file__).parent / "keys.yaml"

#: How a key name relates to the value under it.
CONTEXT_HINTING = "hinting"
CONTEXT_NEUTRAL = "neutral"
CONTEXT_MISLEADING = "misleading"
CONTEXTS = (CONTEXT_HINTING, CONTEXT_NEUTRAL, CONTEXT_MISLEADING)


@dataclass(frozen=True, slots=True)
class Lexicon:
    """The key-name vocabulary, grouped by the signal a name carries.

    Attributes:
        sensitive: Key names that point at a secret, grouped by the class they
            suggest.
        neutral: Key names that carry no signal.
        misleading: Key names containing a credential-suggesting word while the
            value under them is not a credential.
        scopes: Scope values for the ``scope`` field.
    """

    sensitive: dict[str, tuple[str, ...]]
    neutral: tuple[str, ...]
    misleading: tuple[str, ...]
    scopes: tuple[str, ...]

    def sensitive_for(self, label: str) -> tuple[str, ...]:
        """Return the key names that suggest a given class.

        Args:
            label: One of the seven classes.

        Returns:
            The names for that class, or every sensitive name if the class has
            no group of its own.
        """
        names = self.sensitive.get(label)
        if names:
            return names
        return tuple(name for group in self.sensitive.values() for name in group)

    def all_sensitive(self) -> tuple[str, ...]:
        """Return every sensitive key name across all classes.

        Returns:
            The flattened names, in file order.
        """
        return tuple(name for group in self.sensitive.values() for name in group)

    def pick_key(self, rng: random.Random, label: str, context: str) -> str:
        """Choose a key name for a value of a given class and context.

        Args:
            rng: The source of randomness.
            label: The class of the value the key will sit next to.
            context: One of :data:`CONTEXTS`.

        Returns:
            A key name.

        Raises:
            ValueError: If ``context`` is not one of :data:`CONTEXTS`.
        """
        if context == CONTEXT_HINTING:
            return rng.choice(self.sensitive_for(label))
        if context == CONTEXT_NEUTRAL:
            return rng.choice(self.neutral)
        if context == CONTEXT_MISLEADING:
            return rng.choice(self.misleading)
        raise ValueError(f"unknown context: {context!r}")

    def pick_scope(self, rng: random.Random) -> str:
        """Choose a scope value.

        Args:
            rng: The source of randomness.

        Returns:
            A scope string.
        """
        return rng.choice(self.scopes)


def _as_tuple(value: Any, field: str) -> tuple[str, ...]:
    """Coerce a parsed YAML list of strings into a tuple.

    Args:
        value: The parsed value.
        field: Field name, for the error message.

    Returns:
        The values as a tuple.

    Raises:
        ValueError: If the value is not a list of strings.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"lexicon field {field!r} must be a list of strings")
    return tuple(value)


@lru_cache(maxsize=4)
def load_lexicon(path: Path = LEXICON_PATH) -> Lexicon:
    """Read and validate the lexicon.

    Args:
        path: Lexicon location. Defaults to the file shipped with this package.

    Returns:
        The parsed lexicon.

    Raises:
        ValueError: If a required group is missing or malformed. The generator
            depends on all three groups being populated — an empty
            ``misleading`` group would silently drop the adversarial negatives,
            which are the point of NR-0001 §2.2 item 4.
    """
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    raw_sensitive = document.get("sensitive")
    if not isinstance(raw_sensitive, dict) or not raw_sensitive:
        raise ValueError(f"{path}: 'sensitive' must be a non-empty mapping of class to names")
    sensitive = {
        label: _as_tuple(names, f"sensitive.{label}") for label, names in raw_sensitive.items()
    }

    lexicon = Lexicon(
        sensitive=sensitive,
        neutral=_as_tuple(document.get("neutral"), "neutral"),
        misleading=_as_tuple(document.get("misleading"), "misleading"),
        scopes=_as_tuple(document.get("scopes"), "scopes"),
    )
    for field_name, values in (
        ("neutral", lexicon.neutral),
        ("misleading", lexicon.misleading),
        ("scopes", lexicon.scopes),
    ):
        if not values:
            raise ValueError(f"{path}: '{field_name}' must not be empty")
    return lexicon
