"""Body-mode tokenisation (NR-0005 §2.4, from NR-0003 §2.0).

Splits a rendered message into the fragments worth classifying. This is the
same decision NR-0003 §2.0 made for the own-log corpus, moved into the
dependency-free package because inference needs it and `corpus/` is a
development directory that does not ship.

NR-0001's export produced 1,545 body-mode units of which **1,510 were prose
words** — place names, participles, a bare `connection` and an `established.`
that had absorbed its full stop. A fragment is kept only when something points
at it being a value: an assignment, enclosing quotes, a key-like name, or its
own character composition.

The .NET library ports this, and NR-0005 §2.4 requires the two to agree on
shared cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Finds candidates; :func:`is_candidate` decides which to keep. Separating the
#: two is what makes the decision testable on its own, and the decision is
#: where NR-0001's defect was.
_CANDIDATE = re.compile(
    r"""
    (?:(?P<key>[A-Za-z_][A-Za-z0-9_.\-]{0,40})\s*(?P<sep>[=:])\s*)?
    (?:
        "(?P<quoted>[^"]{8,})"
      | (?P<value>[A-Za-z0-9+/_\-.:~%@]{8,})
    )
    """,
    re.VERBOSE,
)

#: Trailing sentence punctuation the pattern absorbs. Not `=`, which is base64
#: padding.
_TRAILING_PUNCTUATION = ":.,;!?)"

#: A single run of letters and nothing else: prose.
_PROSE_WORD = re.compile(r"^[A-Za-z]+$")

_UNMISTAKABLE = (
    re.compile(r"^[0-9a-fA-F]{16,}$"),
    re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$"),
    re.compile(r"^[a-z][a-z0-9+.-]*://"),
    re.compile(r"^-----BEGIN "),
)

#: Shortest pure-letter fragment that may still be a value, and only in mixed
#: case.
_PURE_LETTER_VALUE_LENGTH = 20

#: Key names that make a following fragment worth keeping whatever it looks
#: like, so `password: hunter` survives.
_CREDENTIAL_KEY_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "key",
        "passwd",
        "password",
        "pwd",
        "secret",
        "session",
        "signature",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class BodyCandidate:
    """One fragment of a message body.

    Attributes:
        key: The adjacent key name, or ``<slot:value>`` when there is none.
        value: The fragment.
        scope: ``message.kv`` when a key carried it, ``message.slot`` otherwise.
    """

    key: str
    value: str
    scope: str


def has_value_shape(value: str) -> bool:
    """Decide whether a fragment looks like a value rather than prose.

    Args:
        value: The fragment, already stripped of trailing punctuation.

    Returns:
        True if its character composition is that of an identifier, digest or
        credential.

    **This will reject real secrets**, and that is the trade NR-0003 §2.0 asks
    for. A password can be a single lowercase word, but with no key name and no
    delimiter there is no evidence to label on, and admitting such fragments
    cost 1,510 units of 1,545. A bare lowercase word under a key name is still
    kept, because then there is a key name.
    """
    for pattern in _UNMISTAKABLE:
        if pattern.search(value):
            return True

    mixed_case = value != value.lower() and value != value.upper()
    if _PROSE_WORD.match(value):
        return len(value) >= _PURE_LETTER_VALUE_LENGTH and mixed_case

    has_digit = any(character.isdigit() for character in value)
    has_letter = any(character.isalpha() for character in value)
    if len(value) >= 8 and has_digit and has_letter:
        return True
    separated = any(character in "+/_-.:~%@" for character in value)
    return len(value) >= 16 and (separated or mixed_case)


def is_key_like(key: str) -> bool:
    """Decide whether a prefix reads as an attribute name rather than a word.

    Args:
        key: The candidate key.

    Returns:
        True for `access_token`, `db.host`, `retry2` and bare credential words;
        False for an ordinary word such as `received`.
    """
    lowered = key.lower()
    if lowered in _CREDENTIAL_KEY_WORDS:
        return True
    if any(character in "_.-" for character in key):
        return True
    return any(character.isdigit() for character in key)


def is_candidate(key: str | None, separator: str | None, quoted: bool, value: str) -> bool:
    """Decide whether a fragment becomes a detection unit.

    Args:
        key: An adjacent key-like prefix, if the pattern found one.
        separator: The delimiter between key and value, ``=`` or ``:``.
        quoted: Whether the fragment was enclosed in double quotes.
        value: The fragment, stripped of trailing punctuation.

    Returns:
        True if the fragment should be kept.

    **The two delimiters are not equally informative.** ``=`` does not occur
    between two words of prose, so an assignment is evidence on its own. ``:``
    occurs in prose constantly: `ack received: downstream authorised` parses as
    key `received`, value `downstream`, and admitting it reintroduces exactly
    the units this tokenizer exists to remove. After a colon, something else
    must also point at a value.
    """
    if len(value) < 8:
        return False
    if quoted:
        return True
    if key is not None and separator == "=":
        return True
    if key is not None and separator == ":" and is_key_like(key):
        return True
    return has_value_shape(value)


def body_candidates(message: str) -> list[BodyCandidate]:
    """Split a message body into the fragments worth classifying.

    Args:
        message: The rendered message.

    Returns:
        The candidates, in order, without duplicates.
    """
    seen: set[str] = set()
    candidates: list[BodyCandidate] = []
    for match in _CANDIDATE.finditer(message):
        quoted = match.group("quoted") is not None
        raw = match.group("quoted") if quoted else match.group("value")
        value = raw.rstrip(_TRAILING_PUNCTUATION)
        key = match.group("key")
        if not is_candidate(key, match.group("sep"), quoted, value) or value in seen:
            continue
        seen.add(value)
        candidates.append(
            BodyCandidate(
                key=key if key else "<slot:value>",
                value=value,
                scope="message.kv" if key else "message.slot",
            )
        )
    return candidates
