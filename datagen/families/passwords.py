"""Password and HTTP Basic authentication families.

Passwords are the hardest positive class to generate honestly. A random
28-character string is trivially detectable by entropy alone, but the passwords
that actually leak into logs are often short, memorable and low-entropy — which
is exactly where a detector has to lean on the key name instead of the value.
The families below cover that whole range on purpose, so the training set does
not teach the model that "password" means "high entropy".

The weak-password family composes common English words and digits. Those are
generic words, not credentials taken from any breach corpus.
"""

from __future__ import annotations

import base64
import random

from datagen.families import Family, register
from datagen.families._primitives import strong_password, url_safe_password, username

LABEL = "password"

#: Ordinary English words used to compose memorable passwords. Chosen to be
#: unremarkable: no breach list, no wordlist with provenance to worry about.
_WORDS = (
    "summer",
    "winter",
    "orange",
    "guitar",
    "harbour",
    "monkey",
    "dragon",
    "coffee",
    "bicycle",
    "thunder",
    "garden",
    "pepper",
    "falcon",
    "mantle",
    "sunset",
    "pelican",
)

#: The handful of passwords that genuinely recur in default configurations.
#: These are placeholders, not secrets: a log line reading ``Password=changeme``
#: still needs redacting, and the model must learn that.
_PLACEHOLDERS = (
    "changeme",
    "password",
    "Password1",
    "admin",
    "secret",
    "letmein",
    "P@ssw0rd",
    "test123",
    "root",
    "welcome1",
)


def _password_strong_random(rng: random.Random) -> str:
    """Generate a high-entropy password of the kind a secret manager produces."""
    return strong_password(rng)


def _password_memorable(rng: random.Random) -> str:
    """Generate a human-chosen, moderate-entropy password."""
    word = rng.choice(_WORDS)
    second = rng.choice(_WORDS)
    separator = rng.choice(["", "-", "_", "."])
    suffix = rng.choice([str(rng.randint(1, 9999)), str(rng.randint(70, 99)), "!"])
    if rng.random() < 0.5:
        word = word.capitalize()
    return f"{word}{separator}{second}{suffix}"


def _password_placeholder(rng: random.Random) -> str:
    """Generate a default or placeholder password."""
    return rng.choice(_PLACEHOLDERS)


def _password_url_safe(rng: random.Random) -> str:
    """Generate a password from the restricted alphabet config files tend to use."""
    return url_safe_password(rng)


def _basic_auth_header(rng: random.Random) -> str:
    """Generate an HTTP Basic Authorization header value.

    The credential is base64 of ``user:password``, so the value looks like
    ordinary base64 — the ``Basic `` prefix is what makes it recognisable.
    """
    secret = rng.choice([strong_password(rng, 16), _password_memorable(rng)])
    pair = f"{username(rng)}:{secret}".encode()
    return "Basic " + base64.b64encode(pair).decode("ascii")


def _basic_auth_bare_base64(rng: random.Random) -> str:
    """Generate the base64 of ``user:password`` with no ``Basic`` prefix.

    This is the hard case: without the prefix the value is just base64, and only
    the key name and the decoded colon reveal what it is.
    """
    secret = strong_password(rng, rng.randint(10, 18))
    pair = f"{username(rng)}:{secret}".encode()
    return base64.b64encode(pair).decode("ascii")


register(
    Family(
        name="password.strong_random",
        label=LABEL,
        generate=_password_strong_random,
        doc_url="No vendor format: shaped after secret-manager output, with one character from each class (cf. NIST SP 800-63B composition guidance).",
        note="14 to 28 characters across all four character classes.",
    )
)
register(
    Family(
        name="password.memorable",
        label=LABEL,
        generate=_password_memorable,
        doc_url="No vendor format: human-chosen word-plus-digits passwords, the low-entropy case a detector cannot catch on entropy alone.",
        note="Two ordinary words with a separator and a numeric suffix.",
    )
)
register(
    Family(
        name="password.placeholder",
        label=LABEL,
        generate=_password_placeholder,
        doc_url="No vendor format: the default placeholder passwords that appear in sample configuration files.",
        note="Short, well-known strings that still must be redacted when they appear under a password key.",
    )
)
register(
    Family(
        name="password.url_safe",
        label=LABEL,
        generate=_password_url_safe,
        doc_url="No vendor format: the restricted alphabet used where a password must survive being put in a URL or connection string.",
        note="12 to 24 characters from alphanumerics plus -._~",
    )
)
register(
    Family(
        name="basic_auth.header",
        label=LABEL,
        generate=_basic_auth_header,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7617",
        note="'Basic ' plus base64 of user:password.",
    )
)
register(
    Family(
        name="basic_auth.bare_base64",
        label=LABEL,
        generate=_basic_auth_bare_base64,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7617#section-2",
        note="The same credential without its prefix: base64 whose decoding contains a colon.",
    )
)
