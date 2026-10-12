"""Families whose values cannot be classified from the value alone.

Plan §3.3 states the problem: a 32- or 64-character hex string may be a key or a
digest, and a UUID may be a session id. The value carries no information that
separates them, so the key name, template and scope have to, and where they
cannot the project's convention is to bias toward redaction.

These families exist separately from :mod:`datagen.families.negatives` because
they are labelled differently. The assembler pairs an ambiguous family with a
sensitive or misleading key name and labels the result as a secret with
``ambiguous: true``; the same value shape under a decisive key name
(``TraceId``, ``ETag``, ``ContentHash``) is produced by the negative families
instead. Evaluation reports these separately (plan §3.3), which is only possible
because the flag is in the data.
"""

from __future__ import annotations

import random

from datagen.families import Family, register
from datagen.families._primitives import base64_body, hex_lower, uuid4_like


def _hex32(rng: random.Random) -> str:
    """Generate 32 hex characters: MD5 digest, trace id, or API key."""
    return hex_lower(rng, 32)


def _hex64(rng: random.Random) -> str:
    """Generate 64 hex characters: SHA-256 digest, or API key."""
    return hex_lower(rng, 64)


def _hex40(rng: random.Random) -> str:
    """Generate 40 hex characters: SHA-1 digest, git commit, or session id."""
    return hex_lower(rng, 40)


def _uuid(rng: random.Random) -> str:
    """Generate a UUID: correlation id, or session identifier."""
    return uuid4_like(rng)


def _base64_32(rng: random.Random) -> str:
    """Generate 32 bytes of base64: an encoded blob, or a secret key."""
    return base64_body(rng, 32)


register(
    Family(
        name="ambiguous.hex32",
        label="api_key",
        generate=_hex32,
        doc_url="No vendor format: plan §3.3's stated ambiguity between a 32-hex key and a 32-hex digest.",
        note="Labelled api_key under a sensitive key name, by the bias-toward-redaction convention.",
        ambiguous=True,
    )
)
register(
    Family(
        name="ambiguous.hex64",
        label="api_key",
        generate=_hex64,
        doc_url="No vendor format: plan §3.3's stated ambiguity between a 64-hex key and a SHA-256 digest.",
        note="Labelled api_key under a sensitive key name.",
        ambiguous=True,
    )
)
register(
    Family(
        name="ambiguous.hex40",
        label="token",
        generate=_hex40,
        doc_url="No vendor format: 40 hex characters as a session token, a SHA-1 digest or a commit id.",
        note="Labelled token under a sensitive key name.",
        ambiguous=True,
    )
)
register(
    Family(
        name="ambiguous.uuid",
        label="token",
        generate=_uuid,
        doc_url="No vendor format: plan §3.3's stated ambiguity between a UUID correlation id and a session id.",
        note="Labelled token under a sensitive key name.",
        ambiguous=True,
    )
)
register(
    Family(
        name="ambiguous.base64_32",
        label="api_key",
        generate=_base64_32,
        doc_url="No vendor format: 32 bytes of base64 as either an encoded payload or a symmetric key.",
        note="Labelled api_key under a sensitive key name.",
        ambiguous=True,
    )
)
