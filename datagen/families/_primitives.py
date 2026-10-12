"""Random-string primitives shared by the format families.

Every function here takes an explicit :class:`random.Random`, so generation is
reproducible from a seed and no module-level random state is ever touched
(EXECUTION.md §2 item 7).

Nothing in this module reads from the network, from the environment, or from
any credential store. The values it produces are format-correct and random;
they are not credentials for anything.
"""

from __future__ import annotations

import base64
import random
import zlib

DIGITS = "0123456789"
LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEX_LOWER = "0123456789abcdef"
HEX_UPPER = "0123456789ABCDEF"
ALNUM = DIGITS + LOWER + UPPER
BASE62 = ALNUM
BASE64_CHARS = UPPER + LOWER + DIGITS + "+/"
BASE64URL_CHARS = UPPER + LOWER + DIGITS + "-_"
PUNCTUATION = "!@#$%^&*()-_=+[]{};:,.?"


def pick(rng: random.Random, alphabet: str, length: int) -> str:
    """Build a random string of a given length from an alphabet.

    Args:
        rng: The source of randomness.
        alphabet: Characters to choose from, with replacement.
        length: Number of characters to produce.

    Returns:
        The generated string.
    """
    return "".join(rng.choice(alphabet) for _ in range(length))


def hex_lower(rng: random.Random, length: int) -> str:
    """Generate a lower-case hexadecimal string.

    Args:
        rng: The source of randomness.
        length: Number of hex digits.

    Returns:
        The generated string.
    """
    return pick(rng, HEX_LOWER, length)


def alnum(rng: random.Random, length: int) -> str:
    """Generate a mixed-case alphanumeric string.

    Args:
        rng: The source of randomness.
        length: Number of characters.

    Returns:
        The generated string.
    """
    return pick(rng, ALNUM, length)


def base64_body(rng: random.Random, byte_length: int, *, url_safe: bool = False) -> str:
    """Generate base64 text by encoding random bytes.

    Encoding real random bytes, rather than picking base64 characters directly,
    keeps the result decodable — a detector that tries to base64-decode a
    candidate sees what it would see in production.

    Args:
        rng: The source of randomness.
        byte_length: Number of random bytes to encode.
        url_safe: Use the URL-safe alphabet and strip padding, as JWTs do.

    Returns:
        The encoded string.
    """
    raw = bytes(rng.getrandbits(8) for _ in range(byte_length))
    if url_safe:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return base64.b64encode(raw).decode("ascii")


def base62_crc32(payload: str, width: int = 6) -> str:
    """Encode the CRC32 of a token body as a fixed-width base62 checksum.

    GitHub's token formats end in a base62-encoded CRC32 checksum of the random
    part, which lets a scanner reject a typo'd token without calling the API —
    described in GitHub's "Behind GitHub's new authentication token formats"
    (https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/).
    The post documents the checksum algorithm but not the base62 alphabet
    ordering, so the ordering here is this project's own convention. That is
    enough for the purpose: the generated tokens are the right shape and their
    checksums are self-consistent, which is what a checksum-aware detector and
    its tests need.

    Args:
        payload: The random part of the token.
        width: Number of base62 characters in the checksum.

    Returns:
        The checksum, left-padded with the alphabet's zero character.
    """
    checksum = zlib.crc32(payload.encode("ascii"))
    digits = []
    for _ in range(width):
        checksum, remainder = divmod(checksum, len(BASE62))
        digits.append(BASE62[remainder])
    return "".join(reversed(digits))


def verify_base62_crc32(payload: str, checksum: str) -> bool:
    """Check a base62 CRC32 checksum against its payload.

    Args:
        payload: The random part of the token.
        checksum: The checksum characters that followed it.

    Returns:
        True when the checksum matches.
    """
    return base62_crc32(payload, width=len(checksum)) == checksum


def uuid4_like(rng: random.Random) -> str:
    """Generate a string with UUID version 4 layout.

    Args:
        rng: The source of randomness.

    Returns:
        A canonical 8-4-4-4-12 hex string with the version nibble set to 4 and
        the variant bits set to the RFC 4122 pattern.
    """
    parts = [
        hex_lower(rng, 8),
        hex_lower(rng, 4),
        "4" + hex_lower(rng, 3),
        rng.choice("89ab") + hex_lower(rng, 3),
        hex_lower(rng, 12),
    ]
    return "-".join(parts)


def pem_block(rng: random.Random, header: str, body_bytes: int) -> str:
    """Build a PEM block with the right structure and a random body.

    The body is random base64, not a real key: generating an actual private key
    would put usable key material in a public dataset for no benefit, since what
    the model learns from is the PEM envelope and the base64 texture.

    Args:
        rng: The source of randomness.
        header: The label between the dashes, such as ``RSA PRIVATE KEY``.
        body_bytes: Number of random bytes to encode into the body.

    Returns:
        A multi-line PEM block, wrapped at 64 characters as RFC 7468 requires.
    """
    encoded = base64_body(rng, body_bytes)
    lines = [encoded[index : index + 64] for index in range(0, len(encoded), 64)]
    return "\n".join([f"-----BEGIN {header}-----", *lines, f"-----END {header}-----"])


def hostname(rng: random.Random) -> str:
    """Generate a plausible, non-resolving host name.

    Uses only the reserved names from RFC 2606 and RFC 6761, so a generated
    host can never point at a real service.

    Args:
        rng: The source of randomness.

    Returns:
        A host name under ``.example``, ``.invalid``, ``.test`` or
        ``example.com``.
    """
    label = rng.choice(
        ["db", "cache", "queue", "api", "svc", "primary", "replica", "cluster0", "node1"]
    )
    domain = rng.choice(["example.com", "internal.example", "svc.invalid", "corp.test"])
    return f"{label}.{domain}"


def username(rng: random.Random) -> str:
    """Generate a plausible service or application user name.

    Args:
        rng: The source of randomness.

    Returns:
        A user name.
    """
    return rng.choice(
        [
            "appuser",
            "svc_orders",
            "reporting",
            "payments_rw",
            "admin",
            "nebulog",
            "etl",
            "readonly",
        ]
    )


def strong_password(rng: random.Random, length: int | None = None) -> str:
    """Generate a high-entropy password.

    Args:
        rng: The source of randomness.
        length: Password length; a random length in [14, 28] when omitted.

    Returns:
        A password containing at least one character from each class, shuffled
        so the required characters are not all at the front.
    """
    size = length if length is not None else rng.randint(14, 28)
    required = [
        rng.choice(LOWER),
        rng.choice(UPPER),
        rng.choice(DIGITS),
        rng.choice(PUNCTUATION),
    ]
    rest = [rng.choice(ALNUM + PUNCTUATION) for _ in range(max(0, size - len(required)))]
    characters = required + rest
    rng.shuffle(characters)
    return "".join(characters)


def url_safe_password(rng: random.Random, length: int | None = None) -> str:
    """Generate a password safe to embed in a URL or connection string.

    Connection strings break on ``@``, ``:``, ``/`` and ``;``, so real ones use
    a restricted alphabet; mirroring that keeps the generated connection
    strings parseable.

    Args:
        rng: The source of randomness.
        length: Password length; a random length in [12, 24] when omitted.

    Returns:
        A password drawn from alphanumerics plus a few safe symbols.
    """
    size = length if length is not None else rng.randint(12, 24)
    return pick(rng, ALNUM + "-._~", size)
