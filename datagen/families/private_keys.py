r"""Private key and key-material families.

The bodies are random base64, not real keys. Generating actual key pairs would
put usable private keys into a dataset meant for publication, and would teach
the model nothing extra: what identifies these values is the PEM envelope, the
line wrapping and the base64 texture, all of which are reproduced exactly.

The one-line variant matters more than it looks. Structured logging frequently
collapses a PEM block into a single string with literal ``\\n`` escapes, and a
detector keyed only on a leading ``-----BEGIN`` at the start of a line misses it.
"""

from __future__ import annotations

import random

from datagen.families import Family, register
from datagen.families._primitives import base64_body, pem_block

LABEL = "private_key"


def _pem_rsa_private_key(rng: random.Random) -> str:
    """Generate a PKCS#1 RSA private key block."""
    return pem_block(rng, "RSA PRIVATE KEY", rng.choice([608, 1192]))


def _pem_pkcs8_private_key(rng: random.Random) -> str:
    """Generate a PKCS#8 private key block (the unlabelled ``PRIVATE KEY`` form)."""
    return pem_block(rng, "PRIVATE KEY", rng.choice([632, 1216]))


def _pem_ec_private_key(rng: random.Random) -> str:
    """Generate a SEC1 elliptic-curve private key block."""
    return pem_block(rng, "EC PRIVATE KEY", 121)


def _pem_openssh_private_key(rng: random.Random) -> str:
    """Generate an OpenSSH-format private key block."""
    return pem_block(rng, "OPENSSH PRIVATE KEY", rng.choice([399, 1743]))


def _pem_encrypted_private_key(rng: random.Random) -> str:
    """Generate an encrypted PKCS#8 private key block.

    Still a secret: the passphrase is usually in the same configuration file.
    """
    return pem_block(rng, "ENCRYPTED PRIVATE KEY", 680)


def _pem_single_line(rng: random.Random) -> str:
    r"""Generate a PEM block flattened to one line with literal ``\\n`` escapes."""
    header = rng.choice(["RSA PRIVATE KEY", "PRIVATE KEY", "OPENSSH PRIVATE KEY"])
    block = pem_block(rng, header, 608)
    return block.replace("\n", "\\n")


def _ssh_private_key_body_only(rng: random.Random) -> str:
    """Generate the base64 body of a key with the PEM envelope stripped.

    This is what ends up in an environment variable when someone pastes a key
    without its header. There is no marker left at all — only length, entropy
    and the key name remain.
    """
    return base64_body(rng, rng.choice([608, 1192]))


register(
    Family(
        name="pem.rsa_private_key",
        label=LABEL,
        generate=_pem_rsa_private_key,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7468#section-11",
        note="A BEGIN RSA PRIVATE KEY marker line with a 64-column base64 body.",
        multiline=True,
    )
)
register(
    Family(
        name="pem.pkcs8_private_key",
        label=LABEL,
        generate=_pem_pkcs8_private_key,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7468#section-10",
        note="A bare BEGIN PRIVATE KEY marker: the algorithm-agnostic PKCS#8 form.",
        multiline=True,
    )
)
register(
    Family(
        name="pem.ec_private_key",
        label=LABEL,
        generate=_pem_ec_private_key,
        doc_url="https://datatracker.ietf.org/doc/html/rfc5915",
        note="A BEGIN EC PRIVATE KEY marker; much shorter than an RSA key.",
        multiline=True,
    )
)
register(
    Family(
        name="pem.openssh_private_key",
        label=LABEL,
        generate=_pem_openssh_private_key,
        doc_url="https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.key",
        note="A BEGIN OPENSSH PRIVATE KEY marker: OpenSSH's own container format.",
        multiline=True,
    )
)
register(
    Family(
        name="pem.encrypted_private_key",
        label=LABEL,
        generate=_pem_encrypted_private_key,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7468#section-11",
        note="A BEGIN ENCRYPTED PRIVATE KEY marker: encrypted, still a secret.",
        multiline=True,
    )
)
register(
    Family(
        name="pem.single_line_escaped",
        label=LABEL,
        generate=_pem_single_line,
        doc_url="No vendor format: the flattened form structured logging produces when a PEM block becomes one string field.",
        note="Literal \\n escapes instead of real newlines.",
    )
)
register(
    Family(
        name="pem.body_without_header",
        label=LABEL,
        generate=_ssh_private_key_body_only,
        doc_url="No vendor format: key material pasted without its PEM envelope, leaving no structural marker.",
        note="Long base64 with nothing to key on but length and entropy.",
    )
)
