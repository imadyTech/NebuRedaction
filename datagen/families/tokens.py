"""Bearer, session, OAuth and JWT token families.

A token differs from an API key in that an authorisation server issued it and it
expires. The distinction matters for the model mostly through shape: JWTs and
bearer headers have strong structural markers that an API key does not.
"""

from __future__ import annotations

import json
import random

from datagen.families import Family, register
from datagen.families._primitives import (
    ALNUM,
    alnum,
    base64_body,
    hex_lower,
    pick,
    uuid4_like,
)

LABEL = "token"


def _jwt(rng: random.Random, algorithm: str) -> str:
    """Build a JWT with a real header and payload and a random signature.

    The header and payload are genuine base64url-encoded JSON, so the ``eyJ``
    prefix and the two dots are where they belong and a detector that decodes
    the header sees a real ``alg`` field. The signature is random bytes of the
    right length for the algorithm: it verifies against nothing, which is the
    point.

    Args:
        rng: The source of randomness.
        algorithm: The ``alg`` header value.

    Returns:
        A three-part JWT.
    """
    import base64

    def encode(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header = encode({"alg": algorithm, "typ": "JWT"})
    issued = rng.randint(1_700_000_000, 1_800_000_000)
    claims = encode(
        {
            "sub": uuid4_like(rng),
            "iss": "https://auth.example.com/",
            "aud": rng.choice(["orders-api", "payments-api", "nebulog"]),
            "iat": issued,
            "exp": issued + rng.choice([900, 3600, 86400]),
            "scope": rng.choice(["read:orders", "write:payments", "openid profile"]),
        }
    )
    signature_bytes = {"HS256": 32, "RS256": 256, "ES256": 64}[algorithm]
    signature = base64_body(rng, signature_bytes, url_safe=True)
    return f"{header}.{claims}.{signature}"


def _jwt_hs256(rng: random.Random) -> str:
    """Generate an HS256-signed JWT (32-byte signature)."""
    return _jwt(rng, "HS256")


def _jwt_rs256(rng: random.Random) -> str:
    """Generate an RS256-signed JWT (256-byte signature, so a very long value)."""
    return _jwt(rng, "RS256")


def _jwt_es256(rng: random.Random) -> str:
    """Generate an ES256-signed JWT (64-byte signature)."""
    return _jwt(rng, "ES256")


def _bearer_header(rng: random.Random) -> str:
    """Generate an Authorization header value carrying a bearer token."""
    token = rng.choice([_jwt(rng, "HS256"), pick(rng, ALNUM + "-._~+/", 40) + "="])
    return f"Bearer {token}"


def _opaque_access_token(rng: random.Random) -> str:
    """Generate an opaque OAuth access token with no internal structure."""
    return pick(rng, ALNUM + "-._~", rng.choice([32, 43, 64]))


def _google_refresh_token(rng: random.Random) -> str:
    """Generate a Google OAuth refresh token: ``1//`` plus a long secret."""
    return "1//" + pick(rng, ALNUM + "-_", rng.choice([60, 103]))


def _session_id_hex(rng: random.Random) -> str:
    """Generate a hand-rolled hex session identifier."""
    return hex_lower(rng, rng.choice([32, 40, 64]))


def _signed_session_value(rng: random.Random) -> str:
    """Generate a value-dot-signature session token, as Express and Flask use."""
    payload = base64_body(rng, 24, url_safe=True)
    signature = base64_body(rng, 20, url_safe=True)
    return f"{payload}.{signature}"


def _personal_api_token_opaque(rng: random.Random) -> str:
    """Generate an opaque personal API token of the kind internal portals issue."""
    return f"tok_{alnum(rng, 32)}"


register(
    Family(
        name="jwt.hs256",
        label=LABEL,
        generate=_jwt_hs256,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7519",
        note="Three base64url parts; the eyJ prefix comes from the encoded JSON header.",
    )
)
register(
    Family(
        name="jwt.rs256",
        label=LABEL,
        generate=_jwt_rs256,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7518#section-3.3",
        note="Same layout as HS256 but a 256-byte signature, so the value is far longer.",
    )
)
register(
    Family(
        name="jwt.es256",
        label=LABEL,
        generate=_jwt_es256,
        doc_url="https://datatracker.ietf.org/doc/html/rfc7518#section-3.4",
        note="64-byte signature.",
    )
)
register(
    Family(
        name="oauth.bearer_header",
        label=LABEL,
        generate=_bearer_header,
        doc_url="https://datatracker.ietf.org/doc/html/rfc6750#section-2.1",
        note="The 'Bearer ' prefix inside the value, as it appears in an Authorization header.",
    )
)
register(
    Family(
        name="oauth.opaque_access_token",
        label=LABEL,
        generate=_opaque_access_token,
        doc_url="https://datatracker.ietf.org/doc/html/rfc6749#section-1.4",
        note="RFC 6749 leaves access-token format to the server; opaque strings are the common case.",
    )
)
register(
    Family(
        name="oauth.google_refresh_token",
        label=LABEL,
        generate=_google_refresh_token,
        doc_url="https://developers.google.com/identity/protocols/oauth2",
        note="The '1//' prefix is distinctive and survives truncation.",
    )
)
register(
    Family(
        name="session.id_hex",
        label=LABEL,
        generate=_session_id_hex,
        doc_url="No vendor format: hand-rolled hex session identifiers, the case plan §3.3 calls out as indistinguishable from a digest.",
        note="32, 40 or 64 hex characters.",
    )
)
register(
    Family(
        name="session.signed_value",
        label=LABEL,
        generate=_signed_session_value,
        doc_url="https://expressjs.com/en/resources/middleware/cookie-session.html",
        note="payload.signature, the shape used by signed-cookie middleware.",
    )
)
register(
    Family(
        name="generic.opaque_api_token",
        label=LABEL,
        generate=_personal_api_token_opaque,
        doc_url="No vendor format: the 'tok_' convention many internal portals adopt.",
        note="tok_ plus 32 alphanumerics.",
    )
)
