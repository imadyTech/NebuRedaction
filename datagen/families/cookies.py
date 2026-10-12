"""Cookie families.

Cookies sit awkwardly between ``token`` and ``none``. A session cookie
authenticates a browser and must be redacted; a consent or locale cookie is
harmless. Since the whole ``Cookie`` header is usually logged as one value, the
families below produce both the single-cookie form and the full header, and the
harmless-cookie cases live in :mod:`datagen.families.negatives` so the model
sees both sides of the distinction.
"""

from __future__ import annotations

import random

from datagen.families import Family, register
from datagen.families._primitives import (
    ALNUM,
    UPPER,
    base64_body,
    hex_lower,
    pick,
)

LABEL = "cookie"


def _aspnet_auth_cookie(rng: random.Random) -> str:
    """Generate an ASP.NET Core authentication cookie value."""
    return pick(rng, ALNUM + "-_", rng.choice([180, 240, 320]))


def _jsessionid(rng: random.Random) -> str:
    """Generate a Java servlet JSESSIONID value."""
    suffix = rng.choice(["", f".node{rng.randint(1, 4)}", f".{pick(rng, UPPER, 6)}"])
    return pick(rng, "0123456789ABCDEF", 32) + suffix


def _set_cookie_header(rng: random.Random) -> str:
    """Generate a Set-Cookie header value with attributes."""
    name = rng.choice([".AspNet.Cookies", "sessionid", "connect.sid", "laravel_session"])
    value = rng.choice([base64_body(rng, 32, url_safe=True), hex_lower(rng, 48)])
    attributes = ["Path=/", "HttpOnly", "Secure", "SameSite=Lax"]
    rng.shuffle(attributes)
    return f"{name}={value}; " + "; ".join(attributes[: rng.randint(2, 4)])


def _cookie_header_with_session(rng: random.Random) -> str:
    """Generate a full Cookie header mixing a session cookie with harmless ones."""
    session_name = rng.choice(["sessionid", "JSESSIONID", "connect.sid", "PHPSESSID"])
    session = f"{session_name}={base64_body(rng, 28, url_safe=True)}"
    harmless = [
        f"locale={rng.choice(['en-GB', 'zh-CN', 'de-DE'])}",
        f"theme={rng.choice(['light', 'dark'])}",
        f"consent={rng.choice(['granted', 'denied'])}",
    ]
    rng.shuffle(harmless)
    parts = [*harmless[: rng.randint(1, 3)], session]
    rng.shuffle(parts)
    return "; ".join(parts)


def _laravel_session_cookie(rng: random.Random) -> str:
    """Generate a Laravel session cookie value (base64 of an encrypted payload)."""
    return base64_body(rng, rng.choice([264, 336]))


def _csrf_token_cookie(rng: random.Random) -> str:
    """Generate an anti-forgery cookie value.

    A CSRF token is not an authenticator on its own, but it is paired with the
    session and is treated as a secret by every framework that issues it.
    """
    return pick(rng, ALNUM + "-_", rng.choice([43, 64, 155]))


register(
    Family(
        name="cookie.aspnet_auth",
        label=LABEL,
        generate=_aspnet_auth_cookie,
        doc_url="https://learn.microsoft.com/en-us/aspnet/core/security/authentication/cookie",
        note="Long URL-safe value; the cookie name does the identifying.",
    )
)
register(
    Family(
        name="cookie.jsessionid",
        label=LABEL,
        generate=_jsessionid,
        doc_url="https://jakarta.ee/specifications/servlet/6.0/jakarta-servlet-spec-6.0#session-tracking-mechanisms",
        note="32 upper-case hex characters, sometimes with a node suffix for sticky sessions.",
    )
)
register(
    Family(
        name="cookie.set_cookie_header",
        label=LABEL,
        generate=_set_cookie_header,
        doc_url="https://datatracker.ietf.org/doc/html/rfc6265#section-4.1",
        note="name=value plus attributes, as a response header is logged.",
    )
)
register(
    Family(
        name="cookie.request_header",
        label=LABEL,
        generate=_cookie_header_with_session,
        doc_url="https://datatracker.ietf.org/doc/html/rfc6265#section-4.2",
        note="A whole Cookie header: one session cookie among harmless ones, so partial masking is not an option.",
    )
)
register(
    Family(
        name="cookie.laravel_session",
        label=LABEL,
        generate=_laravel_session_cookie,
        doc_url="https://laravel.com/docs/12.x/session",
        note="Long standard-alphabet base64 with padding.",
    )
)
register(
    Family(
        name="cookie.csrf_token",
        label=LABEL,
        generate=_csrf_token_cookie,
        doc_url="https://learn.microsoft.com/en-us/aspnet/core/security/anti-request-forgery",
        note="Anti-forgery value, paired with the session cookie.",
    )
)
