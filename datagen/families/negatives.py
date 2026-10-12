"""Hard-negative families: values that look like secrets and are not.

These decide the false-positive rate. Plan §4.5 targets a false-positive rate
of 0.1% measured per attribute, and in a real log almost every attribute is a
negative — so the negatives have to be the ones that actually fool a detector,
not filler. Every type listed in plan §5.2 is covered here, plus the ones that
turn up in OpenTelemetry output: trace and span ids, content digests, ETags.

Note what is *not* here: the adversarial cases where a misleading key name is
the trap rather than the value. Those are the same values under a different key
name, so they are produced by the assembler (:mod:`datagen.assemble`) from the
misleading group of the key lexicon, and marked ``meta.adversarial``.
"""

from __future__ import annotations

import base64
import random

from datagen.families import Family, register
from datagen.families._primitives import (
    ALNUM,
    UPPER,
    base64_body,
    hex_lower,
    hostname,
    pick,
    uuid4_like,
)

LABEL = "none"

_SENTENCES = (
    "Order 1042 shipped to the London warehouse",
    "Retrying after transient timeout",
    "Cache warm-up completed in 412 ms",
    "User preferences saved successfully",
    "Scheduled maintenance begins at 02:00 UTC",
    "Inventory reconciliation finished with 3 adjustments",
)

_MIME_TYPES = (
    "application/json",
    "text/html; charset=utf-8",
    "application/vnd.api+json",
    "image/webp",
    "application/octet-stream",
)

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "python-requests/2.32.3",
    "NebuLog.Client/1.4.0 (+https://github.com/imadyTech/NebuLog)",
)


def _uuid_v4(rng: random.Random) -> str:
    """Generate a version 4 UUID — a correlation id, not a credential."""
    return uuid4_like(rng)


def _uuid_v7(rng: random.Random) -> str:
    """Generate a version 7 (time-ordered) UUID."""
    timestamp = rng.randint(0x1800000000, 0x1A00000000)
    head = f"{timestamp:012x}"
    return (
        f"{head[:8]}-{head[8:12]}-7{hex_lower(rng, 3)}-"
        f"{rng.choice('89ab')}{hex_lower(rng, 3)}-{hex_lower(rng, 12)}"
    )


def _sha256_hex(rng: random.Random) -> str:
    """Generate a 64-character hex digest, indistinguishable from a 64-hex key."""
    return hex_lower(rng, 64)


def _sha1_hex(rng: random.Random) -> str:
    """Generate a 40-character hex digest."""
    return hex_lower(rng, 40)


def _md5_hex(rng: random.Random) -> str:
    """Generate a 32-character hex digest, the twin of a Twilio auth token."""
    return hex_lower(rng, 32)


def _digest_with_algorithm_prefix(rng: random.Random) -> str:
    """Generate an algorithm-prefixed digest such as ``sha256:abc...``."""
    algorithm, width = rng.choice([("sha256", 64), ("sha512", 128), ("sha1", 40)])
    return f"{algorithm}:{hex_lower(rng, width)}"


def _crc32_hex(rng: random.Random) -> str:
    """Generate an 8-character checksum."""
    return hex_lower(rng, 8)


def _base64_of_text(rng: random.Random) -> str:
    """Generate base64 of an ordinary English sentence.

    High-entropy-looking, decodes to plain text. A detector that only measures
    entropy over the encoded form gets this wrong.
    """
    return base64.b64encode(rng.choice(_SENTENCES).encode("utf-8")).decode("ascii")


def _base64_of_json(rng: random.Random) -> str:
    """Generate base64 of a small JSON document — the shape of a JWT payload."""
    import json

    payload = {
        "page": rng.randint(1, 40),
        "pageSize": rng.choice([20, 50, 100]),
        "sort": rng.choice(["createdAt", "-total", "status"]),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _semver(rng: random.Random) -> str:
    """Generate a semantic version string."""
    core = f"{rng.randint(0, 9)}.{rng.randint(0, 24)}.{rng.randint(0, 40)}"
    if rng.random() < 0.3:
        core += rng.choice(["-rc.1", "-beta.3", "-preview.2"])
    if rng.random() < 0.2:
        core += f"+build.{rng.randint(100, 99999)}"
    return core


def _assembly_version(rng: random.Random) -> str:
    """Generate a four-part .NET assembly version."""
    return f"{rng.randint(1, 12)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}.{rng.randint(0, 65535)}"


def _trace_id(rng: random.Random) -> str:
    """Generate a W3C trace id: 32 hex characters, the same width as an MD5."""
    return hex_lower(rng, 32)


def _span_id(rng: random.Random) -> str:
    """Generate a W3C span id: 16 hex characters."""
    return hex_lower(rng, 16)


def _traceparent(rng: random.Random) -> str:
    """Generate a full traceparent header value."""
    return f"00-{hex_lower(rng, 32)}-{hex_lower(rng, 16)}-{rng.choice(['00', '01'])}"


def _build_number(rng: random.Random) -> str:
    """Generate a CI build identifier."""
    style = rng.randint(0, 2)
    if style == 0:
        return str(rng.randint(1000, 99999))
    if style == 1:
        return f"2026{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}.{rng.randint(1, 9)}"
    return f"build-{rng.randint(1, 4000)}"


def _git_commit_sha(rng: random.Random) -> str:
    """Generate a git commit id, full or abbreviated."""
    return hex_lower(rng, rng.choice([7, 8, 40]))


def _file_path_windows(rng: random.Random) -> str:
    """Generate a Windows file path."""
    root = rng.choice(["C:\\", "D:\\", "E:\\"])
    parts = rng.sample(["Program Files", "inetpub", "logs", "app", "data", "temp"], 2)
    name = rng.choice(["appsettings.json", "service.log", "cert.pfx", "index.dat"])
    return root + "\\".join([*parts, name])


def _file_path_posix(rng: random.Random) -> str:
    """Generate a POSIX file path."""
    parts = rng.sample(["var", "opt", "srv", "usr", "home", "etc"], 2)
    name = rng.choice(["config.yaml", "app.log", "id_rsa.pub", "manifest.json"])
    return "/" + "/".join([*parts, name])


def _short_link(rng: random.Random) -> str:
    """Generate a short link whose path is a random-looking slug."""
    host = rng.choice(["s.example.com", "go.example", "lnk.invalid"])
    return f"https://{host}/{pick(rng, ALNUM, rng.choice([6, 7, 8, 11]))}"


def _url_without_credentials(rng: random.Random) -> str:
    """Generate an ordinary URL with query parameters but no credential."""
    path = rng.choice(["/api/v1/orders", "/search", "/v2/catalog/items"])
    query = rng.choice(
        [
            f"?page={rng.randint(1, 50)}&pageSize=20",
            f"?q={rng.choice(['boots', 'kettle', 'lamp'])}&sort=price",
            f"?traceId={hex_lower(rng, 32)}",
        ]
    )
    return f"https://{hostname(rng)}{path}{query}"


def _iso8601_timestamp(rng: random.Random) -> str:
    """Generate an ISO 8601 timestamp."""
    return (
        f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T"
        f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
        f".{rng.randint(0, 999):03d}Z"
    )


def _unix_epoch_millis(rng: random.Random) -> str:
    """Generate a millisecond Unix timestamp."""
    return str(rng.randint(1_750_000_000_000, 1_800_000_000_000))


def _snowflake_id(rng: random.Random) -> str:
    """Generate a Snowflake-style 64-bit identifier."""
    return str(rng.randint(10**17, 10**19))


def _etag(rng: random.Random) -> str:
    """Generate an HTTP ETag, which is a quoted digest."""
    weak = "W/" if rng.random() < 0.3 else ""
    return f'{weak}"{hex_lower(rng, rng.choice([16, 32]))}"'


def _email_address(rng: random.Random) -> str:
    """Generate an email address at a reserved domain."""
    local = rng.choice(["alice", "bob.smith", "ops", "no-reply", "s.chen"])
    return f"{local}@{rng.choice(['example.com', 'example.org', 'corp.test'])}"


def _ipv4_address(rng: random.Random) -> str:
    """Generate a private-range IPv4 address."""
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _hostname_value(rng: random.Random) -> str:
    """Generate a host name."""
    return hostname(rng)


def _count_integer(rng: random.Random) -> str:
    """Generate a plain count."""
    return str(rng.randint(0, 50000))


def _byte_size(rng: random.Random) -> str:
    """Generate a human-readable byte size."""
    return f"{rng.randint(1, 999)}{rng.choice(['B', 'KB', 'MB', 'GB'])}"


def _duration(rng: random.Random) -> str:
    """Generate a duration."""
    return rng.choice(
        [
            f"{rng.randint(1, 9999)}ms",
            f"{rng.random() * 10:.4f}",
            f"00:00:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}",
        ]
    )


def _mime_type(rng: random.Random) -> str:
    """Generate a media type."""
    return rng.choice(_MIME_TYPES)


def _user_agent(rng: random.Random) -> str:
    """Generate a User-Agent string: long, structured, entirely harmless."""
    return rng.choice(_USER_AGENTS)


def _correlation_slug(rng: random.Random) -> str:
    """Generate a prefixed correlation identifier, as OpenStack's ``req-`` ids are."""
    prefix = rng.choice(["req", "corr", "msg", "job", "run"])
    return f"{prefix}-{uuid4_like(rng)}"


def _feature_flag_value(rng: random.Random) -> str:
    """Generate a configuration value that is a word, not a secret."""
    return rng.choice(["true", "false", "Enabled", "Disabled", "Warning", "Information"])


def _harmless_cookie_pair(rng: random.Random) -> str:
    """Generate a cookie that authenticates nothing: locale, theme, consent."""
    name, value = rng.choice(
        [
            ("locale", rng.choice(["en-GB", "zh-CN", "fr-FR"])),
            ("theme", rng.choice(["light", "dark"])),
            ("consent", rng.choice(["granted", "denied"])),
            ("tz", rng.choice(["Europe/London", "Asia/Shanghai"])),
        ]
    )
    return f"{name}={value}"


def _public_key_pem(rng: random.Random) -> str:
    """Generate a PEM *public* key block.

    The strongest possible hard negative for the ``private_key`` class: same
    envelope, same base64 texture, one word different, and publishing it is
    harmless.
    """
    from datagen.families._primitives import pem_block

    header = rng.choice(["PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE"])
    return pem_block(rng, header, rng.choice([162, 294, 800]))


def _ssh_public_key(rng: random.Random) -> str:
    """Generate an OpenSSH public key line."""
    return f"ssh-ed25519 {base64_body(rng, 51)} {_email_address(rng)}"


def _license_key(rng: random.Random) -> str:
    """Generate a product licence key: grouped, high-entropy, and not a credential.

    Debatable as a negative, and deliberately so — a licence key is not an
    authentication credential under plan §3.2's definitions, but it looks like
    one. Keeping it on the negative side forces the model to rely on the key
    name here, which is the behaviour the project wants.
    """
    groups = [pick(rng, UPPER + "0123456789", 5) for _ in range(5)]
    return "-".join(groups)


def _phone_like_number(rng: random.Random) -> str:
    """Generate a long digit string of the kind order or account numbers use."""
    return str(rng.randint(10**9, 10**12))


def _lorem_identifier(rng: random.Random) -> str:
    """Generate a readable slug identifier."""
    return "-".join(rng.sample(["bright", "copper", "kettle", "quiet", "river", "amber"], 3))


def _mixed_case_word_value(rng: random.Random) -> str:
    """Generate an ordinary camel-case configuration value."""
    return rng.choice(["ReadCommitted", "RequireHttps", "BackgroundService", "RoundRobin"])


def _alnum_short_id(rng: random.Random) -> str:
    """Generate a short alphanumeric id — too short to be a secret, high entropy per character."""
    return pick(rng, ALNUM, rng.choice([6, 8, 10]))


def _nanoid(rng: random.Random) -> str:
    """Generate a Nano ID, the 21-character URL-safe identifier."""
    return pick(rng, ALNUM + "-_", 21)


def _ulid(rng: random.Random) -> str:
    """Generate a ULID: 26 characters of Crockford base32."""
    return pick(rng, "0123456789ABCDEFGHJKMNPQRSTVWXYZ", 26)


def _docker_image_digest(rng: random.Random) -> str:
    """Generate a container image reference pinned by digest."""
    name = rng.choice(["nebulog/api", "library/redis", "orders/worker"])
    return f"{name}@sha256:{hex_lower(rng, 64)}"


def _query_string_without_secret(rng: random.Random) -> str:
    """Generate a query string whose parameters are all harmless."""
    return (
        f"?page={rng.randint(1, 30)}&pageSize={rng.choice([20, 50])}&traceId={hex_lower(rng, 32)}"
    )


_NEGATIVES: tuple[tuple[str, object, str, str], ...] = (
    (
        "neg.uuid_v4",
        _uuid_v4,
        "https://datatracker.ietf.org/doc/html/rfc9562#section-5.4",
        "Correlation ids; plan §3.3 names UUIDs as possibly being session ids.",
    ),
    (
        "neg.uuid_v7",
        _uuid_v7,
        "https://datatracker.ietf.org/doc/html/rfc9562#section-5.7",
        "Time-ordered UUID, increasingly common as a primary key.",
    ),
    (
        "neg.sha256_hex",
        _sha256_hex,
        "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf",
        "64 hex characters: the exact twin of a 64-hex API key.",
    ),
    (
        "neg.sha1_hex",
        _sha1_hex,
        "https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf",
        "40 hex characters.",
    ),
    (
        "neg.md5_hex",
        _md5_hex,
        "https://datatracker.ietf.org/doc/html/rfc1321",
        "32 hex characters: the twin of a Twilio auth token and a Datadog API key.",
    ),
    (
        "neg.digest_with_prefix",
        _digest_with_algorithm_prefix,
        "https://github.com/opencontainers/image-spec/blob/main/descriptor.md#digests",
        "sha256:... form; the prefix is the signal that resolves the ambiguity.",
    ),
    (
        "neg.crc32_hex",
        _crc32_hex,
        "https://datatracker.ietf.org/doc/html/rfc1952#section-8",
        "Short checksum.",
    ),
    (
        "neg.base64_of_text",
        _base64_of_text,
        "https://datatracker.ietf.org/doc/html/rfc4648#section-4",
        "Base64 of an English sentence: defeats entropy-only heuristics.",
    ),
    (
        "neg.base64_of_json",
        _base64_of_json,
        "https://datatracker.ietf.org/doc/html/rfc4648#section-5",
        "Base64url of small JSON, the same shape as a JWT payload segment.",
    ),
    (
        "neg.semver",
        _semver,
        "https://semver.org/spec/v2.0.0.html",
        "Version strings.",
    ),
    (
        "neg.assembly_version",
        _assembly_version,
        "https://learn.microsoft.com/en-us/dotnet/api/system.version",
        "Four-part .NET version.",
    ),
    (
        "neg.trace_id",
        _trace_id,
        "https://www.w3.org/TR/trace-context/#trace-id",
        "32 hex characters, and extremely frequent in OTel logs.",
    ),
    (
        "neg.span_id",
        _span_id,
        "https://www.w3.org/TR/trace-context/#parent-id",
        "16 hex characters.",
    ),
    (
        "neg.traceparent",
        _traceparent,
        "https://www.w3.org/TR/trace-context/#traceparent-header-field-values",
        "The whole traceparent header: dashes and hex, looks structured and secret-ish.",
    ),
    (
        "neg.build_number",
        _build_number,
        "No vendor format: CI build identifiers in their common shapes.",
        "Numeric, date-based or prefixed.",
    ),
    (
        "neg.git_commit_sha",
        _git_commit_sha,
        "https://git-scm.com/docs/git-rev-parse#_specifying_revisions",
        "Full or abbreviated commit id.",
    ),
    (
        "neg.file_path_windows",
        _file_path_windows,
        "https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats",
        "Windows paths; note that one of them ends in .pfx.",
    ),
    (
        "neg.file_path_posix",
        _file_path_posix,
        "https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_267",
        "POSIX paths; one of them names a public key file.",
    ),
    (
        "neg.short_link",
        _short_link,
        "No vendor format: shortener URLs, whose slug is random-looking but public.",
        "6 to 11 alphanumerics after the host.",
    ),
    (
        "neg.url_without_credentials",
        _url_without_credentials,
        "https://datatracker.ietf.org/doc/html/rfc3986",
        "An ordinary URL: the negative counterpart of credentials-in-URL.",
    ),
    (
        "neg.iso8601_timestamp",
        _iso8601_timestamp,
        "https://datatracker.ietf.org/doc/html/rfc3339",
        "Timestamps.",
    ),
    (
        "neg.unix_epoch_millis",
        _unix_epoch_millis,
        "https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_16",
        "13-digit epoch milliseconds.",
    ),
    (
        "neg.snowflake_id",
        _snowflake_id,
        "https://en.wikipedia.org/wiki/Snowflake_ID",
        "Long numeric ids.",
    ),
    (
        "neg.etag",
        _etag,
        "https://datatracker.ietf.org/doc/html/rfc9110#field.etag",
        "Quoted digest, sometimes weak-prefixed.",
    ),
    (
        "neg.email_address",
        _email_address,
        "https://datatracker.ietf.org/doc/html/rfc5322#section-3.4.1",
        "Email addresses: out of v0.1 scope as PII, and must not be redacted as secrets.",
    ),
    (
        "neg.ipv4_address",
        _ipv4_address,
        "https://datatracker.ietf.org/doc/html/rfc1918",
        "Private-range addresses.",
    ),
    (
        "neg.hostname",
        _hostname_value,
        "https://datatracker.ietf.org/doc/html/rfc1123#section-2",
        "Host names.",
    ),
    (
        "neg.count_integer",
        _count_integer,
        "No vendor format: plain counts, the most common attribute value of all.",
        "Bare integers.",
    ),
    (
        "neg.byte_size",
        _byte_size,
        "No vendor format: human-readable sizes as logging libraries render them.",
        "Number plus unit.",
    ),
    (
        "neg.duration",
        _duration,
        "No vendor format: elapsed-time values in the shapes .NET and OTel emit.",
        "Milliseconds, fractional seconds or a TimeSpan.",
    ),
    (
        "neg.mime_type",
        _mime_type,
        "https://datatracker.ietf.org/doc/html/rfc9110#field.content-type",
        "Media types.",
    ),
    (
        "neg.user_agent",
        _user_agent,
        "https://datatracker.ietf.org/doc/html/rfc9110#field.user-agent",
        "Long, structured, harmless.",
    ),
    (
        "neg.correlation_slug",
        _correlation_slug,
        "No vendor format: the prefixed request ids seen in OpenStack and similar services.",
        "req-<uuid> and friends.",
    ),
    (
        "neg.config_word_value",
        _feature_flag_value,
        "No vendor format: boolean and enum configuration values.",
        "true/false/Enabled/Warning.",
    ),
    (
        "neg.harmless_cookie",
        _harmless_cookie_pair,
        "https://datatracker.ietf.org/doc/html/rfc6265",
        "A cookie that authenticates nothing: the negative side of the cookie class.",
    ),
    (
        "neg.public_key_pem",
        _public_key_pem,
        "https://datatracker.ietf.org/doc/html/rfc7468#section-4",
        "Same PEM envelope as a private key, one word different, harmless to publish.",
    ),
    (
        "neg.ssh_public_key",
        _ssh_public_key,
        "https://datatracker.ietf.org/doc/html/rfc4716",
        "An authorized_keys line: long base64 that is meant to be shared.",
    ),
    (
        "neg.license_key",
        _license_key,
        "No vendor format: grouped product licence keys, high-entropy but not an authenticator.",
        "Five groups of five upper-case alphanumerics.",
    ),
    (
        "neg.long_numeric_id",
        _phone_like_number,
        "No vendor format: account and order numbers.",
        "10 to 12 digits.",
    ),
    (
        "neg.slug_identifier",
        _lorem_identifier,
        "No vendor format: readable generated names, as container runtimes and CI produce.",
        "Three hyphenated words.",
    ),
    (
        "neg.camel_case_value",
        _mixed_case_word_value,
        "No vendor format: camel-case enum values from .NET configuration.",
        "Ordinary words.",
    ),
    (
        "neg.short_alnum_id",
        _alnum_short_id,
        "No vendor format: short ids, too short to be a credential but dense per character.",
        "6 to 10 alphanumerics.",
    ),
    (
        "neg.nanoid",
        _nanoid,
        "https://github.com/ai/nanoid",
        "21 URL-safe characters: the length and alphabet of a short token.",
    ),
    (
        "neg.ulid",
        _ulid,
        "https://github.com/ulid/spec",
        "26 characters of Crockford base32.",
    ),
    (
        "neg.image_digest_reference",
        _docker_image_digest,
        "https://github.com/opencontainers/distribution-spec/blob/main/spec.md",
        "Image reference pinned by digest.",
    ),
    (
        "neg.query_string_clean",
        _query_string_without_secret,
        "https://datatracker.ietf.org/doc/html/rfc3986#section-3.4",
        "A query string with no credential in it: the counterpart of the access_token case.",
    ),
)

#: Negative families whose values span several lines. Only the PEM one does: a
#: public key block has the same envelope and line wrapping as a private key,
#: which is what makes it the hardest negative in the set.
_MULTILINE_NEGATIVES = frozenset({"neg.public_key_pem"})

for _name, _generate, _doc_url, _note in _NEGATIVES:
    register(
        Family(
            name=_name,
            label=LABEL,
            generate=_generate,  # type: ignore[arg-type]
            doc_url=_doc_url,
            note=_note,
            multiline=_name in _MULTILINE_NEGATIVES,
        )
    )
