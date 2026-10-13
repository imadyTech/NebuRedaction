"""Export NebuLog's own demo-service logs as detection units (NR-0001 §2.5).

These are the project's own logs, so unlike CredData they may be trained on and
published (plan §5.1). What they contribute that synthetic data cannot is the
real *shape* of OpenTelemetry output: the actual key names, scopes, templates
and attribute mixtures a .NET service emits, including the dull ones the
generator would never think to invent.

How the export works, and why this way. NR-0001 §2.5 item 1 leaves the method to
the executor and names API export as an option. NebuLog's server exposes
``GET /api/logs`` behind a viewer policy, and ``POST /api/auth/demo`` signs in as
the seeded demo user, so the whole path is: start the server, start the demo
producer against it, let traffic accumulate, read it back over the API.

Nothing in the NebuLog repository is modified. Everything the server and
producer need is supplied as environment variables for the two local processes —
the connection string, the data-protection key path and the producer API key —
which is what NebuLog's own configuration is designed for. The API key is
generated here for this run, written to a git-ignored file under ``data/``, and
is a key for nothing but a locally running copy of the application.

Usage:
    python corpus/export_nebulog_own.py --seconds 90
    python corpus/export_nebulog_own.py --from-file runs/NR-0001/<ts>/api-logs.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
from datetime import UTC, datetime
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from datagen.unit import DetectionUnit  # noqa: E402

NEBULOG_ROOT = REPOSITORY_ROOT.parent / "NebuLog"
SERVER_PROJECT = NEBULOG_ROOT / "src" / "NebuLog.Server.Host"
PRODUCER_PROJECT = NEBULOG_ROOT / "samples" / "NebuLog.Samples.DemoProducer"

OWN_DIR = REPOSITORY_ROOT / "data" / "own"
DEFAULT_OUTPUT = OWN_DIR / "own_v0.jsonl"
#: Local run state: the generated API key and the server's database. Under
#: data/, so .gitignore keeps all of it out of git.
LOCAL_STATE_DIR = REPOSITORY_ROOT / "data" / "own" / ".local"

SOURCE = "own"

#: Alphabet NebuLog's API keys are drawn from (see its ApiKeyGenerator).
_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

#: Attributes the server or the SDK adds for transport, not for the application.
#: They carry no credential and would only dilute the corpus.
_TRANSPORT_ATTRIBUTES = frozenset(
    {"nebulog.ingest_source", "nebulog.received_at", "otel.status_code"}
)

#: Splits a rendered message body into candidate fragments, for the body-mode
#: units of plan §3.1. Keeps a key-ish prefix where one is adjacent, so
#: `access_token=abc` yields the key as well as the value.
#:
#: This finds candidates; :func:`is_body_candidate` decides which to keep.
#: Separating the two is what makes the decision testable on its own, and the
#: decision is where NR-0001's defect was.
_BODY_CANDIDATE = re.compile(
    r"""
    (?:(?P<key>[A-Za-z_][A-Za-z0-9_.\-]{0,40})\s*(?P<sep>[=:])\s*)?
    (?:
        "(?P<quoted>[^"]{8,})"
      | (?P<value>[A-Za-z0-9+/_\-.:~%@]{8,})
    )
    """,
    re.VERBOSE,
)

#: Trailing sentence punctuation, which the candidate pattern happily absorbs:
#: `established.` and `received:` both arrived as "values" in NR-0001's export.
#: Not `=`, which is base64 padding.
_TRAILING_PUNCTUATION = ":.,;!?)"

#: A single run of letters and nothing else — `authorised`, `auckland`,
#: `downstream`. Prose, not a value.
_PROSE_WORD = re.compile(r"^[A-Za-z]+$")

#: Shapes that are a value whatever context they appear in.
_UNMISTAKABLE_VALUE = (
    re.compile(r"^[0-9a-fA-F]{16,}$"),  # hex digest or key
    re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),  # uuid
    re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$"),  # jwt
    re.compile(r"^[a-z][a-z0-9+.-]*://"),  # url, with or without credentials
    re.compile(r"^-----BEGIN "),  # pem block
)

#: Shortest pure-letter fragment that may still be a value, and only in mixed
#: case. Below this a letters-only fragment is a word; above it, mixed case is
#: the only remaining evidence that it is not.
_PURE_LETTER_VALUE_LENGTH = 20


def has_value_shape(value: str) -> bool:
    """Decide whether a fragment looks like a *value* rather than prose.

    Used for body fragments that carry no key and no quotes, where the shape of
    the fragment is the only evidence available.

    Args:
        value: The candidate fragment, already stripped of trailing punctuation.

    Returns:
        True if the fragment has the character composition of an identifier,
        digest or credential.

    The asymmetry here is deliberate. A password can be a single lowercase word,
    so this test *will* reject real secrets when they appear bare in a message
    body with nothing around them to suggest it. That is the trade NR-0003 §2.0
    asks for: without a key name or a delimiter there is no evidence to label
    on, and NR-0001 shows what admitting such fragments costs — 1,510 of 1,545
    body units were place names and participles. A bare lowercase word under a
    key name is still kept, because then there is a key name.
    """
    for pattern in _UNMISTAKABLE_VALUE:
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


#: Key names that make a following fragment worth keeping whatever it looks
#: like, so that `password: hunter` survives. A short local list rather than a
#: reuse of the NR-0002 rule layer's vocabulary: the export runs before any
#: labelling and should not depend on `eval/`, and a key name admitting a
#: *candidate* is a much weaker decision than a rule assigning a *label*.
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


def _is_key_like(key: str) -> bool:
    """Decide whether a prefix reads as an attribute name rather than a word.

    Args:
        key: The candidate key.

    Returns:
        True for `access_token`, `db.host`, `retry2` and for bare credential
        words; False for an ordinary word such as `received`.
    """
    lowered = key.lower()
    if lowered in _CREDENTIAL_KEY_WORDS:
        return True
    if any(character in "_.-" for character in key):
        return True
    return any(character.isdigit() for character in key)


def is_body_candidate(key: str | None, separator: str | None, quoted: bool, value: str) -> bool:
    """Decide whether a body fragment becomes a detection unit (NR-0003 §2.0).

    Args:
        key: An adjacent key-like prefix, if the pattern found one.
        separator: The delimiter between key and value, ``=`` or ``:``.
        quoted: Whether the fragment was enclosed in double quotes.
        value: The fragment, stripped of trailing punctuation.

    Returns:
        True if the fragment should be kept. A fragment is kept when it carries
        its own evidence of being a value — an assignment, quotes, or a key-like
        name — or, failing that, when its shape says so.

    The two delimiters are not equally informative, which §2.0's wording does
    not distinguish but the data demands. ``=`` does not occur between two words
    of prose, so an assignment is evidence on its own. ``:`` occurs in prose
    constantly: `ack received: downstream authorised` parses as key `received`,
    value `downstream`, and admitting it would reintroduce exactly the units
    §2.0 exists to remove. So after a colon, something else must also point at a
    value — a key-like name, or the fragment's own shape.
    """
    if len(value) < 8:
        return False
    if quoted:
        return True
    if key is not None and separator == "=":
        return True
    if key is not None and separator == ":" and _is_key_like(key):
        return True
    return has_value_shape(value)


def _port_is_open(port: int) -> bool:
    """Check whether something is listening on a local port.

    Args:
        port: TCP port.

    Returns:
        True if a connection succeeds.
    """
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def build(project: Path) -> None:
    """Build a project before running it.

    Args:
        project: The project directory.

    Raises:
        RuntimeError: If the build fails, with the compiler output. Building as
            its own step is what makes a build failure distinguishable from a
            server that started but never bound its port.
    """
    print(f"building {project.name} ...")
    completed = subprocess.run(
        ["dotnet", "build", str(project), "-c", "Release", "--nologo", "-v", "quiet"],
        cwd=NEBULOG_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).strip().splitlines()[-25:]
        tail = "\n".join(output)
        raise RuntimeError(f"building {project.name} failed:\n{tail}")


def start_server(port: int, api_key: str) -> subprocess.Popen[bytes]:
    """Start the NebuLog server with local, throwaway configuration.

    Args:
        port: Port to bind.
        api_key: Producer API key to seed. Only its hash is stored by the
            server.

    Returns:
        The running process.

    Raises:
        FileNotFoundError: If the NebuLog checkout is not where it is expected.
    """
    if not SERVER_PROJECT.is_dir():
        raise FileNotFoundError(
            f"{SERVER_PROJECT} not found; NebuLog must be checked out beside this repository"
        )

    LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "ASPNETCORE_URLS": f"http://127.0.0.1:{port}",
            "DOTNET_ENVIRONMENT": "Development",
            "ConnectionStrings__Identity": f"Data Source={LOCAL_STATE_DIR / 'nebulog-local.db'}",
            "NebuLog__DataProtection__KeysPath": str(LOCAL_STATE_DIR / "keys"),
            "NebuLog__Demo__Enabled": "true",
            "NebuLog__DemoProducer__ApiKey": api_key,
        }
    )
    return subprocess.Popen(
        [
            "dotnet",
            "run",
            "--project",
            str(SERVER_PROJECT),
            "-c",
            "Release",
            "--no-build",
            # Without this, launchSettings.json's applicationUrl (port 5080)
            # wins over ASPNETCORE_URLS and the server binds a port this script
            # is not watching — which looks exactly like a server that never
            # started.
            "--no-launch-profile",
        ],
        cwd=NEBULOG_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_producer(port: int, api_key: str) -> subprocess.Popen[bytes]:
    """Start the NebuLog demo producer against the local server.

    Args:
        port: The server's port.
        api_key: The producer API key seeded into the server.

    Returns:
        The running process.
    """
    environment = dict(os.environ)
    environment.update(
        {
            "DOTNET_ENVIRONMENT": "Development",
            "NebuLog__Endpoint": f"http://127.0.0.1:{port}",
            "NebuLog__DemoProducer__ApiKey": api_key,
        }
    )
    return subprocess.Popen(
        [
            "dotnet",
            "run",
            "--project",
            str(PRODUCER_PROJECT),
            "-c",
            "Release",
            "--no-build",
            "--no-launch-profile",
        ],
        cwd=NEBULOG_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def fetch_logs(port: int, limit: int) -> list[dict[str, Any]]:
    """Sign in as the demo user and read the server's log history.

    Args:
        port: The server's port.
        limit: Maximum entries to request.

    Returns:
        The entries as the API returns them.

    Raises:
        RuntimeError: If sign-in or the query fails.
    """
    import urllib.request as request

    jar = CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(jar))
    base = f"http://127.0.0.1:{port}"

    # NebuLog refuses cookie-authenticated state-changing requests that do not
    # carry this header (CsrfHeaderFilter): a browser cannot attach a custom
    # header cross-site without passing a CORS preflight, so requiring one stops
    # cross-site posts. Without it the sign-in returns 403, not 401, which looks
    # like a permissions problem rather than a missing header.
    login = request.Request(
        f"{base}/api/auth/demo",
        data=b"",
        method="POST",
        headers={"X-NebuLog-Csrf": "1"},
    )
    try:
        with opener.open(login, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"POST /api/auth/demo returned {exc.code}. 404 means the demo account is "
            "disabled or unseeded; 403 means the CSRF header was rejected"
        ) from exc

    query = request.Request(f"{base}/api/logs?limit={limit}", method="GET")
    try:
        with opener.open(query, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GET /api/logs returned {exc.code}") from exc


def entry_to_units(entry: dict[str, Any], index: int) -> list[DetectionUnit]:
    """Convert one log entry into detection units.

    Produces an attribute-mode unit per structured attribute, plus the entry's
    trace and span ids where present — those are the single most common
    high-entropy non-secret in OpenTelemetry output and belong in the corpus —
    and body-mode units for candidate fragments of the rendered message.

    Units are unlabelled: labelling happens in NR-0003.

    Args:
        entry: One entry from ``GET /api/logs``.
        index: The entry's position, used to build unique ids.

    Returns:
        The units derived from the entry.
    """
    units: list[DetectionUnit] = []
    service = entry.get("serviceName") or entry.get("ServiceName") or ""
    scope = entry.get("scopeName") or entry.get("ScopeName") or ""
    body = entry.get("body") or entry.get("Body") or ""
    attributes = entry.get("attributes") or entry.get("Attributes") or {}

    base_meta = {
        "service_name": service,
        "scope_name": scope,
        "severity": entry.get("severityText") or entry.get("SeverityText"),
    }

    for position, (key, value) in enumerate(sorted(attributes.items())):
        if key in _TRANSPORT_ATTRIBUTES or not isinstance(value, str) or not value.strip():
            continue
        units.append(
            DetectionUnit(
                id=f"own-{index:06d}-attr-{position:02d}",
                key=key,
                value=value,
                source=SOURCE,
                label=None,
                template=body or None,
                scope="otel.attribute",
                meta={**base_meta, "mode": "attribute"},
            )
        )

    for name, field in (("TraceId", "traceId"), ("SpanId", "spanId")):
        value = entry.get(field) or entry.get(name)
        if isinstance(value, str) and value.strip():
            units.append(
                DetectionUnit(
                    id=f"own-{index:06d}-{name.lower()}",
                    key=name,
                    value=value,
                    source=SOURCE,
                    label=None,
                    scope="otel.resource",
                    meta={**base_meta, "mode": "attribute"},
                )
            )

    seen: set[str] = set()
    for position, match in enumerate(_BODY_CANDIDATE.finditer(body)):
        quoted = match.group("quoted") is not None
        raw = match.group("quoted") if quoted else match.group("value")
        # The candidate pattern absorbs sentence punctuation, so `established.`
        # and `received:` arrived as values in NR-0001's export.
        value = raw.rstrip(_TRAILING_PUNCTUATION)
        key = match.group("key")
        if not is_body_candidate(key, match.group("sep"), quoted, value) or value in seen:
            continue
        seen.add(value)
        units.append(
            DetectionUnit(
                id=f"own-{index:06d}-body-{position:02d}",
                key=key if key else "<slot:value>",
                value=value,
                source=SOURCE,
                label=None,
                template=body,
                scope="message.kv" if key else "message.slot",
                meta={**base_meta, "mode": "body"},
            )
        )

    return units


def scan_for_real_secrets(units: list[DetectionUnit]) -> list[dict[str, Any]]:
    """Look for anything in the exported units that might be a real credential.

    NR-0001 §2.5 item 3 requires this check, and requires stopping and reporting
    if it finds something. The demo producer is configured with an API key, so
    the realistic risk is that key — or a fragment of it — turning up in a log
    body or attribute.

    The patterns are deliberately broad: a false alarm costs a minute of
    reading, a miss publishes a credential.

    Args:
        units: The exported units.

    Returns:
        One record per suspicious unit, with the unit's id, key and the pattern
        that matched. Values are **not** included, so the report can be read and
        committed without copying a possible secret into it.
    """
    patterns = {
        "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
        "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "pem_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "url_credentials": re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s:@]+@"),
        "password_assignment": re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token)\b\s*[=:]\s*\S{8,}"
        ),
        "long_high_entropy": re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
    }
    findings: list[dict[str, Any]] = []
    for unit in units:
        haystack = f"{unit.key} {unit.value} {unit.template or ''}"
        for name, pattern in patterns.items():
            if pattern.search(haystack):
                findings.append({"id": unit.id, "key": unit.key, "pattern": name})
                break
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the demo, export its logs and convert them to detection units.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``2`` if the demo could not be run, ``3`` if the
        real-credential check found something.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5099, help="port for the local server")
    parser.add_argument("--seconds", type=int, default=90, help="how long to let traffic run")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help=(
            "maximum entries to read back. NebuLog's HistoryQuery.MaxLimit is 10,000 and a "
            "larger value is rejected with 400 rather than clamped"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output JSON Lines file")
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="convert a previously saved API response instead of running the demo",
    )
    parser.add_argument(
        "--runs", type=Path, default=REPOSITORY_ROOT / "runs" / "NR-0001", help="run log directory"
    )
    args = parser.parse_args(argv)

    stamped = args.runs / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stamped.mkdir(parents=True, exist_ok=True)

    if args.from_file is not None:
        entries = json.loads(args.from_file.read_text(encoding="utf-8"))
    else:
        if _port_is_open(args.port):
            print(f"FAILED: port {args.port} is already in use", file=sys.stderr)
            return 2

        # A key for a server that exists only for the next few minutes, in
        # NebuLog's own format: nbl_{8 base62}_{32 base62}, per
        # ApiKeyGenerator. An arbitrary random string is rejected at startup
        # with "not a well-formed NebuLog key" and silently ignored, which
        # leaves the producer unable to authenticate. Written to a git-ignored
        # file rather than printed, so it does not reach a transcript or report.
        api_key = (
            f"nbl_{''.join(secrets.choice(_BASE62) for _ in range(8))}"
            f"_{''.join(secrets.choice(_BASE62) for _ in range(32))}"
        )
        LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_STATE_DIR / "run.env").write_text(
            f"NebuLog__DemoProducer__ApiKey={api_key}\n", encoding="utf-8"
        )

        try:
            build(SERVER_PROJECT)
            build(PRODUCER_PROJECT)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 2

        server = start_server(args.port, api_key)
        producer = None
        try:
            print(f"starting server on 127.0.0.1:{args.port} ...")
            deadline = time.time() + 120
            while time.time() < deadline and not _port_is_open(args.port):
                if server.poll() is not None:
                    print("FAILED: the server exited while starting", file=sys.stderr)
                    return 2
                time.sleep(1)
            if not _port_is_open(args.port):
                print("FAILED: the server did not bind within 120s", file=sys.stderr)
                return 2

            print(f"starting demo producer, running for {args.seconds}s ...")
            producer = start_producer(args.port, api_key)
            time.sleep(args.seconds)

            print("reading logs back over the API ...")
            entries = fetch_logs(args.port, args.limit)
            (stamped / "api-logs.json").write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        finally:
            for process in (producer, server):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()

    print(f"{len(entries)} entries")
    units: list[DetectionUnit] = []
    for index, entry in enumerate(entries):
        units.extend(entry_to_units(entry, index))

    findings = scan_for_real_secrets(units)
    report: dict[str, Any] = {
        "work_order": "NR-0001",
        "section": "2.5",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "entries": len(entries),
        "units": len(units),
        "by_mode": {
            mode: sum(1 for unit in units if unit.meta.get("mode") == mode)
            for mode in ("attribute", "body")
        },
        "distinct_keys": len({unit.key for unit in units}),
        "real_secret_check": {
            "patterns_checked": 8,
            "suspicious_units": len(findings),
            "findings": findings[:50],
        },
    }
    (stamped / "own-export.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if findings:
        print(
            f"STOPPING: the real-credential check flagged {len(findings)} unit(s). "
            f"Nothing was written to {args.out}. See {stamped / 'own-export.json'} "
            "and report to Frank before continuing (NR-0001 §2.5 item 3).",
            file=sys.stderr,
        )
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(unit.to_json() + "\n" for unit in units), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{args.out}: {len(units)} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
