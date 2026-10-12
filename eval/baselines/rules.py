"""Control group 1: a pure rule set (NR-0002 §2.2).

Two halves, matching what a production rule-based redactor actually does:

1. **Value patterns** — vendor credential formats. Each rule records where its
   pattern came from. Most are derived from the same vendor documentation the
   generator's families cite, which makes them *independent* derivations rather
   than ports; the few taken from gitleaks' rule set are marked ``gitleaks`` and
   attributed under its MIT licence in ``THIRD_PARTY_NOTICES.md``.

2. **Key-name matching** — the lexicon's sensitive names, matched as **whole
   words after splitting**, never as substrings. NR-0002 §2.2 is explicit about
   this, and the reason is the adversarial slice: a substring match on ``key``
   fires on ``PartitionKey``, ``CacheKey`` and ``KeyCount``, which is precisely
   the failure the project is trying to measure rather than reproduce.

**A methodological caveat that belongs in the report, not buried here.** The
value patterns and the generator's positive families are both derived from the
same vendor documentation. On synthetic data this baseline is therefore being
tested against its own specification, and will score higher than it would in the
wild. That makes it a *demanding* baseline for G2 — which is the safe direction
for a go/no-go gate — but it also means CredData, derived from neither, is the
only honest read on rule-baseline quality. The report says so.

This module is also the rule layer reused by NR-0004's "rules + student model"
combination, so its interface is kept narrow: patterns in, decision out.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from datagen.unit import DetectionUnit

from eval.system import SystemOutput

VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ValueRule:
    """One value pattern.

    Attributes:
        name: Rule identifier, used in the report's error analysis.
        label: The class a match implies.
        pattern: Compiled regular expression, applied with ``search``.
        provenance: Where the pattern came from — a vendor documentation URL for
            an independent derivation, or ``gitleaks`` for a ported rule.
    """

    name: str
    label: str
    pattern: re.Pattern[str]
    provenance: str


def _rule(name: str, label: str, pattern: str, provenance: str) -> ValueRule:
    """Compile a rule.

    Args:
        name: Rule identifier.
        label: Class implied by a match.
        pattern: Regular expression source.
        provenance: Where the pattern came from.

    Returns:
        The compiled rule.
    """
    return ValueRule(name, label, re.compile(pattern), provenance)


#: Value patterns. `provenance` is a vendor documentation URL where the format is
#: published by the vendor — in which case the pattern is derived from that
#: document, the same source the generator's family uses — or "gitleaks (MIT)"
#: where the pattern itself was taken from gitleaks' default rules.
VALUE_RULES: tuple[ValueRule, ...] = (
    _rule(
        "aws.access_key_id",
        "api_key",
        r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b",
        "gitleaks (MIT) — aws-access-token; the prefix set is gitleaks', not AWS documentation",
    ),
    _rule(
        "github.token",
        "api_key",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b",
        "https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/",
    ),
    _rule(
        "github.fine_grained",
        "api_key",
        r"\bgithub_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}\b",
        "https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/",
    ),
    _rule(
        "gitlab.pat",
        "api_key",
        r"\bglpat-[A-Za-z0-9_-]{20}\b",
        "https://docs.gitlab.com/security/tokens/",
    ),
    _rule(
        "slack.bot_token",
        "api_key",
        r"\bxox[baprs]-[0-9A-Za-z-]{10,}",
        "https://api.slack.com/authentication/token-types",
    ),
    _rule(
        "slack.webhook",
        "api_key",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+",
        "https://api.slack.com/messaging/webhooks",
    ),
    _rule(
        "stripe.key",
        "api_key",
        r"\b(?:sk|rk)_live_[A-Za-z0-9]{10,}\b",
        "https://docs.stripe.com/keys",
    ),
    _rule(
        "sendgrid.api_key",
        "api_key",
        r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b",
        "https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys",
    ),
    _rule(
        "google.api_key",
        "api_key",
        r"\bAIza[A-Za-z0-9_-]{35}\b",
        "https://cloud.google.com/docs/authentication/api-keys",
    ),
    _rule(
        "openai.api_key",
        "api_key",
        r"\bsk-proj-[A-Za-z0-9_-]{20,}\b",
        "https://platform.openai.com/docs/api-reference/authentication",
    ),
    _rule(
        "anthropic.api_key",
        "api_key",
        r"\bsk-ant-api\d{2}-[A-Za-z0-9_-]{20,}\b",
        "https://docs.claude.com/en/api/getting-started",
    ),
    _rule(
        "npm.token",
        "api_key",
        r"\bnpm_[A-Za-z0-9]{36}\b",
        "https://docs.npmjs.com/about-access-tokens",
    ),
    _rule(
        "digitalocean.pat",
        "api_key",
        r"\bdop_v1_[0-9a-f]{64}\b",
        "https://docs.digitalocean.com/reference/api/create-personal-access-token/",
    ),
    _rule(
        "hashicorp.vault_token",
        "api_key",
        r"\bhvs\.[A-Za-z0-9_-]{20,}\b",
        "https://developer.hashicorp.com/vault/docs/concepts/tokens",
    ),
    _rule(
        "pypi.token",
        "api_key",
        r"\bpypi-[A-Za-z0-9_-]{32,}\b",
        "https://pypi.org/help/#apitoken",
    ),
    _rule(
        "nuget.api_key",
        "api_key",
        r"\boy2[a-z0-9]{43}\b",
        "https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package",
    ),
    _rule(
        "twilio.api_key_sid",
        "api_key",
        r"\bSK[0-9a-f]{32}\b",
        "https://www.twilio.com/docs/iam/api-keys",
    ),
    _rule(
        "azure.ad_client_secret",
        "api_key",
        r"[A-Za-z0-9_~.-]{3}8Q~[A-Za-z0-9_~.-]{30,}",
        "gitleaks (MIT) — azure-ad-client-secret",
    ),
    _rule(
        "jwt",
        "token",
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        "https://datatracker.ietf.org/doc/html/rfc7519 — the eyJ prefix is base64url of '{\"'",
    ),
    _rule(
        "oauth.bearer_header",
        "token",
        r"\bBearer\s+[A-Za-z0-9_\-.~+/]{16,}=*",
        "https://datatracker.ietf.org/doc/html/rfc6750#section-2.1",
    ),
    _rule(
        "google.refresh_token",
        "token",
        r"\b1//[A-Za-z0-9_-]{40,}",
        "https://developers.google.com/identity/protocols/oauth2",
    ),
    _rule(
        "basic_auth.header",
        "password",
        r"\bBasic\s+[A-Za-z0-9+/]{12,}={0,2}",
        "https://datatracker.ietf.org/doc/html/rfc7617",
    ),
    _rule(
        "pem.private_key",
        "private_key",
        r"-----BEGIN(?:\s+[A-Z0-9]+)*\s+PRIVATE KEY-----",
        "https://datatracker.ietf.org/doc/html/rfc7468#section-11",
    ),
    _rule(
        "pem.private_key_escaped",
        "private_key",
        r"-----BEGIN(?:\s+[A-Z0-9]+)*\s+PRIVATE KEY-----\\n",
        "Derived from the flattened form structured logging produces (no vendor document).",
    ),
    _rule(
        "url.embedded_credentials",
        "connection_string",
        r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s:@]+@",
        "https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1",
    ),
    _rule(
        "connection_string.password_field",
        "connection_string",
        r"(?i)(?:^|;)\s*(?:password|pwd)\s*=\s*[^;\s]{4,}",
        "https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax",
    ),
    _rule(
        "azure.storage_connection_string",
        "connection_string",
        r"(?i)AccountKey\s*=\s*[A-Za-z0-9+/]{64,}={0,2}",
        "https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string",
    ),
)

#: Words that mark a key name as sensitive, matched whole after splitting a key
#: name into words. Derived from the lexicon's sensitive group, reduced to the
#: words that carry the signal.
SENSITIVE_KEY_WORDS: dict[str, frozenset[str]] = {
    "password": frozenset({"password", "passwd", "pwd", "passphrase", "pass"}),
    "token": frozenset({"token", "authorization", "jwt", "bearer"}),
    "api_key": frozenset(
        {"apikey", "secret", "clientsecret", "accesskey", "secretkey", "subscriptionkey"}
    ),
    "private_key": frozenset({"privatekey", "signingkey", "pem"}),
    "connection_string": frozenset({"connectionstring", "dsn"}),
    "cookie": frozenset({"cookie", "setcookie", "jsessionid", "phpsessid", "csrftoken"}),
}

#: Key-name words that look sensitive but are not. Matching these must *not*
#: fire: they are the adversarial slice, and a rule set that trips on them is
#: reproducing the failure the project measures rather than avoiding it.
NEUTRALISING_KEY_WORDS: frozenset[str] = frozenset(
    {
        "count",
        "length",
        "size",
        "policy",
        "expiry",
        "expires",
        "expiresin",
        "type",
        "scheme",
        "algorithm",
        "name",
        "id",
        "version",
        "hash",
        "fingerprint",
        "thumbprint",
        "public",
        "publickey",
        "partition",
        "partitionkey",
        "rowkey",
        "sortkey",
        "cachekey",
        "routingkey",
        "idempotencykey",
        "strength",
        "attempts",
        "lastchanged",
        "rotatedat",
        "min",
        "max",
        "minlength",
        "requires",
        "has",
        "is",
        "tokenizer",
        "tokenizerversion",
        "secretary",
    }
)

#: Splits a key name into words across the four casing conventions real logs mix
#: (plan §4.2): PascalCase, snake_case, SCREAMING_SNAKE and kebab-case.
_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_key_words(key: str) -> list[str]:
    """Split a key name into lower-case words.

    Args:
        key: The attribute name.

    Returns:
        The words, lower-cased, with empties dropped. ``<slot:...>`` markers are
        stripped first so a body-mode fragment contributes no spurious words.
    """
    cleaned = re.sub(r"<slot:([^>]*)>", r"\1", key)
    return [word.lower() for word in _WORD_SPLIT.split(cleaned) if word]


def classify_key(key: str) -> str | None:
    """Classify an attribute name by its words.

    Args:
        key: The attribute name.

    Returns:
        The class the name suggests, or ``None`` when it suggests nothing.

    The neutralising check comes first and wins. ``PasswordPolicy`` contains
    ``password``, and a rule set that reads it as a password is exactly the
    false positive the adversarial slice measures; ``KeyCount`` and
    ``TokenExpiresIn`` are the same shape. Precedence rather than scoring keeps
    the behaviour explainable, which matters for a baseline whose job is to be
    a fair comparison rather than to win.
    """
    words = split_key_words(key)
    if not words:
        return None
    joined = "".join(words)

    # Neutralising words are checked individually, as adjacent pairs, and
    # against the whole joined name. The pair check is what catches
    # `PasswordLastChanged`, whose words split to ["password", "last",
    # "changed"] — neither "last" nor "changed" neutralises anything on its own,
    # but "lastchanged" does. Pairs rather than substrings keeps this from
    # suppressing a genuine name that merely contains a neutral word.
    bigrams = {words[i] + words[i + 1] for i in range(len(words) - 1)}
    if any(word in NEUTRALISING_KEY_WORDS for word in words):
        return None
    if bigrams & NEUTRALISING_KEY_WORDS:
        return None
    if joined in NEUTRALISING_KEY_WORDS:
        return None

    for label, markers in SENSITIVE_KEY_WORDS.items():
        if joined in markers or any(word in markers for word in words):
            return label
    return None


@dataclass(slots=True)
class RuleBaseline:
    """Control group 1: value patterns plus whole-word key-name matching.

    Attributes:
        name: System name in reports.
        kind: ``binary`` — a rule set fires or it does not.
        use_key_names: Whether key-name matching contributes. Turning it off
            isolates how much of the baseline's performance comes from the value
            patterns alone, which the report uses.
        matched: Per-unit record of which rule fired, for the error analysis
            NR-0002 §6 asks for.
    """

    name: str = "rules"
    kind: str = "binary"
    use_key_names: bool = True
    matched: list[str | None] = field(default_factory=list)

    def version(self) -> str:
        """Return the baseline's version."""
        return VERSION

    def decide(self, unit: DetectionUnit) -> tuple[str | None, str | None]:
        """Decide a single unit.

        Args:
            unit: The unit to classify.

        Returns:
            ``(label, rule_name)``, both ``None`` when nothing fired. Value
            patterns are tried before key names: a value that matches a vendor
            format is evidence in itself, while a key name is only ever a hint
            about a value that did not match anything.
        """
        for rule in VALUE_RULES:
            if rule.pattern.search(unit.value):
                return rule.label, rule.name

        if self.use_key_names:
            from_key = classify_key(unit.key)
            if from_key is not None:
                return from_key, f"key:{from_key}"
            # The template is weaker evidence than the attribute name, and is
            # only consulted for body-mode units, which have no real key.
            if unit.template and unit.key.startswith("<slot:"):
                from_template = classify_key(unit.template)
                if from_template is not None:
                    return from_template, f"template:{from_template}"
        return None, None

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score a sequence of units.

        Args:
            units: The units to score.

        Returns:
            Binary output: 1.0 on the decided class, nothing otherwise.
        """
        scores: list[dict[str, float]] = []
        self.matched = []
        for unit in units:
            label, rule_name = self.decide(unit)
            self.matched.append(rule_name)
            scores.append({label: 1.0} if label else {})
        return SystemOutput(
            name=self.name,
            kind="binary",
            scores=scores,
            version=self.version(),
            notes={
                "value_rules": str(len(VALUE_RULES)),
                "key_names": "on" if self.use_key_names else "off",
            },
        )
