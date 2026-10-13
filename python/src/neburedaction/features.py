"""The shared feature extractor — the normative definition (NR-0004 §2 item 1).

Plan §4.2 specifies the features; this module **is** the specification. NR-0005
implements the same extraction in .NET and in the published Python inference
path, and plan §6.4's G3 requires the two to agree to within 1e-6 per sample.
``tests/training/feature-vectors.json`` is generated from this module and
committed so the .NET port has something to be checked against rather than a
prose description to interpret.

Three properties make that port possible, and each one rules out an obvious
shortcut:

**The hash is FNV-1a, not Python's ``hash()``.** Python's built-in hash of a
string is salted per process unless ``PYTHONHASHSEED`` is fixed, so a model
trained in one process would read different features in the next — and no .NET
implementation could ever match it. FNV-1a 32-bit is fully specified in a dozen
lines, is byte-identical anywhere, and ports to C# without a dependency.

**Nothing here needs a dependency.** The package ships with ``dependencies =
[]`` and this module keeps it that way: standard library only, no numpy. That
is also what makes the .NET port a transcription rather than a reimplementation.

**Feature order is fixed and declared.** :data:`DENSE_FEATURES` is the order a
model's dense coefficients are in. Reordering it silently invalidates every
trained model, so the version is bumped when it changes and the loader checks
the version.

**On value-side memorisation (plan §7.5 item 4).** Character n-grams of a value
are exactly the features that can retain a fragment of a real credential, which
matters for v0.2's user-retrained models. Two mitigations are built in here
rather than bolted on later: the dense features are *morphological* — they
describe shape, not content, and cannot be inverted to a value — and the hashed
n-gram features support a minimum-occurrence mask (:func:`rare_bucket_mask`)
that training applies so a bucket seen only once or twice never gets a weight.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: Version of the feature definition. Any change to the dense feature list, the
#: hashing, the tokenisation or the n-gram range is a new version, because a
#: model trained under one definition reads nonsense under another.
FEATURE_VERSION = "1.0"

#: FNV-1a 32-bit parameters (fnv-1a, 32-bit offset basis and prime).
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF

#: Character n-gram sizes taken from the value (plan §4.2: 1-4).
NGRAM_SIZES: tuple[int, ...] = (1, 2, 3, 4)

#: Hashed-feature bucket counts plan §4.2 allows, 2^14 to 2^16.
MIN_BUCKETS = 1 << 14
MAX_BUCKETS = 1 << 16
DEFAULT_BUCKETS = 1 << 15

#: Namespaces keep the three hashed feature families from colliding with each
#: other: the same token as a key word and as a context word is two features.
NS_VALUE_NGRAM = "v"
NS_KEY_WORD = "k"
NS_CONTEXT_WORD = "c"

#: Key-name splitting: camelCase, snake_case, kebab-case, dots, and digits.
_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")

#: `<slot:name>` markers are stripped so a body-mode fragment contributes the
#: slot's name rather than the punctuation around it.
_SLOT = re.compile(r"<slot:([^>]*)>")

#: Template placeholders, which carry no information and would otherwise become
#: high-frequency context words: `<*>`, `{Name}`, `%s`, `$1`.
_PLACEHOLDER = re.compile(r"<\*>|\{[^}]*\}|%[sdv]|\$\d+")

_HEX = re.compile(r"^[0-9a-fA-F]+$")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_BASE64URL = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_JWT = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$")
_URL_CREDENTIALS = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]+:[^/\s:@]+@")

#: Key-name words that suggest a credential, and the abbreviations for them.
#: A *flag*, not a rule: the model decides what to do with it. Deliberately
#: small — NR-0002 measured that key-name matching produced every one of the
#: rule baseline's 1,296 false positives, so this is one input among many
#: rather than a decision.
SENSITIVE_KEY_WORDS: frozenset[str] = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "jwt",
        "key",
        "oauth",
        "passphrase",
        "passwd",
        "password",
        "pem",
        "pin",
        "pwd",
        "secret",
        "session",
        "sig",
        "signature",
        "token",
    }
)

#: Words that *neutralise* a credential-suggesting name: `PasswordPolicy`,
#: `TokenExpiresIn`, `KeyCount`. Same vocabulary idea as NR-0002's rule layer,
#: again as a flag rather than a veto.
NEUTRALISING_KEY_WORDS: frozenset[str] = frozenset(
    {
        "algorithm",
        "attempt",
        "attempts",
        "changed",
        "count",
        "enabled",
        "expired",
        "expires",
        "expiry",
        "failed",
        "format",
        "hash",
        "id",
        "kind",
        "last",
        "lastchanged",
        "length",
        "name",
        "policy",
        "prefix",
        "required",
        "rotation",
        "strength",
        "ttl",
        "type",
        "valid",
        "version",
    }
)

#: Dense feature names, **in the order a model's coefficients expect**.
DENSE_FEATURES: tuple[str, ...] = (
    "value_length_log",
    "value_entropy",
    "value_entropy_normalised",
    "ratio_lower",
    "ratio_upper",
    "ratio_digit",
    "ratio_symbol",
    "ratio_whitespace",
    "distinct_ratio",
    "longest_run_log",
    "class_transitions_ratio",
    "is_hex",
    "is_base64",
    "is_base64url",
    "is_uuid",
    "is_jwt_shaped",
    "has_pem_header",
    "has_url_credentials",
    "has_connection_string_password",
    "starts_with_eyj",
    "is_multiline",
    "has_dot_segments",
    "has_dash_segments",
    "has_underscore",
    "key_word_count",
    "key_has_sensitive_word",
    "key_has_neutralising_word",
    "key_is_slot",
    "template_has_sensitive_word",
    "scope_is_attribute",
    "has_template",
)


def fnv1a(text: str) -> int:
    """Hash a string with FNV-1a, 32-bit.

    Args:
        text: The string to hash.

    Returns:
        A 32-bit unsigned hash.

    Specified rather than borrowed: Python's ``hash()`` is salted per process,
    so using it would make a trained model unreadable by the next process and
    unportable to .NET. The bytes hashed are UTF-8, which is the one detail a
    port has to match and the one a prose description usually omits.
    """
    value = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK32
    return value


def bucket_of(token: str, namespace: str, buckets: int) -> tuple[int, float]:
    """Map a token to a bucket and a sign.

    Args:
        token: The token.
        namespace: One of :data:`NS_VALUE_NGRAM`, :data:`NS_KEY_WORD`,
            :data:`NS_CONTEXT_WORD`.
        buckets: Number of buckets; must be a power of two in range.

    Returns:
        ``(bucket, sign)`` with the sign in ``{-1.0, +1.0}``.

    Raises:
        ValueError: If ``buckets`` is out of the range plan §4.2 allows or is
            not a power of two. A non-power-of-two would make the masking below
            a modulo, which biases the low buckets.

    The sign is the standard hashing-trick trick: two colliding tokens cancel
    on average instead of always reinforcing, which keeps a collision from
    looking like a strong signal. It is taken from a bit the bucket does not
    use, so bucket and sign are independent.
    """
    if not MIN_BUCKETS <= buckets <= MAX_BUCKETS or buckets & (buckets - 1):
        raise ValueError(
            f"buckets must be a power of two in [{MIN_BUCKETS}, {MAX_BUCKETS}], got {buckets}"
        )
    digest = fnv1a(f"{namespace}\x1f{token}")
    bucket = digest & (buckets - 1)
    sign = 1.0 if (digest >> 31) & 1 else -1.0
    return bucket, sign


def split_key_words(key: str) -> list[str]:
    """Split a key name into lower-case words.

    Args:
        key: The attribute name.

    Returns:
        The words, lower-cased, empties dropped. ``<slot:user>`` contributes
        ``user``.
    """
    cleaned = _SLOT.sub(r"\1", key)
    return [word.lower() for word in _WORD_SPLIT.split(cleaned) if word]


def template_words(template: str | None) -> list[str]:
    """Split a message template into lower-case context words.

    Args:
        template: The template, or None.

    Returns:
        The words, with placeholders removed first so that `<*>` and `{Name}`
        do not become the most frequent context tokens in the corpus.
    """
    if not template:
        return []
    cleaned = _PLACEHOLDER.sub(" ", template)
    return [word.lower() for word in _WORD_SPLIT.split(cleaned) if word]


def shannon_entropy(value: str) -> float:
    """Shannon entropy of a string's characters, in bits.

    Args:
        value: The string.

    Returns:
        Entropy in bits per character, 0.0 for an empty string.
    """
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for character in value:
        counts[character] = counts.get(character, 0) + 1
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _character_class(character: str) -> str:
    """Classify one character for the transition count."""
    if character.islower():
        return "l"
    if character.isupper():
        return "u"
    if character.isdigit():
        return "d"
    if character.isspace():
        return "w"
    return "s"


@dataclass(frozen=True, slots=True)
class Features:
    """One unit's extracted features.

    Attributes:
        dense: Values in :data:`DENSE_FEATURES` order.
        hashed: Bucket index to accumulated signed count.
        version: The feature definition version these were produced under.
    """

    dense: tuple[float, ...]
    hashed: dict[int, float] = field(default_factory=dict)
    version: str = FEATURE_VERSION

    def as_dict(self) -> dict[str, float]:
        """Return the dense features by name, for tests and reports."""
        return dict(zip(DENSE_FEATURES, self.dense, strict=True))


def dense_features(
    key: str, value: str, template: str | None = None, scope: str | None = None
) -> tuple[float, ...]:
    """Compute the morphological feature vector.

    Args:
        key: The attribute name.
        value: The attribute value.
        template: The message template, if any.
        scope: The scope name, if any.

    Returns:
        Values in :data:`DENSE_FEATURES` order.

    These describe *shape*, never content, which is what plan §7.5 item 4 asks
    the value side to be built from: none of them can be inverted to recover a
    credential, so a model that relies on them cannot memorise one.
    """
    length = len(value)
    stripped = value.strip()
    counts = {"l": 0, "u": 0, "d": 0, "s": 0, "w": 0}
    transitions = 0
    previous: str | None = None
    longest_run = 0
    run = 0
    for character in value:
        kind = _character_class(character)
        counts[kind] += 1
        if previous is not None and kind != previous:
            transitions += 1
        run = run + 1 if previous == kind else 1
        longest_run = max(longest_run, run)
        previous = kind

    def ratio(kind: str) -> float:
        return counts[kind] / length if length else 0.0

    entropy = shannon_entropy(value)
    # Normalised against the alphabet actually used, so a 64-character hex
    # digest and a 40-character base64 secret are comparable — the same reason
    # NR-0002's entropy baseline normalised, and plan §3.3 names that case as
    # the hard one.
    distinct = len(set(value))
    ceiling = math.log2(distinct) if distinct > 1 else 1.0

    words = split_key_words(key)
    context = template_words(template)

    return (
        math.log1p(length),
        entropy,
        min(1.0, entropy / ceiling) if ceiling > 0 else 0.0,
        ratio("l"),
        ratio("u"),
        ratio("d"),
        ratio("s"),
        ratio("w"),
        distinct / length if length else 0.0,
        math.log1p(longest_run),
        transitions / (length - 1) if length > 1 else 0.0,
        float(bool(_HEX.match(stripped)) and length >= 8),
        float(bool(_BASE64.match(stripped)) and length >= 8),
        float(bool(_BASE64URL.match(stripped)) and length >= 8),
        float(bool(_UUID.match(stripped))),
        float(bool(_JWT.match(stripped)) and stripped.count(".") == 2),
        float("-----BEGIN" in value),
        float(bool(_URL_CREDENTIALS.search(value))),
        float(";password=" in value.lower() or ";pwd=" in value.lower()),
        float(stripped.startswith("eyJ")),
        float("\n" in value),
        float(stripped.count(".") >= 2),
        float(stripped.count("-") >= 2),
        float("_" in stripped),
        float(len(words)),
        float(any(word in SENSITIVE_KEY_WORDS for word in words)),
        float(any(word in NEUTRALISING_KEY_WORDS for word in words)),
        float(key.startswith("<slot:")),
        float(any(word in SENSITIVE_KEY_WORDS for word in context)),
        float((scope or "").startswith("otel.attribute")),
        float(bool(template)),
    )


def hashed_features(
    key: str,
    value: str,
    template: str | None = None,
    scope: str | None = None,
    buckets: int = DEFAULT_BUCKETS,
) -> dict[int, float]:
    """Compute the hashed feature map.

    Args:
        key: The attribute name.
        value: The attribute value.
        template: The message template, if any.
        scope: The scope name, if any.
        buckets: Number of hash buckets.

    Returns:
        Bucket index to accumulated signed count. Absent buckets are zero; the
        map is sparse because 2^15 buckets against a few hundred active
        features would otherwise be 99.9% zeros in every row.

    The three families are namespaced, so the token ``token`` appearing as a
    key word and in a template are different features rather than the same one
    counted twice.
    """
    accumulated: dict[int, float] = {}

    def add(token: str, namespace: str) -> None:
        bucket, sign = bucket_of(token, namespace, buckets)
        accumulated[bucket] = accumulated.get(bucket, 0.0) + sign

    for size in NGRAM_SIZES:
        for start in range(len(value) - size + 1):
            add(value[start : start + size], NS_VALUE_NGRAM)
    for word in split_key_words(key):
        add(word, NS_KEY_WORD)
    for word in template_words(template):
        add(word, NS_CONTEXT_WORD)
    for word in split_key_words(scope or ""):
        add(word, NS_CONTEXT_WORD)
    return accumulated


def extract(
    key: str,
    value: str,
    template: str | None = None,
    scope: str | None = None,
    buckets: int = DEFAULT_BUCKETS,
) -> Features:
    """Extract both feature families for one detection unit.

    Args:
        key: The attribute name.
        value: The attribute value.
        template: The message template, if any.
        scope: The scope name, if any.
        buckets: Number of hash buckets.

    Returns:
        The features.
    """
    return Features(
        dense=dense_features(key, value, template, scope),
        hashed=hashed_features(key, value, template, scope, buckets),
        version=FEATURE_VERSION,
    )


def rare_bucket_mask(
    rows: Iterable[dict[int, float]], minimum_count: int, buckets: int = DEFAULT_BUCKETS
) -> set[int]:
    """Find hashed buckets seen in fewer than ``minimum_count`` rows.

    Args:
        rows: The hashed feature maps of the training set.
        minimum_count: Smallest number of rows a bucket must appear in.
        buckets: Number of buckets, for validation only.

    Returns:
        The buckets to suppress.

    Raises:
        ValueError: If ``minimum_count`` is below one.

    This is plan §7.5 item 4's "minimum occurrence count for rare n-grams". The
    concern is a user-retrained model memorising a fragment of a real
    credential: a 4-gram that occurs in exactly one training row can carry a
    large weight that is, in effect, a lookup for that row's value. Suppressing
    such buckets costs almost nothing — they generalise to nothing by
    definition — and removes the mechanism.

    Applied at **training** time, by zeroing the masked buckets before fitting,
    so the learned weight is zero and inference needs no mask. That keeps the
    extractor's output independent of which corpus a model was trained on,
    which is what lets one extractor serve every model and the .NET port stay
    simple.
    """
    if minimum_count < 1:
        raise ValueError(f"minimum_count must be at least 1, got {minimum_count}")
    occurrences: dict[int, int] = {}
    for row in rows:
        for bucket in row:
            occurrences[bucket] = occurrences.get(bucket, 0) + 1
    return {bucket for bucket, count in occurrences.items() if count < minimum_count}


def apply_mask(row: dict[int, float], mask: set[int]) -> dict[int, float]:
    """Drop masked buckets from one feature map.

    Args:
        row: A hashed feature map.
        mask: Buckets to suppress, from :func:`rare_bucket_mask`.

    Returns:
        A new map without the masked buckets.
    """
    return {bucket: weight for bucket, weight in row.items() if bucket not in mask}


def to_dense_array(
    features: Features, mask: set[int] | None = None, buckets: int = DEFAULT_BUCKETS
) -> list[float]:
    """Flatten features into one vector, dense part first.

    Args:
        features: The extracted features.
        mask: Buckets to suppress.
        buckets: Number of hash buckets.

    Returns:
        A list of ``len(DENSE_FEATURES) + buckets`` floats.

    Provided for completeness and for the test vectors. Training uses the
    sparse representation: at 2^15 buckets this is 32,799 floats per row and
    200,000 rows would be 26 GB.
    """
    row = [0.0] * (len(DENSE_FEATURES) + buckets)
    row[: len(DENSE_FEATURES)] = list(features.dense)
    suppressed = mask or set()
    for bucket, weight in features.hashed.items():
        if bucket not in suppressed:
            row[len(DENSE_FEATURES) + bucket] = weight
    return row


def feature_names(buckets: int = DEFAULT_BUCKETS) -> Sequence[str]:
    """Return every feature's name, in vector order.

    Args:
        buckets: Number of hash buckets.

    Returns:
        Dense names followed by ``hashed_<index>`` names.
    """
    return list(DENSE_FEATURES) + [f"hashed_{index}" for index in range(buckets)]
