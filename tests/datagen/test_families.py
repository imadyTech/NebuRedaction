"""Tests for the format-family registry and the individual families."""

from __future__ import annotations

import random
import re

import pytest
from datagen.families import (
    REGISTRY,
    Family,
    families_for_label,
    negative_families,
    positive_families,
    register,
)
from datagen.families._primitives import base62_crc32, verify_base62_crc32
from datagen.unit import LABELS

#: Minimum family counts required by NR-0001 §2.2 items 1 and 3.
REQUIRED_POSITIVE_TOTAL = 30
REQUIRED_NEGATIVE_TOTAL = 20
REQUIRED_PER_LABEL = {
    "api_key": 12,
    "token": 5,
    "connection_string": 5,
    "password": 3,
    "private_key": 2,
    "cookie": 3,
}

SEED = 20261011


def test_at_least_thirty_positive_families():
    assert len(positive_families()) >= REQUIRED_POSITIVE_TOTAL


def test_at_least_twenty_negative_families():
    assert len(negative_families()) >= REQUIRED_NEGATIVE_TOTAL


@pytest.mark.parametrize(("label", "minimum"), sorted(REQUIRED_PER_LABEL.items()))
def test_per_label_family_minimum(label: str, minimum: int):
    assert len(families_for_label(label)) >= minimum


def test_every_label_is_known():
    for family in REGISTRY.values():
        assert family.label in LABELS


def test_negative_prefix_matches_label():
    for family in REGISTRY.values():
        assert family.name.startswith("neg.") == (family.label == "none"), family.name


def test_family_names_are_well_formed():
    pattern = re.compile(r"^[a-z0-9]+(?:[._][a-z0-9]+)*$")
    for name in REGISTRY:
        assert pattern.match(name), name


def test_every_family_documents_its_format():
    """NR-0001 §2.2 item 1: each positive family must state where its format came from.

    Families with no vendor documentation say so explicitly, which is why the
    assertion is on substance rather than on the string starting with 'http'.
    """
    for family in REGISTRY.values():
        assert family.doc_url, family.name
        assert len(family.doc_url) > 20, family.name
        assert family.doc_url.startswith("http") or family.doc_url.startswith("No vendor format"), (
            f"{family.name}: doc_url must be a URL or an explicit statement that no "
            f"vendor format exists, got {family.doc_url!r}"
        )


def test_positive_families_cite_a_url_or_explain_why_not():
    documented = [f for f in positive_families() if f.doc_url.startswith("http")]
    assert len(documented) >= REQUIRED_POSITIVE_TOTAL, (
        f"at least 30 positive families should cite public documentation; only {len(documented)} do"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_family_generates_a_non_empty_string(name: str):
    family = REGISTRY[name]
    rng = random.Random(SEED)
    for _ in range(20):
        value = family.generate(rng)
        assert isinstance(value, str)
        assert value.strip(), name


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_family_is_deterministic_for_a_given_seed(name: str):
    family = REGISTRY[name]
    first = [family.generate(random.Random(SEED + index)) for index in range(10)]
    second = [family.generate(random.Random(SEED + index)) for index in range(10)]
    assert first == second, name


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_family_produces_varied_values(name: str):
    """A family that returns one value would silently collapse the dataset."""
    family = REGISTRY[name]
    values = {family.generate(random.Random(SEED + index)) for index in range(40)}
    # Some families draw from a small fixed set by design (placeholder passwords,
    # MIME types, user agents), so the floor is low but not one.
    assert len(values) >= 2, name


def test_multiline_flag_matches_reality():
    rng = random.Random(SEED)
    for family in REGISTRY.values():
        values = [family.generate(rng) for _ in range(10)]
        if family.multiline:
            assert any("\n" in value for value in values), family.name
        else:
            assert not any("\n" in value for value in values), family.name


def test_register_rejects_a_duplicate_name():
    existing = next(iter(REGISTRY))
    with pytest.raises(ValueError, match="duplicate family name"):
        register(
            Family(
                name=existing,
                label="api_key",
                generate=lambda rng: "x",
                doc_url="No vendor format: test double for the duplicate-name guard.",
            )
        )


def test_register_rejects_a_mislabelled_negative():
    with pytest.raises(ValueError, match="must agree"):
        register(
            Family(
                name="neg.wrongly_labelled",
                label="api_key",
                generate=lambda rng: "x",
                doc_url="No vendor format: test double for the polarity guard.",
            )
        )


def test_register_rejects_a_positive_named_as_negative():
    with pytest.raises(ValueError, match="must agree"):
        register(
            Family(
                name="vendor.wrongly_labelled",
                label="none",
                generate=lambda rng: "x",
                doc_url="No vendor format: test double for the polarity guard.",
            )
        )


class TestChecksummedFamilies:
    """NR-0001 §6: families whose format carries a check value need unit tests.

    GitHub's token formats are the ones in this corpus that do. The checksum is
    what lets a scanner reject a mistyped token without calling the API, so a
    generated token whose checksum did not verify would be the wrong shape in a
    way no length or alphabet assertion would notice.
    """

    def test_base62_crc32_round_trips(self):
        for payload in ("abc", "0" * 30, "Zz9" * 10, ""):
            checksum = base62_crc32(payload)
            assert len(checksum) == 6
            assert verify_base62_crc32(payload, checksum)

    def test_base62_crc32_rejects_a_changed_payload(self):
        checksum = base62_crc32("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert not verify_base62_crc32("aaaaaaaaaaaaaaaaaaaaaaaaaaaaab", checksum)

    @pytest.mark.parametrize(
        ("name", "prefix"),
        [
            ("github.pat_classic", "ghp_"),
            ("github.app_installation_token", "ghs_"),
        ],
    )
    def test_github_classic_token_checksum_verifies(self, name: str, prefix: str):
        family = REGISTRY[name]
        for index in range(50):
            token = family.generate(random.Random(SEED + index))
            assert token.startswith(prefix)
            body = token[len(prefix) :]
            assert len(body) == 36
            assert verify_base62_crc32(body[:30], body[30:]), token

    def test_github_fine_grained_token_checksum_verifies(self):
        family = REGISTRY["github.pat_fine_grained"]
        for index in range(50):
            token = family.generate(random.Random(SEED + index))
            assert token.startswith("github_pat_")
            identifier, _, secret = token[len("github_pat_") :].partition("_")
            assert len(identifier) == 22
            assert len(secret) == 59
            assert verify_base62_crc32(secret[:53], secret[53:]), token


class TestFamilyShapes:
    """Spot checks that the headline formats really have the documented shape."""

    def test_aws_access_key_id(self):
        family = REGISTRY["aws.access_key_id"]
        for index in range(30):
            value = family.generate(random.Random(SEED + index))
            assert re.fullmatch(r"AKIA[A-Z2-7]{16}", value), value

    def test_jwt_has_three_base64url_parts_and_a_decodable_header(self):
        import base64
        import json

        for name in ("jwt.hs256", "jwt.rs256", "jwt.es256"):
            family = REGISTRY[name]
            value = family.generate(random.Random(SEED))
            parts = value.split(".")
            assert len(parts) == 3, name
            padded = parts[0] + "=" * (-len(parts[0]) % 4)
            header = json.loads(base64.urlsafe_b64decode(padded))
            assert header["typ"] == "JWT"
            assert header["alg"] == name.split(".")[1].upper()

    def test_pem_blocks_are_wrapped_at_64_columns(self):
        family = REGISTRY["pem.rsa_private_key"]
        value = family.generate(random.Random(SEED))
        lines = value.splitlines()
        assert lines[0] == "-----BEGIN RSA PRIVATE KEY-----"
        assert lines[-1] == "-----END RSA PRIVATE KEY-----"
        assert all(len(line) <= 64 for line in lines[1:-1])

    def test_connection_strings_contain_a_credential_separator(self):
        for name in ("postgres.connection_uri", "mysql.connection_uri", "redis.connection_uri"):
            value = REGISTRY[name].generate(random.Random(SEED))
            assert "@" in value and "://" in value, name

    def test_base64_negatives_decode_to_text(self):
        import base64

        value = REGISTRY["neg.base64_of_text"].generate(random.Random(SEED))
        decoded = base64.b64decode(value).decode("utf-8")
        assert decoded.isprintable()

    def test_hosts_are_always_reserved_domains(self):
        """No generated value may point at a domain someone could actually own."""
        reserved = (".example", ".invalid", ".test", "example.com", "example.org")
        rng = random.Random(SEED)
        for name in (
            "postgres.connection_uri",
            "http.basic_credentials_in_url",
            "neg.url_without_credentials",
            "neg.hostname",
        ):
            for _ in range(20):
                value = REGISTRY[name].generate(rng)
                assert any(token in value for token in reserved), (name, value)
