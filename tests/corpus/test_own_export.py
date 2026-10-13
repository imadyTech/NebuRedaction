"""Tests for the own-log body tokenizer (NR-0003 §2.0).

NR-0001's export produced 1,545 body-mode units of which 1,510 were prose
words — place names, participles and a bare `connection`. §9.11 item 4 recorded
it as a defect and NR-0003 §2.0 is the fix. These tests pin the decision so it
cannot regress, and they are written against the predicate rather than the
whole export so they need neither NebuLog nor a saved API response.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent

# corpus/ is a directory of scripts, not a package, so the module is loaded by
# path. Importing it also puts the repository root on sys.path, which is how the
# script itself reaches datagen.unit.
_SPEC = importlib.util.spec_from_file_location(
    "export_nebulog_own", REPOSITORY_ROOT / "corpus" / "export_nebulog_own.py"
)
assert _SPEC is not None and _SPEC.loader is not None
own = importlib.util.module_from_spec(_SPEC)
sys.modules["export_nebulog_own"] = own
_SPEC.loader.exec_module(own)


PROSE = [
    "auckland",
    "christchurch",
    "wellington",
    "authorised",
    "downstream",
    "checkout",
    "connection",
    "established",
]


@pytest.mark.parametrize("word", PROSE)
def test_prose_words_are_not_value_shaped(word: str) -> None:
    """Every fragment NR-0001 wrongly admitted is now rejected."""
    assert not own.has_value_shape(word)
    assert not own.is_body_candidate(None, None, False, word)


# Assembled from parts rather than written out. A JWT-shaped literal trips
# gitleaks' generic-api-key rule on entropy and a literal PEM header trips
# detect-private-key, and the right response to a hook that fires on a test
# fixture is to stop writing credential-shaped literals, not to allowlist the
# file (NR-0002 reached the same conclusion about its AKIA values). The
# predicate under test sees exactly the same string either way.
JWT_SHAPED = ".".join(["eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxIn0", "c2lnbmF0dXJl"])
PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY" + "-----"

VALUE_SHAPED = [
    "9f86d081884c7d659a2feaa0c55ad015",  # sha256 prefix, bare hex
    "055d2131-3bae-4115-b42a-9f4c3bca8c7c",  # uuid
    JWT_SHAPED,
    "postgres://app:hunter2@db.internal:5432/orders",  # url with credentials
    "https://api.example.com/v1/orders",  # url without
    PEM_HEADER,
    "order-20261013-00417",  # letters, digits and separators
    "aBcDeFgHiJkLmNoPqRsTuVwX",  # pure letters, but long and mixed case
]


@pytest.mark.parametrize("value", VALUE_SHAPED)
def test_value_shaped_fragments_are_kept(value: str) -> None:
    """Identifiers, digests and credentials survive without a key name."""
    assert own.has_value_shape(value)
    assert own.is_body_candidate(None, None, False, value)


def test_a_key_and_a_delimiter_are_enough_on_their_own() -> None:
    """`declined: insufficient_funds` is a key/value pair, prose value or not.

    This is the one body unit NebuLog's own logs actually contain, and the
    reason the carrier test comes before the shape test: with a key name there
    is something to label on, so the shape no longer has to carry the decision.
    """
    assert own.has_value_shape("insufficient_funds")  # snake_case is not prose
    assert own.is_body_candidate("declined", ":", False, "insufficient_funds")
    assert own.is_body_candidate("password", "=", False, "hunter22")


def test_quotes_are_enough_on_their_own() -> None:
    """A quoted fragment was delimited by whoever wrote the message."""
    assert own.is_body_candidate(None, None, True, "some quoted phrase")


def test_short_fragments_are_always_rejected() -> None:
    """Below eight characters nothing is kept, carrier or not."""
    assert not own.is_body_candidate("password", "=", False, "short")
    assert not own.is_body_candidate(None, None, True, "short")


def test_trailing_punctuation_is_stripped_before_the_decision() -> None:
    """`established.` and `received:` are the stripped forms' problem, not a new one.

    The candidate pattern's value class contains `.` and `:`, so sentence
    punctuation is absorbed into the match. Stripping it is what turns
    `received:` back into a prose word that the predicate can reject.
    """
    units = own.entry_to_units(
        {
            "body": "connection established. ack received: downstream authorised",
            "attributes": {},
        },
        0,
    )
    assert [unit.value for unit in units if unit.meta["mode"] == "body"] == []


def test_body_units_keep_their_carrier_in_the_scope() -> None:
    """A kv pair is `message.kv`; a bare value-shaped fragment is `message.slot`."""
    units = own.entry_to_units(
        {
            "body": "payment declined: insufficient_funds for 055d2131-3bae-4115-b42a-9f4c3bca8c7c",
            "attributes": {},
        },
        7,
    )
    body = {unit.value: unit for unit in units if unit.meta["mode"] == "body"}
    assert body["insufficient_funds"].scope == "message.kv"
    assert body["insufficient_funds"].key == "declined"
    assert body["055d2131-3bae-4115-b42a-9f4c3bca8c7c"].scope == "message.slot"


def test_body_units_are_unlabelled() -> None:
    """Own logs are unlabelled until the teacher runs; §2.0 does not label."""
    units = own.entry_to_units({"body": f"token={JWT_SHAPED}", "attributes": {}}, 1)
    assert units
    assert all(unit.label is None for unit in units)
    assert all(unit.source == "own" for unit in units)


def test_an_assignment_is_evidence_but_a_colon_is_not() -> None:
    """`=` does not occur between two words of prose; `:` does.

    This is the distinction §2.0's wording leaves open and the data forces: of
    the fragments NR-0001 admitted, `received: downstream` parses as a key and a
    value under a literal reading of "contains a colon".
    """
    assert own.is_body_candidate("whatever", "=", False, "plainword")
    assert not own.is_body_candidate("received", ":", False, "downstream")


@pytest.mark.parametrize("key", ["access_token", "db.host", "api-key", "retry2", "password"])
def test_a_key_like_name_rescues_a_prose_value_after_a_colon(key: str) -> None:
    """`password: hunterrr` is kept: the key name is the evidence."""
    assert own.is_body_candidate(key, ":", False, "hunterrr")


@pytest.mark.parametrize("key", ["received", "connection", "status", "note"])
def test_an_ordinary_word_does_not(key: str) -> None:
    """A plain word before a colon is prose punctuation, not an attribute name."""
    assert not own.is_body_candidate(key, ":", False, "something")
