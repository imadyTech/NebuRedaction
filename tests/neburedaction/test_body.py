"""Checks the body tokenizer against the committed shared cases.

The same fixture drives `dotnet/tests/.../BodyTokenizerTests.cs`, which is the
point: NR-0005 §2.4 requires the two tokenizers to agree, and they can only be
shown to agree against something neither of them computes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neburedaction.body import body_candidates, has_value_shape, is_key_like

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "body-cases-v1.json"


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_candidates_match_the_fixture(case: dict) -> None:
    """The tokenizer reproduces the committed expectation for each case."""
    expected = [(item["key"], item["value"], item["scope"]) for item in case["candidates"]]
    actual = [
        (candidate.key, candidate.value, candidate.scope)
        for candidate in body_candidates(case["message"])
    ]
    assert actual == expected


def test_the_fixture_is_regenerated_rather_than_edited() -> None:
    """Every case carries the reason it exists.

    A case whose reason was deleted is a case nobody can argue with, which is
    how a fixture stops being a specification and becomes a record of whatever
    the code happened to do.
    """
    for case in _cases():
        assert case["why"].strip(), case["name"]


def test_prose_is_rejected_and_structure_is_not() -> None:
    """The decision `has_value_shape` exists to make."""
    assert not has_value_shape("established")
    assert not has_value_shape("auckland")
    assert has_value_shape("9f86d081884c7d659a2feaa0c55ad015")
    assert has_value_shape("3f2504e0-4f89-41d3-9a0c-0305e82c3301")
    assert has_value_shape("a1b2c3d4e5")


def test_pure_letters_need_length_and_mixed_case() -> None:
    """Both conditions, not either."""
    assert not has_value_shape("averylongallloweralpha")
    assert not has_value_shape("ShortMixed")
    assert has_value_shape("VeryLongMixedCaseWordHere")


def test_key_likeness() -> None:
    """A key name is key-like by a credential word, a separator or a digit."""
    assert is_key_like("password")
    assert is_key_like("PASSWORD")
    assert is_key_like("access_token")
    assert is_key_like("db.host")
    assert is_key_like("retry2")
    assert not is_key_like("received")
