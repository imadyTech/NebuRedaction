"""Checks the committed fixture model against its recorded outputs.

The same two files drive `dotnet/tests/.../FixtureModelTests.cs`. Having both
sides read them means a change to the Python reader that quietly alters a
prediction fails here, rather than being absorbed into a regenerated fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neburedaction.detector import SecretDetector, SecretDetectorOptions

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MODEL = FIXTURES / "fixture-model-v1.nrm"
EXPECTED = json.loads((FIXTURES / "fixture-model-expected-v1.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    EXPECTED["cases"],
    ids=[f"{case['key']}-{len(case['value'])}" for case in EXPECTED["cases"]],
)
def test_the_reference_reproduces_the_recorded_outputs(case: dict) -> None:
    """Regenerating the fixture must not be the way a change gets accepted."""
    balanced = SecretDetector.load(MODEL, SecretDetectorOptions(profile="balanced"))
    conservative = SecretDetector.load(MODEL, SecretDetectorOptions(profile="conservative"))

    result = balanced.classify(case["key"], case["value"], case["template"], case["scope"])
    for label, probability in case["probabilities"].items():
        assert result.probabilities[label] == pytest.approx(probability, abs=1e-12)
    assert result.category == case["category"]
    assert result.score == pytest.approx(case["score"], abs=1e-12)
    assert result.should_redact is case["should_redact_balanced"]
    assert (
        conservative.classify(
            case["key"], case["value"], case["template"], case["scope"]
        ).should_redact
        is case["should_redact_conservative"]
    )


def test_the_fixture_exercises_more_than_one_branch() -> None:
    """A fixture where every case agrees proves less than it appears to."""
    balanced = {case["should_redact_balanced"] for case in EXPECTED["cases"]}
    assert balanced == {True, False}
    assert any(
        case["should_redact_balanced"] != case["should_redact_conservative"]
        for case in EXPECTED["cases"]
    )
