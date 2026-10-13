"""Tests for the rule labeller (NR-0003 §2.1).

What matters here is the *confidence tier*, because §2.2 item 3 builds the
teacher's call scope from it: a unit wrongly promoted to high confidence never
gets a teacher opinion, so the tier is a decision about what goes unchecked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datagen.loader import ThirdPartyDataError
from datagen.unit import DetectionUnit
from labeling.rules import (
    HIGH,
    LOW,
    NO_EVIDENCE,
    RuleLabel,
    label_unit,
    label_units,
    read_labels,
    statistics,
    write_labels,
)
from labeling.rules.__main__ import DEFAULT_CORPORA, run


def unit(key: str, value: str, **extra: object) -> DetectionUnit:
    """Build a synthetic unit for a test case."""
    return DetectionUnit(id=f"t-{key}", key=key, value=value, source="synth", **extra)  # type: ignore[arg-type]


#: An AWS access key id, assembled from parts the way NR-0002's tests do, so
#: no credential-shaped literal enters the repository for gitleaks to find.
ACCESS_KEY_ID = "AKIA" + "Q" * 16


def test_a_matched_value_pattern_is_high_confidence() -> None:
    """Structural evidence, so the teacher only samples these."""
    label = label_unit(unit("whatever", ACCESS_KEY_ID))
    assert label.label == "api_key"
    assert label.confidence == HIGH
    assert label.evidence == "aws.access_key_id"


def test_a_value_pattern_outranks_a_misleading_key_name() -> None:
    """The value decided it, so a neutral or wrong key name cannot demote it."""
    assert label_unit(unit("PartitionKey", ACCESS_KEY_ID)).confidence == HIGH


def test_a_structureless_secret_is_low_confidence() -> None:
    """An AWS *secret* key is 40 base64 characters with no marker at all.

    NR-0002 finding 3 lists it among the formats the value patterns cannot
    touch, and that is the point of the tier: this unit must reach the teacher,
    because nothing structural can settle it. Only the key name labels it, and
    only weakly.
    """
    label = label_unit(unit("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE"))
    assert label.confidence == LOW
    assert label.evidence == "key:api_key"


def test_a_key_name_alone_is_low_confidence() -> None:
    """NR-0002 traced all 1,296 rule false positives to key names and templates.

    So a key-name hit produces a label, but never a confident one: these are
    exactly the units that need a second opinion.
    """
    label = label_unit(unit("password", "hunter2"))
    assert label.label == "password"
    assert label.confidence == LOW
    assert label.evidence == "key:password"


def test_a_neutralised_key_name_yields_nothing() -> None:
    """`PartitionKey` is not a key. Reused from the NR-0002 layer, pinned here.

    If this regressed, the labeller would hand the teacher a confident wrong
    label on every partition key in the corpus.
    """
    label = label_unit(unit("PartitionKey", "users-2026"))
    assert label.label == "none"
    assert label.evidence == NO_EVIDENCE


def test_silence_is_low_confidence_not_high() -> None:
    """A unit nothing fires on is `none`, but not confidently so.

    Absence of evidence is not evidence of absence, and the secrets with no
    structural marker (plan §3.3, NR-0002 finding 3) all land here. Promoting
    them would exclude from review precisely the cases the project exists for.
    """
    label = label_unit(unit("city", "auckland"))
    assert label.label == "none"
    assert label.confidence == LOW


def test_raised_entropy_signals_without_claiming_a_class() -> None:
    """Entropy says "random", not "which class", so the label stays `none`."""
    label = label_unit(unit("digest", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822c"))
    assert label.evidence == "entropy"
    assert label.label == "none"
    assert label.confidence == LOW
    assert label.entropy >= 0.9


def test_a_template_is_consulted_only_for_body_mode_units() -> None:
    """A real key name outranks the template; `<slot:...>` has no real key.

    This is the NR-0000 failure mode — the Laya teacher read `password` in
    `Failed password for invalid user ...` and scored the user name 0.972. The
    rule layer reproduces it 416 times, so the template is used, but it can
    never produce high confidence.
    """
    body = label_unit(
        unit("<slot:user>", "bright-copper-kettle", template="Failed password for invalid user <*>")
    )
    assert body.evidence == "template:password"
    assert body.confidence == LOW

    attribute = label_unit(
        unit("username", "bright-copper-kettle", template="Failed password for invalid user <*>")
    )
    assert attribute.label == "none"


def test_the_units_own_label_is_never_read() -> None:
    """The labeller must behave identically on labelled and unlabelled corpora.

    Reading `unit.label` would make the rule labels agree with the generator by
    construction, and §2.5 item 2 reports the agreement rate between them as a
    *measurement*.
    """
    value = "hunter2"
    with_label = DetectionUnit(
        id="x", key="password", value=value, source="synth", label="password"
    )
    without = DetectionUnit(id="x", key="password", value=value, source="own", label=None)
    assert label_unit(with_label) == label_unit(without)


def test_statistics_separate_the_evidence_kinds() -> None:
    """The §2.5 breakdown keeps key, template, entropy and silence apart."""
    labels = [
        RuleLabel("a", "api_key", HIGH, "aws.secret_access_key", 0.8),
        RuleLabel("b", "password", LOW, "key:password", 0.0),
        RuleLabel("c", "password", LOW, "template:password", 0.0),
        RuleLabel("d", "none", LOW, "entropy", 0.95),
        RuleLabel("e", "none", LOW, NO_EVIDENCE, 0.0),
    ]
    stats = statistics(labels)
    assert stats["units"] == 5
    assert stats["by_confidence"] == {HIGH: 1, LOW: 4}
    assert stats["by_evidence"] == {
        "entropy": 1,
        "key": 1,
        "none": 1,
        "template": 1,
        "value_rule": 1,
    }


def test_labels_round_trip_through_the_file(tmp_path: Path) -> None:
    """Written and read back, a label is unchanged."""
    labels = [
        RuleLabel("a", "api_key", HIGH, "aws.secret_access_key", 0.812345678),
        RuleLabel("b", "none", LOW, NO_EVIDENCE, 0.0),
    ]
    path = tmp_path / "labels.jsonl"
    assert write_labels(labels, path) == 2
    recovered = read_labels(path)
    assert [item.unit_id for item in recovered] == ["a", "b"]
    assert [item.confidence for item in recovered] == [HIGH, LOW]
    assert recovered[0].entropy == pytest.approx(0.812346, abs=1e-6)


def test_no_value_is_ever_written_to_a_label_file(tmp_path: Path) -> None:
    """Labels carry a unit id and a decision, never the value.

    The same labeller runs over the own logs, and a label file that embedded
    values would inherit every constraint the corpus has.
    """
    secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    path = tmp_path / "labels.jsonl"
    write_labels(label_units([unit("AWS_SECRET_ACCESS_KEY", secret)]), path)
    written = path.read_text(encoding="utf-8")
    assert secret not in written
    assert set(json.loads(written).keys()) == {
        "unit_id",
        "label",
        "confidence",
        "evidence",
        "entropy",
    }


def test_third_party_data_cannot_be_labelled(tmp_path: Path) -> None:
    """§6 item 6: no path in the code reads `data/third_party/` to make a label.

    Enforced twice over by the loader — once on the path, once per record — and
    this asserts the labeller goes through it rather than around it.
    """
    third_party = tmp_path / "data" / "third_party" / "creddata.jsonl"
    third_party.parent.mkdir(parents=True)
    third_party.write_text(
        json.dumps({"id": "c1", "key": "password", "value": "x", "source": "third_party"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ThirdPartyDataError):
        run(third_party, tmp_path / "out.jsonl")


def test_the_default_corpora_do_not_include_third_party_data() -> None:
    """Belt and braces: the default list itself names only own and synthetic data."""
    assert DEFAULT_CORPORA
    for data, out in DEFAULT_CORPORA:
        assert "third_party" not in data
        assert "third_party" not in out
