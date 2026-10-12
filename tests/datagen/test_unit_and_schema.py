"""Tests for the detection-unit record, its schema and the validator CLI."""

from __future__ import annotations

import json

import pytest
from datagen.unit import LABELS, SECRET_LABELS, DetectionUnit, is_third_party
from datagen.validate import check_label_polarity, load_schema
from datagen.validate import main as validate_main
from jsonschema import Draft202012Validator


@pytest.fixture(scope="module")
def validator():
    return Draft202012Validator(load_schema())


def _unit(**overrides) -> dict:
    record = {
        "id": "t-1",
        "key": "ApiKey",
        # Deliberately not vendor-shaped: a literal matching a real key format
        # is itself a secret-scanner finding, whatever it actually is.
        "value": "placeholder-value-not-a-credential",
        "source": "synth",
        "label": "api_key",
    }
    record.update(overrides)
    return record


class TestSchema:
    def test_the_schema_is_itself_valid(self):
        Draft202012Validator.check_schema(load_schema())

    def test_a_minimal_unit_validates(self, validator):
        validator.validate({"id": "a", "key": "k", "value": "v", "source": "synth"})

    def test_a_full_unit_validates(self, validator):
        validator.validate(
            _unit(
                template="Calling {Url} with {ApiKey}",
                scope="otel.attribute",
                family="aws.access_key_id",
                ambiguous=False,
                meta={"generator_version": "0.1.0", "seed": 1, "split": "train"},
            )
        )

    @pytest.mark.parametrize("missing", ["id", "key", "value", "source"])
    def test_a_missing_required_field_fails(self, validator, missing):
        record = _unit()
        del record[missing]
        assert list(validator.iter_errors(record))

    def test_an_unknown_label_fails(self, validator):
        assert list(validator.iter_errors(_unit(label="pii")))

    def test_a_null_label_fails_because_the_field_must_be_omitted(self, validator):
        assert list(validator.iter_errors(_unit(label=None)))

    @pytest.mark.parametrize(
        "source", ["synth", "own", "third_party:creddata", "third_party:some-set"]
    )
    def test_permitted_sources(self, validator, source):
        validator.validate(_unit(source=source))

    @pytest.mark.parametrize("source", ["Synth", "third_party", "third_party:", "other", ""])
    def test_rejected_sources(self, validator, source):
        assert list(validator.iter_errors(_unit(source=source)))

    def test_an_unknown_top_level_field_fails(self, validator):
        assert list(validator.iter_errors(_unit(unexpected="x")))

    def test_a_non_string_value_fails(self, validator):
        assert list(validator.iter_errors(_unit(value=42)))

    @pytest.mark.parametrize("family", ["aws.access_key_id", "neg.uuid_v4", "jwt.hs256", "a"])
    def test_accepted_family_names(self, validator, family):
        validator.validate(_unit(family=family))

    @pytest.mark.parametrize("family", ["AWS.Key", "aws..key", ".aws", "aws-key"])
    def test_rejected_family_names(self, validator, family):
        assert list(validator.iter_errors(_unit(family=family)))

    def test_every_label_in_the_code_is_in_the_schema(self):
        enum = load_schema()["properties"]["label"]["enum"]
        assert sorted(enum) == sorted(LABELS)


class TestDetectionUnit:
    def test_round_trip_through_a_dict(self):
        unit = DetectionUnit(
            id="a",
            key="k",
            value="v",
            source="synth",
            label="token",
            template="t",
            scope="s",
            family="jwt.hs256",
            ambiguous=True,
            meta={"seed": 7},
        )
        assert DetectionUnit.from_dict(unit.to_dict()) == unit

    def test_absent_optionals_are_omitted(self):
        unit = DetectionUnit(id="a", key="k", value="v", source="synth")
        assert unit.to_dict() == {"id": "a", "key": "k", "value": "v", "source": "synth"}

    def test_empty_meta_is_omitted(self):
        unit = DetectionUnit(id="a", key="k", value="v", source="synth", meta={})
        assert "meta" not in unit.to_dict()

    def test_to_json_is_one_line(self):
        unit = DetectionUnit(id="a", key="k", value="line1\nline2", source="synth")
        text = unit.to_json()
        assert "\n" not in text
        assert json.loads(text)["value"] == "line1\nline2"

    def test_to_json_keeps_non_ascii_legible(self):
        unit = DetectionUnit(id="a", key="密码", value="值", source="synth")
        assert "密码" in unit.to_json()

    @pytest.mark.parametrize("label", sorted(SECRET_LABELS))
    def test_should_redact_is_true_for_secret_labels(self, label):
        assert DetectionUnit(id="a", key="k", value="v", source="synth", label=label).should_redact

    def test_should_redact_is_false_for_none_and_for_unlabelled(self):
        assert not DetectionUnit(
            id="a", key="k", value="v", source="synth", label="none"
        ).should_redact
        assert not DetectionUnit(id="a", key="k", value="v", source="synth").should_redact

    def test_from_dict_reports_a_missing_field(self):
        with pytest.raises(KeyError):
            DetectionUnit.from_dict({"id": "a", "key": "k", "value": "v"})

    @pytest.mark.parametrize(
        ("source", "expected"),
        [("synth", False), ("own", False), ("third_party:creddata", True)],
    )
    def test_is_third_party(self, source, expected):
        assert is_third_party(source) is expected


class TestLabelPolarity:
    def test_a_negative_family_with_a_secret_label_is_rejected(self):
        problem = check_label_polarity({"family": "neg.uuid_v4", "label": "token"})
        assert problem and "negative family" in problem

    def test_a_positive_family_labelled_none_is_rejected(self):
        problem = check_label_polarity({"family": "aws.access_key_id", "label": "none"})
        assert problem and "labelled 'none'" in problem

    def test_consistent_units_pass(self):
        assert check_label_polarity({"family": "neg.uuid_v4", "label": "none"}) is None
        assert check_label_polarity({"family": "jwt.hs256", "label": "token"}) is None

    def test_a_unit_without_a_family_or_label_passes(self):
        assert check_label_polarity({"label": "token"}) is None
        assert check_label_polarity({"family": "neg.uuid_v4"}) is None


class TestValidatorCli:
    def test_a_generated_corpus_validates(self, fixture_corpus):
        paths = [str(fixture_corpus[split]) for split in ("train", "val", "test")]
        assert validate_main(paths) == 0

    def test_a_duplicate_id_is_reported(self, tmp_path, capsys):
        path = tmp_path / "dup.jsonl"
        line = json.dumps(_unit()) + "\n"
        path.write_text(line + line, encoding="utf-8")
        assert validate_main([str(path)]) == 1
        assert "duplicate id" in capsys.readouterr().err

    def test_malformed_json_is_reported(self, tmp_path, capsys):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        assert validate_main([str(path)]) == 1
        assert "not valid JSON" in capsys.readouterr().err

    def test_a_schema_violation_is_reported(self, tmp_path, capsys):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(_unit(label="pii")) + "\n", encoding="utf-8")
        assert validate_main([str(path)]) == 1

    def test_a_missing_file_is_reported(self, tmp_path, capsys):
        assert validate_main([str(tmp_path / "nope.jsonl")]) == 1
        assert "not a file" in capsys.readouterr().err

    def test_blank_lines_are_tolerated(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text("\n" + json.dumps(_unit()) + "\n\n", encoding="utf-8")
        assert validate_main([str(path)]) == 0
