"""Tests for the source restriction on the training and labeling paths.

EXECUTION.md §5 item 2 requires that reading ``data/third_party/`` from a
training or labeling path raise, and that a test prevent regression. These are
those tests, and they check both halves of the guard — the path check and the
per-record check — because either alone can be got around by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datagen.loader import (
    TRAINING_SOURCES,
    ThirdPartyDataError,
    iter_units,
    load_for_evaluation,
    load_for_training,
)
from datagen.unit import DetectionUnit


def _write(path: Path, *records: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _record(source: str, unit_id: str = "a") -> dict:
    return {"id": unit_id, "key": "ApiKey", "value": "x", "source": source, "label": "api_key"}


class TestTrainingSources:
    def test_only_synth_and_own_are_permitted(self):
        assert {"synth", "own"} == TRAINING_SOURCES

    def test_synth_and_own_data_loads(self, tmp_path):
        path = _write(
            tmp_path / "data" / "synth" / "train.jsonl",
            _record("synth", "a"),
            _record("own", "b"),
        )
        units = list(load_for_training([path]))
        assert [unit.id for unit in units] == ["a", "b"]


class TestPathGuard:
    """The first half: refuse the file before it is even opened."""

    def test_a_third_party_directory_is_refused(self, tmp_path):
        path = _write(tmp_path / "data" / "third_party" / "eval_v0.jsonl", _record("synth"))
        with pytest.raises(ThirdPartyDataError, match="third_party"):
            list(load_for_training([path]))

    def test_a_nested_third_party_directory_is_refused(self, tmp_path):
        path = _write(
            tmp_path / "data" / "third_party" / "creddata" / "sub" / "x.jsonl", _record("synth")
        )
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([path]))

    def test_the_guard_is_case_insensitive(self, tmp_path):
        path = _write(tmp_path / "data" / "Third_Party" / "x.jsonl", _record("synth"))
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([path]))

    def test_a_refused_path_is_refused_even_though_its_records_claim_synth(self, tmp_path):
        """A mislabelled file must not get in on its source field."""
        path = _write(tmp_path / "data" / "third_party" / "x.jsonl", _record("synth"))
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([path]))

    def test_a_refusal_stops_the_whole_load(self, tmp_path):
        clean = _write(tmp_path / "data" / "synth" / "a.jsonl", _record("synth", "a"))
        dirty = _write(tmp_path / "data" / "third_party" / "b.jsonl", _record("synth", "b"))
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([clean, dirty]))


class TestRecordGuard:
    """The second half: refuse the record even if the path looks innocent."""

    def test_a_third_party_record_in_a_clean_path_is_refused(self, tmp_path):
        path = _write(
            tmp_path / "data" / "synth" / "train.jsonl",
            _record("synth", "a"),
            _record("third_party:creddata", "b"),
        )
        with pytest.raises(ThirdPartyDataError, match="third_party:creddata"):
            list(load_for_training([path]))

    def test_an_unknown_source_is_refused(self, tmp_path):
        path = _write(tmp_path / "data" / "synth" / "train.jsonl", _record("scraped"))
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([path]))

    def test_the_error_names_the_offending_unit(self, tmp_path):
        path = _write(
            tmp_path / "data" / "synth" / "train.jsonl", _record("third_party:creddata", "bad-1")
        )
        with pytest.raises(ThirdPartyDataError, match="bad-1"):
            list(load_for_training([path]))


class TestEvaluationPath:
    """Evaluation is allowed to see everything; that asymmetry is the design."""

    def test_third_party_data_loads_for_evaluation(self, tmp_path):
        path = _write(
            tmp_path / "data" / "third_party" / "eval_v0.jsonl",
            _record("third_party:creddata", "a"),
        )
        units = list(load_for_evaluation([path]))
        assert len(units) == 1
        assert units[0].source == "third_party:creddata"

    def test_evaluation_and_training_disagree_on_the_same_file(self, tmp_path):
        path = _write(
            tmp_path / "data" / "third_party" / "eval_v0.jsonl",
            _record("third_party:creddata"),
        )
        assert list(load_for_evaluation([path]))
        with pytest.raises(ThirdPartyDataError):
            list(load_for_training([path]))


class TestLowLevelReader:
    def test_iter_units_parses_records(self, tmp_path):
        path = _write(tmp_path / "x.jsonl", _record("synth"))
        assert list(iter_units(path)) == [
            DetectionUnit(id="a", key="ApiKey", value="x", source="synth", label="api_key")
        ]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text("\n" + json.dumps(_record("synth")) + "\n\n", encoding="utf-8")
        assert len(list(iter_units(path))) == 1

    def test_malformed_json_raises_value_error_with_a_location(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text("{oops}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r":1:"):
            list(iter_units(path))

    def test_a_missing_required_field_raises_value_error(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text(json.dumps({"id": "a", "key": "k", "value": "v"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field"):
            list(iter_units(path))


def test_third_party_error_is_not_a_value_error():
    """A caller skipping malformed records must not accidentally skip a policy breach."""
    assert not issubclass(ThirdPartyDataError, ValueError)
    assert issubclass(ThirdPartyDataError, RuntimeError)
