"""Tests for the teacher runner (NR-0003 §2.2 item 4, §2.2A items 2 and 4, §6).

§6 asks for the teacher flow to be tested with a small model or a stub, and for
resumption to be covered by a test or a demonstration. A stub is used here: the
runner's job is batching, bucketing, checkpointing and refusing to pool labels
from different configurations, and none of that is about the model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from datagen.unit import LABELS, DetectionUnit
from labeling.teacher.prompt import load_prompt
from labeling.teacher.run import (
    ConfigurationChangedError,
    RunConfig,
    blocks,
    completed_ids,
    run_teacher,
)


class StubTeacher:
    """A deterministic stand-in for Laya.

    Records every batch it was given, so the tests can assert on batching and
    ordering rather than on an answer.
    """

    def __init__(self, fail_after: int | None = None) -> None:
        """Build a stub, optionally interrupting after a number of batches."""
        self.batches: list[list[str]] = []
        self.fail_after = fail_after
        self.calls = 0

    def predict(self, states: Sequence[str]) -> list[dict[str, float]]:
        """Return a fixed distribution, optionally failing partway through."""
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated interruption")
        self.batches.append(list(states))
        # A distribution that is a real distribution: the merge step and the
        # calibration both assume it sums to one.
        head, *rest = LABELS
        return [{head: 0.7, **{label: 0.3 / len(rest) for label in rest}} for _ in states]


def make_units(count: int, *, long_every: int = 0) -> list[DetectionUnit]:
    """Build units, optionally with some much longer values.

    Args:
        count: How many units.
        long_every: If non-zero, every Nth unit gets a long value, so that
            length bucketing has something to do.
    """
    units = []
    for index in range(count):
        long = long_every and index % long_every == 0
        units.append(
            DetectionUnit(
                id=f"u-{index:04d}",
                key="ApiKey",
                # Distinct per unit: units that render identically would make
                # any assertion about *which* units were processed vacuous.
                value=(f"{index:04d}-" + "x" * 600) if long else f"short-value-{index:04d}",
                source="synth",
                label="none",
            )
        )
    return units


def config(scope_digest: str = "abc", prompt_version: str = "v1") -> RunConfig:
    """Build a run configuration for a test."""
    prompt = load_prompt()
    return RunConfig(
        prompt_version=prompt_version,
        prompt_digest=prompt.digest,
        model="stub",
        scope_digest=scope_digest,
        threads=10,
        batch_size=4,
        block_size=8,
    )


def test_every_unit_is_labelled_once(tmp_path: Path) -> None:
    """The basic contract: one record per unit, no duplicates, no omissions."""
    prompt = load_prompt()
    units = make_units(20)
    summary = run_teacher(
        units,
        StubTeacher(),
        prompt,
        tmp_path / "teacher.jsonl",
        config=config(),
        block_size=8,
        batch_size=4,
    )
    assert summary["labelled"] == 20
    records = [
        json.loads(line)
        for line in (tmp_path / "teacher.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 20
    assert {record["unit_id"] for record in records} == {unit.id for unit in units}


def test_every_record_carries_all_seven_classes(tmp_path: Path) -> None:
    """Every record carries all seven classes, zeros included.

    A soft label with a varying key set would make NR-0004's distillation target
    depend on the serialisation.
    """
    run_teacher(
        make_units(4),
        StubTeacher(),
        load_prompt(),
        tmp_path / "teacher.jsonl",
        config=config(),
        block_size=8,
        batch_size=4,
    )
    first = json.loads((tmp_path / "teacher.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(first["probabilities"]) == set(LABELS)
    assert first["choice"] in LABELS


def test_the_prompt_digest_travels_with_every_label(tmp_path: Path) -> None:
    """A label is only interpretable with the wording that produced it."""
    prompt = load_prompt()
    run_teacher(
        make_units(2),
        StubTeacher(),
        prompt,
        tmp_path / "teacher.jsonl",
        config=config(),
        block_size=8,
        batch_size=4,
    )
    for line in (tmp_path / "teacher.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["prompt_digest"] == prompt.digest
        assert record["prompt_version"] == prompt.version


def test_blocks_are_sorted_by_length_but_not_globally(tmp_path: Path) -> None:
    """§2.2A item 2, and the reason it is per block.

    Within a block, lengths ascend — that is what removes the padding waste. But
    the long units are spread across blocks rather than collected at the end,
    so each block's length mix stays comparable and §2.2A item 6 can read a
    falling throughput as throttling rather than as longer inputs.
    """
    prompt = load_prompt()
    units = make_units(24, long_every=4)
    produced = list(blocks(units, prompt, 8))
    assert len(produced) == 3
    for block in produced:
        lengths = [len(state) for _, state in block]
        assert lengths == sorted(lengths)
        # Each block got its share of long units rather than one block getting
        # all of them.
        assert any(length > 500 for length in lengths)


def test_batches_respect_the_batch_size(tmp_path: Path) -> None:
    """Batches are what the model sees; the measured optimum was 8."""
    teacher = StubTeacher()
    run_teacher(
        make_units(20),
        teacher,
        load_prompt(),
        tmp_path / "teacher.jsonl",
        config=config(),
        block_size=8,
        batch_size=4,
    )
    assert [len(batch) for batch in teacher.batches] == [4, 4, 4, 4, 4]


def test_an_interrupted_run_resumes_without_relabelling(tmp_path: Path) -> None:
    """§6: resumption is covered by a test.

    The first run is killed partway through. The second must label only what is
    left — anything else wastes hours of an overnight run, and on a paid model
    would be charged twice.
    """
    prompt = load_prompt()
    units = make_units(20)
    out = tmp_path / "teacher.jsonl"

    interrupted = StubTeacher(fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        run_teacher(units, interrupted, prompt, out, config=config(), block_size=8, batch_size=4)
    done = completed_ids(out)
    assert 0 < len(done) < 20

    resumed = StubTeacher()
    summary = run_teacher(units, resumed, prompt, out, config=config(), block_size=8, batch_size=4)
    assert summary["resumed"] is True
    assert summary["already_labelled"] == len(done)
    assert summary["labelled"] == 20 - len(done)

    # Nothing already in the file was sent to the model a second time.
    states_by_unit = {prompt.render(unit): unit.id for unit in units}
    reprocessed = {states_by_unit[state] for batch in resumed.batches for state in batch}
    assert reprocessed == {unit.id for unit in units} - done
    recovered = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len({record["unit_id"] for record in recovered}) == 20


def test_a_completed_run_resumes_to_nothing(tmp_path: Path) -> None:
    """Re-running a finished job must be free, not a second full pass."""
    prompt = load_prompt()
    units = make_units(8)
    out = tmp_path / "teacher.jsonl"
    run_teacher(units, StubTeacher(), prompt, out, config=config(), block_size=8, batch_size=4)

    second = StubTeacher()
    summary = run_teacher(units, second, prompt, out, config=config(), block_size=8, batch_size=4)
    assert summary["labelled"] == 0
    assert second.batches == []


def test_a_truncated_final_line_is_tolerated(tmp_path: Path) -> None:
    """A process killed mid-write leaves a partial line; that unit is redone.

    Raising here would mean an overnight run that died at the wrong instant
    could not be resumed at all, which defeats the purpose of checkpointing.
    """
    out = tmp_path / "teacher.jsonl"
    out.write_text(json.dumps({"unit_id": "u-0000"}) + "\n" + '{"unit_id": "u-00', encoding="utf-8")
    assert completed_ids(out) == {"u-0000"}


def test_resuming_under_a_different_prompt_is_refused(tmp_path: Path) -> None:
    """Labels from two prompts must never be pooled in one file.

    This is the failure that would be invisible otherwise: the file would look
    well-formed and every record would carry its own digest, but the set as a
    whole would be two incomparable populations, and NR-0004 would train on
    their average.
    """
    prompt = load_prompt()
    units = make_units(20)
    out = tmp_path / "teacher.jsonl"
    run_teacher(units[:8], StubTeacher(), prompt, out, config=config(), block_size=8, batch_size=4)
    with pytest.raises(ConfigurationChangedError, match="prompt_version"):
        run_teacher(
            units,
            StubTeacher(),
            prompt,
            out,
            config=config(prompt_version="v2"),
            block_size=8,
            batch_size=4,
        )


def test_resuming_under_a_different_scope_is_refused(tmp_path: Path) -> None:
    """A changed scope means the ids in the file no longer describe the job."""
    prompt = load_prompt()
    out = tmp_path / "teacher.jsonl"
    run_teacher(
        make_units(8), StubTeacher(), prompt, out, config=config("one"), block_size=8, batch_size=4
    )
    with pytest.raises(ConfigurationChangedError, match="scope_digest"):
        run_teacher(
            make_units(16),
            StubTeacher(),
            prompt,
            out,
            config=config("two"),
            block_size=8,
            batch_size=4,
        )


def test_the_block_log_records_throughput_over_time(tmp_path: Path) -> None:
    """§2.2A item 6 judges throttling from this series, so it must exist."""
    log = tmp_path / "run.jsonl"
    summary = run_teacher(
        make_units(24),
        StubTeacher(),
        load_prompt(),
        tmp_path / "teacher.jsonl",
        config=config(),
        log=log,
        block_size=8,
        batch_size=4,
    )
    assert len(summary["blocks"]) == 3
    logged = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(logged) == 3
    assert [record["block"] for record in logged] == [0, 1, 2]
    for record in logged:
        assert record["samples_per_second"] is not None
        assert record["cumulative_labelled"] > 0
        assert "median_state_chars" in record
    assert summary["throughput"]["first_block"] is not None


def test_an_unknown_class_from_the_teacher_is_an_error() -> None:
    """A class outside the seven means the prompt and the label set have drifted."""
    from labeling.teacher.run import _distribution

    with pytest.raises(ValueError, match="unknown classes"):
        _distribution({"answers": {"secret_class": {"probabilities": {"banana": 1.0}}}})
    with pytest.raises(ValueError, match="unknown class"):
        _distribution({"answers": {"secret_class": {"probabilities": {}, "choice": "banana"}}})
