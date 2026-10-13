"""The teacher labelling run (NR-0003 §2.2 item 4, §2.2A items 2, 4 and 6).

An overnight job on CPU, so interruption is the normal case rather than the
exceptional one: EXECUTION.md §4 requires batching, checkpointing and resumption,
and §2.2A adds a thread cap, length bucketing and a record of throughput over
time.

**The output file is the checkpoint.** Records are appended per block and
flushed, and a resumed run reads back the unit ids already present and skips
them. A separate progress file that could disagree with the output would need
reconciling, and the reconciliation is the part that goes wrong at 3 a.m.; the
sidecar here holds only the run's *configuration*, and its purpose is to refuse
a resume whose prompt, model or scope has changed since the records were
written. Labels produced under two different prompts must never end up pooled in
one file.

**Blocks, not a global sort.** Length bucketing removes padding waste — the
trial measured +32% — but sorting the whole corpus by length would put every PEM
block at the end, so throughput would decline all night in a way
indistinguishable from thermal throttling. Since §2.2A item 6 asks for
throttling to be judged by throughput drift, that confound would destroy the
only signal available. Sorting within blocks of 1,000 keeps each block's length
mix comparable and keeps essentially all of the gain.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from datagen.unit import LABELS, DetectionUnit

from labeling.teacher.prompt import Prompt

#: Units per block. One block is one checkpoint (§2.2A item 4) and one
#: length-bucketing group, so the two are the same unit of work.
BLOCK_SIZE = 1000

#: Batch size handed to the model. Measured best on this machine; larger is
#: slower on CPU, which is the opposite of the usual expectation.
BATCH_SIZE = 8

#: Threads. §2.2A item 1 caps this at 10 and the trial found 10 fastest anyway.
THREADS = 10

#: The checkpoint's model checkpoint name.
MODEL = "multilingual"


class Teacher(Protocol):
    """Anything that can answer the seven-class question for a batch of states.

    Narrow by design: the runner's batching, bucketing, checkpointing and
    logging are all testable against a stub that implements this one method,
    which is what §6's "老师部分以小模型或打桩替身测试流程" asks for.
    """

    def predict(self, states: Sequence[str]) -> list[dict[str, float]]:
        """Return a probability distribution over :data:`datagen.unit.LABELS`.

        Args:
            states: Rendered prompts.

        Returns:
            One distribution per state, in the same order.
        """
        ...


@dataclass
class LayaTeacher:
    """The Laya teacher (NR-0003 §2.2 item 1).

    Attributes:
        prompt: The versioned prompt.
        model: Checkpoint name.
        threads: Torch thread cap.
        batch_size: Batch handed to ``predict_batch``.
    """

    prompt: Prompt
    model: str = MODEL
    threads: int = THREADS
    batch_size: int = BATCH_SIZE
    _router: Any = field(default=None, init=False, repr=False)

    def load(self) -> float:
        """Load the checkpoint and set the thread cap.

        Returns:
            Load time in seconds.

        The thread cap is set before the first forward pass, because torch does
        not reliably re-read it once intra-op parallel work has started — which
        is also why the §2.2A item 1 trial used one process per thread count
        rather than looping inside one.
        """
        import torch

        torch.set_num_threads(self.threads)

        from laya import Router

        started = time.perf_counter()
        self._router = Router(device="cpu", default=self.model)
        self._router.load(self.model)
        return time.perf_counter() - started

    def predict(self, states: Sequence[str]) -> list[dict[str, float]]:
        """Answer the question for a batch of states.

        Args:
            states: Rendered prompts.

        Returns:
            One probability distribution per state.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
        """
        if self._router is None:
            raise RuntimeError("call load() before predict()")
        questions = self.prompt.questions()
        requests = [
            {"state": state, "questions": questions, "model": self.model} for state in states
        ]
        results = self._router.predict_batch(requests, batch_size=self.batch_size)
        return [_distribution(result) for result in results]


def _distribution(result: Any) -> dict[str, float]:
    """Extract a full seven-class distribution from one Laya answer.

    Args:
        result: One element of ``predict_batch``'s output.

    Returns:
        A probability per class in :data:`datagen.unit.LABELS`, with absent
        classes filled in as 0.0.

    Raises:
        ValueError: If the answer names a class outside the seven, which would
            mean the prompt and the label set had drifted apart.

    Missing classes are filled rather than omitted so that every record has the
    same seven keys. A soft label with a varying key set would make the
    distillation target in NR-0004 depend on the serialisation.
    """
    from labeling.teacher.prompt import QUESTION_KEY

    answer = result["answers"][QUESTION_KEY]
    raw = answer.get("probabilities") or {}
    unexpected = set(raw) - set(LABELS)
    if unexpected:
        raise ValueError(f"teacher returned unknown classes {sorted(unexpected)}")
    distribution = {label: float(raw.get(label, 0.0)) for label in LABELS}
    chosen = answer.get("choice")
    if chosen is not None and chosen not in LABELS:
        raise ValueError(f"teacher chose unknown class {chosen!r}")
    return distribution


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The configuration a resumed run must match.

    Attributes:
        prompt_version: Which prompt file was used.
        prompt_digest: Its SHA-256.
        model: Checkpoint name.
        scope_digest: A digest of the selected unit ids, so a resume against a
            different scope is refused rather than silently relabelling.
        threads: Thread cap.
        batch_size: Batch size.
        block_size: Units per block and per checkpoint.
    """

    prompt_version: str
    prompt_digest: str
    model: str
    scope_digest: str
    threads: int
    batch_size: int
    block_size: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the sidecar file."""
        return {
            "prompt_version": self.prompt_version,
            "prompt_digest": self.prompt_digest,
            "model": self.model,
            "scope_digest": self.scope_digest,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
        }


class ConfigurationChangedError(RuntimeError):
    """A resume was attempted against a changed configuration.

    Deliberately not a subclass of ``ValueError``: the same distinction
    ``ThirdPartyDataError`` draws in the loader. This is not a bad argument, it
    is a refusal to mix two sets of labels, and a caller catching ``ValueError``
    for a malformed input should not swallow it.
    """


def completed_ids(path: Path) -> set[str]:
    """Read the unit ids already labelled in an output file.

    Args:
        path: The output JSON Lines file; may not exist.

    Returns:
        The ids present. A truncated final line — the normal result of a process
        killed mid-write — is ignored rather than raising, because the whole
        point of resuming is to tolerate however the previous run ended. The
        unit is simply relabelled.
    """
    if not path.is_file():
        return set()
    found: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                found.add(json.loads(line)["unit_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return found


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    """Read the run's configuration sidecar, if present."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def blocks(
    units: Sequence[DetectionUnit], prompt: Prompt, block_size: int
) -> Iterator[list[tuple[DetectionUnit, str]]]:
    """Split units into length-sorted blocks.

    Args:
        units: The units to process, in scope order.
        prompt: The prompt, used to render each state.
        block_size: Units per block.

    Yields:
        Blocks of ``(unit, rendered_state)`` pairs, each block sorted by state
        length so that batches within it hold similar lengths.

    Sorting is stable on the rendered length and then the unit id, so a block's
    processing order is fully determined by its contents — a resumed run
    processes what remains in the same order a fresh one would.
    """
    for start in range(0, len(units), block_size):
        chunk = units[start : start + block_size]
        rendered = [(unit, prompt.render(unit)) for unit in chunk]
        rendered.sort(key=lambda pair: (len(pair[1]), pair[0].id))
        yield rendered


def run_teacher(
    units: Sequence[DetectionUnit],
    teacher: Teacher,
    prompt: Prompt,
    out: Path,
    *,
    config: RunConfig,
    log: Path | None = None,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    on_block: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Label units with the teacher, resumably.

    Args:
        units: The units in scope, in scope order.
        teacher: The model, or a stub.
        prompt: The versioned prompt.
        out: Output JSON Lines file, appended to.
        config: The configuration a resume must match.
        log: Optional per-block log, appended as JSON Lines. This is what
            §2.2A item 6's throttling judgement reads.
        block_size: Units per block and per checkpoint.
        batch_size: Units per model call.
        on_block: Called with each block's log record, for progress reporting.

    Returns:
        A summary: counts, timings and the throughput series.

    Raises:
        ConfigurationChangedError: If an existing output was produced under a
            different prompt, model or scope.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = out.with_suffix(out.suffix + ".config.json")

    existing = _load_sidecar(sidecar_path)
    already = completed_ids(out)
    if existing is not None and existing != config.to_dict():
        wanted = config.to_dict()
        differences = sorted(
            key for key in set(existing) | set(wanted) if existing.get(key) != wanted.get(key)
        )
        raise ConfigurationChangedError(
            f"{out} was written under a different configuration (differs in: "
            f"{', '.join(differences)}). Labels from two configurations must not be pooled: "
            "move the existing file aside to start again, or restore the previous "
            "configuration to resume it."
        )
    sidecar_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pending = [unit for unit in units if unit.id not in already]
    summary: dict[str, Any] = {
        "in_scope": len(units),
        "already_labelled": len(already),
        "to_label": len(pending),
        "resumed": bool(already),
        "prompt_version": prompt.version,
        "prompt_digest": prompt.digest,
        "model": config.model,
        "threads": config.threads,
        "batch_size": batch_size,
        "block_size": block_size,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "blocks": [],
    }
    if not pending:
        summary["finished_utc"] = summary["started_utc"]
        summary["labelled"] = 0
        return summary

    labelled = 0
    run_started = time.perf_counter()
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        for index, block in enumerate(blocks(pending, prompt, block_size)):
            block_started = time.perf_counter()
            for batch_start in range(0, len(block), batch_size):
                batch = block[batch_start : batch_start + batch_size]
                distributions = teacher.predict([state for _, state in batch])
                for (unit, _), distribution in zip(batch, distributions, strict=True):
                    handle.write(
                        json.dumps(
                            {
                                "unit_id": unit.id,
                                "source": unit.source,
                                "probabilities": {
                                    label: round(value, 6) for label, value in distribution.items()
                                },
                                "choice": max(distribution, key=lambda label: distribution[label]),
                                "prompt_version": prompt.version,
                                "prompt_digest": prompt.digest,
                                "model": config.model,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                labelled += len(batch)
            # Flushed per block: this is the checkpoint. A kill between blocks
            # loses at most one block, and `completed_ids` tolerates a partial
            # final line if it lands mid-write.
            handle.flush()
            elapsed = time.perf_counter() - block_started
            record = {
                "block": index,
                "units": len(block),
                "seconds": round(elapsed, 3),
                "samples_per_second": round(len(block) / elapsed, 3) if elapsed else None,
                "cumulative_labelled": labelled,
                "cumulative_seconds": round(time.perf_counter() - run_started, 1),
                "median_state_chars": sorted(len(state) for _, state in block)[len(block) // 2],
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            summary["blocks"].append(record)
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8", newline="\n") as log_handle:
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if on_block is not None:
                on_block(record)

    total = time.perf_counter() - run_started
    rates = [
        block["samples_per_second"]
        for block in summary["blocks"]
        if block["samples_per_second"] is not None
    ]
    summary.update(
        {
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "labelled": labelled,
            "seconds": round(total, 1),
            "samples_per_second": round(labelled / total, 3) if total else None,
            "throughput": {
                "first_block": rates[0] if rates else None,
                "last_block": rates[-1] if rates else None,
                "min": min(rates) if rates else None,
                "max": max(rates) if rates else None,
                "note": (
                    "§2.2A item 6 judges throttling by drift here. Blocks are length-sorted "
                    "internally but not against each other, so median_state_chars is roughly "
                    "constant across blocks and a falling rate is not explained by longer inputs."
                ),
            },
        }
    )
    return summary
