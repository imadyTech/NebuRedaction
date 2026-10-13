"""Control group 3: CredSweeper (NR-0002 §2.2).

CredSweeper is Samsung's credential scanner — rules plus an ML re-check. It is
run unmodified, which is the only fair way to use somebody else's detector as a
control: tuning it would turn "how does a mature tool do on this task" into "how
does our configuration of it do".

**How detection units are presented to it.** CredSweeper scans files and reasons
about lines, not about ``(key, value)`` pairs. NR-0002 §2.2 specifies
constructing an input line per unit, so each unit becomes ``key = "value"`` in a
temporary file, one unit per line, and a finding on line *n* is attributed to
unit *n*. Two consequences worth stating plainly, because they bound what this
control's numbers mean:

* Multi-line values (PEM blocks) cannot keep their newlines without breaking the
  line-to-unit mapping, so their escaped form is used — the same form
  ``pem.single_line_escaped`` produces, which CredSweeper does detect.
* The rendering is a reconstruction, not the original file CredSweeper would
  have seen in the wild. Its line-context features therefore see a synthetic
  context. This cannot be avoided while comparing on detection units, and it is
  recorded in the report rather than glossed over.

**Its output is a decision, not a score.** CredSweeper applies its own ML
threshold internally and reports credentials it believes in, so it is registered
as a binary system and reported at its own operating point (NR-0002 §2.1 item 2).
Its ``ml_threshold`` is left at the shipped default.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from datagen.unit import DetectionUnit

from eval.system import SystemOutput

#: CredSweeper's rule names mapped onto the seven classes. Anything unmapped
#: falls back to ``api_key``, the project's generic secret class, and is counted
#: so the report can say how often the fallback was used.
_RULE_TO_LABEL: dict[str, str] = {
    "Password": "password",
    "SQL Password": "password",
    "CMD Password": "password",
    "Nonce": "api_key",
    "Secret": "api_key",
    "API": "api_key",
    "Key": "api_key",
    "Credential": "api_key",
    "AWS Client ID": "api_key",
    "AWS Multi": "api_key",
    "AWS MWS Key": "api_key",
    "Google API Key": "api_key",
    "Google OAuth Access Token": "token",
    "Google Multi": "api_key",
    "Github Old Token": "api_key",
    "Github Classic Token": "api_key",
    "Github Fine-grained Token": "api_key",
    "Gitlab Token": "api_key",
    "Slack Token": "api_key",
    "Slack Webhook": "api_key",
    "Stripe Standard API Key": "api_key",
    "Stripe Restricted API Key": "api_key",
    "SendGrid API Key": "api_key",
    "Azure Secret Value": "api_key",
    "Twilio API Key": "api_key",
    "Twilio Credentials": "api_key",
    "Dropbox App secret": "api_key",
    "Atlassian Token": "api_key",
    "JSON Web Token": "token",
    "Bearer Authorization": "token",
    "Basic Authorization": "password",
    "Auth": "token",
    "Token": "token",
    "PEM Private Key": "private_key",
    "BASE64 Private Key": "private_key",
    "Certificate": "private_key",
    "URL Credentials": "connection_string",
    "SQLAlchemy Connection String": "connection_string",
    "Mongo Connection String": "connection_string",
}

#: Characters that would break the one-unit-per-line rendering.
_NEWLINES = re.compile(r"[\r\n]+")


@dataclass(slots=True)
class CredSweeperBaseline:
    """Control group 3: CredSweeper, run unmodified.

    Attributes:
        name: System name in reports.
        kind: ``binary`` — it applies its own threshold and reports decisions.
        pool_count: Worker processes CredSweeper may use. Affects speed only.
        unmapped_rules: Rule names seen in findings that have no class mapping,
            counted for the report.
        available: Whether the package imported. Recorded rather than raised so
            a missing optional control degrades the report instead of failing
            the run.
    """

    name: str = "credsweeper"
    kind: str = "binary"
    pool_count: int = max(1, (os.cpu_count() or 2) - 2)
    #: Smallest number of units worth giving a worker of its own.
    #:
    #: Measured on this machine (12 workers, Windows, which spawns rather than
    #: forks), with decisions identical in both cases:
    #:
    #:   2,000 units    1 chunk 5.0 s   vs  10 chunks 5.9 s   (18% SLOWER)
    #:  20,000 units    1 chunk 30.1 s  vs  12 chunks 26.1 s  (13% faster)
    #:
    #: So chunking is a loss at small sizes and a modest win at realistic ones,
    #: and the threshold sits above the crossover rather than below it. 2,000 is
    #: the larger of the two measured points, chosen in the conservative
    #: direction: at worst a mid-sized run forgoes a 13% gain, where a lower
    #: threshold would have made small runs reliably slower.
    #:
    #: Worth recording that 13% is nothing like the ~12x the `pool_count`
    #: setting implies. The single-process defect NR-0002 found is real and is
    #: fixed, but it was never costing an order of magnitude.
    min_units_per_chunk: int = 2000
    unmapped_rules: dict[str, int] = field(default_factory=dict)
    available: bool = True

    def _chunk_count(self, units: int) -> int:
        """Decide how many files to split the units across.

        Args:
            units: How many units will be scanned.

        Returns:
            At least one, never more than :attr:`pool_count`, and never so many
            that a chunk falls below :attr:`min_units_per_chunk`. A small
            evaluation therefore stays on a single file, where the pool would
            cost more in process start-up than it saved.
        """
        if units <= 0:
            return 1
        affordable = max(1, units // self.min_units_per_chunk)
        return max(1, min(self.pool_count, affordable))

    def version(self) -> str:
        """Return CredSweeper's version, or why it is unavailable."""
        try:
            from credsweeper import __version__

            return str(__version__)
        except ImportError as exc:  # pragma: no cover - exercised only without the package
            self.available = False
            return f"unavailable ({exc})"

    @staticmethod
    def render_line(unit: DetectionUnit) -> str:
        """Render one unit as a line CredSweeper can scan.

        Args:
            unit: The unit to render.

        Returns:
            ``key = "value"`` on a single line, with newlines escaped so a
            multi-line value cannot shift the line-to-unit mapping.
        """
        key = unit.key if not unit.key.startswith("<slot:") else "value"
        value = _NEWLINES.sub("\\\\n", unit.value).replace('"', '\\"')
        return f'{key} = "{value}"'

    def score(self, units: Sequence[DetectionUnit]) -> SystemOutput:
        """Score a sequence of units by scanning a rendered file.

        Args:
            units: The units to score.

        Returns:
            Binary output, empty for every unit when CredSweeper is unavailable.
        """
        version = self.version()
        if not self.available:
            return SystemOutput(
                name=self.name,
                kind="binary",
                scores=[{} for _ in units],
                version=version,
                notes={"status": "unavailable; control group skipped"},
            )

        from credsweeper.app import CredSweeper
        from credsweeper.file_handler.files_provider import FilesProvider

        self.unmapped_rules = {}
        scores: list[dict[str, float]] = [{} for _ in units]

        with tempfile.TemporaryDirectory() as workspace:
            # One file per worker, not one file for everything. CredSweeper's
            # pool parallelises **across files**, so the previous single-file
            # rendering left pool_count set to the machine's cores while exactly
            # one process did the work — recorded as a defect in NR-0002 §9.10,
            # and NR-0004's outline asks for it first.
            #
            # Chunking changes nothing about the decisions. CredSweeper's
            # line-context features look at neighbouring lines, but the units
            # are already in an arbitrary order with synthetic neighbours (the
            # caveat in `notes` says so), so a chunk boundary is the same kind of
            # artefact as every other line boundary here — and there are only
            # `chunks` of them against len(units) line boundaries.
            chunks = self._chunk_count(len(units))
            sources: list[Path] = []
            # offsets[name] is the global index of that chunk's first line,
            # which is what turns a per-file line number back into a unit.
            offsets: dict[str, int] = {}
            for index in range(chunks):
                start = index * len(units) // chunks
                stop = (index + 1) * len(units) // chunks
                if start == stop:
                    continue
                path = Path(workspace) / f"units-{index:03d}.txt"
                path.write_text(
                    "".join(self.render_line(unit) + "\n" for unit in units[start:stop]),
                    encoding="utf-8",
                )
                sources.append(path)
                offsets[path.name] = start

            report_path = Path(workspace) / "report.json"
            # CredSweeper defaults to a single worker, and its ML validator is
            # the dominant cost of the whole evaluation. Giving it the machine's
            # cores is a configuration choice, not a tuning one: it changes how
            # fast the tool runs, never what it decides.
            sweeper = CredSweeper(json_filename=str(report_path), pool_count=self.pool_count)
            # run() takes a content provider, not a path list: FilesProvider is
            # what turns paths into the scannable files it expects.
            sweeper.run(FilesProvider(sources))

            findings = []
            if report_path.is_file():
                text = report_path.read_text(encoding="utf-8").strip()
                if text:
                    findings = json.loads(text)

        for finding in findings:
            rule = str(finding.get("rule", ""))
            label = _RULE_TO_LABEL.get(rule)
            if label is None:
                self.unmapped_rules[rule] = self.unmapped_rules.get(rule, 0) + 1
                label = "api_key"
            for occurrence in finding.get("line_data_list") or []:
                line_number = occurrence.get("line_num")
                if not isinstance(line_number, int):
                    continue
                # A finding now carries which chunk it came from, so its line
                # number is local to that file and has to be shifted. An
                # unrecognised path is dropped rather than guessed at:
                # attributing a finding to the wrong unit would move a true
                # positive onto a negative and corrupt both numbers.
                name = Path(str(occurrence.get("path", ""))).name
                offset = offsets.get(name)
                if offset is None:
                    continue
                index = offset + line_number - 1
                if 0 <= index < len(scores):
                    scores[index][label] = 1.0

        return SystemOutput(
            name=self.name,
            kind="binary",
            scores=scores,
            version=version,
            notes={
                "input_rendering": 'one unit per line as key = "value"',
                "ml_threshold": "shipped default",
                "pool_count": str(self.pool_count),
                "input_files": (
                    "units are split across up to pool_count files, because "
                    "CredSweeper's pool parallelises across files rather than within one"
                ),
                "unmapped_rules": json.dumps(
                    dict(sorted(self.unmapped_rules.items())), ensure_ascii=False
                ),
                "caveat": (
                    "Units are reconstructed into lines, so CredSweeper's line-context "
                    "features see a synthetic context rather than the original file."
                ),
            },
        )
