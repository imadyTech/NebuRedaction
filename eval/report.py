"""Report assembly (NR-0002 §2.1 item 4).

Two files, and the split between them is the design:

* ``report.json`` holds **only deterministic content**. NR-0002 §2.1 item 4
  requires two runs on the same input to produce the same file, so a wall-clock
  timestamp, a duration or a hostname cannot go in it. A test enforces this.
* ``run-meta.json`` holds everything that legitimately varies — when the run
  happened, how long it took, which machine. It is useful and it is kept, just
  not where it would break reproducibility.

Putting the timestamp in the report and then comparing reports with the
timestamp excluded would be the other way to satisfy the letter of the
requirement; this way satisfies it literally, and a reader can diff two reports
with ordinary tools.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.evaluate import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    TARGET_FALSE_POSITIVE_RATES,
    SystemRun,
    TestSet,
)

#: Report format version. Raise it when the structure changes, so a stored
#: report can be read by the code that understands it.
REPORT_VERSION = "1.0"


def build_report(
    runs: Sequence[SystemRun],
    test_sets: Sequence[TestSet],
    selection_set: TestSet,
    *,
    schema_version: str,
    testsets_lock: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the deterministic report.

    Args:
        runs: Each system's results.
        test_sets: The test sets evaluated against.
        selection_set: The set thresholds were chosen on.
        schema_version: The frozen detection-unit schema version.
        testsets_lock: The test-set manifest (files and digests).

    Returns:
        The report, containing no non-deterministic values.
    """
    return {
        "report_version": REPORT_VERSION,
        "schema_version": schema_version,
        "method": {
            "main_task": "redact or not: a unit is positive when its label is not 'none'",
            "target_false_positive_rates": list(TARGET_FALSE_POSITIVE_RATES),
            "threshold_selection": (
                f"Continuous systems' thresholds are chosen on '{selection_set.name}' "
                "(synthetic validation data) and applied unchanged to every test set. "
                "EXECUTION.md §8 forbids selecting thresholds on third-party data."
            ),
            "binary_systems": (
                "A rule set has one operating point, so its own measured false-positive rate "
                "and recall are reported instead of recall at a chosen rate."
            ),
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seed": BOOTSTRAP_SEED,
            },
            "average_precision": (
                "PR-AUC is average precision; ties are broken pessimistically, with negatives "
                "ranked ahead of positives at equal score."
            ),
        },
        "testsets_lock": testsets_lock,
        "test_sets": {
            test_set.name: {
                "count": len(test_set.units),
                "description": test_set.description,
            }
            for test_set in test_sets
        },
        "systems": {
            run.name: {
                "kind": run.kind,
                "version": run.version,
                "notes": dict(sorted(run.notes.items())),
                "thresholds": {
                    name: (round(value, 8) if value is not None else None)
                    for name, value in sorted(run.thresholds.items())
                },
                "results": run.results,
            }
            for run in sorted(runs, key=lambda r: r.name)
        },
    }


def build_run_meta(duration_seconds: float, systems: Sequence[str]) -> dict[str, Any]:
    """Assemble the non-deterministic run metadata.

    Args:
        duration_seconds: How long the run took.
        systems: The system names evaluated.

    Returns:
        Metadata kept out of ``report.json`` so that file stays reproducible.
    """
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 2),
        "systems": list(systems),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "note": (
            "Everything here varies between runs and is therefore kept out of report.json, "
            "which NR-0002 §2.1 item 4 requires to be identical for identical input."
        ),
    }


def _format_rate(value: float) -> str:
    """Format a rate as a percentage with enough precision to be useful.

    Args:
        value: A rate in [0, 1].

    Returns:
        A percentage string. Three decimals, because the project's target is
        0.1% and two decimals cannot tell 0.1% from 0.14%.
    """
    return f"{value * 100:.3f}%"


def _slice_table(slices: dict[str, Any]) -> list[str]:
    """Render one operating point's slices as a Markdown table.

    Args:
        slices: Slice name to metrics.

    Returns:
        Markdown lines.
    """
    lines = [
        "| slice | n | positives | recall | 95% CI | measured FPR | precision | PR-AUC |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, data in slices.items():
        confidence = data.get("recall_ci") or {}
        interval = (
            f"{confidence.get('low', 0):.3f}-{confidence.get('high', 0):.3f}" if confidence else "—"
        )
        lines.append(
            f"| `{name}` | {data['count']} | {data['positives']} | "
            f"{data['recall']:.3f} | {interval} | "
            f"{_format_rate(data['false_positive_rate'])} | "
            f"{data['precision']:.3f} | {data['average_precision']:.3f} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown.

    Args:
        report: The assembled report.

    Returns:
        The Markdown document.
    """
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Report format {report['report_version']}; "
        f"detection-unit schema {report['schema_version']}.",
        "",
        "## Method",
        "",
        f"- **Main task**: {report['method']['main_task']}.",
        f"- **Threshold selection**: {report['method']['threshold_selection']}",
        f"- **Binary systems**: {report['method']['binary_systems']}",
        f"- **Confidence intervals**: bootstrap, "
        f"{report['method']['bootstrap']['resamples']} resamples, "
        f"{report['method']['bootstrap']['confidence']:.0%}, "
        f"seed {report['method']['bootstrap']['seed']}.",
        f"- **PR-AUC**: {report['method']['average_precision']}",
        "",
        "## Test sets",
        "",
        "| test set | units | what it answers |",
        "|---|---:|---|",
    ]
    for name, data in report["test_sets"].items():
        lines.append(f"| `{name}` | {data['count']} | {data['description']} |")

    for system_name, system in report["systems"].items():
        lines += ["", f"## {system_name}", ""]
        lines.append(f"- kind: **{system['kind']}**, version `{system['version']}`")
        if system["thresholds"]:
            rendered = ", ".join(
                f"`{name}` = {value}" for name, value in system["thresholds"].items()
            )
            lines.append(f"- thresholds chosen on the selection set: {rendered}")
        for key, value in system["notes"].items():
            lines.append(f"- {key}: {value}")

        for set_name, set_result in system["results"].items():
            lines += ["", f"### {system_name} on `{set_name}`", ""]
            lines.append(
                f"{set_result['count']} units, {set_result['positives']} positive, "
                f"{set_result['negatives']} negative."
            )
            for point_name, point in set_result["operating_points"].items():
                lines += ["", f"**Operating point `{point_name}`**"]
                if point.get("threshold") is None:
                    lines += ["", f"_{point.get('note', 'not operable')}_"]
                    continue
                lines += [f" (threshold {point['threshold']})", ""]
                lines += _slice_table(point["slices"])

            robustness = set_result.get("robustness")
            if robustness:
                lines += ["", "**Robustness**", ""]
                lines += [
                    "| transform | operating point | recall | Δ vs canonical | measured FPR |",
                    "|---|---|---:|---:|---:|",
                ]
                for transform_name, transform in robustness.items():
                    for point_name, point in transform["operating_points"].items():
                        delta = point.get("recall_delta_vs_canonical")
                        lines.append(
                            f"| `{transform_name}` | `{point_name}` | "
                            f"{point['recall']:.3f} | "
                            f"{'—' if delta is None else f'{delta:+.3f}'} | "
                            f"{_format_rate(point['false_positive_rate'])} |"
                        )
    return "\n".join(lines) + "\n"


def write_report(
    report: dict[str, Any], run_meta: dict[str, Any], out_dir: Path
) -> dict[str, Path]:
    """Write the report files.

    Args:
        report: The deterministic report.
        run_meta: The non-deterministic metadata.
        out_dir: Directory to write into.

    Returns:
        The paths written, keyed by short name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report_json": out_dir / "report.json",
        "report_md": out_dir / "report.md",
        "run_meta": out_dir / "run-meta.json",
    }
    paths["report_json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    paths["report_md"].write_text(render_markdown(report), encoding="utf-8")
    paths["run_meta"].write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths
