"""Feasibility probe for the Laya teacher model on CPU (work order NR-0000 §2.7).

Measures what R3 planning needs from the machine that will run teacher labeling:
checkpoint load time, peak process memory, single-sample latency percentiles and
batched throughput, and it scores the model's zero-shot ``choice`` answers against
a labelled sample file.

The sample file is supplied **by path** and is only read, never copied: the R0
sample set contains third-party material that must not enter this repository
(plan §5.1, work order NR-0000 §2.7.3).

Example:
    uv run --group teacher python labeling/probe/laya_probe.py
    --samples ../docs/neburedaction/plan/samples-v0.jsonl --out runs/NR-0000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The seven v0.1 classes, from plan §3.2. The descriptions are the `choice`
# criteria handed to the teacher; they are deliberately phrased in terms of what
# the value is, not of how a rule would match it.
CLASSES: dict[str, str] = {
    "none": (
        "not a secret: an identifier, hash, UUID, trace or span id, version, "
        "timestamp, file path, hostname, count, size, user name or ordinary text"
    ),
    "password": "a password, passphrase or PIN used to authenticate a human or a service account",
    "token": (
        "a bearer, session, OAuth, refresh or JWT token: a credential issued by an "
        "authorisation server and valid for a limited time"
    ),
    "api_key": (
        "an API key, client secret, access key id or secret access key for a cloud or "
        "SaaS platform, or any other long high-entropy application secret"
    ),
    "private_key": "a private key or certificate key material, such as a PEM or PKCS block",
    "connection_string": (
        "a database, broker or service connection string or URL that carries an embedded "
        "user name and password"
    ),
    "cookie": "an HTTP cookie or session cookie value that authenticates a browser session",
}

QUESTION_KEY = "secret_class"


def build_questions() -> dict[str, Any]:
    """Build the Laya question schema for one detection unit.

    Returns:
        A single ``choice`` question over the seven classes of plan §3.2, in the
        typed-question form that plan §4.4 specifies for the teacher.
    """
    return {
        QUESTION_KEY: {
            "type": "choice",
            "instructions": (
                "The state is one attribute taken from an application log record. "
                "Decide what kind of value it holds. Answer 'none' unless the value "
                "really is a secret credential that must be redacted before the log "
                "is written."
            ),
            "criteria": dict(CLASSES),
        }
    }


def render_state(unit: dict[str, Any]) -> str:
    """Render a detection unit as the text state handed to the teacher.

    Args:
        unit: A detection unit with a ``key`` and ``value`` and optionally a
            ``template`` and ``scope`` (plan §3.1).

    Returns:
        A compact, stable textual rendering of the unit.
    """
    lines = [
        f"attribute name: {unit.get('key', '')}",
        f"attribute value: {unit.get('value', '')}",
    ]
    if unit.get("template"):
        lines.append(f"message template: {unit['template']}")
    if unit.get("scope"):
        lines.append(f"scope: {unit['scope']}")
    return "\n".join(lines)


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Read detection units from a JSON Lines sample file.

    Args:
        path: Path to the sample file. It is read in place and never copied.

    Returns:
        The parsed sample records, in file order.

    Raises:
        SystemExit: If the file does not exist.
    """
    if not path.is_file():
        sys.exit(f"sample file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def peak_rss_bytes() -> int | None:
    """Return the peak resident set size of this process, in bytes.

    Returns:
        The peak working set if the platform reports one, otherwise the current
        resident set size, or ``None`` if neither is available.
    """
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_wset", None) or info.rss)


def probe_accelerators() -> dict[str, Any]:
    """Probe optional Intel integrated-GPU acceleration paths (§2.7.5).

    Returns:
        What each path reports. The result is recorded only; no adaptation is
        made either way.
    """
    result: dict[str, Any] = {}
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_xpu_is_available"] = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        result["torch_xpu_device_count"] = (
            int(torch.xpu.device_count()) if result["torch_xpu_is_available"] else 0
        )
        result["torch_cuda_is_available"] = bool(torch.cuda.is_available())
    # A probe records what each path reports, including a failure to import or
    # initialise it; it must never fail the run.
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import openvino  # type: ignore[import-not-found]

        result["openvino_version"] = openvino.__version__
        result["openvino_devices"] = list(openvino.Core().available_devices)
    # OpenVINO is not a declared dependency; absence is a result, not an error.
    except Exception as exc:
        result["openvino"] = f"not available ({type(exc).__name__})"
    return result


def main() -> int:
    """Run the probe and write a JSON report.

    Returns:
        A process exit code: ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", required=True, type=Path, help="path to a JSON Lines sample file"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/NR-0000"), help="directory for the run log"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for the throughput run"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="passes over the samples for the latency run"
    )
    parser.add_argument("--seed", type=int, default=20261011, help="seed for every random choice")
    parser.add_argument("--model", default="english", help="Laya checkpoint to use")
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    samples = load_samples(args.samples)
    questions = build_questions()
    states = [render_state(record["unit"]) for record in samples]

    from laya import Router

    load_started = time.perf_counter()
    router = Router(device="cpu", default=args.model)
    router.load(args.model)
    load_seconds = time.perf_counter() - load_started

    # Warm-up: the first forward pass pays one-off allocation and kernel costs that
    # would otherwise land in the p95.
    warmup_started = time.perf_counter()
    router.predict(states[0], questions, model=args.model)
    warmup_seconds = time.perf_counter() - warmup_started

    single_latencies: list[float] = []
    predictions: list[dict[str, Any]] = []
    for pass_index in range(args.repeats):
        for sample, state in zip(samples, states, strict=True):
            started = time.perf_counter()
            result = router.predict(state, questions, model=args.model)
            elapsed = time.perf_counter() - started
            single_latencies.append(elapsed)
            if pass_index == 0:
                answer = result["answers"][QUESTION_KEY]
                predictions.append(
                    {
                        "id": sample.get("id"),
                        "source": sample.get("source"),
                        "key": sample["unit"].get("key"),
                        "expected": sample.get("label"),
                        "predicted": answer.get("choice"),
                        "probability": (answer.get("probabilities") or {}).get(
                            answer.get("choice")
                        ),
                        "confidence": answer.get("confidence"),
                        "answer_confidence": answer.get("answer_confidence"),
                        "latency_seconds": round(elapsed, 4),
                    }
                )

    requests = [{"state": state, "questions": questions, "model": args.model} for state in states]
    batch_started = time.perf_counter()
    router.predict_batch(requests, batch_size=args.batch_size)
    batch_seconds = time.perf_counter() - batch_started

    sorted_latencies = sorted(single_latencies)
    agreements = sum(1 for p in predictions if p["expected"] == p["predicted"])
    binary_agreements = sum(
        1 for p in predictions if (p["expected"] == "none") == (p["predicted"] == "none")
    )

    report = {
        "work_order": "NR-0000",
        "section": "2.7",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "model": {"checkpoint": args.model, "device": "cpu"},
        "samples": {"path": str(args.samples), "count": len(samples)},
        "timings_seconds": {
            "load": round(load_seconds, 3),
            "warmup_first_predict": round(warmup_seconds, 3),
            "single_p50": round(statistics.median(sorted_latencies), 4),
            "single_p95": round(
                sorted_latencies[min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))],
                4,
            ),
            "single_mean": round(statistics.fmean(sorted_latencies), 4),
            "single_min": round(sorted_latencies[0], 4),
            "single_max": round(sorted_latencies[-1], 4),
            "batch_total": round(batch_seconds, 3),
        },
        "throughput_samples_per_second": {
            "single": round(len(single_latencies) / sum(single_latencies), 2),
            f"batch_{args.batch_size}": round(len(states) / batch_seconds, 2),
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "accuracy": {
            "exact_class_agreement": f"{agreements}/{len(predictions)}",
            "redact_or_not_agreement": f"{binary_agreements}/{len(predictions)}",
        },
        "predictions": predictions,
        "accelerators": probe_accelerators(),
    }

    out_dir = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "laya-probe.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
