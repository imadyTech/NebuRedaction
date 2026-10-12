"""Record what the antivirus quarantined during third-party data acquisition.

Required by work order NR-0001 §2.4 item 2A. CredData's acquisition clones
around 330 public repositories, and some of them are penetration-testing tools
whose code antivirus software classifies as malicious — correctly, in the sense
that the code really is what the signature says it is.

The rule this script serves: quarantined files are **not restored**, and **no
antivirus exclusion is added** for the data directory. What is done instead is
to record the event — path, threat name, time — and to count separately any
labelled samples lost as a result, without making them up.

This script only reads Defender's detection history. It never restores a file,
never changes a setting, and never adds an exclusion.

Usage:
    python corpus/record_av_quarantine.py
"""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CREDDATA_CHECKOUT = REPOSITORY_ROOT / "data" / "third_party" / "creddata" / "CredData"

#: Reads Defender's detection history and emits it as JSON. Read-only: the
#: cmdlets used here report state and cannot alter it.
_POWERSHELL_QUERY = r"""
$ErrorActionPreference = 'Stop'
$threats = @{}
foreach ($t in (Get-MpThreat)) { $threats[[string]$t.ThreatID] = $t }
$out = foreach ($d in (Get-MpThreatDetection)) {
  $t = $threats[[string]$d.ThreatID]
  [pscustomobject]@{
    detected_utc  = $d.InitialDetectionTime.ToUniversalTime().ToString('o')
    threat_id     = [string]$d.ThreatID
    threat_name   = if ($t) { $t.ThreatName } else { $null }
    severity      = if ($t) { $t.SeverityID } else { $null }
    action_success= [bool]$d.ActionSuccess
    status_id     = [string]$d.ThreatStatusID
    resources     = @($d.Resources)
  }
}
ConvertTo-Json -InputObject @($out) -Depth 5
"""


def query_defender() -> list[dict[str, Any]]:
    """Read Windows Defender's detection history.

    Returns:
        One record per detection, or an empty list if Defender is unavailable
        (a non-Windows machine, or the cmdlets not present).
    """
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_QUERY],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"could not read Defender history: {exc}", file=sys.stderr)
        return []
    text = completed.stdout.strip()
    if not text:
        return []
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def _repo_hash_from_resource(resource: str) -> str | None:
    r"""Extract the CredData clone directory name from a quarantined path.

    Args:
        resource: A Defender resource string, such as ``file:_E:\\...\\tmp\\<hash>\\...``.

    Returns:
        The 64-character clone directory name, or ``None`` if the path is not
        inside CredData's clone directory.
    """
    match = re.search(r"[\\/]tmp[\\/]([0-9a-f]{64})[\\/]", resource)
    return match.group(1) if match else None


def _meta_name_for(repo_id: str) -> str:
    """Return the meta CSV base name CredData uses for a snapshot entry.

    Args:
        repo_id: The 64-character snapshot key.

    Returns:
        The CRC32 of the id's raw bytes, in lower-case hex.
    """
    return f"{binascii.crc32(binascii.unhexlify(repo_id)):08x}"


def assess_sample_loss(repo_hashes: set[str]) -> list[dict[str, Any]]:
    """Work out whether quarantine cost any labelled samples.

    A quarantined file only costs samples if CredData's metadata actually
    references it. The distinction matters: the penetration-testing code that
    triggers the signatures is usually not the file that carries a credential,
    so the honest answer is often "none" — but it has to be checked rather than
    assumed, and checked per file rather than per repository.

    Args:
        repo_hashes: Clone directory names that had a file quarantined.

    Returns:
        One record per affected repository: its URL, its meta row counts, and
        how many of those rows point at a quarantined path.
    """
    snapshot_path = CREDDATA_CHECKOUT / "snapshot.json.orig"
    if not snapshot_path.is_file():
        snapshot_path = CREDDATA_CHECKOUT / "snapshot.json"
    if not snapshot_path.is_file():
        return []
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for repo_hash in sorted(repo_hashes):
        meta_base = _meta_name_for(repo_hash)
        record: dict[str, Any] = {
            "clone_directory": repo_hash,
            "repo_url": snapshot.get(repo_hash),
            "meta_file": f"{meta_base}.csv",
        }
        for candidate in (
            CREDDATA_CHECKOUT / "meta" / f"{meta_base}.csv",
            CREDDATA_CHECKOUT / "meta" / f"{meta_base}.csv.bak",
        ):
            if not candidate.is_file():
                continue
            with candidate.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            record["meta_file_present_as"] = candidate.name
            record["meta_rows"] = len(rows)
            record["by_ground_truth"] = {
                value: sum(1 for row in rows if row.get("GroundTruth") == value)
                for value in ("T", "F", "X")
            }
            break
        else:
            record["meta_rows"] = None
            record["note"] = "meta file not found for this repository"
        results.append(record)
    return results


def main(argv: list[str] | None = None) -> int:
    """Record the quarantine events and the sample loss they caused.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` always: an absence of detections is a valid result, not a failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "NR-0001",
        help="directory for the run log",
    )
    args = parser.parse_args(argv)

    detections = query_defender()
    in_creddata: list[dict[str, Any]] = []
    elsewhere = 0
    repo_hashes: set[str] = set()

    for detection in detections:
        resources = detection.get("resources") or []
        hashes = {h for h in (_repo_hash_from_resource(str(r)) for r in resources) if h}
        if not hashes:
            elsewhere += 1
            continue
        repo_hashes |= hashes
        in_creddata.append(
            {
                "detected_utc": detection.get("detected_utc"),
                "threat_name": detection.get("threat_name"),
                "severity": detection.get("severity"),
                "quarantine_succeeded": detection.get("action_success"),
                "paths": [
                    str(resource).replace("file:_", "").replace("\\", "/") for resource in resources
                ],
            }
        )

    report: dict[str, Any] = {
        "work_order": "NR-0001",
        "section": "2.4 item 2A",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "policy": (
            "Quarantined files are not restored and no antivirus exclusion is added for the "
            "data directory. Cloned third-party code is never executed, imported or installed."
        ),
        "detections_total": len(detections),
        "detections_in_creddata": len(in_creddata),
        "detections_elsewhere": elsewhere,
        "distinct_threat_names": sorted(
            {str(item["threat_name"]) for item in in_creddata if item.get("threat_name")}
        ),
        "affected_clone_directories": len(repo_hashes),
        "events": sorted(in_creddata, key=lambda item: str(item["detected_utc"])),
        "sample_loss": assess_sample_loss(repo_hashes),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "av-quarantine.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
