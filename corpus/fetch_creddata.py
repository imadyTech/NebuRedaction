"""Acquire the CredData evaluation corpus (NR-0001 §2.4 item 1).

CredData is the only third-party dataset v0.1 uses, and it is used for
**evaluation only** — never for training (plan §5.1, EXECUTION.md §5). The data
never enters this repository; only this script and the mapping beside it do.

How acquisition works. CredData does not ship credential values. Its ``meta``
CSVs carry labels plus file paths, line numbers and value offsets, and its
``download_data.py`` reconstructs the data by cloning each of the ~300 source
repositories at a pinned commit, extracting the referenced files, obfuscating
every credential, and deleting the clones. So the values that end up on disk are
already obfuscated by the dataset's authors, and the download is unavoidably a
few hundred git clones.

That makes a full fetch a long task. EXECUTION.md §4 requires estimating such a
task from a small trial first, which is what ``--trial`` does: it clones a
sample of the snapshot, measures time and bytes, and extrapolates. Run that
before committing to the full fetch.

Usage:
    # Estimate the full fetch from a 12-repository sample.
    python corpus/fetch_creddata.py --trial 12

    # Do the real thing, through CredData's own pipeline.
    python corpus/fetch_creddata.py --jobs 8

    # Inventory what is already on disk.
    python corpus/fetch_creddata.py --inventory
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: Where CredData is checked out and materialised. Under data/, so .gitignore
#: keeps every byte of it out of git, and under third_party/, so the unified
#: loader refuses to read it from a training path.
CREDDATA_ROOT = REPOSITORY_ROOT / "data" / "third_party" / "creddata"
CHECKOUT_DIR = CREDDATA_ROOT / "CredData"

#: Where CredData's pipeline puts the extracted dataset.
#:
#: This has to be ``CredData/data``, and ``--data_dir`` has to be passed as the
#: relative string ``data``. CredData's extraction step builds each destination
#: path from ``data_dir`` and compares it to the ``FilePath`` column of its own
#: metadata, which stores paths relative to the repository root
#: (``data/<repo>/<scope>/<id>.<ext>``). An absolute ``--data_dir`` makes every
#: comparison fail and the step aborts on the first file with
#: "Cannot find ...". So the path is relative and the working directory is the
#: checkout.
DATASET_DIR = CREDDATA_ROOT / "CredData" / "data"
DATASET_DIR_ARGUMENT = "data"

CREDDATA_URL = "https://github.com/Samsung/CredData"

#: The CredData commit to pin. Acquisition must be reproducible: an unpinned
#: clone would silently change the evaluation set between runs, which plan §6.4
#: forbids once the evaluation framework is fixed at G1. Set by --ref on the
#: first fetch and recorded in the run log.
DEFAULT_REF = "main"


def git_environment() -> dict[str, str]:
    """Return an environment that makes git tolerate long Windows paths.

    Several of CredData's source repositories (WebKit, kubernetes,
    elasticsearch and a dozen more) have paths longer than Windows' default
    260-character limit, and ``git checkout`` fails on them with "Filename too
    long". CredData's pipeline treats that as fatal and aborts the whole run.

    ``core.longpaths`` makes git use the Windows extended-length path APIs,
    which have no such limit. It is passed through git's ``GIT_CONFIG_*`` environment
    protocol rather than written to the user's global config: the setting is
    this task's business, not a change to how git behaves for everything else
    on the machine. Child processes inherit it, so CredData's own ``git`` calls
    get it too without modifying CredData.

    Returns:
        The current environment plus the git configuration overrides.
    """
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.longpaths",
            "GIT_CONFIG_VALUE_0": "true",
        }
    )
    return environment


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    """Run a command and return its stdout.

    Args:
        command: Argument vector; never a shell string.
        cwd: Working directory.
        check: Raise if the command fails.

    Returns:
        Captured stdout.

    Raises:
        subprocess.CalledProcessError: If the command fails and ``check`` is set.
    """
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=check, env=git_environment()
    )
    return completed.stdout


def clone_creddata(ref: str) -> str:
    """Clone or update the CredData repository itself.

    Args:
        ref: Branch, tag or commit to check out.

    Returns:
        The resolved commit SHA, for the run log.
    """
    if not CHECKOUT_DIR.is_dir():
        CHECKOUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {CREDDATA_URL} -> {CHECKOUT_DIR}")
        run(["git", "clone", CREDDATA_URL, str(CHECKOUT_DIR)])
    else:
        print(f"updating existing checkout at {CHECKOUT_DIR}")
        run(["git", "fetch", "--all"], cwd=CHECKOUT_DIR)
    run(["git", "checkout", ref], cwd=CHECKOUT_DIR)
    return run(["git", "rev-parse", "HEAD"], cwd=CHECKOUT_DIR).strip()


def load_snapshot() -> dict[str, str]:
    """Read CredData's snapshot of source repositories and pinned commits.

    Returns:
        A mapping of ``commit-sha-plus-id`` to repository URL.

    Raises:
        FileNotFoundError: If the checkout is missing.
    """
    path = CHECKOUT_DIR / "snapshot.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; clone CredData first")
    return json.loads(path.read_text(encoding="utf-8"))


def directory_size(path: Path) -> tuple[int, int]:
    """Measure a directory tree.

    Args:
        path: Directory to measure.

    Returns:
        ``(file_count, total_bytes)``.
    """
    count = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return count, total


def human(byte_count: int) -> str:
    """Format a byte count for a report.

    Args:
        byte_count: Size in bytes.

    Returns:
        A short human-readable string.
    """
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def trial(sample_size: int, seed: int, out_dir: Path) -> dict[str, Any]:
    """Clone a sample of the source repositories and extrapolate the full cost.

    The sample is drawn with a fixed seed so the estimate is reproducible, and
    the clones are shallow-ish only in the sense that each is a single commit —
    CredData itself needs full history for its checkout step, so this
    deliberately measures the same operation the real fetch performs, which is
    the only way the extrapolation means anything.

    Args:
        sample_size: How many repositories to clone.
        seed: Seed for choosing the sample.
        out_dir: Directory for the trial clones and the measurement log.

    Returns:
        The measurements and the extrapolation.
    """
    snapshot = load_snapshot()
    items = sorted(snapshot.items())
    rng = random.Random(seed)
    sample = rng.sample(items, min(sample_size, len(items)))

    workspace = out_dir / "trial-clones"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (repo_id, repo_url) in enumerate(sample, start=1):
        commit = repo_id[:40]
        target = workspace / f"repo{index:03d}"
        clone_started = time.perf_counter()
        ok = True
        try:
            run(["git", "clone", "--no-checkout", repo_url, str(target)])
            run(["git", "checkout", commit], cwd=target)
        except subprocess.CalledProcessError:
            ok = False
        elapsed = time.perf_counter() - clone_started
        count, size = directory_size(target) if target.is_dir() else (0, 0)
        results.append(
            {
                "repo_url": repo_url,
                "commit": commit,
                "ok": ok,
                "seconds": round(elapsed, 2),
                "files": count,
                "bytes": size,
            }
        )
        print(
            f"  [{index}/{len(sample)}] {'ok ' if ok else 'FAIL'} "
            f"{elapsed:6.1f}s {human(size):>9}  {repo_url}"
        )

    total_elapsed = time.perf_counter() - started
    successful = [item for item in results if item["ok"]]
    total_repositories = len(items)

    report: dict[str, Any] = {
        "work_order": "NR-0001",
        "section": "2.4",
        "kind": "creddata-fetch-trial",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "sample_size": len(sample),
        "total_repositories_in_snapshot": total_repositories,
        "successful_clones": len(successful),
        "failed_clones": len(results) - len(successful),
        "sample_seconds": round(total_elapsed, 1),
        "sample_bytes": sum(int(item["bytes"]) for item in successful),
        "per_repository": results,
    }
    if successful:
        mean_seconds = sum(float(item["seconds"]) for item in successful) / len(successful)
        mean_bytes = report["sample_bytes"] / len(successful)
        report["estimate"] = {
            "mean_seconds_per_repository": round(mean_seconds, 2),
            "mean_bytes_per_repository": int(mean_bytes),
            "serial_hours_full_fetch": round(mean_seconds * total_repositories / 3600, 2),
            "transient_bytes_full_fetch": int(mean_bytes * total_repositories),
            "transient_size_full_fetch": human(int(mean_bytes * total_repositories)),
            "note": (
                "Serial estimate. download_data.py parallelises with --jobs, so wall time "
                "divides roughly by the job count until the network saturates. The transient "
                "size is the clones, which CredData deletes after extracting the referenced "
                "files; the dataset that remains is far smaller."
            ),
        }

    shutil.rmtree(workspace, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "creddata-fetch-trial.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def inventory(out_dir: Path) -> dict[str, Any]:
    """Record what CredData files are present on disk and how large they are.

    Args:
        out_dir: Directory for the inventory file.

    Returns:
        The inventory, as recorded.
    """
    report: dict[str, Any] = {
        "work_order": "NR-0001",
        "kind": "creddata-inventory",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "root": str(CREDDATA_ROOT),
        "present": CREDDATA_ROOT.is_dir(),
    }
    if CREDDATA_ROOT.is_dir():
        for label, path in (("checkout", CHECKOUT_DIR), ("dataset", DATASET_DIR)):
            if path.is_dir():
                count, size = directory_size(path)
                report[label] = {
                    "path": str(path),
                    "files": count,
                    "bytes": size,
                    "size": human(size),
                }
        if DATASET_DIR.is_dir():
            top = sorted(item.name for item in DATASET_DIR.iterdir())
            report["dataset_entries"] = top[:200]
            report["dataset_entry_count"] = len(top)
        meta_dir = CHECKOUT_DIR / "meta"
        if meta_dir.is_dir():
            metas = sorted(meta_dir.glob("*.csv"))
            report["meta_files"] = len(metas)
            report["meta_bytes"] = sum(path.stat().st_size for path in metas)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "creddata-inventory.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


#: Virtual environment for CredData's own pipeline dependencies (GitPython,
#: pybase62, tabulate, colorama). Kept separate from the project environment:
#: those are CredData's requirements, not this project's, and nothing from a
#: third-party repository belongs in the environment that builds our artifacts.
CREDDATA_VENV = CREDDATA_ROOT / ".venv"


def creddata_interpreter(override: str | None = None) -> str:
    """Return the interpreter to run CredData's pipeline with.

    Args:
        override: An explicit interpreter path from the command line.

    Returns:
        The override if given, else the isolated environment's interpreter if it
        exists, else this process's interpreter. Falling back rather than failing
        keeps the script usable, and the pipeline's own ImportError will say what
        is missing.
    """
    if override:
        return override
    for candidate in (
        CREDDATA_VENV / "Scripts" / "python.exe",
        CREDDATA_VENV / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


#: Source repositories that cannot be checked out on Windows at all.
#:
#: These three contain a directory whose name ends in a space
#: ("MyNotes Room Jetpack /"), which NTFS cannot represent — git reports
#: "invalid path" and the checkout fails whatever is configured. This is not the
#: 260-character limit that `core.longpaths` fixes; it is a filename the
#: filesystem has no way to store.
#:
#: CredData's pipeline treats any failed checkout as fatal, so these have to be
#: excluded for the fetch to complete on this platform. Excluding them loses
#: their meta rows from the evaluation set; `--exclude-unobtainable` reports
#: exactly how many, so the loss is measured rather than assumed.
#: Source repositories that cannot be acquired on this machine, with the reason.
#:
#: Each entry costs its repository's labelled samples. The reasons are kept
#: distinct because they mean different things for the evaluation set and are
#: reported separately (NR-0001 §2.4 item 2A): a filesystem limit is a property
#: of this platform, while an antivirus quarantine is a deliberate decision not
#: to hold the file.
UNOBTAINABLE: dict[str, str] = {
    # NTFS cannot represent a directory name ending in a space.
    "https://github.com/hussien89aa/KotlinUdemy": "ntfs_illegal_name",
    "https://github.com/qibaoguang/Spring-Boot-Reference-Guide": "ntfs_illegal_name",
    "https://github.com/threedr3am/learnjavabug": "ntfs_illegal_name",
    # File names containing characters NTFS reserves. For the payload
    # collection the names *are* the payloads ('shell.gif?shell.php',
    # '"><svg onload=alert(1)>'); archer has migration scripts named
    # 'master->v2.0.sql'.
    "https://github.com/swisskyrepo/PayloadsAllTheThings": "ntfs_illegal_name",
    "https://github.com/jly8866/archer": "ntfs_illegal_name",
    # Antivirus quarantined three files in this clone (CrackMapExec, a
    # penetration-testing tool). Work order §2.4 item 2A forbids restoring them
    # or adding a scanner exclusion, and the extraction step cannot read past
    # the stubs Defender leaves behind, so the repository is dropped and its
    # labelled samples are counted as quarantine loss rather than recovered.
    "https://github.com/byt3bl33d3r/CrackMapExec": "antivirus_quarantine",
    # Never materialised here: the download pool aborted before reaching them.
    # Not known to be unobtainable in principle.
    "https://github.com/tornadoweb/tornado": "not_materialised",
    "https://github.com/tymondesigns/jwt-auth": "not_materialised",
    "https://github.com/microsoft/VFSForGit": "not_materialised",
    # Cloned fine, but CredData's extraction step cannot see all their files.
    # It enumerates a clone with `pathlib.Path(...).glob("**/*")`, and on
    # Windows that cannot return paths beyond the 260-character limit — the
    # same limitation that defeats shutil.rmtree. These repositories all have
    # very deep trees, so the enumeration comes back short, the file set no
    # longer matches the metadata, and CredData drops the repository.
    #
    # `core.longpaths` fixes the *clone*, which is why these download
    # successfully; it cannot fix a caller that uses the non-extended APIs.
    # Patching upstream's enumeration is out of scope, so these are excluded
    # and their 11,425 labelled rows are recorded as lost.
    "https://github.com/cloudfoundry/credhub": "windows_path_enumeration",
    "https://github.com/spring-projects/spring-security-oauth": "windows_path_enumeration",
    "https://github.com/dromara/shenyu": "windows_path_enumeration",
    "https://github.com/rest-assured/rest-assured": "windows_path_enumeration",
    "https://github.com/elastic/elasticsearch": "windows_path_enumeration",
    "https://github.com/aws/aws-sdk-java": "windows_path_enumeration",
    "https://github.com/kubernetes/kubernetes": "windows_path_enumeration",
    "https://github.com/twilio/twilio-ruby": "windows_path_enumeration",
    "https://github.com/redisson/redisson": "windows_path_enumeration",
    "https://github.com/alibaba/Sentinel": "windows_path_enumeration",
}

#: Kept for callers that only need the names.
UNOBTAINABLE_ON_WINDOWS: tuple[str, ...] = tuple(UNOBTAINABLE)


def meta_name_for(repo_id: str) -> str:
    """Return the meta CSV name CredData uses for a snapshot entry.

    Mirrors ``get_new_repo_id`` in CredData's ``download_data.py``: the CRC32 of
    the raw bytes of the repository id, in lower-case hex.

    Args:
        repo_id: The 64-character snapshot key.

    Returns:
        The meta file's base name, without the ``.csv`` suffix.
    """
    import binascii

    return f"{binascii.crc32(binascii.unhexlify(repo_id)):08x}"


def exclude_repositories(urls: tuple[str, ...]) -> dict[str, Any]:
    """Remove repositories from the snapshot and set their meta files aside.

    Setting the meta file aside with a ``.bak`` suffix is CredData's own
    convention for data it could not materialise — ``move_files`` does exactly
    this when files are missing — so the downstream obfuscation step skips them
    without being modified.

    The original snapshot is kept as ``snapshot.json.orig`` so the exclusion is
    reversible and auditable.

    This must be re-applied on every fetch. ``snapshot.json`` and the meta CSVs
    are tracked files in CredData's repository, so the ``git checkout`` that
    pins the CredData revision restores them — an exclusion applied by an
    earlier invocation is gone by the time the next one downloads anything.
    Hence the idempotence below, and hence ``--exclude-unobtainable`` belongs on
    the fetch command, not on a separate preparation step.

    Args:
        urls: Repository URLs to exclude.

    Returns:
        What was excluded and how many meta rows went with it.
    """
    import csv as _csv

    snapshot_path = CHECKOUT_DIR / "snapshot.json"
    original_path = CHECKOUT_DIR / "snapshot.json.orig"
    snapshot = json.loads(
        (original_path if original_path.is_file() else snapshot_path).read_text(encoding="utf-8")
    )
    if not original_path.is_file():
        original_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    wanted = set(urls)
    excluded: list[dict[str, Any]] = []
    kept: dict[str, str] = {}
    for repo_id, repo_url in snapshot.items():
        if repo_url in wanted:
            meta_path = CHECKOUT_DIR / "meta" / f"{meta_name_for(repo_id)}.csv"
            rows = 0
            set_aside = meta_path.with_suffix(".csv.bak")
            if meta_path.is_file():
                with meta_path.open(encoding="utf-8", newline="") as handle:
                    rows = sum(1 for _ in _csv.DictReader(handle))
                set_aside.unlink(missing_ok=True)
                meta_path.rename(set_aside)
            elif set_aside.is_file():
                with set_aside.open(encoding="utf-8", newline="") as handle:
                    rows = sum(1 for _ in _csv.DictReader(handle))
            excluded.append(
                {
                    "repo_url": repo_url,
                    "repo_id": repo_id,
                    "meta_file": meta_path.name,
                    "meta_rows": rows,
                    "reason": UNOBTAINABLE.get(repo_url, "unspecified"),
                }
            )
        else:
            kept[repo_id] = repo_url

    snapshot_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return {
        "requested": list(urls),
        "excluded": excluded,
        "excluded_repositories": len(excluded),
        "excluded_meta_rows": sum(int(item["meta_rows"]) for item in excluded),
        "snapshot_size_before": len(snapshot),
        "snapshot_size_after": len(kept),
        "excluded_meta_rows_by_reason": {
            reason: sum(int(item["meta_rows"]) for item in excluded if item["reason"] == reason)
            for reason in sorted({str(item["reason"]) for item in excluded})
        },
    }


def _force_remove(path: Path) -> None:
    """Remove a directory tree, including paths too long for the normal APIs.

    ``shutil.rmtree`` cannot do this job on Windows. CredData's clones contain
    paths well past the 260-character limit — deeply nested test fixtures in
    kubernetes, WebKit and a PowerShell DSC module among them — and Python's
    ``os`` calls fail on them with "The system cannot find the path specified",
    aborting the walk partway through and leaving most of the tree behind.

    ``robocopy`` uses the extended-length path APIs throughout, so mirroring an
    empty directory onto the target empties it whatever the path lengths are.
    It is also several times faster than a Python walk over 700,000 files. The
    now-empty skeleton goes last, and ``rmtree`` can manage that because the
    long paths are gone with their files.

    Args:
        path: Directory to remove.

    Raises:
        OSError: If the tree is still present afterwards.
    """
    import stat
    import tempfile

    if os.name == "nt":
        with tempfile.TemporaryDirectory() as empty:
            # /MIR makes the target match the (empty) source, which deletes
            # everything in it. The /N* flags silence the per-file log; without
            # them robocopy prints a line per file, which for 700k files is the
            # slowest part of the operation.
            subprocess.run(
                [
                    "robocopy",
                    empty,
                    str(path),
                    "/MIR",
                    "/NFL",
                    "/NDL",
                    "/NJH",
                    "/NJS",
                    "/NC",
                    "/NS",
                    "/NP",
                    "/R:1",
                    "/W:1",
                ],
                capture_output=True,
                check=False,
            )

    def on_error(function: Any, failed_path: str, _excinfo: Any) -> None:
        """Clear the read-only attribute git leaves behind and retry once."""
        Path(failed_path).chmod(stat.S_IWRITE)
        function(failed_path)

    if path.is_dir():
        shutil.rmtree(path, onexc=on_error)
    if path.is_dir():
        raise OSError(f"{path} is still present after removal")


def clear_tmp() -> dict[str, Any]:
    """Remove CredData's clone directory entirely before a fetch.

    **Why a resumed fetch is not possible on Windows.** CredData decides whether
    it can reuse an existing clone by running two commands through
    ``subprocess.check_call(..., shell=True)``, and the second is a shell
    conditional written for ``sh``::

        if [ -n "$(git status --porcelain)" ]; then exit 1; fi

    On Windows the shell is ``cmd.exe``, which cannot parse that, so the check
    fails for *every* existing clone no matter how healthy it is. The code then
    falls through to its download path, which begins with ``os.makedirs`` on a
    directory that already exists, and the run dies with ``FileExistsError``.

    So on this platform the pipeline needs an empty ``tmp`` and one
    uninterrupted run. Clearing it up front is what makes that true; the earlier
    attempt to validate and keep the good clones was wasted effort, because
    CredData will not reuse them regardless of their state.

    Returns:
        What was removed.
    """
    tmp = CHECKOUT_DIR / "tmp"
    if not tmp.is_dir():
        return {"present": False, "removed_directories": 0, "removal_failures": []}

    directories = [item.name for item in tmp.iterdir() if item.is_dir()]
    # NR-0001 §2.4 item 2A requires the volume before and after, so measure
    # rather than estimate. The scan walks ~700k files and takes a minute or two.
    files_before, bytes_before = directory_size(tmp)
    failures: list[str] = []
    try:
        _force_remove(tmp)
    except OSError as exc:
        failures.append(str(exc))
    still_exists = tmp.is_dir()
    files_after, bytes_after = directory_size(tmp) if still_exists else (0, 0)
    return {
        "present": True,
        "removed_directories": len(directories),
        "files_before": files_before,
        "bytes_before": bytes_before,
        "size_before": human(bytes_before),
        "files_after": files_after,
        "bytes_after": bytes_after,
        "size_after": human(bytes_after),
        "still_exists": still_exists,
        "removal_failures": failures,
    }


def full_fetch(
    jobs: int, noise: int, python: str | None = None, *, skip_download: bool = False
) -> int:
    """Run CredData's own download pipeline.

    The pipeline is run unmodified and in its own directory, because that is
    what NR-0001 §2.4 item 1 means by the official acquisition flow: CredData's
    obfuscation step is part of how the data becomes safe to hold, and
    reimplementing it would be both wasteful and riskier.

    Args:
        jobs: Parallel jobs to pass through.
        noise: CredData's obfuscation seed perturbation. Left at 0 so the
            obfuscated values are the dataset's canonical ones and the
            evaluation set is reproducible.
        python: Interpreter to use; defaults to the isolated environment.
        skip_download: Extract from the clones already present instead of
            downloading. This is how an interrupted fetch is resumed on
            Windows, where CredData's own reuse check cannot work.

    Returns:
        The pipeline's exit status.
    """
    script = CHECKOUT_DIR / "download_data.py"
    if not script.is_file():
        print(f"{script} not found; clone CredData first", file=sys.stderr)
        return 2
    command = [
        creddata_interpreter(python),
        "download_data.py",
        "--data_dir",
        DATASET_DIR_ARGUMENT,
        "--jobs",
        str(jobs),
        "--noise",
        str(noise),
    ]
    if skip_download:
        # Extract from the clones already in tmp/. This also sidesteps the
        # Windows reuse bug described in clear_tmp(): --skip_download goes
        # straight to the extraction step and never runs the checkout test that
        # cmd.exe cannot parse.
        command.append("--skip_download")
    print(f"running: {' '.join(command)} (cwd={CHECKOUT_DIR})")
    completed = subprocess.run(command, cwd=CHECKOUT_DIR, check=False, env=git_environment())
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Fetch, trial-fetch or inventory CredData.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial",
        type=int,
        metavar="N",
        help="clone N sampled repositories, measure, extrapolate, then delete them",
    )
    parser.add_argument(
        "--inventory", action="store_true", help="record what is already on disk and exit"
    )
    parser.add_argument("--jobs", type=int, default=4, help="parallel jobs for the full fetch")
    parser.add_argument("--noise", type=int, default=0, help="CredData obfuscation perturbation")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "extract from the clones already in tmp/ instead of downloading. Implies "
            "--keep-tmp, and is the way to resume after an interrupted fetch on Windows"
        ),
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help=(
            "do not clear CredData's tmp directory first. Only useful for inspecting a "
            "previous run: CredData cannot resume from an existing tmp on Windows"
        ),
    )
    parser.add_argument(
        "--exclude-unobtainable",
        action="store_true",
        help=(
            "drop the source repositories that cannot be checked out on Windows, "
            "and report how many meta rows that costs"
        ),
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter for CredData's pipeline (default: its isolated .venv)",
    )
    parser.add_argument("--ref", default=DEFAULT_REF, help="CredData ref to check out")
    parser.add_argument("--seed", type=int, default=20261012, help="seed for the trial sample")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "NR-0001",
        help="directory for run logs",
    )
    args = parser.parse_args(argv)

    stamped = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if args.inventory:
        print(json.dumps(inventory(stamped), indent=2))
        return 0

    commit = clone_creddata(args.ref)
    print(f"CredData at {commit}")

    if args.exclude_unobtainable:
        exclusion = exclude_repositories(UNOBTAINABLE_ON_WINDOWS)
        stamped.mkdir(parents=True, exist_ok=True)
        (stamped / "creddata-exclusions.json").write_text(
            json.dumps(exclusion, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(exclusion, indent=2))

    if args.trial is not None:
        report = trial(args.trial, args.seed, stamped)
        print(json.dumps({k: v for k, v in report.items() if k != "per_repository"}, indent=2))
        print(f"\ntrial log: {stamped / 'creddata-fetch-trial.json'}")
        return 0

    if not args.keep_tmp and not args.skip_download:
        cleared = clear_tmp()
        stamped.mkdir(parents=True, exist_ok=True)
        (stamped / "creddata-tmp-cleared.json").write_text(
            json.dumps(cleared, indent=2) + "\n", encoding="utf-8"
        )
        print(f"cleared tmp: removed {cleared['removed_directories']} clone directories")
        if cleared["removal_failures"]:
            print(f"  removal failures: {cleared['removal_failures']}", file=sys.stderr)
            return 2

    status = full_fetch(args.jobs, args.noise, args.python, skip_download=args.skip_download)
    print(json.dumps(inventory(stamped), indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
