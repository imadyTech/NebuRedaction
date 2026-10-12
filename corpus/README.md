# corpus

Acquisition scripts and processing recipes for third-party datasets. **The data
itself never enters this repository**, and none of it is ever redistributed.

Permitted use differs per source and is not a single rule — see
[../docs/licensing/R0-license-review.md](../docs/licensing/R0-license-review.md)
for what each licence allows and which decision set it:

- **CredData** — local evaluation only; never trained on.
- **LogHub** and **SDLog's data** — **not used at all in v0.1**, for training or
  evaluation, and not downloaded. Permission will be sought from their
  maintainers after publication (plan §5.1 item 0).

## Scripts

| Script | What it does |
|---|---|
| `fetch_creddata.py` | Acquires CredData through its own pipeline. `--trial N` estimates the cost first; `--inventory` reports what is on disk. |
| `convert_creddata.py` | Converts CredData's meta rows into detection units, writing `data/third_party/eval_v0.jsonl` plus statistics. |
| `record_av_quarantine.py` | Reads Defender's detection history and records what acquisition triggered (NR-0001 §2.4 item 2A). Read-only. |
| `export_nebulog_own.py` | Runs NebuLog's demo locally and exports its logs as detection units into `data/own/`. |
| `mappings/creddata.yaml` | CredData's categories mapped to the seven classes, with the reasoning. |

## Acquiring CredData

```bash
# 1. Estimate the cost (clones a sample, measures, deletes it again).
uv run --group datagen python corpus/fetch_creddata.py --trial 10

# 2. Acquire. See the Windows notes below before running this.
uv run --group datagen python corpus/fetch_creddata.py --jobs 8 --exclude-unobtainable

# 3. Convert to detection units.
uv run --group datagen python corpus/convert_creddata.py
```

CredData ships labels, not credential values: its `meta` CSVs carry file paths,
line numbers and value offsets, and its `download_data.py` reconstructs the data
by cloning ~330 source repositories at pinned commits, extracting the referenced
files, obfuscating every credential, and deleting the clones. Measured on this
project's machine: about **2.5 s and 27 MB per repository**, so roughly **11 GB
transient** and 15–25 minutes with `--jobs 8`.

## Safety rules for acquisition

From NR-0001 §2.4 item 2A, and they are not optional.

**Cloned third-party code is never executed, imported or installed.** Among the
~330 source repositories are public penetration-testing tools. Only CredData's
own processing script runs, and it only ever reads the clones as text.

**Antivirus detections are recorded, never undone.** On this machine Defender
quarantined three files in CrackMapExec's clone
(`HackTool:Python/Pourri.A!MTB`, `HackTool:PowerShell/BloodHound`,
`Trojan:PowerShell/Fleisnam.E`). Quarantined files are **not restored**, and
**no antivirus exclusion is added** for the data directory. `record_av_quarantine.py`
records the event and counts separately any labelled samples lost; they are not
substituted for. In that instance none of CrackMapExec's 87 meta rows referenced
a quarantined file, so no labelled sample was lost.

**The clone directory is deleted as soon as the data is extracted**, keeping only
the extracted dataset, with the size recorded before and after. Deleting it
requires the operator's explicit authorisation.

## Windows notes

Two platform limits are worth knowing before a fetch, because both cause
CredData's pipeline to abort rather than degrade.

**Long paths.** Several source repositories (WebKit, kubernetes, elasticsearch
and about a dozen more) have paths beyond the 260-character limit. The fetch
script passes `core.longpaths=true` through git's `GIT_CONFIG_*` environment
protocol, which fixes it without touching the user's global git config and
without modifying CredData.

**Filenames NTFS cannot represent.** Four repositories cannot be checked out at
all: three contain a directory whose name ends in a space, and
`PayloadsAllTheThings` has file names containing `?`, `"`, `<` and `>` — the
names *are* the payloads. `--exclude-unobtainable` drops them and reports the
cost: **4 repositories, 411 of 67,896 meta rows (0.6%)**.

**Resuming is not possible on Windows.** CredData decides whether it can reuse
an existing clone by running a `sh` conditional through `shell=True`:

```sh
if [ -n "$(git status --porcelain)" ]; then exit 1; fi
```

`cmd.exe` cannot parse that, so the check fails for every existing clone however
healthy it is, and the code then calls `os.makedirs` on a directory that already
exists and dies with `FileExistsError`. A fetch therefore needs an empty clone
directory and one uninterrupted run. This is a property of the upstream script,
not something to work around by editing it.

Owner work orders: NR-0001 (acquisition and conversion), NR-0002 (external test
sets).
