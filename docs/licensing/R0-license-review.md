# R0 License Review

Per-source license verification for every dataset, model weight and rule set
nebuRedaction touches. Required by work order NR-0000 §2.6; the usage column
follows the lowest-risk arrangement decided in plan §5.1 (released models are
trained on synthetic and own data only; third-party data is used for local
evaluation only, is never committed to this repository and is never
redistributed).

All URLs were checked on **2026-10-11**.

**This file is an engineering record. It is not legal advice.**

---

## Summary

| # | Source | License as found | Our use | Attribution duty | Verdict |
|---|---|---|---|---|---|
| 1 | CredData (Samsung) | Repo Apache-2.0; each data file under its originating project's license | Local evaluation only | Cite CredData; per-file notices only matter if redistributed, and we do not redistribute | Usable as planned |
| 2 | SDLog annotated data (mooselab) | Repo MIT; data directory carries no separate statement; data is LogHub-derived | Local evaluation only | Cite SDLog and LogHub | **Inherits the LogHub question — see #3** |
| 3 | LogHub (logpai) | No SPDX license; terms of use state research/academic availability | Local evaluation only | Cite the LogHub paper and link the repository | **Recommend removing from evaluation** — Frank's decision |
| 4 | Laya weights (convaiinnovations/laya) | Apache-2.0 | Teacher labeling of synthetic and own samples | Keep the Apache-2.0 notice; name Laya in the model card | Usable as planned |
| 5 | CredSweeper (Samsung) | MIT | Run as an evaluation control group | Reproduce the MIT notice if any of its code is vendored | Usable as planned |
| 6 | gitleaks rules | MIT | Rule porting (patterns as reference) | Reproduce the MIT notice and attribute ported rules | Usable as planned |
| 6 | detect-secrets rules (Yelp) | Apache-2.0 | Rule porting (patterns as reference) | NOTICE/attribution per Apache-2.0 §4 for ported material | Usable as planned |
| 7 | StarPII (bigcode) | No license in the model card metadata; access is gated | **Excluded** | — | Excluded; see #7 |

---

## 1. CredData (Samsung)

- **URL**: <https://github.com/Samsung/CredData>
- **Verified**: 2026-10-11
- **License as found**: the repository is Apache-2.0 (GitHub reports
  `Apache-2.0`). The README's License section states that each data file stays
  under its originating project's license, and that the per-repository licenses
  are recorded in the `license` directory keyed by RepoID: "Each file is under
  the existing project's license." The dataset authors state that they filtered
  the source repositories by whether the license permitted dataset use, with
  manual review where GitHub's license detection was unreliable.
- **Our use**: **local evaluation only.** CredData supplies credential samples
  with their surrounding context; its values are already obfuscated by the
  dataset authors. It never enters the training set, never enters this
  repository, and is never redistributed. Only the acquisition script, the
  processing recipe and aggregate metrics are published.
- **Attribution duty**: cite CredData in the evaluation report, the data card
  and the model card. The per-file originating licenses constrain
  redistribution, which we do not do; if any CredData-derived artifact were
  ever published, each affected file's own license would have to be honoured
  individually, which is the main reason the lowest-risk arrangement avoids
  redistribution entirely.
- **Verdict**: usable for local evaluation as planned.

## 2. SDLog annotated data (mooselab)

- **URL**: <https://github.com/mooselab/SDLog>
- **Verified**: 2026-10-11
- **License as found**: GitHub reports the repository as MIT. The data
  directories (`annotated_data/`, `target_dataset/1-raw_datasets/`) carry no
  license statement of their own. Their contents are the LogHub `*_2k`
  datasets — the file names are literally `Android_2k.log_structured.txt`,
  `OpenStack_2k.log_structured.txt` and so on — so the underlying logs are
  LogHub's, and the repository's MIT grant plausibly covers the *annotations*
  and the code rather than the logs themselves.
- **Our use**: **local evaluation only**, as a source of log structure and of
  non-secret sensitive attributes (negatives). Never trained on, never
  committed, never redistributed.
- **Attribution duty**: cite SDLog, and cite LogHub for the underlying logs.
- **Verdict**: **inherits the LogHub question.** Treating SDLog data on the
  same footing as LogHub is what plan §5.1 already prescribes; whatever is
  decided for #3 applies here too.

## 3. LogHub (logpai)

- **URL**: <https://github.com/logpai/loghub>
- **Verified**: 2026-10-11
- **License as found**: no SPDX license — GitHub reports `NOASSERTION`
  ("Other"). The README's License section states: "The datasets are freely
  available for research or academic work." It additionally asks that any usage
  or distribution refer to the repository URL and cite the LogHub paper, and
  notes the logs are deliberately *not* sanitized or anonymized.
- **Our use as planned**: **local evaluation only** — real log templates and
  value distributions as an external test set. Never trained on, never
  committed, never redistributed.
- **Attribution duty**: cite the LogHub paper (Zhu, He, He, Liu, Lyu, *Loghub:
  A Large Collection of System Log Datasets for AI-driven Log Analytics*,
  ISSRE 2023) and link the repository.
- **Finding**: the grant is scoped to "research or academic work", and there is
  no license text that defines that term or grants anything broader.
  nebuRedaction is an open-source project, but it is produced by a company and
  feeds a commercial product line, so it is not obviously "research or academic
  work" — and the restriction, unlike CredData's, bites on *use*, not merely on
  redistribution. Publishing only aggregate metrics reduces the exposure but
  does not change what the data is being used for. A second consideration
  points the same way: LogHub states the logs are not sanitized, so they may
  contain real values that this project has no need to hold.
- **Verdict**: **recommend removing from evaluation** unless Frank is content
  with the "research or academic work" reading, or obtains written permission
  from the LogHub maintainers. Listed as a pending decision; this review does
  not decide it. If LogHub is removed, SDLog's data (#2) goes with it, and the
  external test set shrinks to CredData plus NebuLog's own logs — which weakens
  but does not invalidate G2, since CredData remains fully independent of the
  training distribution.

## 4. Laya model weights (convaiinnovations/laya)

- **URL**: <https://huggingface.co/convaiinnovations/laya>
- **Verified**: 2026-10-11
- **License as found**: the model card's metadata declares `license:
  apache-2.0`, and the card tags the model for commercial use. The checkpoint
  is distributed ungated.
- **Our use**: **teacher labeling.** Laya answers typed `choice` questions over
  uncertain samples. Per plan §5.1 item 5 it only ever sees synthetic samples
  and NebuLog's own logs — not third-party data. Its soft labels are distilled
  into the student models, so Laya's license has to permit that; Apache-2.0
  does, and it places no condition on model output.
- **Attribution duty**: retain the Apache-2.0 notice for anything derived from
  the weights, and name Laya as the teacher in the model card and the technical
  report. The weights themselves are not redistributed — they are downloaded
  into the Hugging Face cache and stay out of this repository.
- **Verdict**: usable as planned.

## 5. CredSweeper (Samsung)

- **URL**: <https://github.com/Samsung/CredSweeper>
- **Verified**: 2026-10-11
- **License as found**: MIT (GitHub reports `MIT`).
- **Our use**: run unmodified as an **evaluation control group**, and its ML
  review score is used as a second-opinion teacher signal (plan §4.4). Running
  a tool is not a license event under MIT; vendoring its code would be.
- **Attribution duty**: name CredSweeper and its version in the evaluation
  report. If any CredSweeper source is ever copied into this repository,
  reproduce the MIT notice alongside it.
- **Verdict**: usable as planned.

## 6. gitleaks and detect-secrets rule sets

- **URLs**: <https://github.com/gitleaks/gitleaks> (rules at
  `config/gitleaks.toml`), <https://github.com/Yelp/detect-secrets>
- **Verified**: 2026-10-11
- **Licenses as found**: gitleaks is MIT; detect-secrets is Apache-2.0. The
  generated `config/gitleaks.toml` carries no separate license header, so the
  repository license governs it.
- **Our use**: two distinct uses, and they carry different duties.
  1. **gitleaks as a tool** guarding this repository (pre-commit and CI). Using
     the published binary creates no obligation beyond the MIT notice that
     travels with it.
  2. **Rule porting** — reading these projects' patterns while writing
     nebuRedaction's own rule baseline (NR-0002). A regular expression for a
     documented vendor key format is largely dictated by that vendor's format,
     but a pattern copied verbatim is copied code.
- **Attribution duty**: where a rule is ported rather than independently
  derived, record the origin in the rule baseline and reproduce the
  corresponding notice — MIT's notice for gitleaks-derived patterns, and an
  Apache-2.0 attribution in `NOTICE` for detect-secrets-derived patterns
  (Apache-2.0 §4 requires retaining notices and marking modified files).
  NR-0002 should keep a per-rule provenance column so this stays auditable
  rather than being reconstructed later.
- **Verdict**: usable as planned, with per-rule provenance recorded.

## 7. StarPII (bigcode) — excluded

- **URL**: <https://huggingface.co/bigcode/starpii>
- **Verified**: 2026-10-11
- **License as found**: the model card metadata declares **no license**, and
  the repository is **gated** (Hugging Face reports `gated: auto`, so access
  requires accepting terms and requesting access).
- **Our use**: **excluded.** Two independent reasons, either sufficient:
  1. **Out of scope.** StarPII detects PII — names, emails, keys in source
     code. v0.1 covers secret credentials only and explicitly excludes PII
     (decision §12.1 item 2), so there is nothing for it to contribute.
  2. **License unclear and access gated.** A model with no declared license,
     behind an access gate, is exactly the kind of dependency this project
     should not take on for a published artifact.
- **Verdict**: excluded. Recorded here so the exclusion is a documented
  decision rather than an omission. If PII detection is ever brought into
  scope, the license and gate terms must be re-verified first.

---

## Pending decisions

1. **LogHub (and, following it, SDLog's data).** The "research or academic
   work" grant may not cover this project's evaluation use. Options: accept the
   reading and proceed; seek written permission from the LogHub maintainers; or
   drop both sources from the evaluation and rely on CredData plus NebuLog's
   own logs as the external test set. **Frank's decision** — see work order
   NR-0000 §7.

## Re-verification

Licenses change. Re-check every entry before each public release (NR-0006), and
re-check any entry whose source is upgraded to a new version. Record the new
verification date here rather than overwriting the old one.
