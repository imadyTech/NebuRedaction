# Third-Party Notices

nebuRedaction's own code, synthetic dataset, published model weights and
documentation are MIT licensed. This file covers everything else the project
uses: where it came from, what its licence says, what we use it for, and what
we owe its authors.

No third-party dataset is distributed with this repository. The acquisition
scripts in `corpus/` fetch them to a local, git-ignored directory; the data
itself never enters the repository, a release, or a published dataset.

The reasoning behind each entry — the licence wording, the verification date and
why the use was judged acceptable — is in
[docs/licensing/R0-license-review.md](docs/licensing/R0-license-review.md). This
file states the conclusions; that file argues them.

Verified 2026-10-11, decisions updated 2026-10-12.

---

## Datasets

### CredData — Samsung

- **Source**: <https://github.com/Samsung/CredData>
- **Licence**: Apache-2.0 for the repository. Each data file additionally
  remains under its originating project's licence; CredData records those
  per-repository licences in its `license` directory, and its README states
  that "Each file is under the existing project's license."
- **Our use**: **local evaluation only.** CredData is the external test set: it
  takes no part in training, threshold selection or model choice. Its credential
  values are already obfuscated by its authors.
- **Not redistributed.** We publish the acquisition script
  (`corpus/fetch_creddata.py`), the category mapping
  (`corpus/mappings/creddata.yaml`) and aggregate metrics — never the data.
- **Attribution**: cite CredData in the evaluation report, the data card and the
  model card.

### LogHub — logpai, and SDLog — mooselab

- **Sources**: <https://github.com/logpai/loghub>,
  <https://github.com/mooselab/SDLog>
- **Licences**: LogHub declares no SPDX licence; its terms state the datasets
  are "freely available for research or academic work" and ask that use or
  distribution cite the LogHub paper. SDLog's repository is MIT, but its data
  directories carry no licence of their own and their contents are LogHub's
  `*_2k` datasets.
- **Our use in this version: none.** Neither source is downloaded or used, for
  training or for evaluation. The "research or academic work" grant does not
  clearly cover a company-produced project, and the restriction applies to use
  rather than only to redistribution, so v0.1 does without them.
- **Intended**: after this project is published, we will ask the LogHub and
  SDLog maintainers for permission. If it is granted, the sources may be added
  in a later version, and this file and the data card will be updated to say so.
- Should they be added, the attribution owed is a citation of the LogHub paper
  (Zhu, He, He, Liu, Lyu, *Loghub: A Large Collection of System Log Datasets for
  AI-driven Log Analytics*, ISSRE 2023), a link to the LogHub repository, and a
  citation of SDLog.

### StarPII — bigcode

- **Source**: <https://huggingface.co/bigcode/starpii>
- **Status**: **not used.** Its model card declares no licence and access is
  gated. It also detects PII, which v0.1 excludes from scope.

---

## Models

### Laya — convaiinnovations

- **Source**: <https://huggingface.co/convaiinnovations/laya>
- **Licence**: Apache-2.0, per the model card metadata.
- **Our use**: teacher labelling during dataset construction. Laya answers typed
  `choice` questions about uncertain samples, and only ever sees synthetic and
  own-log samples — never third-party data. Its soft labels are distilled into
  the student models; Apache-2.0 permits that and places no condition on model
  output.
- **Not redistributed.** The weights are downloaded to the Hugging Face cache
  and stay out of this repository and out of our releases.
- **Attribution**: Laya is named as the teacher in the model card and the
  technical report, and the Apache-2.0 notice is retained for anything derived
  from the weights.

---

## Tools and rule sets

### gitleaks

- **Source**: <https://github.com/gitleaks/gitleaks>
- **Licence**: MIT.
- **Our use**: run as a tool, to keep credentials out of this repository
  (pre-commit hook, CI full-history scan, and the synthetic-fixture audit in
  `tools/check_synthetic_manifest.py`). Its rule patterns are also read as
  reference while building nebuRedaction's own rule baseline.
- **Attribution**: where a rule is ported rather than independently derived from
  the vendor's published format, the origin is recorded against that rule and
  the MIT notice is reproduced. Rules derived from a vendor's own documentation
  are attributed to that documentation.

### detect-secrets — Yelp

- **Source**: <https://github.com/Yelp/detect-secrets>
- **Licence**: Apache-2.0.
- **Our use**: rule patterns read as reference for the rule baseline.
- **Attribution**: Apache-2.0 §4 requires retaining notices and marking
  modified files, so any ported pattern is attributed in a `NOTICE` entry and
  recorded in the rule baseline's provenance column.

### CredSweeper — Samsung

- **Source**: <https://github.com/Samsung/CredSweeper>
- **Licence**: MIT.
- **Our use**: run unmodified as a control group in the evaluation, and its ML
  review score is used as a second-opinion teacher signal. No code is vendored;
  if any ever is, the MIT notice travels with it.
- **Attribution**: CredSweeper and its version are named in the evaluation
  report.

---

## Build and runtime dependencies

.NET package versions are pinned in `Directory.Packages.props` and Python
dependencies in `uv.lock`; each package's licence is its own and is available
from its registry entry. The shipped `imady.NebuRedaction` library has **no
runtime dependencies**, so nothing in this section reaches a consumer of the
package.
