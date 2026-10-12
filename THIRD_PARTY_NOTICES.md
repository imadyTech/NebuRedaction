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
  `tools/check_synthetic_manifest.py`). Two of its rule patterns are also
  **ported** into the evaluation's rule baseline, `eval/baselines/rules.py`:
  `aws.access_key_id` and `azure.ad_client_secret`. Each carries
  `provenance = "gitleaks (MIT)"` in the source, and a test asserts that every
  ported rule names its origin. The remaining patterns in that file are derived
  independently from the vendors' own published formats — the same documents the
  synthetic generator's families cite — and are attributed to those documents
  rather than to gitleaks.
- **Attribution**: the MIT notice is reproduced below, as its licence requires
  for the ported patterns.

```
MIT License

Copyright (c) 2019 Zachary Rice

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### detect-secrets — Yelp

- **Source**: <https://github.com/Yelp/detect-secrets>
- **Licence**: Apache-2.0.
- **Our use**: its *approach* — a Shannon-entropy threshold over base64 and
  hexadecimal strings — is reimplemented in `eval/baselines/rules_entropy.py` as
  control group 2. No code or pattern is copied: the implementation normalises
  entropy by the alphabet's maximum and emits a continuous score, neither of
  which detect-secrets does. The module names detect-secrets as the source of
  the idea.
- **Attribution**: an idea is not copyrightable and Apache-2.0 §4 attaches to
  the work, so no notice is strictly owed for a reimplementation. The following
  notice is reproduced anyway, because the debt is real and the cost of
  acknowledging it is nil.

```
Copyright 2018 Yelp, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Should a detect-secrets pattern ever be copied verbatim, Apache-2.0 §4 requires
retaining its notices and marking the modified file; the rule baseline's
per-rule `provenance` field is where that would be recorded.

### CredSweeper — Samsung

- **Source**: <https://github.com/Samsung/CredSweeper>
- **Licence**: MIT.
- **Our use**: run unmodified as control group 3 in the evaluation
  (`eval/baselines/credsweeper_adapter.py`), installed from PyPI as a declared
  dependency, and its ML review score will be used as a second-opinion teacher
  signal in NR-0003. No code is vendored; if any ever is, the MIT notice travels
  with it.
- **Attribution**: CredSweeper and its exact version are recorded in every
  evaluation report that includes it.

---

## Build and runtime dependencies

.NET package versions are pinned in `Directory.Packages.props` and Python
dependencies in `uv.lock`; each package's licence is its own and is available
from its registry entry. The shipped `imady.NebuRedaction` library has **no
runtime dependencies**, so nothing in this section reaches a consumer of the
package.
