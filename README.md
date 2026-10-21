# nebuRedaction

A small, fast, offline classifier that decides whether a single log attribute
is a secret credential — and masks it before it is written. It is built for
structured logging pipelines: given a `(key, value)` pair plus optional
message-template and scope context, it returns per-class probabilities and a
redaction decision, and masks matched values as `***#<fingerprint>` so the
same value can still be correlated without ever being exposed.

## Experimental — not recommended for production

**The accuracy gate was not met.** The best candidate reaches 0.804 recall at a
0.1% false-positive rate against a bar of 0.834, and — the more important
result — it learned *the formats its generator produces* rather than the
concept of a credential: 0.075% false positives on format families it trained
on, **29.8% on families held out**, and on real third-party code it is beaten
by the rule baseline it was built to improve on.

Use it as defence in depth, never as the only thing keeping credentials out of
logs. The failures are reported at length, with the measurements, in the
[technical report](docs/reports/v0.1-technical-report.md) — read §6 before
depending on any of this.

The public API, the model format and the evaluation numbers may all change.

## Minimal examples

Both load the model embedded in the package and classify one attribute. The
.NET library targets `net10.0` and `netstandard2.1` and has **zero runtime
dependencies**; the Python package is standard library only.

```csharp
using imady.NebuRedaction;

// Default uses the model embedded in the package, on the balanced profile.
DetectionResult result = SecretDetector.Default.Classify(
    key: "access_token",
    value: "notarealvalue1",
    template: "auth for {access_token}",
    scope: "otel.attribute");

Console.WriteLine($"{result.Category} {result.Score:F4} redact={result.ShouldRedact}");
if (result.ShouldRedact)
{
    Console.WriteLine(SecretDetector.Mask("notarealvalue1"));  // ***#<fingerprint>
}
```

```python
from neburedaction.detector import SecretDetector, SecretDetectorOptions, mask

# The Python package bundles no weights: pass a path to a .nrm file.
detector = SecretDetector.load("neburedaction-v0.1.0.nrm", SecretDetectorOptions())
result = detector.classify(
    key="access_token",
    value="notarealvalue1",
    template="auth for {access_token}",
    scope="otel.attribute",
)
print(result.category, f"{result.score:.4f}", result.should_redact, result.reason)
if result.should_redact:
    print(mask("notarealvalue1"))
```

`RedactionProfile.Conservative` (`profile="conservative"`) is a ten-times
stricter threshold: it redacts less and damages fewer logs, at lower recall.
`AlwaysRedactKeys` and `NeverRedactKeys` compare key names case-insensitively,
and a name in both lists is redacted — `never` loses to `always`.

## Non-goals for v0.1

- **No general PII detection.** v0.1 covers secret credentials only — no
  names, addresses, phone numbers or payment-card data.
- **No sequence labeling.** The model classifies candidate units; splitting a
  message body into candidates is done by a deterministic tokenizer, not by
  the model.
- **No large language model at inference time.** A teacher LLM is used while
  building the training set; what ships is a kilobyte-scale student model.
- **No guarantee of completeness.** This is a probabilistic classifier and a
  defence in depth, not a substitute for keeping credentials out of logs.
- **No cloud service.** Everything runs in-process and offline.

## Repository layout

| Path | Contents |
|---|---|
| `datagen/` | Synthetic data generator |
| `corpus/` | Acquisition scripts for third-party datasets (no data) |
| `labeling/` | Rule, teacher and human-spot-check labeling tools |
| `training/` | Training pipelines (ML.NET, PyTorch CPU) |
| `eval/` | Evaluation framework and baseline adapters |
| `dotnet/` | .NET inference library |
| `python/` | Python reference inference implementation |
| `tools/` | (v0.2) self-training command line tooling |
| `docs/` | Model card, data card, evaluation reports, licensing, user guide |

## Data policy

No dataset, model weight, credential or `.env` file is ever committed, and no
credential-shaped test fixture either: the generator emits format-correct vendor
credentials, so committing them would be indistinguishable from a leak to any
scanner that matches vendor patterns. Fixtures are generated on demand instead —
generation is deterministic from the seed, so tests build them in a temporary
directory.

The guard is still there and still tested. Anything that does appear under
`tests/fixtures/synthetic/` must have its SHA-256 registered in a
synthetic-sample manifest, which the pre-commit hook and CI both verify, and an
end-to-end test proves the guard rejects a credential-shaped string the
generator never produced.

The released models are trained on synthetic data and nebuLog's own demo logs
only. Of the third-party datasets, **CredData (Samsung, n.d.-a) is used for local evaluation
only** — it never enters training, threshold selection or model choice — and
**LogHub (Zhu et al., n.d.) and SDLog (Mooselab, n.d.) are not used at all in this version**. No third-party data is
downloaded into the repository or redistributed; `corpus/` holds the acquisition
scripts and processing recipes, never the data. Details and the licence wording
each decision follows from are in
[docs/licensing/R0-license-review.md](docs/licensing/R0-license-review.md).

## Vendor names

**Vendor names.** Vendor names appearing in this document are used only to identify the credential formats they denote, as descriptive references to publicly published material; **this project is not affiliated with, sponsored by or endorsed by any of them.** Sources are cited in the reference list. See the *Trademarks* section of [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The code, the synthetic dataset, the published model weights and the
documentation are **MIT** licensed. See [LICENSE](LICENSE).

Third-party data is not distributed with this repository. The sources it uses,
their licences, what each is used for and the attribution owed are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## References

Sources named above. Bibliographic details were checked against each source on
2026-10-16, as EXECUTION.md §9 item 3 requires. Software and dataset entries give the version
this project pins where one is pinned, and a retrieval date where the source is
undated and subject to change, as APA 7th edition directs. Where a work carries
no publication date, `n.d.` appears rather than a guess.

Apache Software Foundation. (2004). *Apache license, version 2.0*.
http://www.apache.org/licenses/LICENSE-2.0

BigCode. (n.d.). *StarPII* [Model]. Hugging Face. Retrieved October 16, 2026,
from https://huggingface.co/bigcode/starpii

Convai Innovations. (n.d.). *Laya* [Model]. Hugging Face. Retrieved
October 16, 2026, from https://huggingface.co/convaiinnovations/laya

Mooselab. (n.d.). *SDLog* [Computer software]. GitHub. Retrieved October 16,
2026, from https://github.com/mooselab/SDLog

Rice, Z. (n.d.). *gitleaks* (Version 8.30.1) [Computer software]. GitHub.
Retrieved October 16, 2026, from https://github.com/gitleaks/gitleaks

Samsung. (n.d.-a). *CredData* [Data set]. GitHub. Retrieved October 16, 2026,
from https://github.com/Samsung/CredData

Samsung. (n.d.-b). *CredSweeper* (Version 1.19.2) [Computer software]. GitHub.
Retrieved October 16, 2026, from https://github.com/Samsung/CredSweeper

Yelp. (n.d.). *detect-secrets* [Computer software]. GitHub. Retrieved
October 16, 2026, from https://github.com/Yelp/detect-secrets

Zhu, J., He, S., Liu, J., He, P., Xie, Q., Zheng, Z., & Lyu, M. R. (n.d.).
*LogHub* [Data set]. GitHub. Retrieved October 16, 2026, from
https://github.com/logpai/loghub
