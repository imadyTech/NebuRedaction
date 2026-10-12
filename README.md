# nebuRedaction

A small, fast, offline classifier that decides whether a single log attribute
is a secret credential — and masks it before it is written. It is built for
structured logging pipelines: given a `(key, value)` pair plus optional
message-template and scope context, it returns per-class probabilities and a
redaction decision, and masks matched values as `***#<fingerprint>` so the
same value can still be correlated without ever being exposed.

**Status: pre-release, under construction.** Nothing here is published yet and
the public API, the model format and the evaluation numbers may all change.

The shipped artifacts will be a .NET inference library
(`imady.NebuRedaction`, `net10.0` + `netstandard2.1`, zero runtime
dependencies), a Python reference implementation (`neburedaction`), the model
weights and the synthetic training corpus.

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
only. Of the third-party datasets, **CredData is used for local evaluation
only** — it never enters training, threshold selection or model choice — and
**LogHub and SDLog are not used at all in this version**. No third-party data is
downloaded into the repository or redistributed; `corpus/` holds the acquisition
scripts and processing recipes, never the data. Details and the licence wording
each decision follows from are in
[docs/licensing/R0-license-review.md](docs/licensing/R0-license-review.md).

## License

The code, the synthetic dataset, the published model weights and the
documentation are **MIT** licensed. See [LICENSE](LICENSE).

Third-party data is not distributed with this repository. The sources it uses,
their licences, what each is used for and the attribution owed are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
