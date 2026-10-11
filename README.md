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
| `corpus/` | Acquisition scripts for third-party evaluation data (no data) |
| `labeling/` | Rule, teacher and human-spot-check labeling tools |
| `training/` | Training pipelines (ML.NET, PyTorch CPU) |
| `eval/` | Evaluation framework and baseline adapters |
| `dotnet/` | .NET inference library |
| `python/` | Python reference inference implementation |
| `tools/` | (v0.2) self-training command line tooling |
| `docs/` | Model card, data card, evaluation reports, licensing, user guide |

## Data policy

No dataset, model weight, credential or `.env` file is ever committed. Test
fixtures may only contain synthetic samples, and those are registered in a
synthetic-sample manifest that CI verifies. Third-party datasets are used for
local evaluation only: they never enter the training set and are never
redistributed.

## License

Apache-2.0. See [LICENSE](LICENSE).
