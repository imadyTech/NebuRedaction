# python

Python reference inference implementation, package `neburedaction`. Its
semantics are the ones the .NET library reproduces: plan §6.4's gate G3
requires the two to agree to within 1e-6 per sample, so where they could differ,
this side is the definition.

Standard library only.

Owner work order: NR-0005.

## Experimental

Not recommended for production. NR-0004 measured gate G2 as not met — the best
candidate reached 0.804 recall at a 0.1% false-positive rate against a bar of
0.834 — and its false-positive rate is several times its target on format
families it was not trained on. Treat it as defence in depth, never as the only
thing keeping credentials out of logs.

## The package ships no model

`SecretDetector.load` takes a path to a `.nrm` file you supply; no weights are
bundled and none are published yet. `tests/fixtures/fixture-model-v1.nrm` in
the repository is a 581-byte hand-built fixture, trained on nothing, which
exists so the reader can be tested — fine for checking that the library loads
and runs, and its predictions mean nothing.

## Example

```python
from neburedaction.detector import SecretDetector, SecretDetectorOptions, mask

detector = SecretDetector.load(
    "fixture-model-v1.nrm",
    SecretDetectorOptions(profile="balanced", never_redact_keys=frozenset({"request_id"})),
)

result = detector.classify(
    key="access_token",
    value="notarealvalue1",
    template="auth for {access_token}",
    scope="otel.attribute",
)
print(result.category, f"{result.score:.4f}", result.should_redact, result.reason)
print(mask("notarealvalue1") if result.should_redact else "kept")

# Body mode: the tokenizer picks the fragments worth classifying and drops
# prose, so a message of ordinary words yields nothing.
for fragment, body in detector.classify_message("session token=notarealvalue1 opened"):
    print(fragment, body.should_redact)
```

`profile="conservative"` is a ten-times-stricter threshold: it redacts less and
damages fewer logs, at lower recall. `always_redact_keys` and
`never_redact_keys` compare key names case-insensitively, and a name in both
lists is redacted — `never` loses to `always`.
