# dotnet

.NET inference library `imady.NebuRedaction` — `net10.0` + `netstandard2.1`,
zero runtime dependencies.

- `src/imady.NebuRedaction` — the library.
- `tests/imady.NebuRedaction.Tests` — xUnit tests.
- `benchmarks/imady.NebuRedaction.Benchmarks` — BenchmarkDotNet latency and
  allocation measurement.

Owner work order: NR-0005.

## Experimental

Not recommended for production. NR-0004 measured gate G2 as not met — the best
candidate reached 0.804 recall at a 0.1% false-positive rate against a bar of
0.834 — and its false-positive rate is several times its target on format
families it was not trained on. Treat it as defence in depth, never as the only
thing keeping credentials out of logs.

## The package ships no model

`SecretDetector` loads a `.nrm` file you supply; no weights are embedded and
none are published yet. `tests/fixtures/fixture-model-v1.nrm` is a 581-byte
hand-built fixture, trained on nothing, which exists so the reader can be
tested — it is fine for checking that the library loads and runs, and its
predictions mean nothing.

## Example

```csharp
using imady.NebuRedaction;

var detector = SecretDetector.Load(
    "fixture-model-v1.nrm",
    new SecretDetectorOptions
    {
        Profile = RedactionProfile.Balanced,
        NeverRedactKeys = ["RequestId"],
    });

DetectionResult result = detector.Classify(
    key: "access_token",
    value: "notarealvalue1",
    template: "auth for {access_token}",
    scope: "otel.attribute");

Console.WriteLine($"{result.Category} {result.Score:F4} redact={result.ShouldRedact}");
Console.WriteLine(result.ShouldRedact ? SecretDetector.Mask("notarealvalue1") : "kept");

// Body mode: the tokenizer picks the fragments worth classifying, and drops
// prose. `ClassifyMessage` returns one result per candidate.
foreach ((string fragment, DetectionResult body) in
    detector.ClassifyMessage("session token=notarealvalue1 opened"))
{
    Console.WriteLine($"{fragment}: {body.ShouldRedact}");
}
```

`RedactionProfile.Conservative` is a ten-times-stricter threshold: it redacts
less and damages fewer logs, at lower recall. `AlwaysRedactKeys` and
`NeverRedactKeys` compare key names case-insensitively, and a name in both
lists is redacted — `never` loses to `always`.
