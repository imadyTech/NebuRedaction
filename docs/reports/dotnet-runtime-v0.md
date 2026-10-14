# .NET runtime v0 — consistency and performance

NR-0005 §2.4 and §2.5. What the .NET library agrees with the Python reference
on, and what it costs per call.

**Experimental.** Plan §1.2 makes this release about being correct and
runnable, not about being good. Nothing here measures whether the model is any
good — NR-0004 measured that, found gate G2 not met, and
[`students-v0.md`](students-v0.md) says so. This report measures whether the
port reproduces the reference and how fast it runs.

## G3 — consistency

**Met.** Plan §6.4 requires the two implementations to agree to within 1e-6
absolute on every class probability, with identical redaction decisions. They
do, on every sample of the set §2.4 item 1 specifies.

| | |
|---|---|
| Model | `models/neburedaction-v0.1.0.nrm`, 94.9 KB, model B (LightGBM) |
| Samples | 2,400 — 1,000 synthetic test, 1,000 own logs, 200 adversarial negatives, 200 robustness-transformed |
| Seed | 20261015, fixed in `tools/generate_consistency_set.py` |
| Probability comparisons | 16,800 (7 classes × 2,400) |
| Worst absolute difference | **6.661e-16**, at `synthetic/v0-test-0015003`, class `none` |
| Decisions | identical under both the balanced and the conservative profile |
| Categories | identical |

The margin is worth stating plainly: 6.661e-16 is double-precision rounding —
about 3 ULP at a probability near 1 — roughly **ten orders of magnitude** inside
the 1e-6 tolerance. The two implementations are not merely close enough; they
are doing the same arithmetic in the same order on the same bytes. That is what
the `.nrm` spec's two requirements buy, accumulating in 64-bit and summing
trees in file order, and it means the tolerance is nowhere near load-bearing:
a real divergence would be a feature or an operation differing, which shows up
thousands of times larger, not a rounding difference creeping past 1e-6.

### The gate was tested, not just passed

It passed on the first run, which is a weak signal on its own: a comparison
that silently compares nothing also passes on the first run. So the set was
perturbed — one probability moved by 1e-5, ten times the tolerance, and one
decision flipped — and the gate failed, naming the section, the sample id, the
class, both values and the delta. Then the real set was regenerated and it
passed again.

### What the set does not cover

The 1,000 own-log units are **unlabelled** here and used only for agreement.
Agreement is not accuracy: the two implementations agree on a wrong answer
exactly as readily as on a right one. That is all G3 claims and all this
section reports.

## Performance

§2.5 asks for p50 and p99 latency and bytes allocated per `Classify` call,
against the plan's p99 ≤ 20 µs target. It is a report, not a gate.

| machine | Intel Core Ultra 5 245K @ 4.20 GHz, 14 cores, Windows 11 (10.0.26200.9711) |
|---|---|
| runtime | .NET 10.0.12, X64 RyuJIT x86-64-v3 |
| harness | BenchmarkDotNet 0.15.4, Release |
| date | 2026-10-15 |

| input | mean | P50 | P90 | P95 | allocated |
|---|---:|---:|---:|---:|---:|
| 2-character value | 4.22 µs | 4.22 µs | 4.26 µs | 4.28 µs | 5,608 B |
| 32-character hex | 6.06 µs | 6.06 µs | 6.11 µs | 6.12 µs | 6,584 B |
| 512-character value | 18.5 µs | 18.5 µs | 18.7 µs | 18.8 µs | 5,504 B |
| `Mask` | 129 ns | 129 ns | 130 ns | 130 ns | 192 B |

### P99 is not reported, and P95 is not it

BenchmarkDotNet's `StatisticColumn` offers percentiles up to P95 and no P99. A
P99 computed from the iteration count a 4-microsecond method needs would mostly
be an estimate of its own noise, so this report gives P95 as measured and does
not relabel it.

**Against the 20 µs target:** the two shorter inputs are comfortably inside it
at every percentile measured. The 512-character case at 18.8 µs P95 is inside
the target too, but close enough to it that its p99 may well sit above 20 µs.
Stating that it meets a p99 target would be claiming something not measured.

### Allocation: §2.2 item 5 is improved, not met

§2.2 item 5 asks for a near-zero-allocation hot path. 5.5 KB a call is not
near-zero.

It was far worse. The first measurement put the 512-character case at 70.7 µs
and 202,568 B — 3.5× the latency target. The trees were not the problem.
`Features.Hashed` built a substring, a namespaced string and a UTF-8 byte array
for each of 2,042 n-grams in order to fill a 32,768-bucket map, and `Project`
then read the 256 buckets the model actually has.

`Features.AccumulateHashed` now hashes each n-gram in place — FNV-1a is a
streaming hash, so the namespace and separator fold into a seed once per call,
and an n-gram's UTF-8 bytes go into a stack buffer — and adds it straight into
the projected vector, skipping buckets no tree reads:

| input | before | after | |
|---|---:|---:|---|
| 512 chars | 70.7 µs / 202,568 B | 18.5 µs / 5,504 B | 3.8× faster, 37× less |
| 32 chars | 9.9 µs / 28,520 B | 6.1 µs / 6,584 B | 1.6× faster, 4.3× less |
| 2 chars | 4.7 µs / 6,968 B | 4.2 µs / 5,608 B | 1.1× faster, 1.2× less |

Dropping the unused buckets is **exact, not an approximation**, and only
because the hashed features are not normalised: each bucket's signed count is
independent of the others, so a bucket nothing reads cannot change one that is.
A later feature version that normalises them invalidates the shortcut. A test
runs both paths over all 27 committed feature cases and compares every column
of a full 32,768-bucket vector in both directions, and G3's 16,800 comparisons
still hold.

What remains is the projected vector itself (287 doubles), the dense feature
array, the word lists the key and template splitters return, and the result's
probability dictionary. Removing them means pooling buffers and returning a
struct instead of a class — a different API, and out of scope here.

## Cross-platform

§2.6 item 1 asks CI to run the consistency tests on Windows and Linux. **It
cannot run G3 there**, and this is a consequence of a decision already taken:
§7 decided on 2026-10-14 not to commit the trained model, and `.gitignore`
excludes `*.nrm` and `data/`. A CI runner has neither a model nor a corpus.

Left at that, CI would report a clean pass having exercised the binary reader,
the tree walk and the Platt calibration — the parts most likely to differ
between platforms — on neither of them. So
`tests/fixtures/fixture-model-v1.nrm` is committed under an explicit
`.gitignore` exception: 581 bytes, four hand-specified trees over two classes,
trained on nothing, containing no credential and nothing derived from one. Both
implementations read it and are checked against the same recorded outputs.

**That is a reader-parity check, not G3.** It proves the two read the same
bytes and do the same arithmetic. Agreement on 2,400 real samples stays a local
gate for as long as the model stays out of the repository.

## Packaging

Both packages were built locally and verified in fresh directories (§2.7; not
pushed anywhere).

| | |
|---|---|
| `imady.NebuRedaction` | 0.1.0-preview.1, `net10.0` + `netstandard2.1`, **zero dependencies** in both target groups |
| `neburedaction` | 0.1.0, wheel and sdist, **zero dependencies** |
| Verification | a fresh console project and a fresh venv, each installing only the built package, each running its README example |

Both examples print the same three lines — `token 0.7432`, `***#be37`,
`notarealvalue1 True` — which is the two implementations agreeing from the
packaged artifacts rather than from the working tree.

Both descriptions say "experimental" and state that G2 is not met and that no
model weights are bundled, because a package description is the one piece of
documentation someone reads before depending on it.
