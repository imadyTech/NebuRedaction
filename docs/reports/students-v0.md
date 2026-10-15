# Student models v0 — selection report

What a kilobyte-scale student model trained on NR-0003's labels can and cannot
do. Produced at gate G2 by `python -m eval.students`, under the comparison rule
written into `eval/README.md` before any model existed (plan §6.3).

Plan §6.4 made G2 a **recording gate** on 2026-10-14: measured and reported,
not go/no-go. This report is that record.

- **Feature definition**: `neburedaction.features` v1.0, 31 dense + 2^15 hashed
- **Models**: A — hashed features + multinomial logistic regression, L1, int8.
  B — LightGBM over the dense features plus 256 hashed buckets.
  C — character CNN reading the value's characters and no features at all.
- **Runs**: `runs/eval/students-v0/` (88.4 s), `-b/` (84.9 s), `-c/` (9.1 s)
- **Labels**: NR-0003's `train_v0.jsonl` (200,000) and `val_v0.jsonl` (22,692)

## The headline

**G2 is not met.** The bar is recall ≥ 0.834 at a 0.1% false-positive rate with
the model at ≤ 100 KB. The best candidate — model B combined with the
value-pattern rules — reaches **0.804** at that operating point, three points
short, at 89.4 KB.

| system | recall @0.1% | size | meets G2 |
|---|---:|---:|---|
| **`rules+student-b`** | **0.804** | 91,512 B | no |
| `student-b` | 0.783 | 91,512 B | no |
| `rules+student-c` | 0.751 | 40,092 B | no |
| `student-c` | 0.726 | 40,092 B | no |
| `rules+student-a` | 0.687 | 41,274 B | no |
| `student-a` | 0.654 | 41,274 B | no |
| `rules+entropy` | 0.611 | — | no |
| `rules` | n/a — binary, 0.734 at **11.17%** | — | — |
| `rules-values-only` | n/a — binary, 0.593 at **0.00%** | — | — |

The best candidate falls 3 points short. The size budget was never the
difficulty: A and C use about 40% of the allowance and B 87%.

**Plan §4.3 expected model A to be the default and model C to be an
upper-bound reference. The measurement reverses that**, and the reason is the
subject of this report.

## Model C wins, and what that says

Model C reads the **characters of the value** — no key name, no template, no
scope, none of the 31 engineered features, no hashed n-grams. Model A reads all
of them. Measured on the synthetic test set at the 0.1% operating point:

All three, combined with the value-pattern rules, on the synthetic test set at
the 0.1% operating point:

| | A | B | C |
|---|---:|---:|---:|
| **recall** | 0.687 | **0.804** | 0.751 |
| measured FPR | **2.47%** | 4.02% | 3.49% |
| seen families, recall | 0.702 | **0.831** | 0.749 |
| seen families, FPR | **0.075%** | 0.084% | 0.084% |
| **held-out families, recall** | 0.532 | 0.526 | **0.773** |
| held-out families, FPR | **29.8%** | 48.9% | 42.4% |
| held-out families, AP | 0.778 | 0.737 | **0.812** |
| CredData recall | 0.172 | **0.261** | 0.183 |
| **CredData FPR** | 22.9% | **6.1%** | 7.7% |
| CredData AP | 0.282 | **0.477** | 0.288 |
| adversarial-negative FPR | **2.33%** | 3.74% | 2.77% |
| size | 40.3 KB | 89.4 KB | **39.2 KB** |
| latency, Python side | 292 µs | 388 µs | **21 µs** |

**B wins where the distribution is familiar; C wins where it is not.** B is
best on overall recall, on seen families and on CredData — the last by a wide
margin in both recall and false positives. C is best on held-out format
families, the slice built to measure generalisation to formats never trained
on, and it is the smallest and by far the fastest.

A is beaten by both on recall and is worst on CredData by a factor of four in
false-positive rate. Its one advantage is the lowest false-positive rate on
held-out families and on adversarial negatives — it is the most cautious of the
three, and the least capable.

**The engineered features are not where the value is — they are where the
memorisation is.** 32,768 hashed character n-grams give a linear model an
efficient way to recognise *the generator's formats*, which is exactly what
inflates A's in-distribution ranking and exactly what fails to carry. Reading
raw characters forces C to learn something more like the shape of a credential,
and that generalises further. Plan §7.5 item 4 worried about n-gram features
memorising user credentials; the same mechanism turns out to limit the base
model's generalisation too.

B is the evidence that this is about the feature *surface* rather than about
linear models. B is a tree ensemble — more capacity than A, not less — but it
reads the 31 dense features plus only **256** hashed buckets instead of 32,768.
With two orders of magnitude less memorisation surface it beats A everywhere,
and its CredData false-positive rate is a quarter of A's. The hashed n-gram
space, not the model class, is what A was overfitting.

This is the control doing its job. Had only A been built, the project would
have concluded that a kilobyte-scale student is a sound idea and that the
feature pipeline NR-0005 must port is load-bearing. Both halves of that are
questionable on this evidence.

## What the models are actually good at

Judged without a threshold, the student is the **best system this project has
produced** on the data it was trained on. Average precision measures ranking
quality and does not depend on where a threshold is put:

| system | synth_test | held-out families | CredData |
|---|---:|---:|---:|
| `rules+student-a` | **0.972** | 0.778 | 0.282 |
| `rules+student-c` | 0.961 | 0.812 | 0.288 |
| `student-c` | 0.951 | 0.580 | 0.329 |
| `student-a` | 0.939 | 0.436 | 0.214 |
| `rules+entropy` | 0.916 | **0.961** | **0.709** |
| `rules-values-only` | 0.741 | 0.685 | 0.189 |
| `rules` | 0.566 | 0.581 | 0.341 |

On the synthetic test set it beats every NR-0002 baseline, including
`rules+entropy`, which was the strongest. On the slice it was trained for it is
very good.

And on **real logs it is excellent**. NebuLog's own 2,692 production-shaped
units, at both threshold presets:

| system | false positives on real logs |
|---|---:|
| **student-a** | **0 / 2,692 = 0.00%** |
| rule layer | 0 / 2,692 = 0.00% |
| CredSweeper | 0 / 2,692 = 0.00% |
| Laya teacher (NR-0003) | 50 / 2,692 = **1.86%** |
| plan §4.5 target | 0.10% |

This is the in-distribution calibration reading NR-0002 finding 5 asked NR-0003
to make possible, and the student passes it outright.

## Three failures, and they are the same failure

### 1. The threshold does not transfer

The threshold was chosen on the validation split for a 0.1% false-positive
rate. On the synthetic **test** split it measures **2.47%** — twenty-five times
its target.

NR-0002 finding 5 predicted exactly this for the student, having measured it
for `rules+entropy`: a threshold chosen for 0.1% on synthetic validation data
came out at 1.223% on CredData. The prediction was recorded before any model
existed and it held.

### 2. Held-out format families collapse

Ten format families were excluded from training so that generalisation to
unseen formats could be measured (plan §5.3). Split by that slice, at the same
threshold:

Student alone, without the rules:

| slice | units | A recall | A FPR | B recall | B FPR | C recall | C FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seen_families` | 18,327 | 0.697 | **0.075%** | 0.830 | 0.084% | 0.744 | 0.084% |
| `held_out_families` | 1,673 | 0.212 | **29.8%** | 0.306 | 48.9% | 0.539 | 42.4% |

**On formats they have seen, all three models are at their target
false-positive rate.** 0.075% and 0.084% are inside the 0.1% budget, with
recall from 0.697 to 0.830. Every bit of the overall excess — 2.47% for A,
4.02% for B, 3.49% for C — comes from the 1,673 held-out units, where a third
to half of the negatives are flagged. The models are not badly calibrated; they
are being asked about formats they have never seen.

Average precision tells the same story from the other side: the student ranks
held-out families at 0.436, the **worst of all five systems**, while
`rules+entropy` ranks them at 0.961, the best. Entropy is format-agnostic; a
hashed n-gram model is not.

### 3. CredData is worse still

| system | recall | measured FPR | AP |
|---|---:|---:|---:|
| `rules+student-a` | 0.172 | 23.3% | 0.282 |
| `student-a` | 0.077 | 22.9% | 0.214 |
| `rules+entropy` (NR-0002) | 0.162 | 1.22% | 0.709 |
| `rules` (NR-0002) | 0.503 | 5.96% | 0.341 |
| `credsweeper` (NR-0002) | 0.807 | 1.21% | 0.723 |

On real-world code the student is beaten by everything, including the rule
baseline it was built to improve on.

### The common cause

These are not three problems. **The student learned the generator's formats,
not the concept of a credential.** It is excellent inside the distribution —
0.972 average precision, 0.075% false positives on seen families, 0.00% on real
NebuLog logs — and poor the moment the format is one it has not seen, whether
that is a held-out synthetic family or a real repository.

This is precisely what the held-out families were constructed to detect, and
NR-0002's report said so in advance: "our value patterns and the generator's
positive families are derived from the same vendor documentation, so on
synthetic data the baseline is being tested against its own specification."
The same critique now applies to the student, and the measurement confirms it.

## What the rules contribute

The union with the value patterns is worth having, and more where it matters:

| slice | student alone | rules + student | gain |
|---|---:|---:|---:|
| synth_test overall | 0.654 | 0.687 | +0.033 |
| held-out families | 0.212 | **0.532** | **+0.320** |
| CredData | 0.077 | 0.172 | +0.095 |

The value patterns are format knowledge written by hand rather than learned, so
they carry to formats the model never saw. Where the student is weakest the
rules help most — which is an argument for the hybrid rather than for either
half.

The rule half is **value patterns only**. NR-0002 measured the full rule
baseline at 11.17% false positives, every one traced to a key name or a
template, while `rules-values-only` produced zero. A union with the full set
inherits its rate, so G2's own 0.1% operating point would be unreachable by
construction.

## The adversarial slice

The 2,784 adversarial negatives — non-secrets under credential-suggesting names
and templates — are where this project began, and the student does well:

| system | false-positive rate on adversarial negatives |
|---|---:|
| **student-a** | **2.33%** |
| `rules+entropy` @0.1% (NR-0002) | 0.25% |
| `rules-values-only` (NR-0002) | 0.00% |
| `rules` (NR-0002) | 41.67% |
| Laya teacher (NR-0003) | **81.45%** |

Against the teacher that supplied its soft labels, the student is thirty-five
times better on exactly the cases the teacher was worst at. The distillation
did not inherit the teacher's defining failure.

## Recommendation and the porting choice

**NR-0004 §11.2 fixed the rule before these numbers existed**: the model ported
in NR-0005 is A or B, chosen by `rules+student-*` recall at a 0.1%
false-positive rate on the synthetic test set, with the smaller model taking a
near-tie inside one point.

**B wins: 0.804 against A's 0.687**, eleven points clear of a one-point
threshold. It is exported to `runs/NR-0004/export/` with its calibration, both
threshold presets, the feature version and the class order.

B is also the right answer on the merits, not only by the rule. It is the best
of the three on the operating point, on seen families and on CredData — the
last being the only genuinely external data the project has, where its
false-positive rate of 6.1% is a quarter of A's 22.9% and its average precision
of 0.477 is more than double.

**C is not ported in v0.1**, by §11.2, and the reason is engineering risk
rather than merit: plan §6.4's G3 requires the .NET and Python implementations
to agree to within 1e-6 per sample, which is a far stronger constraint on a
stack of convolutions than on a tree walk, and C has had no hyperparameter
search and no calibration of its own. **That is a deliberate deferral of the
best generalising model, and it should be read as one.** C beats B on held-out
families by 0.773 against 0.526 and is a fifth the size and a twentieth the
latency. It is the main finding for the technical report and the first item for
future work (NR-0008).

For the operating preset, **conservative** (0.01% target) is the defensible
default despite the lower recall, because the failure mode this report
documents is a false-positive explosion on unfamiliar formats, and the
conservative preset degrades more gracefully when the distribution shifts.

### A note on model B's grid

B's recall at the operating point was wildly non-monotonic in the tree budget:

| trees × leaves | size | recall @0.1% |
|---|---:|---:|
| 40 × 15 | 55.2 KB | 0.039 |
| 70 × 15 | 96.2 KB | 0.103 |
| 40 × 31 | 116.4 KB (over budget) | 0.846 |
| **100 × 10** | **89.4 KB** | **0.818** |

Forty trees of fifteen leaves scores 0.039 while a hundred trees of ten leaves
scores 0.818, with *less* capacity per tree. This is the tie effect again: an
under-boosted ensemble leaves many units sharing a leaf value, so their scores
are identical, and a threshold cannot separate them. Under the pessimistic
tie-breaking a false-positive budget requires, recall collapses. The number is
not measuring the model's ability so much as how many distinct scores it
produces, and anyone reading a boosted model's recall at a strict operating
point should check that first.

## What this report cannot tell you

1. **Only one hyperparameter was fitted at full scale.** The grid was reduced
   to C ∈ {0.2, 0.5} and then stopped after C = 0.2, which took two hours.
   C = 0.5 is unexplored, and with 60% of the size budget unused a less
   sparse model might well do better.
2. **B's grid was four points and C's was one.** B searched four tree budgets
   and the best in-budget fit was at the edge of the grid's trees axis, so a
   larger ensemble of small trees is unexplored. C had no search at all.
3. **Latency is a Python-side figure**: 292 µs per unit for A, 388 µs for B
   and 21 µs for C. Plan §6.4's G3 target applies to the .NET runtime, which
   NR-0005 builds.

4. **No model here is tuned.** C was trained once, three epochs at one width.
   A searched two penalties and completed one before the second was stopped.
   The comparison is between three barely-tuned models, and the gaps between
   them are large enough to survive that, but the absolute numbers are not a
   ceiling.
5. **No robustness transforms were run** for the students, so the base64 and
   case-flip sensitivity NR-0002 measured for the baselines is unknown here.

6. **A full candidate comparison was not attempted.** Separating key-name and
   value models and fusing them, a character language model's surprisal, a
   one-class model, an MLP and a GRU are all untried; §11 scoped them out for
   time and they belong in NR-0008.

## Reproducing

```bash
uv run --group train --group datagen python -m training.train_a --grid 0.2
uv run --group train --group datagen python -m training.fit_student_calibration \
    --model-dir runs/NR-0004/<ts>/C-0.2
uv run --group train --group datagen --group eval python -m eval.students \
    --model-dir runs/NR-0004/<ts>/C-0.2 --with-baselines

uv run --group train --group torch --group datagen python -m training.train_c --epochs 3
uv run --group train --group torch --group datagen --group eval python -m eval.students \
    --model-dir runs/NR-0004/<ts> --variant c --out runs/eval/students-v0-c

uv run --group train --group datagen python -m training.train_b
uv run --group train --group datagen --group eval python -m eval.students \
    --model-dir runs/NR-0004/<ts> --variant b --out runs/eval/students-v0-b
uv run --group train --group datagen python -m training.export_for_runtime \
    --model-dir runs/NR-0004/<ts> --variant b
```
