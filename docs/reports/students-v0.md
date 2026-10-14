# Student models v0 — selection report

What a kilobyte-scale student model trained on NR-0003's labels can and cannot
do. Produced at gate G2 by `python -m eval.students`, under the comparison rule
written into `eval/README.md` before any model existed (plan §6.3).

Plan §6.4 made G2 a **recording gate** on 2026-10-14: measured and reported,
not go/no-go. This report is that record.

- **Feature definition**: `neburedaction.features` v1.0, 31 dense + 2^15 hashed
- **Models**: A — hashed features + multinomial logistic regression, L1, int8.
  C — character CNN reading the value's characters and no features at all.
- **Runs**: `runs/eval/students-v0/` (88.4 s), `runs/eval/students-v0-c/` (9.1 s)
- **Labels**: NR-0003's `train_v0.jsonl` (200,000) and `val_v0.jsonl` (22,692)

## The headline

**G2 is not met.** The bar is recall ≥ 0.834 at a 0.1% false-positive rate with
the model at ≤ 100 KB. The best candidate reaches **0.687** at that operating
point, and the model is 40.3 KB.

| system | recall @0.1% | size | meets G2 |
|---|---:|---:|---|
| **`rules+student-c`** | **0.751** | 40,092 B | no |
| `student-c` | 0.726 | 40,092 B | no |
| `rules+student-a` | 0.687 | 41,274 B | no |
| `student-a` | 0.654 | 41,274 B | no |
| `rules+entropy` | 0.611 | — | no |
| `rules` | n/a — binary, 0.734 at **11.17%** | — | — |
| `rules-values-only` | n/a — binary, 0.593 at **0.00%** | — | — |

The size budget was never the difficulty. Both models use about 40% of the
allowance, and plan §4.3 budgeted 16–64 KB for model A.

**Plan §4.3 expected model A to be the default and model C to be an
upper-bound reference. The measurement reverses that**, and the reason is the
subject of this report.

## Model C wins, and what that says

Model C reads the **characters of the value** — no key name, no template, no
scope, none of the 31 engineered features, no hashed n-grams. Model A reads all
of them. Measured on the synthetic test set at the 0.1% operating point:

| slice | A | C |
|---|---:|---:|
| overall recall | 0.654 | **0.726** |
| seen families, recall | 0.697 | **0.744** |
| **held-out families, recall** | 0.212 | **0.539** |
| held-out families, average precision | 0.436 | **0.580** |
| CredData recall | 0.077 | **0.172** |
| **CredData false-positive rate** | 22.9% | **6.7%** |
| CredData average precision | 0.214 | **0.329** |
| adversarial-negative FPR | 2.33% | 2.77% |
| latency, Python side | 292 µs/unit | **21 µs/unit** |

C is better everywhere that matters and fourteen times faster, at the same
size. The one place A is ahead is the false-positive rate on held-out families
(29.8% against 42.4%), and both are catastrophic there.

**The engineered features are not where the value is — they are where the
memorisation is.** 32,768 hashed character n-grams give a linear model an
efficient way to recognise *the generator's formats*, which is exactly what
inflates A's in-distribution score and exactly what fails to carry. Reading raw
characters forces C to learn something more like the shape of a credential, and
that generalises further. Plan §7.5 item 4 worried about n-gram features
memorising user credentials; the same mechanism turns out to limit the base
model's generalisation too.

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

| slice | units | A recall | A FPR | C recall | C FPR |
|---|---:|---:|---:|---:|---:|
| `seen_families` | 18,327 | 0.697 | **0.075%** | 0.744 | **0.084%** |
| `held_out_families` | 1,673 | 0.212 | **29.8%** | 0.539 | **42.4%** |

**On formats they have seen, both models are at their target false-positive
rate.** 0.075% and 0.084% are inside the 0.1% budget. Every bit of the overall
2.47% (A) and 3.49% (C) comes from the 1,673 held-out units, where a third to
nearly half of the negatives are flagged.

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

## Recommendation

**Model C, the character CNN**, against plan §4.3's expectation that A would be
the default. 39.2 KB, 10,023 parameters, three epochs, reading the value's
characters only. It beats A on recall at the operating point (0.726 against
0.654), on held-out families (0.539 against 0.212), on CredData in both recall
and false-positive rate, and it is fourteen times faster to score.

Ship it combined with the value-pattern rules — `rules+student-c` is the best
measured candidate at 0.751 — since the rules are hand-written format knowledge
that carries to formats no model has seen, which is precisely where both
students are weakest.

Two caveats attach to that recommendation:

1. **C has no calibration yet.** §2 item 3's presets were fitted for A. C's
   operating points here come from thresholds the framework chose on the
   selection set, which is sound for the measurement but not a shipped
   artefact.
2. **NR-0005 has to port whatever is chosen**, and a CNN is a harder port than
   a sparse int8 linear model. Plan §6.4's G3 requires the .NET and Python
   implementations to agree to 1e-6, which is a stronger constraint on a
   network of convolutions than on a dot product. If that cost is judged too
   high, A remains viable at a measured cost of 7 points of recall and much
   worse generalisation — but that is a trade to make deliberately, not by
   defaulting to the plan's original expectation.

For the operating preset, **conservative** (0.01% target) is the defensible
default despite the lower recall, because the failure mode this report
documents is a false-positive explosion on unfamiliar formats, and the
conservative preset degrades more gracefully when the distribution shifts.

## What this report cannot tell you

1. **Only one hyperparameter was fitted at full scale.** The grid was reduced
   to C ∈ {0.2, 0.5} and then stopped after C = 0.2, which took two hours.
   C = 0.5 is unexplored, and with 60% of the size budget unused a less
   sparse model might well do better.
2. **Model B is not in this comparison.** It is implemented and its size
   characterised — 70 trees per class × 15 leaves = 96.2 KB in a compact
   encoding — but it was not trained at full scale or evaluated. Given that C
   beat A by generalising rather than by capacity, B's position is genuinely
   open: it reads the dense features plus only 256 hashed buckets, so it has
   far less of the memorisation surface than A.
3. **Latency is a Python-side figure**: 292 µs per unit for A (feature
   extraction plus scoring) and 21 µs for C. Plan §6.4's G3 target applies to
   the .NET runtime, which NR-0005 builds.

4. **Model C was trained once, with no hyperparameter search at all** — three
   epochs at one width. A searched two penalties and completed one. Neither
   model has been tuned, and C's advantage was measured against an A that may
   itself be undertrained.
5. **No robustness transforms were run** for the students, so the base64 and
   case-flip sensitivity NR-0002 measured for the baselines is unknown here.

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
```
