# eval

The evaluation framework, the control-group adapters and the rule baselines.

Plan §6.3 requires the evaluation to be finished before any model exists, so
that "how good is it" cannot be decided after seeing the results. Everything
here was written and fixed at gate G1, before a line of training code.

```bash
# Evaluate the default control groups and write a report.
uv run --group eval python -m eval.run --out runs/eval/<name>

# Pick systems explicitly; see --list for the registered names.
uv run --group eval python -m eval.run --systems rules rules+entropy --out runs/eval/<name>

# Refresh the test-set lock after the data is regenerated.
uv run --group eval python -m eval.run --write-lock
```

## The G2 comparison rule, fixed in advance

Gate G2 (plan §6.4) asks whether "rules + student model" beats "pure rules" by
**at least 10 percentage points of recall at a 0.1% false-positive rate**, with
the model at 100 KB or less.

That comparison has a problem the moment the pure-rule baseline's own
false-positive rate is above 0.1%: there is no operating point at 0.1% to
compare against, because a rule set does not have a dial. Deciding what to do
about that *after* seeing the numbers would be exactly the "look at the results,
then pick the metric" failure plan §6.3 exists to prevent. So the rule is fixed
here, now, before the student models exist.

**Let `FPR_r` be the pure-rule baseline's measured false-positive rate on the
synthetic test set.**

- **If `FPR_r` ≤ 0.1%** — compare at 0.1%. The student model's threshold is
  chosen on the synthetic validation split for a 0.1% false-positive rate, and
  the recall difference at that point decides G2.
- **If `FPR_r` > 0.1%** — report **both** points and judge on the first:
  1. **at 0.1%** (the gate): the student model is operated at 0.1%, and the
     rule baseline is reported at its own `FPR_r` with the gap in false-positive
     rate stated. The student model being held to a stricter false-positive
     budget than the baseline it must beat is the conservative direction, and
     that is deliberate.
  2. **at `FPR_r`** (for interpretation only): the student model is operated at
     the baseline's own false-positive rate, so the two are compared like for
     like. This number says whether the model is better at the same error
     budget, which is the fairer scientific question but the weaker engineering
     one.

**As measured at G1, `FPR_r` is well above 0.1%**, so the second branch applies.
The figure is in `docs/reports/baseline-v0.md`; the short version is that the
rule baseline's key-name matching fires on a large share of the hard negatives,
which is the behaviour the project exists to improve on.

## What the framework does, and the decisions inside it

**The main task is "redact or not"** — a unit is positive when its label is not
`none`. Per-class recall is reported as well, split into *was it flagged at all*
(what matters operationally: a token masked as an `api_key` is still masked) and
*was the class right* (what matters for the mask fingerprint and for diagnosing
confusions).

**Thresholds are chosen on the synthetic validation split and nowhere else.**
EXECUTION.md §8 forbids selecting a threshold on third-party data. The chosen
threshold is then applied unchanged to every test set, and the **measured**
false-positive rate is reported next to recall. When the measured rate on
CredData differs from the target the threshold was chosen for, that gap is a
finding about threshold transfer, not a defect in the report.

**Binary and continuous systems are not measured the same way.** A rule set has
one operating point; asking for "its recall at 0.1%" is meaningless, so its own
measured rate and recall are reported instead. The kind is declared by each
system rather than inferred, because conflating the two is the easiest way to
produce a flattering number by accident.

**PR-AUC is average precision**, with ties broken pessimistically (negatives
ranked ahead of positives at equal score). No interpolation choice is involved,
so two implementations agree, and a system that scores everything identically
cannot look good by luck.

**Confidence intervals are bootstrap**, 1,000 resamples, 95%, fixed seed
(NR-0002 §2.1 item 3).

**`report.json` contains no timestamps.** NR-0002 §2.1 item 4 requires two runs
on the same input to produce the same file, so anything that varies between runs
— when it ran, how long it took, which machine — goes in `run-meta.json`
instead. A test enforces that the report has no such field, and the two files
together lose nothing.

## Slices

A single headline number hides everything that matters, so every result is
broken down by:

| slice | the question it answers |
|---|---|
| `held_out_families` | Does it generalise to formats never seen in training (plan §5.3)? |
| `seen_families` | The comparison that makes the held-out number meaningful. |
| `ambiguous` | How does it do where the value genuinely cannot settle the label (plan §3.3)? |
| `adversarial_negatives` | Non-secrets under credential-suggesting names — where NR-0000's teacher scored a 0.972-confidence false positive. |
| `multiline_values` | PEM blocks and anything else spanning lines. |
| `key_recovery_*` | For CredData: 59% of units carry a whole line as their value and 7% have no recovered key name. Those are different input shapes from attribute mode. |
| `source_*` | Synthetic and third-party data measured apart. |

Robustness slices apply four deterministic transforms — base64 wrapping, case
flipping, affix noise and truncation — and report the recall change against the
untransformed set. Labels are preserved by construction: a base64-wrapped AWS
key is still an AWS key.

## Control groups

| name | control (plan §6.1) | status |
|---|---|---|
| `rules` | 1, pure rules | implemented: value patterns + whole-word key-name matching |
| `rules-values-only` | 1, variant | the same without key names, to separate what the names add from what they cost |
| `rules+entropy` | 2, rules + entropy | implemented, continuous |
| `credsweeper` | 3, CredSweeper | implemented, run unmodified |
| `laya` | 4, Laya zero-shot | placeholder; NR-0003 |
| — | 5, SDLog | **not present**: plan §5.1 item 0 excludes it from v0.1 entirely |
| `student-a/b/c` | 6, student models | placeholder; NR-0004 |

The rule layer in `baselines/rules.py` is also the rule half of NR-0004's
"rules + student model" combination, so its interface is kept narrow.

## A caveat that belongs on the rule baseline's numbers

The rule baseline's value patterns and the generator's positive families are
both derived from the same vendor documentation. On synthetic data the baseline
is therefore being tested against its own specification and scores higher than
it would in the wild.

That cuts in a useful direction for a gate — it makes G2 harder to pass, which
is the safe way for a go/no-go to be wrong — but it means **CredData is the only
honest read on rule-baseline quality**, since neither the rules nor the
generator were derived from it. Both numbers are reported, and the baseline
report says which to trust for what.
