# labeling

Rule labelling, teacher labelling (Laya, CredSweeper ML) and the human
spot-check tooling.

- `rules/` — the rule labeller with a confidence tier (NR-0003 §2.1).
- `teacher/` — the Laya teacher, its versioned prompt and the overnight runner
  (NR-0003 §2.2, §2.2A).
- `review/` — the manual review tool and the review queue (NR-0003 §2.4).
- `probe/` — R0 feasibility probe for the Laya teacher model (NR-0000 §2.7).

Owner work orders: NR-0000 (probe), NR-0003 (labelling pipeline).

```bash
# 1. Rule labels for every corpus (seconds).
uv run --group eval python -m labeling.rules

# 2. The teacher's plan and projection, without loading the model.
uv run --group eval python -m labeling.teacher --dry-run

# 3. Teacher labelling. Resumable: re-run the same command after any
#    interruption and it continues from the last checkpoint.
uv run --group teacher --group torch --group eval python -m labeling.teacher

# 4. Build the review queue, review it, then merge.
uv run --group eval python -m labeling.review --build
uv run --group eval python -m labeling.review
uv run --group eval python -m labeling.merge
```

Nothing in this package may read `data/third_party/`. CredData is
evaluation-only (plan §5.1, EXECUTION.md §8), and units are loaded through
`datagen.loader.load_for_training`, which refuses third-party data by path and
again per record. A test asserts there is no other route into a label.

## The merge strategy (NR-0003 §2.3)

| case | handling |
|---|---|
| synthetic, teacher agrees with the generator | take the **generator's** label, attach the teacher's soft label |
| synthetic, they disagree | **review queue**. If review finds the generator wrong, fix the generator — NR-0001's code — and record the defect in the execution report. Do not edit the single label |
| own log, Laya and CredSweeper agree and the top recalibrated probability ≥ 0.9 | **auto-adopt** |
| own log, anything else | **review queue** |

Four things about this table are decisions rather than transcription, and each
is here because the alternative was worse.

**The generator wins on synthetic data, always.** Where the two agree there is
nothing to decide. Where they disagree the generator is still right by
construction — it built the value from a vendor format and knows what it built —
so a disagreement is evidence about the *teacher*, or about the generator's
specification, and never a reason to overwrite one label. This is why
disagreements go to review as a batch rather than being resolved one at a time:
a disagreement that turns out to be the generator's fault is a defect affecting
every unit of that family, and editing the one unit that happened to be reviewed
would leave the rest wrong and hide the cause.

**"Agreement" is about redaction, not about the class.** Teacher and generator
agree when they agree on whether the unit should be redacted at all. A secret
the teacher calls a `token` and the generator calls an `api_key` is a class
confusion, recorded as such in the statistics, but it is not a disagreement
worth a human's time: the unit gets masked either way, and only the mask's class
fingerprint differs. Per-class agreement is reported separately, which is what
§2.2 item 3's stratified high-confidence sample exists to measure.

**CredSweeper is consulted only on the own logs.** On synthetic data the
generator already supplies ground truth, so a second opinion adds nothing but
cost. On the own logs there is no ground truth at all, so §2.3 requires two
independent systems to agree before a label is adopted without a human. The
agreement is binary — both say "secret" or both say "not a secret" — because
CredSweeper reports a credential score, not one of our seven classes.

**The 0.9 threshold is applied to recalibrated probabilities, never raw ones.**
NR-0000 measured the teacher labelling a user name as a password at 0.972
confidence, so an uncalibrated 0.9 is not a 90% chance of being right and using
it as one would auto-adopt exactly the confident false positives the project
exists to prevent. §2.2 item 6 requires temperature recalibration on the
synthetic validation split before any auto-adoption, and the threshold is
applied after it.

### What the own logs turned out to contribute

Worth stating because it changes how the table's bottom two rows behave in
practice: NebuLog's own logs contain **no credentials and no credential-shaped
key names**. All 2,692 units come back from the rule labeller as `none` with no
evidence at all, and the eight distinct keys are `OrderId`, `Region`, `Product`,
`Amount`, `Grams`, `ElapsedMs`, `Reason` and `declined`.

So the expected outcome for every own-log unit is `none`, and the own corpus is
in practice a second adversarial check: any own-log unit the teacher calls a
secret is a false positive, measured on real log data rather than on constructed
negatives. What the own logs cannot do is calibrate recall, because they contain
no positives — which is a real limit on NR-0002 finding 5's hope that NR-0003
would supply an in-distribution calibration source. They calibrate the
false-positive half of it, which was finding 5's actual complaint.

## Label provenance

Every merged record says where its hard label came from, in `meta.label_source`:

| value | meaning |
|---|---|
| `generator` | synthetic, construction-guaranteed, teacher agreed |
| `teacher_auto` | own log, Laya and CredSweeper agreed at ≥ 0.9 recalibrated |
| `human` | a reviewer decided it (`data/labels/review_v0.jsonl` has the record) |
| `generator_after_review` | reviewed, generator confirmed correct |

A label with no provenance is a bug, not a default: NR-0004 trains on these
files and the distinction between a construction-guaranteed label and a model's
guess is the difference between distillation and self-training.
