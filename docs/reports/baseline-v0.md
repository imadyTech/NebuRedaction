# Baseline report v0 — rule and tool control groups

What credential detection on log attributes looks like **before** this project
trains anything. Produced at gate G1 by `python -m eval.run`, which was written
and frozen before any model existed (plan §6.3).

Only aggregate metrics appear here. No third-party sample is reproduced: plan
§5.1 keeps CredData local, so every example quoted below is synthetic.

- **Framework**: `eval/` v0.1.0; detection-unit schema **v1.0**, frozen
- **Test sets**: pinned in `eval/testsets.lock`
- **Run**: `runs/eval/baseline-v0/`, 329 s

## Test sets

| set | units | role |
|---|---:|---|
| `synth_val` | 20,000 | **selection only** — every threshold is chosen here |
| `synth_test` | 20,000 | main test set; includes the ten held-out format families |
| `creddata` | 49,568 | external set, independent of the training distribution |

Two things about `creddata` belong on every number taken from it. **18.2% of
CredData's labelled rows could not be acquired on this platform** — 11,425 to a
Windows path-enumeration limit in the upstream tooling, 556 to filenames NTFS
cannot represent, 319 never materialised, 87 to an antivirus quarantine that was
deliberately not overridden (NR-0001 §9.5). And **it contains no `cookie`-class
units at all**, so that class is measured on synthetic data only.

NebuLog's own logs are absent: they are unlabelled until NR-0003.

## Headline results

Each binary system at its own operating point; `rules+entropy` at thresholds
chosen on `synth_val` for a 0.1% false-positive rate.

### Synthetic test set

| system | recall | measured FPR | PR-AUC | held-out recall |
|---|---:|---:|---:|---:|
| `rules` | 0.734 | **11.17%** | 0.566 | 0.670 |
| `credsweeper` 1.19.2 | 0.674 | **12.20%** | 0.525 | 0.579 |
| `rules+entropy` @0.1% | 0.611 | 0.06% | 0.916 | 0.605 |
| `rules+entropy` @1% | 0.662 | 1.00% | 0.916 | 0.616 |
| `rules-values-only` | 0.593 | **0.00%** | 0.741 | 0.501 |

### CredData (external)

| system | recall | measured FPR | PR-AUC |
|---|---:|---:|---:|
| `credsweeper` 1.19.2 | **0.807** | 1.21% | 0.723 |
| `rules` | 0.503 | 5.96% | 0.341 |
| `rules+entropy` @0.1% | 0.162 | 1.22% | 0.709 |
| `rules-values-only` | 0.125 | 1.22% | 0.189 |

## What these numbers say

### 1. Every rule-based system misses the project's false-positive target by one to two orders of magnitude

Plan §4.5 sets a target of **0.1% false positives per attribute**. The pure rule
baseline measures **11.17%** and CredSweeper **12.20%** on the synthetic test
set. These are not tuning failures — a rule set has one operating point and no
dial to turn.

This is the fact G2 has to be judged against, so the comparison rule was fixed
in advance in `eval/README.md` before any model existed: because
**`FPR_r` = 11.17% > 0.1%**, G2 will be reported at **both** 0.1% (the gate,
which holds the student model to a stricter error budget than the baseline it
must beat) and at `FPR_r` (for like-for-like interpretation). Judgement is on
the first.

### 2. All of the rule baseline's false positives come from key names, and most of those from message templates

| system | false negatives | false positives |
|---|---:|---:|
| `rules` (values + key names) | 2,238 | **1,296** |
| `rules-values-only` | 3,422 | **0** |

Turning key-name matching off removes **every** false positive and costs 1,184
detections (false negatives rise from 2,238 to 3,422). The trade is explicit, and the breakdown says where the damage is:

| cause of false positive | count |
|---|---:|
| `template:password` | **416** |
| `key:token` | 278 |
| `key:password` | 188 |
| `key:api_key` | 143 |
| `template:api_key` | 92 |
| `key:private_key` | 77 |

**The single largest source is the message template**, and this reproduces
NR-0000's finding exactly. There, the Laya teacher labelled the user name in
`Failed password for invalid user webmaster` as a password with probability
0.972, reading the word in the template rather than the value. A hand-written
rule set, built independently, fails the same way 416 times. Three synthetic
examples, all labelled `none`:

| key | value | template |
|---|---|---|
| `<slot:user>` | `bright-copper-kettle` | `Failed password for invalid user <*> from <*> port <*> ssh2` |
| `<slot:subject>` | `s.chen@example.com` | `password reset requested for <*>` |
| `<slot:hash>` | `9f86d081884c7d65…` | `Computed hash <*> for {FileName}` |

On the **adversarial-negative slice** specifically, the false-positive rates are
`rules` **41.67%**, `credsweeper` **28.20%**, `rules+entropy` @0.1% 0.25%,
`rules-values-only` 0.00%. Two of the four systems misfire on roughly a third of
the cases the project exists to get right.

### 3. What the rules cannot see at all

`rules-values-only` misses 3,422 of 8,400 secrets. Grouped by family, the top
misses are the formats with **no structural marker**:

| family | missed | why |
|---|---:|---|
| `ambiguous.hex32` / `hex40` / `hex64` / `uuid` / `base64_32` | 124 each | plan §3.3's stated ambiguity: the value alone cannot distinguish a key from a digest |
| `aws.secret_access_key` | 124 | 40 base64 characters, no prefix |
| `datadog.api_key` | 97 | bare 32 hex, identical in shape to an MD5 |
| `generic.oauth_client_secret` | 96 | no vendor shape by definition |
| `generic.high_entropy_key` | 91 | a hand-rolled internal key |

Three synthetic examples the value patterns cannot touch — each is a real secret
whose value is indistinguishable from a common non-secret:

| key | value | label |
|---|---|---|
| `AWS_SECRET_ACCESS_KEY` | 40 base64 characters | `api_key` |
| `DATADOG_API_KEY` | 32 hex characters | `api_key` |
| `ApiKey` | `055d2131-3bae-4115-b42a-9f4c3bca8c7c` | `api_key` |

**This is precisely the gap a learned model is supposed to close**, and it is why
the key name is a necessary feature (plan §3.3) even though keying on it naively
produces finding 2.

### 4. The rule baseline is flattered by synthetic data; CredSweeper is not

`rules` scores 0.734 recall on synthetic data and **0.503** on CredData — a drop
of 23 points. `credsweeper` moves the other way: 0.674 to **0.807**.

This was predicted before the run and recorded in `eval/baselines/rules.py` and
`eval/README.md`: our value patterns and the generator's positive families are
derived from the *same vendor documentation*, so on synthetic data the baseline
is being tested against its own specification. CredData was derived from
neither, which is why **CredData is the honest read on rule-baseline quality**
and the synthetic number is the demanding one for G2.

CredSweeper doing better on CredData than on our synthetic set is consistent
with the same reasoning from the other side: CredData is real-world code of the
kind CredSweeper was built for, while our adversarial negatives are adversarial
for it too.

### 5. Thresholds do not transfer from synthetic to real data

`rules+entropy`'s threshold was chosen on `synth_val` for a 0.1% false-positive
rate. It achieves **0.060%** on `synth_test` — so the choice generalises within
the synthetic distribution — and **1.223%** on CredData, twenty times the target,
while recall collapses from 0.611 to **0.162**.

EXECUTION.md §8 forbids choosing thresholds on third-party data, so this gap
cannot be closed by tuning on CredData, and it should not be. It is a measurement
of how far the synthetic distribution is from a real one, and NR-0004 should
expect the same effect: **a student model calibrated on synthetic validation data
will not hold its false-positive rate on real logs.** Own-log data, labelled in
NR-0003, is the only in-distribution calibration source the project will have.

### 6. Robustness: base64 wrapping defeats everything

Recall change against the untransformed set, on `synth_test`:

| transform | `rules` | `rules+entropy` @0.1% |
|---|---:|---:|
| `affix_noise` | +0.000 | −0.019 |
| `truncated` | −0.133 | −0.187 |
| `case_flipped` | **−0.411** | **−0.549** |
| `base64_wrapped` | **−0.446** | **−0.611** |

`affix_noise` costs the rule baseline nothing, which is what word-boundary
anchoring is for. The other three are severe, and `base64_wrapped` takes
`rules+entropy` to **zero recall** — unsurprising once stated, since every vendor
prefix it keys on is destroyed by the encoding and the re-encoded entropy no
longer reads as a credential. Base64-wrapped secrets are not hypothetical:
Kubernetes secrets and many configuration systems store values that way.

### 7. CredData's input shapes differ enormously, and averaging hides it

`rules` on CredData, by how the key name was recovered (NR-0002 §2.3A item 2):

| slice | units | recall | measured FPR |
|---|---:|---:|---:|
| `multiline_fallback` | 29,241 | **0.917** | 1.37% |
| `assignment` | 16,738 | 0.504 | **23.07%** |
| `defaulted` | 3,483 | 0.079 | 0.82% |
| `option` | 106 | 0.800 | **95.35%** |

The aggregate 0.503 recall is an average over slices that differ by a factor of
eleven. `multiline_fallback` units carry a whole source line as their value, so
there is far more for a pattern to match; `defaulted` units have no recovered key
name and almost nothing is caught. The 23% false-positive rate on `assignment` —
the slice closest to real attribute mode — is the number to watch, and the 95% on
`option` is on 106 units and should not be read as more than a flag.

### 8. Per-class recall is very uneven

`rules` on `synth_test`:

| class | units | recall | correct class |
|---|---:|---:|---:|
| `connection_string` | 1,112 | 0.928 | 0.927 |
| `private_key` | 861 | 0.895 | 0.889 |
| `api_key` | 3,596 | 0.779 | 0.753 |
| `token` | 1,355 | 0.675 | 0.663 |
| `password` | 738 | 0.526 | 0.520 |
| `cookie` | 738 | **0.344** | 0.291 |

Structure is what rules can see. `connection_string` and `private_key` have
unmistakable markers (`://user:pass@`, `-----BEGIN … PRIVATE KEY-----`).
`password` and `cookie` have none: a password is just a string, and a session
cookie is distinguished from a harmless one only by its name. Flagging and
correct attribution track each other closely, so the systems are not confusing
classes much — they are missing them.

## What this means for G2

The bar G2 must clear, with the comparison rule fixed in advance:

- **Pure rules**: recall **0.734** at a measured **11.17%** false-positive rate
  (synthetic); **0.503** at 5.96% (CredData).
- G2 asks for **+10 points of recall at 0.1% false positives**, with the model at
  ≤100 KB.
- The only system here that gets near 0.1% is `rules+entropy`, at **0.611**
  recall — and `rules-values-only` reaches 0.00% false positives at 0.593 recall.
  **So the realistic target for a student model is ~0.70+ recall at 0.1%**, which
  no baseline achieves.

The honest summary: the rule baselines are not weak at finding structured
credentials, and CredSweeper is genuinely good on real data. Where all of them
fail is the combination the project is actually about — **holding a 0.1%
false-positive rate while catching credentials that have no structural marker** —
and that failure is concentrated in the adversarial negatives, where the two
strongest systems misfire on 28% and 42% of cases.

## Reproducing

```bash
uv run --group eval python -m eval.run --out runs/eval/<name>
uv run --group eval python -m eval.error_analysis
```

`report.json` is byte-identical between runs; timestamps and durations live in
`run-meta.json` so that stays true. The error analysis is in
`runs/eval/error-analysis.json`.
