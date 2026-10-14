# `.nrm` v1 — the nebuRedaction model container

The file a nebuRedaction runtime loads. One format, read identically by the
.NET library and the Python reference implementation, which plan §6.4's G3
requires to agree to within 1e-6 per sample.

**Experimental.** This is a pre-release format for a pre-release project. It is
versioned so that a future change is a refusal rather than a misread, but no
compatibility is promised.

## Layout

All integers are **little-endian**. All floating-point values are IEEE 754.

| offset | size | field |
|---|---|---|
| 0 | 4 | magic, ASCII `NRM1` |
| 4 | 2 | format version, `uint16`, currently `1` |
| 6 | 4 | metadata length in bytes, `uint32` |
| 10 | *n* | metadata, UTF-8 JSON, no trailing NUL |
| 10 + *n* | rest | weight section |

A reader **must** reject the file unless the magic matches, the format version
is one it implements, and the metadata's `feature_version` is the one its
feature extractor produces. The third check is the one that matters in
practice: a model read under a different feature definition is not slightly
stale, it is reading different columns, and every prediction would be plausible
nonsense.

## Metadata

```json
{
  "format_version": 1,
  "model_type": "gbdt",
  "feature_version": "1.0",
  "base_model_version": "0.1.0",
  "training_data_sha256": "…",
  "labels": ["none", "password", "token", "api_key",
             "private_key", "connection_string", "cookie"],
  "calibration": {"method": "platt", "slope": …, "intercept": …},
  "thresholds": {"conservative": …, "balanced": …},
  "evaluation": {"g2_met": false, "recall_at_0.001": 0.804, "…": "…"},
  "license": "MIT",
  "experimental": true
}
```

`model_type` is `gbdt` or `linear-hashed`. **Only `gbdt` is implemented in
v0.1**, because NR-0004 §11.2 selected model B; a reader encountering
`linear-hashed` must say so rather than guess.

`labels` is the class order every probability array is in. It is stored rather
than assumed so that a reader never has to agree with the trainer out of band.

`thresholds` are in **calibrated** probability space, so a runtime compares
them against the output of the calibration, not against a raw score.

## Weight section, `model_type: "gbdt"`

```
uint32   class_count
uint32   tree_count            total across all classes
uint32   dense_feature_count   31 for feature version 1.0
uint32   hashed_column_count   how many hash buckets the model reads
uint32   bucket_count          the hashing space, e.g. 32768
int32[hashed_column_count]     bucket ids, ascending

repeated tree_count times:
  uint16  class_index
  uint16  internal_count
  uint16  leaf_count
  repeated internal_count times:
    uint16   feature          index into the projected feature vector
    float32  threshold
    int16    left             >= 0 internal node, < 0 leaf ~left
    int16    right
  repeated leaf_count times:
    float32  value
```

### The projected feature vector

A tree's `feature` indexes a vector of `dense_feature_count +
hashed_column_count` entries:

- entries `0 .. dense_feature_count-1` are the dense features in
  `DENSE_FEATURES` order, as the normative extractor produces them;
- entry `dense_feature_count + i` is the hash bucket named by `bucket ids[i]`.

The model reads a few hundred of the `bucket_count` buckets, not all of them,
which is why the ids are stored. Selecting them was part of training.

### Evaluating a tree

Start at internal node 0.

```
while node is internal:
    node = value[feature] <= threshold ? left : right
```

A child index below zero names a leaf: leaf number `~child`, that is
`-child - 1`. A tree with `internal_count == 0` is a single leaf and its value
is `leaf[0]`.

### From trees to probabilities

```
score[c] = sum of leaf values of every tree whose class_index == c
p = softmax(score)
```

**Accumulate in 64-bit floating point**, even though the stored values are
32-bit, and sum trees in file order. Both are required: 32-bit accumulation
loses the 1e-6 agreement G3 asks for, and floating-point addition is not
associative, so a different order is a different answer. Storing 32-bit and
widening on load costs nothing in agreement — both implementations read the
same bytes — while halving the file.

Softmax subtracts the maximum score before exponentiating, which changes
nothing mathematically and keeps `exp` from overflowing.

### Redaction score and calibration

```
redaction_raw = 1 - p[index of "none"]
redaction     = platt(logit(redaction_raw))
```

where `platt(x) = 1 / (1 + exp(-(slope * x + intercept)))` and
`logit(q) = log(q / (1 - q))` with `q` clamped to `[1e-9, 1 - 1e-9]`.

A unit is redacted when `redaction >= thresholds[profile]`.

## Why a private encoding rather than the trainer's own

LightGBM can serialise a booster itself, and the exporter could have embedded
that text. Three reasons it does not:

1. **Size.** The same 700 trees are 869 KB of LightGBM text and 89 KB here —
   about nine times. The text repeats a 287-name feature list, per-node split
   gains and leaf weights that inference never reads.
2. **One parser, not two.** A .NET implementation of LightGBM's text format is
   a meaningful amount of code to write and to keep correct against a format
   that is not specified for third parties. The layout above is a page.
3. **The container must carry more than weights anyway** — the calibration,
   the thresholds, the class order, the feature version. A format that holds
   the weights opaquely and everything else beside them has two notions of
   version.

The cost is that this file cannot be read by LightGBM, and the exporter is the
only thing that writes it. The exporter is tested against the trainer's own
predictions for exactly that reason.

## Example

A minimal single-tree, two-class file (metadata abbreviated):

```
4E 52 4D 31                 "NRM1"
01 00                       format version 1
2A 00 00 00                 metadata length 42
7B 22 ... 7D                {"format_version":1,...}
02 00 00 00                 class_count    2
02 00 00 00                 tree_count     2
1F 00 00 00                 dense_feature_count 31
01 00 00 00                 hashed_column_count 1
00 80 00 00                 bucket_count   32768
2A 00 00 00                 bucket ids:    [42]

00 00                       tree 0, class 0
01 00                       internal_count 1
02 00                       leaf_count     2
  01 00                     feature 1
  00 00 00 3F               threshold 0.5f
  FF FF                     left  = -1  -> leaf 0
  FE FF                     right = -2  -> leaf 1
  CD CC CC 3D               leaf 0 =  0.1f
  CD CC CC BD               leaf 1 = -0.1f

01 00 00 00 01 00           tree 1, class 1, no internals, one leaf
  00 00 00 00               leaf 0 = 0.0f
```

## Versioning

`format_version` changes when the layout changes. `feature_version` changes
when the feature extractor changes, which invalidates every model fitted under
the old one. The two are independent and both are checked on load.
