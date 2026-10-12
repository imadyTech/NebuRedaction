# datagen

Synthetic data generator. Produces the detection units the released models are
trained on. Every credential-shaped value it emits is format-correct but
randomly generated, so the output can be published and committed as fixtures; no
real credential is ever read, written or verified against a live service.

The dataset it produces is described in
[../docs/data-card-synth.md](../docs/data-card-synth.md), including the
composition, the limitations and how to reproduce it exactly.

## Using it

```bash
# Generate the v0 corpus (200k/20k/20k) and a synthetic-sample manifest.
uv run --group datagen python -m datagen.generate \
  --config datagen/configs/v0.yaml --seed 20261011 \
  --out data/synth/v0 --manifest data/synth/v0/manifest.json

# A quick 500-per-split run, for checking a change.
uv run --group datagen python -m datagen.generate \
  --config datagen/configs/v0.yaml --seed 20261011 --out /tmp/check --size 500

# Validate any detection-unit file.
uv run --group datagen python -m datagen.validate data/synth/v0/train.jsonl
```

The same config and seed produce byte-identical output; the test suite checks
that, and `dataset.json` records each split's SHA-256.

## Layout

| Path | Contents |
|---|---|
| `unit.py` | The detection-unit record and its serialisation. |
| `schema/unit.schema.json` | The record's JSON Schema. |
| `validate.py` | Validation CLI, including the invariants a schema cannot express. |
| `families/` | One value generator per format family, with its documentation citation. |
| `lexicon/` | The key-name vocabulary and its provenance. |
| `assemble.py` | Combines values with key names, templates and scopes. |
| `generate.py` | Split composition and the generation CLI. |
| `manifest.py` | The synthetic-sample manifest. |
| `loader.py` | The unified data loader and the source restriction it enforces. |
| `configs/v0.yaml` | The v0 corpus's configuration, with the reasoning for each number. |

## Adding a format family

1. Write the generator in the right module under `families/`, taking an explicit
   `random.Random` and nothing else.
2. `register()` it with a `doc_url` that either cites the vendor's published
   format or states plainly that no vendor format exists and the shape is a
   convention. This is a requirement, not a nicety: NR-0001 §2.2 item 1 asks for
   the provenance of every format, and a test enforces that the field is
   populated and substantive.
3. If the format carries a check value, add a unit test that verifies a
   generated value satisfies it.
4. Regenerate the corpus and its manifest **together**, since the manifest
   records digests of the values the corpus contains. Nothing needs committing:
   test fixtures are generated on demand (see `tests/datagen/conftest.py` for
   why they are not in git).

Adding a family changes the generated corpus, so bump `GENERATOR_VERSION` and
the config's `name` rather than silently replacing v0.

## A note on the key-name lexicon

`lexicon/keys.yaml` is drawn only from vendor documentation, the configuration
conventions of widely used frameworks, and NebuLog's own demo logs.

Plan §5.2 originally said to take the key-name distribution from open log
templates. That contradicts plan §5.1, which bars third-party data from
training, and §5.1 wins — so no name in the lexicon was obtained by counting key
names in LogHub, CredData or SDLog. The lexicon file records this at the top, and
a test asserts the provenance statement is present, so the constraint travels
with the data rather than living only in a work order. LogHub and SDLog are not
used at all in v0.1.

Owner work orders: NR-0001 (format, generator, manifest), NR-0003 (labeling).
