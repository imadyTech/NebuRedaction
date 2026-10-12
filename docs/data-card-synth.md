# Data card — nebuRedaction synthetic corpus v0

A corpus of labelled log attributes for training a classifier that decides
whether one attribute of one log record is a secret credential. Every
credential-shaped value in it is format-correct and randomly generated.

**No real credential is present in this dataset.** Nothing was collected from a
log, a repository, a breach corpus or any other real source, and no value was
ever sent anywhere to see whether it worked. The generator is deterministic, so
any record can be regenerated from the seed recorded beside it and checked.

- **Licence**: MIT, same as the code. It contains no third-party data.
- **Version**: v0, generator `0.1.0`, seed `20261011`.
- **Work order**: NR-0001. Produced by `datagen/`, config
  `datagen/configs/v0.yaml`.
- **Status**: pre-release. The record format is fixed at G1, not before.

## Composition

| Split | Records | Secrets | Non-secrets | Format families | SHA-256 (first 16) |
|---|---|---|---|---|---|
| train | 200,000 | 84,000 | 116,000 | 104 | `576f3c5205ad32ed` |
| val | 20,000 | 8,400 | 11,600 | 104 | `1ac156bc8666fe62` |
| test | 20,000 | 8,400 | 11,600 | 114 | `492def8dff0710fd` |

Classes follow plan §3.2. Counts for `train` / `test`:

| Class | train | test |
|---|---|---|
| `none` | 116,000 | 11,600 |
| `api_key` | 37,940 | 3,596 |
| `token` | 13,542 | 1,355 |
| `connection_string` | 10,840 | 1,112 |
| `private_key` | 8,128 | 861 |
| `cookie` | 6,775 | 738 |
| `password` | 6,775 | 738 |

**The 42% secret rate is not realistic, and is not meant to be.** In a real log
stream, secrets are rarer than one attribute in a million; training on that
ratio teaches a model to answer `none` unconditionally. The operating threshold
and the reported false-positive rate are what must reflect the real prior
(plan §4.5) — not the training mix. Anyone computing a headline accuracy on
this corpus will get a number that means nothing about production.

## Format families

A *format family* is one recognisable shape a value can take: one vendor's key
layout, one flavour of connection string, one kind of identifier that merely
looks like a secret. There are 114, of which 68 are secrets and 46 are not.

| Class | Families |
|---|---|
| `api_key` | 29 |
| `token` | 11 |
| `connection_string` | 9 |
| `private_key` | 7 |
| `password` | 6 |
| `cookie` | 6 |
| `none` (hard negatives) | 46 |

Every positive family records in its source code where its format is documented
publicly, or states plainly that the vendor publishes no format and the shape is
a convention. GitHub's token formats carry a base62 CRC32 checksum over the
random part; the generated tokens satisfy it, and a test verifies that.

The negatives are the point of the corpus as much as the positives: they are the
values that fool a detector. They include 32- and 64-character hex digests (the
twins of a Twilio auth token and a 64-hex API key), W3C trace and span ids,
base64 of ordinary English, UUIDs, ULIDs, Nano IDs, ETags, image digests, SSH
*public* keys, PEM *public* key blocks, licence keys, user agents and file paths.

## Context, and the adversarial negatives

A value alone is often not classifiable — plan §3.3 says so — so each record
also carries the key name it sat under, the message template it appeared in and
its scope. Key names come from `datagen/lexicon/keys.yaml`, drawn only from
vendor documentation, framework configuration conventions and NebuLog's own
logs; the lexicon file documents that provenance and the prohibition it follows.

Each record records how its key name relates to its value:

- **hinting** — the name points at the right answer (`ApiKey` over an API key);
- **neutral** — the name says nothing (`Value`, `Data`, `Arg0`);
- **misleading** — the name points the wrong way.

**24% of the negatives in every split are adversarial**: a non-secret under a
credential-suggesting key name or template, flagged `meta.adversarial`. The
share is exact rather than sampled. Three carriers:

- `sensitive_key` (40%) — `ApiKey = <uuid>`: a name that really does mean
  "secret", over a value that is not one.
- `misleading_key` (30%) — `tokenCount = 128`: a name that merely contains the
  word.
- `template` (30%) — `Failed password for invalid user <*>`: the trap is in the
  surrounding message, and the attribute name is not involved at all.

These exist because of a measured failure. In NR-0000 the Laya teacher model was
asked to classify ten samples; it labelled the *user name* in `Failed password
for invalid user webmaster` as a password, with probability 0.972. It was reading
the word "password" in the template, not the value. A 420-million-parameter
calibrated model made that mistake, so a kilobyte-scale student will too unless
the training set argues with it.

The remaining 76% of negatives sit under neutral key names, which is what a real
log looks like.

## Ambiguous records

6,775 records in `train` (3.4%) are marked `ambiguous: true`: 32-, 40- and
64-character hex strings, UUIDs and 32-byte base64 under sensitive or misleading
key names — the cases plan §3.3 names as impossible to settle from the value.
They follow the project's bias-toward-redaction convention, so they are labelled
as secrets, and the flag lets the evaluation report them separately rather than
mixing them into the headline numbers.

## Splits

**Ten families are held out of training entirely** and appear only in the test
split, which is how plan §5.3 measures generalisation to formats the model has
never seen. 1,673 test records (8.4%) come from them.

Held out: `azure.ad_client_secret`, `oauth.google_refresh_token`,
`amqp.connection_uri`, `basic_auth.bare_base64`, `pem.body_without_header`,
`cookie.laravel_session`, `neg.public_key_pem`, `neg.ulid`,
`neg.base64_of_text`, `neg.image_digest_reference`.

Two of those are deliberately awkward. `pem.body_without_header` is key material
with its PEM envelope stripped, so nothing structural is left to key on;
`neg.public_key_pem` is the nearest possible miss to a private key — same
envelope, same base64 texture, one word different. The held-out score is meant
to be a hard number, not a comfortable one.

`train` and `val` are generated independently from the same family pool and the
same seed stream, which gives two disjoint samples of one distribution; `val` is
for early stopping and threshold selection only.

## Reproducing it

```bash
uv run --group datagen python -m datagen.generate \
  --config datagen/configs/v0.yaml --seed 20261011 --out data/synth/v0
```

The same config and seed produce byte-identical files; the SHA-256 values above
are the check. Every record carries its own `meta.seed`, so a single record can
be regenerated and verified on its own without rebuilding the corpus.

Validate with:

```bash
uv run --group datagen python -m datagen.validate data/synth/v0/train.jsonl
```

## Record format

One JSON object per line; the schema is `datagen/schema/unit.schema.json`.

```json
{"id":"v0-train-0000005","key":"IdToken","value":"055d2131-3bae-4115-b42a-9f4c3bca8c7c",
 "label":"token","template":"\"IdToken\": \"<VALUE>\"","scope":"config.appsettings",
 "family":"ambiguous.uuid","ambiguous":true,"source":"synth",
 "meta":{"generator_version":"0.1.0","seed":16373712619421723136,"context":"hinting","split":"train"}}
```

## Known limitations

1. **The class prior is wrong on purpose.** See above. This corpus cannot be
   used to estimate a production false-positive rate directly.
2. **English-centric.** Key names, message templates and the words inside
   generated passwords are English. A non-English deployment's key names are not
   represented, and the model's behaviour there is unmeasured.
3. **Generated values are uniformly random within their format.** Real
   credentials are too, mostly — but real *passwords* are not, and real base64
   blobs encode structured data with characteristic byte distributions. The
   `password.memorable` and `password.placeholder` families push back on this for
   passwords; elsewhere the texture is more uniform than production.
4. **Templates are a small fixed set.** Roughly thirty, drawn from ASP.NET Core,
   OpenTelemetry and shell or configuration idiom. Real template diversity is far
   greater.
5. **Not drawn from real logs at all.** That is a licensing and safety decision
   (plan §5.1), and it has a cost: whatever is distinctive about real log
   attributes and absent from the generator's imagination is absent here too.
   CredData serves as the independent external test set precisely because of
   this, and it is the only check on this limitation.
6. **Vendor formats drift.** Each family reflects its vendor's documentation as
   of 2026-10-11. Formats change, and a stale family will train the model on a
   shape that no longer occurs.
7. **Some class boundaries are judgement calls.** A licence key is a negative; a
   TOTP seed is an `api_key`; an AWS access key *id* is an `api_key` even though
   it is not secret by itself. These are defensible, not inevitable, and the
   same calls are applied to CredData's categories in
   `corpus/mappings/creddata.yaml` so the two corpora can be compared.
8. **Cookies are under-represented in evaluation.** CredData has no cookie
   category, so the `cookie` class is only ever tested on synthetic and own data.

## Intended and unintended uses

Intended: training and evaluating small classifiers that decide whether a log
attribute should be redacted, and as a published benchmark others can regenerate
exactly.

Not intended: as a corpus of real credential patterns for threat research (the
values are random); as a source of realistic base rates; or as a password
wordlist (the password families are generated, and the placeholder list is a
handful of generic defaults).
