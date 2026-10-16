# Vendor provenance review

NR-0008 §2.0. This project imitates the *shapes* of real credentials in order
to generate synthetic training data. Three questions follow from that, and this
document answers each: where those shapes came from, whether anything was
copied from third-party code without declaring it, and whether vendor names
appear anywhere EXECUTION.md §9 says they should not.

**Review only.** Findings are listed in §5 with the decision taken on each.
Code was changed in exactly one place, F1, and only after that decision.

**Vendor names are used here as EXECUTION.md §9 item 4 permits**, and this
paragraph is the declaration that item 4 requires: every vendor name, key name
and format below is a **descriptive reference to publicly published material**,
cited in §6, used to audit provenance rather than to illustrate anything.
**This project is not affiliated with, sponsored by or endorsed by any vendor
named.** No vendor's documentation text or sample code is reproduced in whole;
had any been, §9 item 4 condition ④ would require disclosing it in
`THIRD_PARTY_NOTICES.md`, where the two ported regular expressions already are.
In-text citations follow APA 7th edition, as §9 item 1 requires of an English
document.

An earlier draft of this paragraph claimed an *exemption* from §9 item 4 on the
grounds that it forbade naming vendors. That was written against a superseded
text: item 4 withdrew the prohibition on 2026-10-15 and was widened again on
2026-10-16. No exemption is needed, and claiming one misdescribed the rule.

| | |
|---|---|
| Reviewed | 2026-10-15; translated to English and citations added 2026-10-16 |
| Scope | `main` at `618803b`: 170 tracked files, and all 44 commits of history |
| Governing rules | NR-0008 §2.0; EXECUTION.md §9 (2026-10-15) |
| Method | per-file regular-expression inventory; full-history diff scan via `git log -p --all`; runtime introspection of `datagen.families` and `eval.baselines.rules` |

## 1. Vendor-name inventory (§2.0 item 1)

Grouped by **purpose** rather than listed by file. The same vendor name means
something entirely different in a format-family definition than it does in the
body of a public report.

| Purpose | Occurrences | Files | Notes |
|---|---:|---:|---|
| 1 Generator families, `datagen/families/` | 152 | 8 | Format definitions. The vendor name is a **family identifier**, and each family cites its source (§2) |
| 2 Rule baseline, `eval/baselines/` | 68 | 2 | NR-0002's regular-expression baseline and the CredSweeper adapter. All 27 rules carry a provenance field |
| 3 Other code | 38 | 9 | The key-name lexicon `datagen/lexicon/keys.yaml` (25), generator config, unit schema, evaluation transforms |
| 4 Shipping libraries, `python/` `dotnet/` | 8 | 7 | **None is a vendor reference**: `microsoft` comes from `learn.microsoft.com` links, MSBuild property names and analyzer ids in `.editorconfig` |
| 5 Tests | 47 | 13 | Assertions about families and rules |
| 6 **`docs/` (public-facing)** | 43 | 3 | **See finding F4** |
| 7 Third-party notices | 12 | 1 | Sources and licences for datasets, models and tools |
| 8 Corpus tooling, CI, configuration | 68 | 13 | `corpus/fetch_creddata.py` (28) and `corpus/mappings/creddata.yaml` (21) map CredData's labels onto this project's classes; CI and gitleaks configuration |

### 1.1 Full history

- History contains **exactly the 170 paths the current tree contains** — no file
  has ever been deleted, so there is no such thing here as a file that exists
  only in history.
- Scanning the **removed lines** of `git log -p --all` for vendor names matched
  four (`redis`, `pypi`, `logpai`, `mooselab`), and **all four are still in the
  current tree**. No vendor reference exists only in history.
- History does contain two removed vendor-shaped literals in test source: two
  strings of `AKIA` plus 16 upper-case alphanumerics, one containing
  `TESTVALUE` and one `EXAMPLE`, removed by `6f82971` and `68253ce`. **The
  originals are not reproduced here** — EXECUTION.md §5 item 3 forbids
  vendor-shaped credential strings anywhere in this repository, source literals
  included, and a provenance review is the last place to break that rule. They
  are reachable by commit id. Neither was a real credential; both were shapes,
  not secrets. See NR-0001 §9 and NR-0005 §9.11.

## 2. Format provenance (§2.0 item 2)

**All 114 format families carry a provenance field. None is missing.** The
classification below, and the per-family list in Appendix A, are generated from
the registry rather than transcribed.

| Class | Count |
|---|---:|
| Open standard or specification | 43 |
| Vendor documentation | 35 |
| No vendor format (reason stated) | 27 |
| Project's own repository | 5 |
| Vendor's own engineering blog | 3 |
| Encyclopaedia (not official) | 1 |
| **Total** | **114** |

The **27 entries declaring "No vendor format"** deserve a note of their own.
They are not gaps in the record; each is an explicit denial with a stated
reason, for instance:

- `ambiguous.hex32` — "No vendor format: plan §3.3's stated ambiguity between a
  32-hex key and a 32-hex digest."
- `password.memorable` — "No vendor format: human-chosen word-plus-digits
  passwords, the low-entropy case a detector cannot catch on entropy alone."
- `pem.single_line_escaped` — "No vendor format: the flattened form structured
  logging produces when a PEM block becomes one string field."

That pattern is the point of the exercise: where a vendor could not be cited,
the family says why it is constructed the way it is.

### 2.1 Content copied from third-party code

Checked entry by entry. **Nothing copied is undeclared.**

| Third party | How this project uses it | Declaration |
|---|---|---|
| **gitleaks** (Rice, n.d.), MIT | **2 of the 27** baseline rules are ports: `aws.access_key_id` and `azure.ad_client_secret` | **Declared.** `THIRD_PARTY_NOTICES.md` names both under *Tools and rule sets* and reproduces the MIT licence in full; each rule's `provenance` field reads `gitleaks (MIT)` plus the rule name; and a test asserts that every ported rule names its origin |
| **detect-secrets** (Yelp, n.d.), Apache-2.0 | Its *approach* — a Shannon-entropy threshold over base64 and hexadecimal strings — is reimplemented in `eval/baselines/rules_entropy.py` | **Declared**, and explicitly as an approach reimplemented, not code copied |
| **CredSweeper** (Samsung, n.d.-b) | Invoked as an external system through `eval/baselines/credsweeper_adapter.py` | **Declared** |
| **CredData, LogHub, SDLog, StarPII, Laya** | Datasets and a model | **Declared** |

Two cross-checks are worth recording:

1. **The generator's AWS family and the rule baseline are independent.**
   `datagen/families/api_keys.py` uses the single prefix AWS documents, while
   the identically named rule in `rules.py` uses gitleaks' full prefix set and
   says so in its own provenance field: the prefix set is gitleaks', not AWS
   documentation. The boundary is drawn and stated.
2. **The key-name lexicon was not derived from any log dataset.**
   `datagen/lexicon/keys.yaml` declares three permitted sources — vendor
   documentation, the configuration conventions of widely used frameworks, and
   NebuLog's own demo-service logs — and states outright that no name there was
   derived by counting key names in LogHub, CredData or SDLog, recording that
   plan §5.2 and §5.1 conflict on this point and that §5.1 governs.

## 3. Vendor research cited in documentation (§2.0 item 3)

**None.** Every external reference in `docs/`, `README.md` and
`THIRD_PARTY_NOTICES.md` points to a tool, dataset or model repository — nine
in total, listed under F6. No vendor research report, engineering blog or paper
is cited in any document.

A vendor's own engineering blog *is* cited five times — three format families
and two baseline rules — but entirely **in code**, as format provenance. See
F2.

## 4. Conclusion

The provenance record for the generator and the rule baseline is complete: 114
families and 27 rules, **none without a source**; **no undeclared**
third-party port; **no vendor reference** anywhere in history that is not also
in the current tree; and **no vendor research material** cited in any document.

The findings fall into two groups: one factual citation error (F1), and several
mismatches that arose because EXECUTION.md §9 was adopted on 2026-10-15, after
the documents it governs were written (F4, F6).

## 5. Findings and decisions

### F1 — `aws.access_key_id` cited the wrong page — fixed (Frank, 2026-10-15)

The family cited a page about ARN formats, with an anchor for unique
identifiers. Verified against the source (Amazon Web Services, n.d.-a):

1. That page is titled *Identify AWS resources with Amazon Resource Names
   (ARNs)*. **It carries no `identifiers-unique-ids` anchor** — its anchors are
   `reference-arns`, `arns-syntax`, `supported-arns`, `arns-paths` and
   `arns-paths-wildcards`.
2. **It does not mention the `AKIA` prefix at all.**
3. The anchor name is correct but belongs to a **different page** (Amazon Web
   Services, n.d.-b), whose *Understanding unique ID prefixes* section carries
   the prefix table giving `AKIA` for an access key.
4. Half the accompanying note — "plus 16 upper-case alphanumerics" — **is not
   supported by AWS documentation either.** The table gives prefixes, not
   lengths or alphabets.

**Resolution.** The `doc_url` now points at the page and anchor that carry the
prefix table, and the note separates what the documentation states (the prefix)
from what this project chose (the length and alphabet of the remainder), in the
style the 27 "No vendor format" families already use. Generated values are
unchanged — the prefix was always right — and the 507 tests in `tests/datagen/`
pass.

### F2 — three families and two rules cite a vendor's engineering blog — accepted (Frank, 2026-10-15)

| Site | Source |
|---|---|
| `github.pat_classic`, `github.app_installation_token`, and `github.token` in the rule baseline | GitHub's engineering blog on its authentication token formats |
| `github.pat_fine_grained`, and `github.fine_grained` in the rule baseline | GitHub's blog post introducing fine-grained personal access tokens |

The publisher is the vendor itself, so the source is official; but it is a blog
post rather than reference documentation, which is what §2.0 item 2 asks to
have flagged.

Worth stating plainly: **GitHub's reference documentation does not document the
base62 CRC32 checksum** its token formats carry, and the data card's claim rests
on exactly that detail. For this particular fact the blog is the best source
that exists.

**Decision: accepted.** The reasoning given was that nebuRedaction is itself a
blog-level output — at most an unreviewed arXiv posting — and that while it is
written to approach the standards of rigorous publication, **it must state its
research status rather than hold itself to the standard of a reviewed journal.**
Citing a vendor engineering blog is consistent with a document that declares
that status. NR-0008 §2.1 carries the obligation to make the declaration.

### F3 — `neg.snowflake_id` cites an encyclopaedia — accepted (Frank, 2026-10-15)

The family cites Wikipedia, which is neither a vendor nor an official source.
The family is a **negative** — a Snowflake id is not a credential — so an
imprecise shape cannot produce a wrong label, only a less realistic negative.

**Decision: accepted as adequate for this use.**

### F4 — vendor key names in public documents — key names kept (Frank, 2026-10-15)

§9 was adopted on 2026-10-15 and both documents below predate it, so this is a
**mismatch with a new rule, not a violation at the time of writing.**

`docs/reports/baseline-v0.md`, a report and therefore within §9's scope, in
table rows whose own pipes prevent quoting them as a table:

```
116 | `aws.secret_access_key` | 124 | 40 base64 characters, no prefix
117 | `datadog.api_key`       |  97 | bare 32 hex, identical in shape to an MD5
126 | `AWS_SECRET_ACCESS_KEY` | 40 base64 characters | `api_key`
127 | `DATADOG_API_KEY`       | 32 hex characters    | `api_key`
```

Lines 116–117 name vendor format families in a results table; lines 126–127 use
real vendor names **as example key names**, which is the most direct mismatch
with §9 item 4.

`docs/data-card-synth.md`, a data card and explicitly within scope, names
GitHub, Twilio and AWS in body text (lines 63, 68, 189) and lists vendor-named
held-out families (lines 122–124). `students-v0.md` and `dotnet-runtime-v0.md`
contain no vendor-name hits.

**Decision: the key names are kept.** The reasoning given was that
`AWS_SECRET_ACCESS_KEY` and `DATADOG_API_KEY` are the attribute-bearing names
real logs actually carry, whereas `IMADY_API_KEY` is a name that does not
exist.

That decision was taken against measurement rather than argument, because the
natural objection is that changing those names might partially disable the
model. It does not, and the reason is structural rather than lucky. Full data
is in NR-0008 §9.3; in summary:

- The shipped model reads **256 of 32,768** hash buckets. The tokens `aws`,
  `datadog` and `imady` are **not among them**; `secret`, `key` and `token`
  are. What the model learned is that the key name contains secret, key or
  token — never that it contains aws.
- Renaming changes **exactly 2 of 32,799 features** — it switches off the
  vendor token's bucket and switches on the replacement's — and **neither
  bucket is one the model reads**. All 31 engineered features stay
  byte-identical, because the word count is unchanged and the sensitive words
  are preserved.
- Ten probe scores across both pairs of names came out at a difference of
  **exactly zero**, with identical decisions.

So keeping the names is not about avoiding a cost; there is none for this
model. It is that **the lexicon is training data meant to mirror reality, not
an example shown to a reader.** §9 item 4 was written for documentation
examples, and applying it to a training vocabulary is a category error — one
that happens to be harmless here only because the model never learned the
vendor token.

**A limitation follows, and must be declared** (Frank, 2026-10-15). Bucket
selection is by frequency over the training corpus, so a user who trains
further or adds target words changes which 256 buckets are selected, and must
**retrain, recalibrate, re-choose both thresholds and re-run the evaluation** —
four steps, none of them skippable. Every number this project reports holds for
one model, one corpus and one test set. The full statement is NR-0008 §9.6; it
belongs in the technical report's limitations chapter and in NR-0006's model
card.

**Resolved since.** Three values published in a vendor's documentation as
illustrative placeholders had been written verbatim into test source — two in
`tests/labeling/test_rule_labeller.py` and one in
`tests/neburedaction/test_detector.py`, the last added during NR-0005. They had
nothing to do with the model, but they breached EXECUTION.md §5 item 3, and
**gitleaks structurally cannot catch that class**: the values contain
`EXAMPLE`, a stop word, which is the same suppression the project relies on
elsewhere. All three are now produced by the generator at test time, and
`tests/test_no_vendor_examples.py` enforces their absence — a guard whose own
detection is verified against planted values, because a guard that silently
matches nothing passes forever.

**Closed** (Frank, 2026-10-16). Two things settled it. First, §9 item 4 no
longer forbids what this finding flagged: as revised on 2026-10-15 and widened
on 2026-10-16, vendor names, key names, public formats, example code snippets
and example values may all be used as descriptive references to published
material, subject to five conditions — citation, text names only, a
no-affiliation statement in the README and the data card, disclosure in
`THIRD_PARTY_NOTICES.md` of any *complete* use of vendor code or prose, and
prompt removal on a rights-holder's objection. Second, Frank reviewed
`baseline-v0.md` and `docs/data-card-synth.md` directly and found no problem in
them.

The one thing this finding did produce is now done independently: condition ③'s
no-affiliation sentence was absent from both the README and the data card, and
has been added to each.

**This finding was, in part, my error.** I re-read NR-0008 before this work but
not EXECUTION.md, so F4 was raised against a §9 item 4 that had already been
withdrawn the day before. The three copied *values* under F4 were a real
finding; the paragraph about vendor names in prose was litigating a rule that
no longer existed.

### F5 — the evaluation slices are vendor-named — closed without a pause (Frank, 2026-10-15)

The question raised was one of intellectual property: whether what those slices
use is licensed under MIT or another open-source licence, since if not, the
project's licensing would have a hole and work would have to stop.

**No such licence is required, because no copyrighted work is being copied.**
What the project takes from vendors is three distinct things, and only one of
them is a licensing matter:

1. **Names used as identifiers** (`aws.access_key_id`, `DATADOG_API_KEY`) are
   trademarks, not copyrightable works. Using a mark to denote the thing it
   names is nominative use.
2. **Factual format descriptions** — a prefix, a length, an alphabet —
   published by vendors so that others may implement against them. Facts are
   not copyrightable, and writing one's own generator or regular expression
   from a published description does not copy the description. Every secret
   scanner, gitleaks included, rests on this. **The slice names come from (1)
   and the slice contents from (2).**
3. **Verbatim expression from vendor documentation** was the one place actual
   copying occurred — the three placeholder values described under F4, now
   removed. They were almost certainly too short and too arbitrary to attract
   copyright, but "probably de minimis" is a poor position to hold when
   deleting them costs nothing.

The project's one genuine licence dependency is already discharged: the two
regular expressions ported from gitleaks are MIT, declared, reproduced in full,
and enforced by a test.

**Two reservations, stated rather than buried.** The author of this review is
not a lawyer. Whether trademark use in a public repository and a published
report requires an explicit notice, and whether a vendor's documentation site
terms bear on reusing published format facts, are judgement calls that were
referred upward. On the first, a trademark and no-affiliation notice has since
been added to `THIRD_PARTY_NOTICES.md`.

### F6 — citations in existing documents (§9 items 1–2)

The nine external references in the repository sit in just two files:

| Document | References |
|---|---:|
| `THIRD_PARTY_NOTICES.md` | 9 (all of them) |
| `docs/licensing/R0-license-review.md` | 8 (all but the Apache licence URL) |

Neither is named in §9's scope (user guide, technical report, model card, data
card, README, blog draft). **The sharper finding is the inverse:** every
document that *is* in scope — `docs/data-card-synth.md`,
`docs/reports/baseline-v0.md`, `docs/reports/students-v0.md`,
`docs/reports/dotnet-runtime-v0.md`, `docs/spec/nrm-v1.md` and `README.md` —
contained **zero** external references, while discussing credential formats and
naming CredData, gitleaks and Laya in prose. So F6 is not "bare URLs where
formatted citations belong"; it is "no bibliography where §9 requires one". The
provenance URLs in Appendix A are a data table, not a bibliography.

**Decision** (Frank, 2026-10-16): reference lists are added to the documents
that will be **published** — `README.md`, `docs/data-card-synth.md`,
`docs/licensing/R0-license-review.md` and this review. Internal process
documents are left alone.

## 6. References

Bibliographic details were checked against each source on 2026-10-16. Software
and dataset entries give the version this project pins where one is pinned, and
a retrieval date where the source is undated and subject to change, as APA 7th
edition directs. Where a work carries no publication date, `n.d.` is used
rather than a guess.

Amazon Web Services. (n.d.-a). *Identify AWS resources with Amazon Resource
Names (ARNs)*. AWS Identity and Access Management User Guide. Retrieved
October 16, 2026, from
https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html

Amazon Web Services. (n.d.-b). *IAM identifiers*. AWS Identity and Access
Management User Guide. Retrieved October 16, 2026, from
https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_identifiers.html

Apache Software Foundation. (2004). *Apache license, version 2.0*.
http://www.apache.org/licenses/LICENSE-2.0

BigCode. (n.d.). *StarPII* [Model]. Hugging Face. Retrieved October 16, 2026,
from https://huggingface.co/bigcode/starpii

Convai Innovations. (n.d.). *Laya* [Model]. Hugging Face. Retrieved
October 16, 2026, from https://huggingface.co/convaiinnovations/laya

Mooselab. (n.d.). *SDLog* [Computer software]. GitHub. Retrieved October 16,
2026, from https://github.com/mooselab/SDLog

Rice, Z. (n.d.). *gitleaks* (Version 8.30.1) [Computer software]. GitHub.
Retrieved October 16, 2026, from https://github.com/gitleaks/gitleaks

Samsung. (n.d.-a). *CredData* [Data set]. GitHub. Retrieved October 16, 2026,
from https://github.com/Samsung/CredData

Samsung. (n.d.-b). *CredSweeper* (Version 1.19.2) [Computer software]. GitHub.
Retrieved October 16, 2026, from https://github.com/Samsung/CredSweeper

Yelp. (n.d.). *detect-secrets* [Computer software]. GitHub. Retrieved
October 16, 2026, from https://github.com/Yelp/detect-secrets

Zhu, J., He, S., Liu, J., He, P., Xie, Q., Zheng, Z., & Lyu, M. R. (n.d.).
*LogHub* [Data set]. GitHub. Retrieved October 16, 2026, from
https://github.com/logpai/loghub

**Deliberately not formatted as references.** The provenance URLs in Appendix A
are data, not sources this document argues from: they record where each format
family's shape was derived, and they are generated from the registry rather
than curated. Formatting them as a bibliography would imply this review relies
on all of them, which it does not. The two AWS pages it does rely on — the ones
verified under F1 — appear above.

Two further sources named in the text are this project's own internal documents
rather than external publications, and are linked rather than cited:
`THIRD_PARTY_NOTICES.md` and `docs/licensing/R0-license-review.md`.

## Appendix A — provenance by family and by rule

Generated from `datagen.families.REGISTRY` and `eval.baselines.rules`, not
transcribed. Class counts are in §2 and are not repeated here.

| **Total** | **114** |

### A.1 Every family and its provenance

| Format family | Class | Provenance |
|---|---|---|
| `neg.snowflake_id` | Encyclopaedia (not official) | https://en.wikipedia.org/wiki/Snowflake_ID |
| `ambiguous.base64_32` | No vendor format (reason stated) | — No vendor format: 32 bytes of base64 as either an encoded payload or a symmetric key |
| `ambiguous.hex32` | No vendor format (reason stated) | — No vendor format: plan §3.3's stated ambiguity between a 32-hex key and a 32-hex digest |
| `ambiguous.hex40` | No vendor format (reason stated) | — No vendor format: 40 hex characters as a session token, a SHA-1 digest or a commit id |
| `ambiguous.hex64` | No vendor format (reason stated) | — No vendor format: plan §3.3's stated ambiguity between a 64-hex key and a SHA-256 digest |
| `ambiguous.uuid` | No vendor format (reason stated) | — No vendor format: plan §3.3's stated ambiguity between a UUID correlation id and a session id |
| `generic.high_entropy_key` | No vendor format (reason stated) | — No vendor format: this is the hand-rolled internal key that entropy heuristics must catch |
| `generic.oauth_client_secret` | No vendor format (reason stated) | — No vendor format: OAuth 2.0 (RFC 6749) does not prescribe a client-secret shape, so deployments use base64, hex or a UUID |
| `generic.opaque_api_token` | No vendor format (reason stated) | — No vendor format: the 'tok_' convention many internal portals adopt |
| `neg.build_number` | No vendor format (reason stated) | — No vendor format: CI build identifiers in their common shapes |
| `neg.byte_size` | No vendor format (reason stated) | — No vendor format: human-readable sizes as logging libraries render them |
| `neg.camel_case_value` | No vendor format (reason stated) | — No vendor format: camel-case enum values from .NET configuration |
| `neg.config_word_value` | No vendor format (reason stated) | — No vendor format: boolean and enum configuration values |
| `neg.correlation_slug` | No vendor format (reason stated) | — No vendor format: the prefixed request ids seen in OpenStack and similar services |
| `neg.count_integer` | No vendor format (reason stated) | — No vendor format: plain counts, the most common attribute value of all |
| `neg.duration` | No vendor format (reason stated) | — No vendor format: elapsed-time values in the shapes .NET and OTel emit |
| `neg.license_key` | No vendor format (reason stated) | — No vendor format: grouped product licence keys, high-entropy but not an authenticator |
| `neg.long_numeric_id` | No vendor format (reason stated) | — No vendor format: account and order numbers |
| `neg.short_alnum_id` | No vendor format (reason stated) | — No vendor format: short ids, too short to be a credential but dense per character |
| `neg.short_link` | No vendor format (reason stated) | — No vendor format: shortener URLs, whose slug is random-looking but public |
| `neg.slug_identifier` | No vendor format (reason stated) | — No vendor format: readable generated names, as container runtimes and CI produce |
| `password.memorable` | No vendor format (reason stated) | — No vendor format: human-chosen word-plus-digits passwords, the low-entropy case a detector cannot catch on entropy alone |
| `password.placeholder` | No vendor format (reason stated) | — No vendor format: the default placeholder passwords that appear in sample configuration files |
| `password.strong_random` | No vendor format (reason stated) | — No vendor format: shaped after secret-manager output, with one character from each class (cf. NIST SP 800-63B composition guidance) |
| `password.url_safe` | No vendor format (reason stated) | — No vendor format: the restricted alphabet used where a password must survive being put in a URL or connection string |
| `pem.body_without_header` | No vendor format (reason stated) | — No vendor format: key material pasted without its PEM envelope, leaving no structural marker |
| `pem.single_line_escaped` | No vendor format (reason stated) | — No vendor format: the flattened form structured logging produces when a PEM block becomes one string field |
| `session.id_hex` | No vendor format (reason stated) | — No vendor format: hand-rolled hex session identifiers, the case plan §3.3 calls out as indistinguishable from a digest |
| `basic_auth.bare_base64` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7617#section-2 |
| `basic_auth.header` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7617 |
| `cookie.jsessionid` | Open standard or specification | https://jakarta.ee/specifications/servlet/6.0/jakarta-servlet-spec-6.0#session-tracking-mechanisms |
| `cookie.request_header` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc6265#section-4.2 |
| `cookie.set_cookie_header` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc6265#section-4.1 |
| `http.basic_credentials_in_url` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `jwt.es256` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7518#section-3.4 |
| `jwt.hs256` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7519 |
| `jwt.rs256` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7518#section-3.3 |
| `neg.base64_of_json` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc4648#section-5 |
| `neg.base64_of_text` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc4648#section-4 |
| `neg.crc32_hex` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc1952#section-8 |
| `neg.email_address` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc5322#section-3.4.1 |
| `neg.etag` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc9110#field.etag |
| `neg.file_path_posix` | Open standard or specification | https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap03.html#tag_03_267 |
| `neg.harmless_cookie` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc6265 |
| `neg.hostname` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc1123#section-2 |
| `neg.ipv4_address` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc1918 |
| `neg.iso8601_timestamp` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc3339 |
| `neg.md5_hex` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc1321 |
| `neg.mime_type` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc9110#field.content-type |
| `neg.public_key_pem` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7468#section-4 |
| `neg.query_string_clean` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc3986#section-3.4 |
| `neg.semver` | Open standard or specification | https://semver.org/spec/v2.0.0.html |
| `neg.sha1_hex` | Open standard or specification | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf |
| `neg.sha256_hex` | Open standard or specification | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf |
| `neg.span_id` | Open standard or specification | https://www.w3.org/TR/trace-context/#parent-id |
| `neg.ssh_public_key` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc4716 |
| `neg.trace_id` | Open standard or specification | https://www.w3.org/TR/trace-context/#trace-id |
| `neg.traceparent` | Open standard or specification | https://www.w3.org/TR/trace-context/#traceparent-header-field-values |
| `neg.unix_epoch_millis` | Open standard or specification | https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_16 |
| `neg.url_without_credentials` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc3986 |
| `neg.user_agent` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc9110#field.user-agent |
| `neg.uuid_v4` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc9562#section-5.4 |
| `neg.uuid_v7` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc9562#section-5.7 |
| `oauth.bearer_header` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc6750#section-2.1 |
| `oauth.opaque_access_token` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc6749#section-1.4 |
| `pem.ec_private_key` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc5915 |
| `pem.encrypted_private_key` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `pem.pkcs8_private_key` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7468#section-10 |
| `pem.rsa_private_key` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `redis.connection_uri` | Open standard or specification | https://www.iana.org/assignments/uri-schemes/prov/redis |
| `smtp.connection_uri` | Open standard or specification | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `neg.digest_with_prefix` | Project's own repository | https://github.com/opencontainers/image-spec/blob/main/descriptor.md#digests |
| `neg.image_digest_reference` | Project's own repository | https://github.com/opencontainers/distribution-spec/blob/main/spec.md |
| `neg.nanoid` | Project's own repository | https://github.com/ai/nanoid |
| `neg.ulid` | Project's own repository | https://github.com/ulid/spec |
| `pem.openssh_private_key` | Project's own repository | https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.key |
| `amqp.connection_uri` | Vendor documentation | https://www.rabbitmq.com/docs/uri-spec |
| `anthropic.api_key` | Vendor documentation | https://docs.claude.com/en/api/getting-started |
| `aws.access_key_id` | Vendor documentation | https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_identifiers.html#identifiers-prefixes |
| `aws.secret_access_key` | Vendor documentation | https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html |
| `azure.ad_client_secret` | Vendor documentation | https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app |
| `azure.storage_account_key` | Vendor documentation | https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage |
| `azure.storage_connection_string` | Vendor documentation | https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string |
| `cookie.aspnet_auth` | Vendor documentation | https://learn.microsoft.com/en-us/aspnet/core/security/authentication/cookie |
| `cookie.csrf_token` | Vendor documentation | https://learn.microsoft.com/en-us/aspnet/core/security/anti-request-forgery |
| `cookie.laravel_session` | Vendor documentation | https://laravel.com/docs/12.x/session |
| `datadog.api_key` | Vendor documentation | https://docs.datadoghq.com/account_management/api-app-keys/ |
| `digitalocean.personal_access_token` | Vendor documentation | https://docs.digitalocean.com/reference/api/create-personal-access-token/ |
| `gitlab.personal_access_token` | Vendor documentation | https://docs.gitlab.com/security/tokens/ |
| `google.api_key` | Vendor documentation | https://cloud.google.com/docs/authentication/api-keys |
| `hashicorp.vault_service_token` | Vendor documentation | https://developer.hashicorp.com/vault/docs/concepts/tokens |
| `mongodb.srv_connection_string` | Vendor documentation | https://www.mongodb.com/docs/manual/reference/connection-string/ |
| `mssql.connection_string` | Vendor documentation | https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax |
| `mysql.connection_uri` | Vendor documentation | https://dev.mysql.com/doc/refman/8.4/en/connecting-using-uri-or-key-value-pairs.html |
| `neg.assembly_version` | Vendor documentation | https://learn.microsoft.com/en-us/dotnet/api/system.version |
| `neg.file_path_windows` | Vendor documentation | https://learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats |
| `neg.git_commit_sha` | Vendor documentation | https://git-scm.com/docs/git-rev-parse#_specifying_revisions |
| `npm.access_token` | Vendor documentation | https://docs.npmjs.com/about-access-tokens |
| `nuget.api_key` | Vendor documentation | https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package |
| `oauth.google_refresh_token` | Vendor documentation | https://developers.google.com/identity/protocols/oauth2 |
| `openai.api_key` | Vendor documentation | https://platform.openai.com/docs/api-reference/authentication |
| `postgres.connection_uri` | Vendor documentation | https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING |
| `pypi.upload_token` | Vendor documentation | https://pypi.org/help/#apitoken |
| `sendgrid.api_key` | Vendor documentation | https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys |
| `session.signed_value` | Vendor documentation | https://expressjs.com/en/resources/middleware/cookie-session.html |
| `slack.bot_token` | Vendor documentation | https://api.slack.com/authentication/token-types |
| `slack.webhook_url` | Vendor documentation | https://api.slack.com/messaging/webhooks |
| `stripe.restricted_key` | Vendor documentation | https://docs.stripe.com/keys#limit-access |
| `stripe.secret_key` | Vendor documentation | https://docs.stripe.com/keys |
| `twilio.api_key_sid` | Vendor documentation | https://www.twilio.com/docs/iam/api-keys |
| `twilio.auth_token` | Vendor documentation | https://www.twilio.com/docs/iam/api-keys |
| `github.app_installation_token` | Vendor's own engineering blog | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.pat_classic` | Vendor's own engineering blog | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.pat_fine_grained` | Vendor's own engineering blog | https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/ |

### A.2 The 27 rules in `eval/baselines/rules.py`

| Rule | Class | Provenance |
|---|---|---|
| `aws.access_key_id` | `api_key` | gitleaks (MIT) — aws-access-token; the prefix set is gitleaks', not AWS documentation |
| `github.token` | `api_key` | https://github.blog/engineering/platform-security/behind-githubs-new-authentication-token-formats/ |
| `github.fine_grained` | `api_key` | https://github.blog/security/application-security/introducing-fine-grained-personal-access-tokens-for-github/ |
| `gitlab.pat` | `api_key` | https://docs.gitlab.com/security/tokens/ |
| `slack.bot_token` | `api_key` | https://api.slack.com/authentication/token-types |
| `slack.webhook` | `api_key` | https://api.slack.com/messaging/webhooks |
| `stripe.key` | `api_key` | https://docs.stripe.com/keys |
| `sendgrid.api_key` | `api_key` | https://www.twilio.com/docs/sendgrid/ui/account-and-settings/api-keys |
| `google.api_key` | `api_key` | https://cloud.google.com/docs/authentication/api-keys |
| `openai.api_key` | `api_key` | https://platform.openai.com/docs/api-reference/authentication |
| `anthropic.api_key` | `api_key` | https://docs.claude.com/en/api/getting-started |
| `npm.token` | `api_key` | https://docs.npmjs.com/about-access-tokens |
| `digitalocean.pat` | `api_key` | https://docs.digitalocean.com/reference/api/create-personal-access-token/ |
| `hashicorp.vault_token` | `api_key` | https://developer.hashicorp.com/vault/docs/concepts/tokens |
| `pypi.token` | `api_key` | https://pypi.org/help/#apitoken |
| `nuget.api_key` | `api_key` | https://learn.microsoft.com/en-us/nuget/nuget-org/publish-a-package |
| `twilio.api_key_sid` | `api_key` | https://www.twilio.com/docs/iam/api-keys |
| `azure.ad_client_secret` | `api_key` | gitleaks (MIT) — azure-ad-client-secret |
| `jwt` | `token` | https://datatracker.ietf.org/doc/html/rfc7519 — the eyJ prefix is base64url of '{"' |
| `oauth.bearer_header` | `token` | https://datatracker.ietf.org/doc/html/rfc6750#section-2.1 |
| `google.refresh_token` | `token` | https://developers.google.com/identity/protocols/oauth2 |
| `basic_auth.header` | `password` | https://datatracker.ietf.org/doc/html/rfc7617 |
| `pem.private_key` | `private_key` | https://datatracker.ietf.org/doc/html/rfc7468#section-11 |
| `pem.private_key_escaped` | `private_key` | Derived from the flattened form structured logging produces (no vendor document). |
| `url.embedded_credentials` | `connection_string` | https://datatracker.ietf.org/doc/html/rfc3986#section-3.2.1 |
| `connection_string.password_field` | `connection_string` | https://learn.microsoft.com/en-us/sql/connect/ado-net/connection-string-syntax |
| `azure.storage_connection_string` | `connection_string` | https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string |
