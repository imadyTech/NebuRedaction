# corpus

Acquisition scripts and processing recipes for third-party datasets
(CredData, SDLog, LogHub). **The data itself never enters this repository**,
and none of it is ever redistributed.

Permitted use differs per source and is not a single rule — see
[../docs/licensing/R0-license-review.md](../docs/licensing/R0-license-review.md)
for what each licence allows and which decision set it:

- **CredData** — local evaluation only; never trained on.
- **LogHub** and **SDLog's data** — internal training only, removed from
  evaluation (decided 2026-10-12). Whether a *publicly released* model may be
  trained on them is still open; see that review's consequences section.

Owner work orders: NR-0001 (acquisition scripts), NR-0002 (external test sets).
