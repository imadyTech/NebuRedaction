# training

Training pipelines: ML.NET for models A (hashed features + multinomial
logistic regression) and B (LightGBM), PyTorch on CPU for model C
(tiny char-level CNN).

Training input is not a single rule. Synthetic and own data are always
permitted. Of the third-party sources, LogHub and SDLog's data may be used for
**internal** training (decided 2026-10-12) while CredData may not be trained on
at all — see
[../docs/licensing/R0-license-review.md](../docs/licensing/R0-license-review.md).
Whether a publicly released model may be trained on the permitted third-party
sources is still open and must be settled before NR-0006.

Owner work orders: NR-0004 (student models), NR-0009 (train CLI).
