# training

Training pipelines: ML.NET for models A (hashed features + multinomial
logistic regression) and B (LightGBM), PyTorch on CPU for model C
(tiny char-level CNN).

Training input is restricted to synthetic and own data; reading
`data/third_party/` from a training script is a defect (plan §5.1).

Owner work orders: NR-0004 (student models), NR-0009 (train CLI).
