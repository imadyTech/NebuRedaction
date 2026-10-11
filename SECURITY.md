# Security Policy

## This repository contains no real credentials

Every credential-shaped string in this repository is synthetic: format-correct
but randomly generated, produced by the generator in `datagen/` and registered
in the synthetic-sample manifest. No real secret, key, token or password is
present in the working tree or in the git history.

Two mechanisms enforce this:

- a `gitleaks` pre-commit hook and a full-history `gitleaks` scan in CI;
- a manifest check that every credential-shaped fixture under
  `tests/fixtures/synthetic/` is a registered synthetic sample.

If you believe you have found a real credential here, please report it as a
vulnerability using the channel below — do not open a public issue.

## Reporting a vulnerability

Report security issues privately to **security@imady.com**. Please include:

- what you found and where (file path, commit, or a reproduction);
- why you believe it is exploitable or sensitive;
- any suggested remediation.

We aim to acknowledge reports within five working days. Please give us a
reasonable opportunity to fix an issue before disclosing it publicly.

## Scope note on the models

nebuRedaction is a probabilistic classifier. Missed detections and false
positives are expected behaviour, not vulnerabilities — see the evaluation
report for measured rates. Treat it as defence in depth, never as the only
control keeping credentials out of your logs.
