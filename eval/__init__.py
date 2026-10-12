"""Evaluation framework for nebuRedaction (work order NR-0002, gate G1).

Plan §6.3 requires the evaluation to be fixed before any model exists, so that
"how good is it" cannot be decided after seeing the results. This package is
that fixture: the interface every system implements, the metrics, the slices, the
report format, and the rule baselines.

Modules:
    system: the interface and the continuous/binary distinction.
    metrics: recall at a target false-positive rate, PR-AUC, bootstrap intervals.
    transforms: deterministic robustness transforms.
    slices: the subsets the report breaks results down by.
    evaluate: threshold selection and the evaluation itself.
    report: report assembly and Markdown rendering.
    testsets: test-set definitions and the lock file.
    run: the command-line runner.
    baselines: the control-group adapters.
"""

__all__ = ["VERSION"]

#: Version of the evaluation framework. Raise it when a metric definition
#: changes; EXECUTION.md §8 requires such a change to be declared in the report.
VERSION = "0.1.0"
