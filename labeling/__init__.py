"""Labelling: rule labels, teacher labels, merge strategy and manual review.

NR-0003. Synthetic labels are guaranteed by construction, so what the teacher
adds is three things (§1): soft labels for distillation, labels for the
unlabelled own logs, and — where it disagrees with the generator — evidence of
either a generator defect or a genuine ambiguity.

Nothing here may read ``data/third_party/``. CredData is evaluation-only
(plan §5.1, EXECUTION.md §8), and the loader enforces it: this package loads
units through :func:`datagen.loader.load_for_training`, which refuses
third-party data both by path and per record. A test in ``tests/labeling``
asserts there is no other path into a label.
"""

VERSION = "0.1.0"
