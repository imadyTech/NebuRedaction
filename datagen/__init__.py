"""Synthetic data generation for nebuRedaction (work order NR-0001).

The generator produces the detection units that the released models are trained
on. Every credential-shaped value it emits is format-correct but randomly
generated, so the output can be published and committed as fixtures; no real
credential is ever read, written or verified against a live service.

Modules:
    unit: the detection-unit record and its serialisation.
    validate: schema validation CLI (``python -m datagen.validate``).
    generate: generation CLI (``python -m datagen.generate``).
    assemble: combines values with key names, templates and scopes.
    families: one value generator per format family.
    lexicon: the key-name vocabulary.
    manifest: the synthetic-sample manifest.
    loader: the unified data loader that enforces the source restriction.
"""

__all__ = ["GENERATOR_VERSION"]

#: Version of the generator itself, recorded in every unit's ``meta`` so that a
#: dataset can be traced back to the code that produced it. Bump the minor part
#: when a family's output changes, since that breaks byte-for-byte
#: reproducibility of earlier datasets.
GENERATOR_VERSION = "0.1.0"
