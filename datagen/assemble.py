"""Assembling detection units from values, key names, templates and scopes.

A format family produces a *value*. On its own that is not a training example:
plan §3.3 is explicit that the value alone is often insufficient, so what the
model learns from is the value together with the name it sat under, the message
template it appeared in and the scope it came from. This module builds those
combinations, in the three proportions NR-0001 §2.2 item 5 asks for — key name
hinting, neutral, or misleading.

Two rules here carry most of the weight:

* A **misleading key name over a secret value** is still a secret. Moving the
  key name does not change what the value is, and a detector that lets a
  reassuring name talk it out of a redaction is worse than useless.
* A **misleading key name over a non-secret value** is an *adversarial
  negative*, marked ``meta.adversarial``. These are the cases NR-0000's teacher
  probe failed: ``Failed password for invalid user webmaster`` was classified as
  a password with probability 0.972 because the template contained the word.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datagen import GENERATOR_VERSION
from datagen.families import Family
from datagen.lexicon import (
    CONTEXT_HINTING,
    CONTEXT_MISLEADING,
    CONTEXT_NEUTRAL,
    Lexicon,
)
from datagen.unit import DetectionUnit

#: Message templates for attribute-mode units, keyed by the scope they suit.
#: The ``{key}`` placeholder is replaced with the unit's key name, mirroring how
#: a structured logger renders a named hole.
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "otel.attribute": (
        "Request starting {{Protocol}} {{Method}} {{Path}} with {{{key}}}",
        "Calling {{Url}} with {{{key}}}",
        "Configuration loaded: {{{key}}}",
        "Handling {{EventName}} for {{TenantId}} using {{{key}}}",
        "Publishing to {{Topic}} with {{{key}}}",
    ),
    "otel.resource": ("Resource attributes: {{ServiceName}} {{{key}}}",),
    "config.env": ("<KEY>=<VALUE>",),
    "config.appsettings": ('"{key}": "<VALUE>"',),
    "config.yaml": ("{key}: <VALUE>",),
    "http.header": ("{key}: <VALUE>",),
    "http.query": ("GET {{Path}}?{key}=<VALUE>",),
    "message.kv": ("{key}=<VALUE>",),
    "message.slot": (
        "Failed password for invalid user <*> from <*> port <*> ssh2",
        "Authenticated {{User}} with <*> in {{Elapsed}}ms",
        "Connecting to <*> as <*>",
    ),
    "code.kwarg": ("{key}=<VALUE>",),
    "db.connection": ("Opening connection using {{{key}}}",),
    "k8s.secret_ref": ("env {key} from secretKeyRef {{SecretName}}",),
    "ci.variable": ("{key}=<VALUE>",),
}

#: Scopes whose keys are body-mode fragments rather than attribute names.
_BODY_SCOPES = ("message.kv", "message.slot", "http.query")

#: Templates used for the adversarial negatives whose trap is in the template
#: rather than the key name. The ``<*>`` marks the slot the unit's value filled,
#: following the convention of the R0 sample file.
_ADVERSARIAL_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Failed password for invalid user <*> from <*> port <*> ssh2", "<slot:user>"),
    ("Failed password for <*> from <*> port <*> ssh2", "<slot:user>"),
    ("password reset requested for <*>", "<slot:subject>"),
    ("Rotating secret <*> in vault {Vault}", "<slot:secret_name>"),
    ("Token <*> expires in {Seconds}s", "<slot:token_name>"),
    ("api key <*> was revoked by {Actor}", "<slot:key_name>"),
    ("Verifying credential <*> against {Store}", "<slot:credential_id>"),
    ("Loaded signing key <*> for {Issuer}", "<slot:key_id>"),
    ("authorization policy <*> evaluated to {Result}", "<slot:policy>"),
    ("Computed hash <*> for {FileName}", "<slot:hash>"),
)


@dataclass(frozen=True, slots=True)
class ContextMix:
    """How often each key-name context is chosen.

    Attributes:
        hinting: Weight for a key name that points at the right answer.
        neutral: Weight for a key name that says nothing.
        misleading: Weight for a key name that points the wrong way.
    """

    hinting: float
    neutral: float
    misleading: float

    def pick(self, rng: random.Random) -> str:
        """Choose a context according to the weights.

        Args:
            rng: The source of randomness.

        Returns:
            One of the context constants.
        """
        return rng.choices(
            [CONTEXT_HINTING, CONTEXT_NEUTRAL, CONTEXT_MISLEADING],
            weights=[self.hinting, self.neutral, self.misleading],
        )[0]


def _template_for(rng: random.Random, scope: str, key: str) -> str | None:
    """Choose a message template for a scope, or none.

    Args:
        rng: The source of randomness.
        scope: The unit's scope.
        key: The unit's key name, substituted into the template.

    Returns:
        A template string, or ``None`` when the scope has no template or the
        draw says to omit it. Roughly a sixth of units carry no template, since
        plan §3.1 makes the template optional and the model must work without it.
    """
    candidates = _TEMPLATES.get(scope)
    if not candidates or rng.random() < 0.15:
        return None
    return rng.choice(candidates).format(key=key)


def _scope_for(rng: random.Random, lexicon: Lexicon, family: Family) -> str:
    """Choose a scope appropriate to a family.

    Args:
        rng: The source of randomness.
        lexicon: The key-name lexicon, which supplies the scope vocabulary.
        family: The family whose value is being placed.

    Returns:
        A scope string. Multi-line values are kept out of the body-mode and
        header scopes, where a real logger would never put them.
    """
    scope = lexicon.pick_scope(rng)
    if family.multiline and scope in (*_BODY_SCOPES, "http.header"):
        return rng.choice(["config.env", "config.appsettings", "k8s.secret_ref"])
    return scope


def _body_key(rng: random.Random, key: str, scope: str) -> str:
    """Convert an attribute name into a body-mode slot name where appropriate.

    Args:
        rng: The source of randomness.
        key: The chosen key name.
        scope: The unit's scope.

    Returns:
        The key name to record. Body-mode fragments cut from a message have no
        attribute name of their own, so ``message.slot`` units use the
        ``<slot:...>`` form from plan §3.1.
    """
    if scope == "message.slot":
        return rng.choice([f"<slot:{rng.randint(0, 5)}>", "<slot:value>", key])
    return key


def assemble_unit(
    rng: random.Random,
    *,
    unit_id: str,
    family: Family,
    lexicon: Lexicon,
    context_mix: ContextMix,
    split: str,
    seed: int,
    held_out_family: bool,
    force_context: str | None = None,
) -> DetectionUnit:
    """Build one detection unit from a family.

    Args:
        rng: The source of randomness, already seeded for this unit.
        unit_id: Identifier to record.
        family: The format family supplying the value.
        lexicon: The key-name lexicon.
        context_mix: How often each key-name context is chosen.
        split: ``train``, ``val`` or ``test``, recorded in ``meta``.
        seed: The seed that produced this unit, recorded in ``meta`` so a single
            record can be regenerated on its own.
        held_out_family: Whether the family is held out of training.
        force_context: Use this context instead of drawing from
            ``context_mix``. The generator sets it to ``misleading`` for the
            quota of adversarial negatives, so their share is exact rather than
            left to the draw.

    Returns:
        The assembled unit.
    """
    value = family.generate(rng)
    context = force_context if force_context is not None else context_mix.pick(rng)

    # An ambiguous value is only ambiguous because the key name fails to settle
    # it. Under a neutral name there is nothing left to go on, so such a unit is
    # labelled by the bias-toward-redaction convention and flagged; under a
    # hinting or misleading name the key name is the whole signal.
    if family.ambiguous and context == CONTEXT_NEUTRAL:
        context = rng.choice([CONTEXT_HINTING, CONTEXT_MISLEADING])

    key = lexicon.pick_key(rng, family.label, context)
    scope = _scope_for(rng, lexicon, family)
    key = _body_key(rng, key, scope)
    template = _template_for(rng, scope, key)

    # A non-secret sitting under a credential-suggesting name is the adversarial
    # case, and both key-name groups qualify: `ApiKey` (a sensitive name) over a
    # UUID is as much of a trap as `tokenCount` (a misleading name) over an
    # integer, and NR-0001 §2.2 item 4's definition covers either. For a secret
    # value a misleading name changes nothing — it is still a secret.
    adversarial = family.is_negative and context in (CONTEXT_HINTING, CONTEXT_MISLEADING)

    meta: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "context": context,
        "split": split,
    }
    if adversarial:
        meta["adversarial"] = True
    if held_out_family:
        meta["held_out_family"] = True

    return DetectionUnit(
        id=unit_id,
        key=key,
        value=value,
        source="synth",
        label=family.label,
        template=template,
        scope=scope,
        family=family.name,
        ambiguous=True if family.ambiguous else None,
        meta=meta,
    )


def assemble_template_adversarial(
    rng: random.Random,
    *,
    unit_id: str,
    family: Family,
    split: str,
    seed: int,
    held_out_family: bool,
) -> DetectionUnit:
    """Build an adversarial negative whose trap is the template, not the key.

    This is the exact shape that defeated the teacher model in NR-0000 §9.5: the
    value is an ordinary user name or identifier, and the surrounding template
    says "password". Nothing about the value is secret.

    Args:
        rng: The source of randomness, already seeded for this unit.
        unit_id: Identifier to record.
        family: A negative family supplying the value.
        split: ``train``, ``val`` or ``test``.
        seed: The seed that produced this unit.
        held_out_family: Whether the family is held out of training.

    Returns:
        The assembled unit, labelled ``none`` and marked adversarial.

    Raises:
        ValueError: If a positive family is passed. Labelling a real secret
            ``none`` would poison the training set, so it is refused rather than
            silently relabelled.
    """
    if not family.is_negative:
        raise ValueError(
            f"template-adversarial units must come from a negative family, got {family.name!r}"
        )

    template, slot_key = rng.choice(_ADVERSARIAL_TEMPLATES)
    value = family.generate(rng)
    meta: dict[str, object] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "context": CONTEXT_MISLEADING,
        "split": split,
        "adversarial": True,
        "adversarial_carrier": "template",
    }
    if held_out_family:
        meta["held_out_family"] = True

    return DetectionUnit(
        id=unit_id,
        key=slot_key,
        value=value,
        source="synth",
        label="none",
        template=template,
        scope="message.slot",
        family=family.name,
        meta=meta,
    )
