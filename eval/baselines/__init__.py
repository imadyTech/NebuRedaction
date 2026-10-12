"""Control-group adapters, and the registry the runner resolves names against.

The six control groups come from plan §6.1. Three are implemented here, two are
placeholders with a working interface so the work orders that fill them do not
have to touch the framework, and one is deliberately absent:

===========================  ==========================================
``rules``                    Control 1, implemented (NR-0002)
``rules+entropy``            Control 2, implemented (NR-0002)
``credsweeper``              Control 3, implemented (NR-0002)
``laya``                     Control 4, placeholder — NR-0003
``student-a`` / ``-b`` / ``-c``  Control 6, placeholder — NR-0004
SDLog                        Control 5, not present in v0.1
===========================  ==========================================

SDLog has no placeholder on purpose. Plan §5.1 item 0 says it is not used in
v0.1 at all, and an entry that could be invoked invites someone to invoke it.
"""

from __future__ import annotations

from collections.abc import Callable

from eval.baselines.credsweeper_adapter import CredSweeperBaseline
from eval.baselines.placeholders import PlaceholderSystem
from eval.baselines.rules import RuleBaseline
from eval.baselines.rules_entropy import RulesPlusEntropy

__all__ = [
    "REGISTRY",
    "CredSweeperBaseline",
    "PlaceholderSystem",
    "RuleBaseline",
    "RulesPlusEntropy",
    "available_systems",
    "build",
]


def _rules_values_only() -> RuleBaseline:
    """Build the rule baseline with key-name matching disabled.

    Returns:
        A rule baseline that judges values alone. Reported alongside the full
        rule baseline so the report can separate what the value patterns achieve
        from what the key names add — and, on the adversarial slice, what they
        cost.
    """
    return RuleBaseline(name="rules-values-only", use_key_names=False)


#: Name to constructor. Names are what ``--systems`` accepts.
REGISTRY: dict[str, Callable[[], object]] = {
    "rules": RuleBaseline,
    "rules-values-only": _rules_values_only,
    "rules+entropy": RulesPlusEntropy,
    "credsweeper": CredSweeperBaseline,
    "laya": lambda: PlaceholderSystem(
        name="laya",
        work_order="NR-0003",
        reason="Laya zero-shot shares the teacher inference code, which NR-0003 builds.",
    ),
    "student-a": lambda: PlaceholderSystem(
        name="student-a",
        work_order="NR-0004",
        reason="Student model A (hashed features + multinomial logistic regression).",
    ),
    "student-b": lambda: PlaceholderSystem(
        name="student-b",
        work_order="NR-0004",
        reason="Student model B (gradient-boosted trees).",
    ),
    "student-c": lambda: PlaceholderSystem(
        name="student-c",
        work_order="NR-0004",
        reason="Student model C (tiny char-level CNN).",
    ),
}

#: The systems NR-0002 actually runs. The placeholders are registered so the
#: interface is exercised and named, but running them produces no findings.
DEFAULT_SYSTEMS: tuple[str, ...] = (
    "rules",
    "rules-values-only",
    "rules+entropy",
    "credsweeper",
)


def available_systems() -> list[str]:
    """Return every registered system name.

    Returns:
        The names, sorted.
    """
    return sorted(REGISTRY)


def build(name: str) -> object:
    """Construct a system by name.

    Args:
        name: A registered name.

    Returns:
        The system.

    Raises:
        KeyError: If the name is not registered, listing what is — a typo on the
            command line should not look like a system that scored nothing.
    """
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown system {name!r}; registered: {', '.join(available_systems())}"
        ) from None
    return factory()
