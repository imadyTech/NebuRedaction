"""Loading and rendering the versioned teacher prompt (NR-0003 §2.2 item 1).

The prompt lives in ``prompts/v<n>.txt`` and is parsed here rather than written
inline, for one reason: a label is only interpretable together with the wording
that produced it. The file's SHA-256 travels with every checkpoint and every
output record, so two labels produced under different prompts can never be
pooled by accident, and a resumed run whose prompt was edited mid-flight is
detected rather than silently averaged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from datagen.unit import LABELS, DetectionUnit

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: The prompt version used by this work order.
DEFAULT_VERSION = "v1"

#: The question's key in the Laya answer dictionary.
QUESTION_KEY = "secret_class"

#: `[section]` headers. A criteria section is `[criteria:<class>]`.
_SECTION = re.compile(r"^\[([a-z_]+(?::[a-z_]+)?)\]\s*$")


@dataclass(frozen=True, slots=True)
class Prompt:
    """A parsed, versioned teacher prompt.

    Attributes:
        version: The file's version, such as ``v1``.
        instructions: The question instructions.
        warning: Guidance about not labelling from the key name or template.
            Kept as its own section because it is the one part of the prompt
            that exists to counter a measured failure, and the execution report
            has to be able to say when it was introduced.
        state_template: The rendering of a unit, with ``{key}``, ``{value}``,
            ``{template}`` and ``{scope}`` placeholders.
        criteria: One description per class, in :data:`datagen.unit.LABELS`
            order.
        digest: SHA-256 of the prompt file's bytes.
    """

    version: str
    instructions: str
    warning: str
    state_template: str
    criteria: dict[str, str]
    digest: str

    def questions(self) -> dict[str, Any]:
        """Build the Laya question schema.

        Returns:
            A single ``choice`` question over the seven classes, in the
            typed-question form plan §4.4 specifies for the teacher. The
            warning is appended to the instructions because Laya's schema has
            one instructions field; keeping the two apart in the file and
            joining them here is what lets the report say which part is new.
        """
        return {
            QUESTION_KEY: {
                "type": "choice",
                "instructions": f"{self.instructions}\n\n{self.warning}".strip(),
                "criteria": dict(self.criteria),
            }
        }

    def render(self, unit: DetectionUnit) -> str:
        """Render one unit as the text state handed to the teacher.

        Args:
            unit: The unit to render.

        Returns:
            The filled state template. Absent optional fields render as
            ``(none)`` rather than an empty line, so that a unit without a
            template and a unit whose template is empty produce the same
            string and the prompt's shape never varies.
        """
        return self.state_template.format(
            key=unit.key,
            value=unit.value,
            template=unit.template or "(none)",
            scope=unit.scope or "(none)",
        )


def _parse(text: str, version: str, digest: str) -> Prompt:
    """Parse a prompt file.

    Args:
        text: The file's contents.
        version: The version label to record.
        digest: The file's SHA-256.

    Returns:
        The parsed prompt.

    Raises:
        ValueError: If a section is missing, unknown, or duplicated, or if the
            criteria do not cover exactly the seven classes. A prompt that is
            missing a class would make that class unreachable for the teacher
            while every output file still looked well-formed, so this is a hard
            failure rather than a default.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        header = _SECTION.match(line)
        if header:
            current = header.group(1)
            if current in sections:
                raise ValueError(f"{version}: duplicate section [{current}]")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    def body(name: str) -> str:
        if name not in sections:
            raise ValueError(f"{version}: missing section [{name}]")
        return "\n".join(sections[name]).strip()

    criteria = {}
    for name in sections:
        if name.startswith("criteria:"):
            criteria[name.split(":", 1)[1]] = " ".join(body(name).split())

    if set(criteria) != set(LABELS):
        missing = sorted(set(LABELS) - set(criteria))
        extra = sorted(set(criteria) - set(LABELS))
        raise ValueError(f"{version}: criteria missing {missing}, unexpected {extra}")

    # The state template keeps its newlines; everything else is collapsed to a
    # single paragraph so that reflowing a line in the file does not change the
    # prompt the model sees, and therefore does not change the labels.
    state_template = "\n".join(sections["state"]).strip()
    if not {"{key}", "{value}"} <= set(re.findall(r"\{[a-z]+\}", state_template)):
        raise ValueError(f"{version}: [state] must use at least {{key}} and {{value}}")

    return Prompt(
        version=version,
        instructions=" ".join(body("instructions").split()),
        warning=" ".join(body("warning").split()),
        state_template=state_template,
        criteria={label: criteria[label] for label in LABELS},
        digest=digest,
    )


@lru_cache(maxsize=4)
def load_prompt(version: str = DEFAULT_VERSION) -> Prompt:
    """Load a versioned prompt from ``prompts/``.

    Args:
        version: The version to load, such as ``v1``.

    Returns:
        The parsed prompt, with the file's digest.

    Raises:
        FileNotFoundError: If no such version exists.
    """
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.is_file():
        available = sorted(item.stem for item in PROMPTS_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no prompt {version!r}; available: {available}")
    raw = path.read_bytes()
    return _parse(raw.decode("utf-8"), version, hashlib.sha256(raw).hexdigest())
