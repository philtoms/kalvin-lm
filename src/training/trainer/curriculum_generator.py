"""CurriculumGenerator — LLM-based curriculum generation.

Generates a structured curriculum markdown document from a natural language
goal using an LLM client. Validates the output via CurriculumDocument
parsing, retries once on failure, and writes the result to the curricula
directory.
"""

from __future__ import annotations

import re
from pathlib import Path

from training.harness.llm import LLMClient
from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
)

# System prompt

_SYSTEM_PROMPT = """\
You are a curriculum designer for the Kalvin training system. Generate a \
structured curriculum document in markdown format.

The document must have exactly three sections:
## Objective — what the curriculum teaches (1-3 sentences)
## Approach — the pedagogical strategy (1-3 sentences)
## Lessons — ordered lessons, each with an ### heading

Lesson headings use stable labels: whole numbers (1, 2, 3) for distinct \
conceptual steps, or numbers with a single trailing lowercase letter \
(2a, 2b) for refinements of the parent concept.

Each lesson body contains:
- Human-readable prose explaining the lesson's intent
- Fenced code blocks (triple backticks) containing KScript source

KScript syntax:
- Identity: single character (e.g. M)
- Denote: Q = V  →  {V: [Q]}  (Q denotes V)
- Countersign: Q == V  →  {Q: [V]}, {V: [Q]}
- Connote: Q > V  →  {Q: [V]}  (Q connotes V)
- Canonize: Q => V1 V2  →  {Q: [V1, V2]}
- Indented chaining: value on one line becomes query of indented block

Example curriculum:
## Objective
Teach Kalvin the SVO structure of "Mary had a little lamb".
## Approach
Introduce subject, verb, and object as independent identities, then \
combine them into the SVO structure.
## Lessons
### 1
Introduce the subject as an identity.
```
M = S
```
### 2
Introduce the verb.
```
H = V
```
### 3
Combine into the full SVO structure.
```
MHALL = SVO =>
  M = S
  H = V
  ALL = O
```
"""

# Exception


class CurriculumGenerationError(Exception):
    """Raised when curriculum generation fails after retry."""


# Slug generation


def _slug_from_goal(goal: str) -> str:
    """Derive a filesystem-safe slug from a goal string.

    Lowercase, non-alphanumeric replaced with hyphens, leading/trailing
    hyphens stripped, truncated to 60 characters.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    return slug[:60]


# Generator


class CurriculumGenerator:
    """LLM-based generator that produces a curriculum from a goal string.

    Parameters
    ----------
    client:
        An LLM client for chat completions.
    curricula_dir:
        Directory to write generated curriculum files.
    """

    def __init__(self, client: LLMClient, curricula_dir: Path) -> None:
        self._client = client
        self._curricula_dir = Path(curricula_dir)

    def generate(self, goal: str) -> Path:
        """Generate a curriculum from *goal* and write to disk.

        Makes one LLM call. If the response fails validation, retries
        once with the error included in the prompt. Raises
        :class:`CurriculumGenerationError` on second failure.

        Returns the path to the written curriculum file.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a curriculum for: {goal}"},
        ]

        response = self._client.complete(messages)
        content = response.content or ""

        try:
            doc = CurriculumDocument.from_string(content)
        except CurriculumParseError as exc:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"The previous output was invalid: {exc}. Please try again.",
                }
            )
            response = self._client.complete(messages)
            content = response.content or ""
            try:
                doc = CurriculumDocument.from_string(content)
            except CurriculumParseError as exc2:
                raise CurriculumGenerationError(
                    f"Curriculum generation failed after retry: {exc2}"
                ) from exc2

        slug = _slug_from_goal(goal)
        self._curricula_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._curricula_dir / f"{slug}.md"

        markdown = _serialize_document(doc)
        file_path.write_text(markdown, encoding="utf-8")

        return file_path


# Serialization


def _serialize_document(doc: CurriculumDocument) -> str:
    """Serialize a CurriculumDocument back to markdown."""
    parts: list[str] = []

    parts.append("## Objective\n")
    parts.append(doc.objective.strip())
    parts.append("")

    parts.append("## Approach\n")
    parts.append(doc.approach.strip())
    parts.append("")

    parts.append("## Lessons\n")
    for lesson in doc.lessons:
        parts.append(f"### {lesson.label}\n")
        if lesson.prose.strip():
            parts.append(lesson.prose.strip())
            parts.append("")
        for block in lesson.kscript:
            parts.append("```")
            parts.append(block.strip())
            parts.append("```")
            parts.append("")

    return "\n".join(parts)
