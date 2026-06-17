"""CurriculumDocument — markdown parser for structured curriculum files.

Parses a markdown document with three required sections (Objective, Approach,
Lessons) and extracts ordered Lesson objects with stable labels, prose, and
KScript source. Supports amendment operations that mutate the document and
write it back to the source file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Label validation

_LABEL_RE = re.compile(r"^\d+[a-z]?$")
"""Valid lesson label: one or more digits optionally followed by a single
lowercase letter (e.g. ``"1"``, ``"2a"``, ``"12"``)."""


# Data types


@dataclass(frozen=True)
class Lesson:
    """A single lesson within a curriculum, bounded by an ``### <label>`` heading.

    Attributes
    ----------
    label:
        Stable label from the heading (e.g. ``"1"``, ``"2a"``).
    prose:
        Non-code-block text — human-readable context.
    kscript:
        Contents of fenced code blocks (KScript source lines).
    """

    label: str
    prose: str
    kscript: list[str] = field(default_factory=list)


class CurriculumDocument:
    """Parsed markdown curriculum with three required sections and ordered lessons.

    Parameters
    ----------
    objective:
        Content of the ``## Objective`` section.
    approach:
        Content of the ``## Approach`` section.
    lessons:
        Ordered lesson list parsed from ``## Lessons`` section.
    source_path:
        File path the document was loaded from (for amendment write-back).
    """

    def __init__(
        self,
        objective: str,
        approach: str,
        lessons: list[Lesson],
        source_path: Path | None = None,
    ) -> None:
        self._objective = objective
        self._approach = approach
        self._lessons = list(lessons)
        self._source_path = Path(source_path) if source_path else None

    # Properties

    @property
    def objective(self) -> str:
        """Content of the Objective section."""
        return self._objective

    @property
    def approach(self) -> str:
        """Content of the Approach section."""
        return self._approach

    @property
    def lessons(self) -> list[Lesson]:
        """Ordered lesson list."""
        return list(self._lessons)

    @property
    def source_path(self) -> Path | None:
        """File path the document was loaded from."""
        return self._source_path

    # Lookup

    def find_lesson(self, label: str) -> Lesson | None:
        """Find a lesson by label. Returns ``None`` if not found."""
        for lesson in self._lessons:
            if lesson.label == label:
                return lesson
        return None

    def all_labels(self) -> list[str]:
        """Return all lesson labels in document order."""
        return [lesson.label for lesson in self._lessons]

    # Parsing

    @classmethod
    def from_file(cls, path: str | Path) -> CurriculumDocument:
        """Parse and validate a curriculum markdown file.

        Parameters
        ----------
        path:
            Path to the markdown file.

        Returns
        -------
        CurriculumDocument
            Parsed and validated document with ``source_path`` set.

        Raises
        ------
        CurriculumParseError
            If the file cannot be read or validation fails.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CurriculumParseError(f"Cannot read file {path}: {exc}") from exc
        doc = cls.from_string(text)
        doc._source_path = path
        return doc

    @classmethod
    def from_string(cls, text: str) -> CurriculumDocument:
        """Parse and validate a curriculum from a string.

        Parameters
        ----------
        text:
            Markdown text to parse.

        Returns
        -------
        CurriculumDocument
            Parsed and validated document with ``source_path`` set to ``None``.

        Raises
        ------
        CurriculumParseError
            If validation fails.
        """
        sections = _split_sections(text)

        required = {"objective", "approach", "lessons"}
        missing = required - sections.keys()
        if missing:
            names = ", ".join(t.title() for t in sorted(missing))
            raise CurriculumParseError(f"Missing required section(s): {names}")

        lessons = _parse_lessons(sections["lessons"])

        if not lessons:
            raise CurriculumParseError("At least one lesson is required")

        labels = [lesson.label for lesson in lessons]
        seen: set[str] = set()
        for label in labels:
            if label in seen:
                raise CurriculumParseError(f"Duplicate lesson label: {label!r}")
            seen.add(label)

        return cls(
            objective=sections["objective"].strip(),
            approach=sections["approach"].strip(),
            lessons=lessons,
        )

    # Amendment

    def amend(self, action: str, **kwargs: object) -> None:
        """Mutate the document and write back to the source file.

        Parameters
        ----------
        action:
            One of ``"insert"``, ``"append"``, ``"modify"``.
        **kwargs:
            Action-specific parameters:
            - ``insert``: ``after_label`` (str), ``lesson`` (Lesson)
            - ``append``: ``lesson`` (Lesson)
            - ``modify``: ``label`` (str), ``lesson`` (Lesson)

        Raises
        ------
        ValueError
            If the action is invalid, target label doesn't exist, new label
            duplicates an existing one, or ``source_path`` is ``None``.
        """
        if self._source_path is None:
            raise ValueError("Cannot amend: no source_path set")

        if action == "insert":
            self._amend_insert(
                after_label=str(kwargs["after_label"]),
                lesson=kwargs["lesson"],  # type: ignore[arg-type]
            )
        elif action == "append":
            self._amend_append(kwargs["lesson"])  # type: ignore[arg-type]
        elif action == "modify":
            self._amend_modify(
                label=str(kwargs["label"]),
                lesson=kwargs["lesson"],  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"Unknown amend action: {action!r}")

        self._write_back()

    def _amend_insert(self, after_label: str, lesson: Lesson) -> None:
        """Insert *lesson* after the lesson with *after_label*."""
        self._check_no_duplicate_label(lesson.label)

        for i, existing in enumerate(self._lessons):
            if existing.label == after_label:
                self._lessons.insert(i + 1, lesson)
                return
        raise ValueError(f"Label {after_label!r} not found")

    def _amend_append(self, lesson: Lesson) -> None:
        """Append *lesson* at the end of the lesson list."""
        self._check_no_duplicate_label(lesson.label)
        self._lessons.append(lesson)

    def _amend_modify(self, label: str, lesson: Lesson) -> None:
        """Replace the lesson at *label* with *lesson*."""
        if lesson.label != label:
            self._check_no_duplicate_label(lesson.label, exclude=label)

        for i, existing in enumerate(self._lessons):
            if existing.label == label:
                self._lessons[i] = lesson
                return
        raise ValueError(f"Label {label!r} not found")

    def _check_no_duplicate_label(self, label: str, *, exclude: str | None = None) -> None:
        """Raise ``ValueError`` if *label* already exists (excluding *exclude*)."""
        for existing in self._lessons:
            if existing.label == label and existing.label != exclude:
                raise ValueError(f"Duplicate label: {label!r}")

    def _write_back(self) -> None:
        """Serialize the document to markdown and write to source_path."""
        if self._source_path is None:
            raise ValueError("Cannot write back: no source_path")
        markdown = self._serialize()
        self._source_path.write_text(markdown, encoding="utf-8")

    def _serialize(self) -> str:
        """Serialize the document to a markdown string."""
        parts: list[str] = []

        parts.append("## Objective\n")
        parts.append(self._objective.strip())
        parts.append("")

        parts.append("## Approach\n")
        parts.append(self._approach.strip())
        parts.append("")

        parts.append("## Lessons\n")
        for lesson in self._lessons:
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


# Parsing helpers


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into sections by ``##`` headings.

    Returns a dict mapping lowercase heading names to their body text.
    """
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines)
            current_name = line[3:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_lines)

    return sections


def _parse_lessons(text: str) -> list[Lesson]:
    """Parse the Lessons section body into a list of Lesson objects.

    Splits on ``### <label>`` headings and extracts prose and code blocks
    from each lesson body.
    """
    lessons: list[Lesson] = []
    current_label: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("### "):
            if current_label is not None:
                lessons.append(_build_lesson(current_label, current_lines))
            current_label = line[4:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_label is not None:
        lessons.append(_build_lesson(current_label, current_lines))

    for lesson in lessons:
        if not _LABEL_RE.match(lesson.label):
            raise CurriculumParseError(
                f"Invalid lesson label: {lesson.label!r} "
                f"(expected digits with optional single trailing letter)"
            )

    return lessons


def _build_lesson(label: str, lines: list[str]) -> Lesson:
    """Build a Lesson from its label and body lines.

    Extracts fenced code blocks as kscript entries. All other text is prose.
    """
    body = "\n".join(lines)

    kscript_entries: list[str] = []
    prose_parts: list[str] = []
    in_code = False
    code_lines: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()
        if stripped == "```" and not in_code:
            in_code = True
            code_lines = []
        elif stripped == "```" and in_code:
            in_code = False
            kscript_entries.append("\n".join(code_lines))
            code_lines = []
        elif in_code:
            code_lines.append(line)
        else:
            prose_parts.append(line)

    prose = "\n".join(prose_parts).strip()
    return Lesson(label=label, prose=prose, kscript=kscript_entries)


class CurriculumParseError(Exception):
    """Raised when a curriculum document fails parsing or validation."""
