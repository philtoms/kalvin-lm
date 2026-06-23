"""Curriculum data model and persistence for the Trainer.

Provides the ordered lesson list (Curriculum) and per-session tracking
state (CurriculumState) including submitted/satisfied/pending sets and
JSON persistence for restart recovery.

Supports two input forms:
- A :class:`CurriculumDocument` (label-based lesson tracking).
- A flat ``list[str]`` of KScript sources (positional tracking; wrapped
  in a synthetic :class:`CurriculumDocument`).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)

logger = logging.getLogger(__name__)

# Type alias for entry identity keys used in tracking sets.
# Mirrors the pattern from ui/kscript/app.py line 23.
EntryKey: TypeAlias = tuple[int, tuple[int, ...]]


class Curriculum:
    """Ordered lesson container with position tracking.

    Accepts either a :class:`CurriculumDocument` or a flat ``list[str]``
    of KScript sources (wrapped in a synthetic document).

    Parameters
    ----------
    lessons_or_document:
        A :class:`CurriculumDocument` or a ``list[str]`` of KScript sources.
    position:
        Current position in the lesson list (0-indexed). Position indexes
        into the document's lesson list (one position per lesson, not per
        kscript block).
    """

    def __init__(
        self,
        lessons_or_document: list[str] | CurriculumDocument,
        *,
        position: int = 0,
    ) -> None:
        if isinstance(lessons_or_document, CurriculumDocument):
            self._document: CurriculumDocument = lessons_or_document
            self.position: int = position
        else:
            self._document = self._make_synthetic_document(lessons_or_document)
            self.position = position

    def _make_synthetic_document(self, lessons: list[str]) -> CurriculumDocument:
        """Create a synthetic CurriculumDocument from a flat lessons list."""
        lesson_objects = [
            Lesson(label=str(i + 1), prose="", kscript=[src]) for i, src in enumerate(lessons)
        ]
        return CurriculumDocument(
            objective="(auto-generated)",
            approach="(auto-generated)",
            lessons=lesson_objects,
        )

    @property
    def document(self) -> CurriculumDocument:
        """The underlying CurriculumDocument."""
        return self._document

    @property
    def lessons(self) -> list[str]:
        """Flat list of all KScript source strings across all lessons.

        Preserved for backward compatibility and JSON persistence.
        Each lesson contributes its kscript blocks in order.
        """
        result: list[str] = []
        for lesson in self._document.lessons:
            for block in lesson.kscript:
                result.append(block)
        return result

    def current(self) -> str | None:
        """Return the current lesson's joined KScript, or ``None`` if complete.

        Returns the concatenated kscript blocks for the lesson at the
        current position, separated by newlines.
        """
        lesson = self.current_lesson()
        if lesson is None:
            return None
        return "\n".join(lesson.kscript) if lesson.kscript else ""

    def current_lesson(self) -> Lesson | None:
        """Return the current Lesson object, or ``None`` if complete."""
        all_lessons = self._document.lessons
        if self.position >= len(all_lessons):
            return None
        return all_lessons[self.position]

    def current_label(self) -> str | None:
        """Return the label of the current lesson, or ``None`` if complete."""
        lesson = self.current_lesson()
        return lesson.label if lesson else None

    def advance(self) -> None:
        """Move to the next lesson in the curriculum."""
        self.position += 1

    def is_complete(self) -> bool:
        """Return ``True`` if all lessons have been consumed."""
        return self.position >= len(self._document.lessons)

    def total(self) -> int:
        """Return the total number of lessons."""
        return len(self._document.lessons)

    def remaining(self) -> int:
        """Return the number of lessons remaining (including current)."""
        return max(0, len(self._document.lessons) - self.position)

    def label_at_position(self, pos: int) -> str | None:
        """Return the label at the given position, or None if out of range."""
        all_lessons = self._document.lessons
        if 0 <= pos < len(all_lessons):
            return all_lessons[pos].label
        return None

    def position_of_label(self, label: str) -> int | None:
        """Return the position index for a label, or None if not found."""
        for i, lesson in enumerate(self._document.lessons):
            if lesson.label == label:
                return i
        return None


class CurriculumState:
    """Per-session tracking state for the Trainer.

    Holds the curriculum, per-entry tracking sets (submitted, satisfied,
    pending), label-based lesson tracking, and an append-only event log.
    Supports JSON persistence for restart recovery.

    Parameters
    ----------
    curriculum:
        The :class:`Curriculum` instance to track.
    save_path:
        Optional default path for :meth:`save` / :meth:`load`.
    curriculum_file:
        Optional path to the curriculum file for persistence.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        *,
        save_path: str | Path | None = None,
        curriculum_file: str | None = None,
    ) -> None:
        self.curriculum: Curriculum = curriculum
        self.submitted: set[EntryKey] = set()
        self.satisfied: set[EntryKey] = set()
        self.pending: set[EntryKey] = set()
        self.event_log: list[dict] = []
        self._save_path: Path | None = Path(save_path) if save_path else None
        self._curriculum_file: str | None = curriculum_file

        self.lesson_submitted: set[str] = set()
        self.lesson_satisfied: set[str] = set()

    # Label-based tracking

    @property
    def current_label(self) -> str | None:
        """Label of the next unsatisfied lesson.

        Scans labels in document order and returns the first one not in
        ``lesson_satisfied``. Returns ``None`` if all lessons are satisfied.
        """
        for lesson in self.curriculum.document.lessons:
            if lesson.label not in self.lesson_satisfied:
                return lesson.label
        return None

    @property
    def curriculum_file(self) -> str | None:
        """Path to the curriculum file."""
        return self._curriculum_file

    @curriculum_file.setter
    def curriculum_file(self, value: str | None) -> None:
        self._curriculum_file = value

    def mark_lesson_submitted(self, label: str) -> None:
        """Add *label* to the lesson-level submitted set."""
        self.lesson_submitted.add(label)

    def mark_lesson_satisfied(self, label: str) -> None:
        """Add *label* to the lesson-level satisfied set.

        Also advances the curriculum position to the next unsatisfied lesson.
        """
        self.lesson_satisfied.add(label)
        self._advance_to_next_unsatisfied()

    def is_lesson_submitted(self, label: str) -> bool:
        """Return ``True`` if *label* is in the lesson submitted set."""
        return label in self.lesson_submitted

    def is_lesson_satisfied(self, label: str) -> bool:
        """Return ``True`` if *label* is in the lesson satisfied set."""
        return label in self.lesson_satisfied

    def _advance_to_next_unsatisfied(self) -> None:
        """Move the curriculum position to the first unsatisfied lesson."""
        for i, lesson in enumerate(self.curriculum.document.lessons):
            if lesson.label not in self.lesson_satisfied:
                self.curriculum.position = i
                return
        self.curriculum.position = len(self.curriculum.document.lessons)

    # Set operations (EntryKey-level)

    def mark_submitted(self, key: EntryKey) -> None:
        """Add *key* to the submitted set, removing it from pending if present."""
        self.submitted.add(key)
        self.pending.discard(key)

    def mark_satisfied(self, key: EntryKey) -> None:
        """Add *key* to the satisfied set."""
        self.satisfied.add(key)

    def is_submitted(self, key: EntryKey) -> bool:
        """Return ``True`` if *key* is in the submitted set."""
        return key in self.submitted

    def is_satisfied(self, key: EntryKey) -> bool:
        """Return ``True`` if *key* is in the satisfied set."""
        return key in self.satisfied

    # Event log

    def log_event(self, event_type: str, data: dict) -> None:
        """Append an event to the append-only event log with an ISO timestamp.

        The event log persists across session resets.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "data": data,
        }
        self.event_log.append(entry)

    # Session reset

    def reset_session(self) -> None:
        """Clear submitted/satisfied/pending sets but preserve curriculum
        position and the append-only event log.
        """
        self.submitted.clear()
        self.satisfied.clear()
        self.pending.clear()
        self.lesson_submitted.clear()
        self.lesson_satisfied.clear()

    # Persistence

    def save(self, path: str | Path | None = None) -> None:
        """Serialize state to a JSON file.

        Parameters
        ----------
        path:
            File path. Falls back to ``save_path`` from construction.
        """
        target = Path(path) if path else self._save_path
        if target is None:
            raise ValueError("No save path specified")

        data = {
            "version": 2,
            "position": self.curriculum.position,
            "lessons": self.curriculum.lessons,
            "curriculum_file": self._curriculum_file,
            "current_label": self.current_label,
            "lesson_submitted": sorted(self.lesson_submitted),
            "lesson_satisfied": sorted(self.lesson_satisfied),
            "submitted": _serialize_set(self.submitted),
            "satisfied": _serialize_set(self.satisfied),
            "pending": _serialize_set(self.pending),
            "event_log": self.event_log,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CurriculumState:
        """Deserialize state from a JSON file.

        Supports both new format (version 2 with curriculum_file and labels)
        and legacy format (flat lessons list with positional index).

        When ``curriculum_file`` is present and the file exists, the
        curriculum is loaded from the file (preserving the original
        label scheme). Otherwise, a synthetic document is created from
        the flat lessons list.

        Returns a fully reconstructed :class:`CurriculumState`.
        """
        target = Path(path)
        raw = json.loads(target.read_text())

        lessons_list = raw["lessons"]
        position = raw["position"]
        curriculum_file = raw.get("curriculum_file")

        curriculum = cls._load_curriculum(lessons_list, position, curriculum_file)

        state = cls(curriculum, save_path=target, curriculum_file=curriculum_file)
        state.submitted = _deserialize_set(raw.get("submitted", []))
        state.satisfied = _deserialize_set(raw.get("satisfied", []))
        state.pending = _deserialize_set(raw.get("pending", []))
        state.event_log = raw.get("event_log", [])

        state.lesson_submitted = set(raw.get("lesson_submitted", []))
        state.lesson_satisfied = set(raw.get("lesson_satisfied", []))

        return state

    @staticmethod
    def _load_curriculum(
        lessons_list: list[str],
        position: int,
        curriculum_file: str | None,
    ) -> Curriculum:
        """Construct a Curriculum from saved data.

        If ``curriculum_file`` is present and the file exists, load the
        CurriculumDocument from it. Otherwise, create a Curriculum from
        the flat lessons list (legacy compat).
        """
        if curriculum_file:
            file_path = Path(curriculum_file)
            if file_path.exists():
                try:
                    doc = CurriculumDocument.from_file(file_path)
                    return Curriculum(doc, position=position)
                except (CurriculumParseError, OSError) as exc:
                    logger.warning(
                        "Failed to load curriculum from %s, falling back to flat list: %s",
                        curriculum_file,
                        exc,
                    )
        return Curriculum(lessons_list, position=position)


# Serialization helpers


def _serialize_set(s: set[EntryKey]) -> list[list]:
    """Convert a set of EntryKey tuples to a JSON-serializable list.

    Each ``(sig, (nodes...))`` becomes ``[sig, [nodes...]]``.
    """
    return [[sig, list(nodes)] for sig, nodes in sorted(s)]


def _deserialize_set(items: list[list]) -> set[EntryKey]:
    """Convert a JSON list back to a set of EntryKey tuples.

    Each ``[sig, [nodes...]]`` becomes ``(sig, tuple(nodes))``.
    """
    return {(item[0], tuple(item[1])) for item in items}
