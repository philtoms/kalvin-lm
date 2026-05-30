"""Curriculum data model and persistence for the Trainer.

Provides the ordered lesson list (Curriculum) and per-session tracking
state (CurriculumState) including submitted/satisfied/pending sets and
JSON persistence for restart recovery.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

# Type alias for entry identity keys used in tracking sets.
# Mirrors the pattern from ui/kscript/app.py line 23.
EntryKey: TypeAlias = tuple[int, tuple[int, ...]]


class Curriculum:
    """Ordered list of KScript source strings (lessons) with position tracking.

    Parameters
    ----------
    lessons:
        Ordered list of KScript source strings.
    position:
        Current position in the lesson list (0-indexed).
    """

    def __init__(self, lessons: list[str], *, position: int = 0) -> None:
        self.lessons: list[str] = list(lessons)
        self.position: int = position

    def current(self) -> str | None:
        """Return the current lesson source, or ``None`` if curriculum is complete."""
        if self.position >= len(self.lessons):
            return None
        return self.lessons[self.position]

    def advance(self) -> None:
        """Move to the next lesson in the curriculum."""
        self.position += 1

    def is_complete(self) -> bool:
        """Return ``True`` if all lessons have been consumed."""
        return self.position >= len(self.lessons)

    def total(self) -> int:
        """Return the total number of lessons."""
        return len(self.lessons)

    def remaining(self) -> int:
        """Return the number of lessons remaining (including current)."""
        return max(0, len(self.lessons) - self.position)


class CurriculumState:
    """Per-session tracking state for the Trainer.

    Holds the curriculum, per-entry tracking sets (submitted, satisfied,
    pending), and an append-only event log. Supports JSON persistence
    for restart recovery.

    Parameters
    ----------
    curriculum:
        The :class:`Curriculum` instance to track.
    save_path:
        Optional default path for :meth:`save` / :meth:`load`.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        *,
        save_path: str | Path | None = None,
    ) -> None:
        self.curriculum: Curriculum = curriculum
        self.submitted: set[EntryKey] = set()
        self.satisfied: set[EntryKey] = set()
        self.pending: set[EntryKey] = set()
        self.event_log: list[dict] = []
        self._save_path: Path | None = Path(save_path) if save_path else None

    # ── Set operations ────────────────────────────────────────────────

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

    # ── Event log ─────────────────────────────────────────────────────

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

    # ── Session reset ─────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Clear submitted/satisfied/pending sets but preserve curriculum
        position and the append-only event log.
        """
        self.submitted.clear()
        self.satisfied.clear()
        self.pending.clear()

    # ── Persistence ───────────────────────────────────────────────────

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
            "position": self.curriculum.position,
            "lessons": self.curriculum.lessons,
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

        Returns a fully reconstructed :class:`CurriculumState`.
        """
        target = Path(path)
        raw = json.loads(target.read_text())

        lessons = raw["lessons"]
        position = raw["position"]
        curriculum = Curriculum(lessons, position=position)

        state = cls(curriculum, save_path=target)
        state.submitted = _deserialize_set(raw.get("submitted", []))
        state.satisfied = _deserialize_set(raw.get("satisfied", []))
        state.pending = _deserialize_set(raw.get("pending", []))
        state.event_log = raw.get("event_log", [])

        return state


# ── Serialization helpers ─────────────────────────────────────────────


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
