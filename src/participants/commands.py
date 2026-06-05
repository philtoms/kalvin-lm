"""Shared command parser for supervisor participants.

Maps human free-text input to structured commands, which are then dispatched
as bus messages. Used by both TUI and Slack participants to interpret user
input uniformly.

Spec ref: specs/harness-server.md §Shared Command Protocol (HRNS-32)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from harness.constants import TRAINEE_ROLE, TRAINER_ROLE


@dataclass
class Command(ABC):
    """Base class for all supervisor commands.

    Each command stores the original input text so it can be re-dispatched
    verbatim on the bus.
    """

    original_text: str

    @abstractmethod
    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        """Return bus messages as (role, action, payload) tuples.

        Args:
            latest_proposal: The most recent pending proposal from the Trainer,
                used only by RatifyCommand. May be None.

        Returns:
            List of (role, action, payload) tuples ready for bus dispatch.
        """
        ...


@dataclass
class StartCommand(Command):
    """Session start command."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class StopCommand(Command):
    """Session stop command."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class PauseCommand(Command):
    """Session pause command."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class ResumeCommand(Command):
    """Session resume command."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class GoalCommand(Command):
    """Set a training goal."""

    text: str

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class RatifyCommand(Command):
    """Countersign the latest pending proposal."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        if latest_proposal is None:
            return []
        return [(TRAINEE_ROLE, "countersign", latest_proposal)]


@dataclass
class FileGoalCommand(Command):
    """Submit a file path as a training goal."""

    path: str

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class GuidanceCommand(Command):
    """Freeform guidance for the trainer."""

    text: str

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


def parse_command(text: str) -> Command:
    """Parse free-text input into a structured Command.

    Recognition rules (evaluated in order):
        - "start" (case-insensitive, stripped) → StartCommand
        - "stop" (case-insensitive, stripped) → StopCommand
        - "pause" (case-insensitive, stripped) → PauseCommand
        - "resume" (case-insensitive, stripped) → ResumeCommand
        - starts with "goal:" or "goal " (case-insensitive) → GoalCommand
        - equals "ratify" (case-insensitive, stripped) → RatifyCommand
        - looks like a file path → FileGoalCommand
        - everything else → GuidanceCommand

    Args:
        text: Raw user input string.

    Returns:
        The parsed Command instance.
    """
    original = text
    stripped = text.strip()
    lower = stripped.lower()

    # Session control keywords
    if lower == "start":
        return StartCommand(original_text=original)
    if lower == "stop":
        return StopCommand(original_text=original)
    if lower == "pause":
        return PauseCommand(original_text=original)
    if lower == "resume":
        return ResumeCommand(original_text=original)

    # Goal prefix
    if lower.startswith("goal:") or lower.startswith("goal "):
        # Extract the goal text after "goal:" or "goal "
        goal_text = stripped[5:].strip()
        return GoalCommand(original_text=original, text=goal_text)

    # Ratify keyword
    if lower == "ratify":
        return RatifyCommand(original_text=original)

    # File path heuristic
    if _looks_like_file_path(stripped):
        return FileGoalCommand(original_text=original, path=stripped)

    # Default: freeform guidance
    return GuidanceCommand(original_text=original, text=original)


def _looks_like_file_path(text: str) -> bool:
    """Heuristic to detect file path inputs.

    Matches paths that:
        - end with .md
        - start with / or ./
        - contain a path separator with a file extension
    """
    if text.endswith(".md"):
        return True
    if text.startswith("/") or text.startswith("./"):
        return True
    if re.search(r"[/\\].+\.[a-zA-Z0-9]+$", text):
        return True
    return False
