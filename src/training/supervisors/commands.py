"""Shared command parser for supervisors.

Maps supervisor free-text input to structured commands, which are then dispatched
as bus messages. Used by both the TUI and Slack supervisors to interpret user
input uniformly.

Spec ref: specs/harness-server.md §Shared Command Protocol (HRNS-32).
The ``scaffold`` command (reactive scaffolding) is defined in
specs/reactive-delegation.md §Scaffold Command.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from training.harness.constants import TRAINEE_ROLE, TRAINER_ROLE


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
class RestartCommand(Command):
    """Clear training state and restart session from the beginning."""

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class SaveCommand(Command):
    """Persist Kalvin's model to disk via agent_codec."""

    path: str | None = None

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINEE_ROLE, "save", self.path)]


@dataclass
class LoadCommand(Command):
    """Load Kalvin's model from disk via agent_codec."""

    path: str | None = None

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINEE_ROLE, "load", self.path)]


@dataclass
class GoalCommand(Command):
    """Set a training goal."""

    text: str

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        return [(TRAINER_ROLE, "input", self.original_text)]


@dataclass
class RatifyCommand(Command):
    """Accept the latest pending proposal — a supervisor decision.

    Routes to the ``trainer`` role as a ``supervisor_decision`` so the Trainer
    applies the countersign itself after replaying any held events
    (`@specs/supervisor-decision.md` SD-9). The proposal is carried verbatim
    so the Trainer countersigns the exact kline Kalvin proposed.
    """

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        if latest_proposal is None:
            return []
        return [
            (
                TRAINER_ROLE,
                "supervisor_decision",
                {"decision": "ratify", "proposal": latest_proposal},
            )
        ]


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


@dataclass
class ScaffoldCommand(Command):
    """Submit reactive scaffolding (KScript) — a supervisor decision.

    When a proposal is pending, routes to the ``trainer`` role as a
    ``supervisor_decision`` so the Trainer applies the scaffold answer
    (`@specs/supervisor-decision.md` SD-10); the KScript is then submitted to
    Kalvin and compiled like any lesson submission (compile failures surface
    as ``error`` events — SD-12).

    When no proposal is pending, this is a free submission to Kalvin (not a
    decision answer) and goes directly to the ``trainee`` ``submit`` action.
    """

    text: str

    def to_messages(self, latest_proposal: Any) -> list[tuple[str, str, Any]]:
        if latest_proposal is not None:
            return [
                (
                    TRAINER_ROLE,
                    "supervisor_decision",
                    {
                        "decision": "scaffold",
                        "proposal": latest_proposal,
                        "text": self.text,
                    },
                )
            ]
        return [(TRAINEE_ROLE, "submit", self.text)]


def parse_command(text: str) -> Command:
    """Parse free-text input into a structured Command.

    Recognition rules (evaluated in order):
        - "start" (case-insensitive, stripped) → StartCommand
        - "stop" (case-insensitive, stripped) → StopCommand
        - "pause" (case-insensitive, stripped) → PauseCommand
        - "resume" (case-insensitive, stripped) → ResumeCommand
        - "restart" (case-insensitive, stripped) → RestartCommand
        - "save" or "save:<path>" (case-insensitive) → SaveCommand
        - "load" or "load:<path>" (case-insensitive) → LoadCommand
        - starts with "goal:" or "goal " (case-insensitive) → GoalCommand
        - equals "ratify" (case-insensitive, stripped) → RatifyCommand
        - starts with "scaffold:" or "scaffold " (case-insensitive) → ScaffoldCommand
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
    if lower == "restart":
        return RestartCommand(original_text=original)

    # Model persistence
    if lower == "save" or lower.startswith("save:") or lower.startswith("save "):
        save_path = stripped[5:].strip() if len(stripped) > 4 else None
        return SaveCommand(original_text=original, path=save_path or None)
    if lower == "load" or lower.startswith("load:") or lower.startswith("load "):
        load_path = stripped[5:].strip() if len(stripped) > 4 else None
        return LoadCommand(original_text=original, path=load_path or None)

    # Goal prefix
    if lower.startswith("goal:") or lower.startswith("goal "):
        goal_text = stripped[5:].strip()
        return GoalCommand(original_text=original, text=goal_text)

    # Ratify keyword
    if lower == "ratify":
        return RatifyCommand(original_text=original)

    # Scaffold prefix (KScript source). Placed before the file-path heuristic so
    # that multi-line KScript (which may contain "/" and ".") is not
    # misclassified as a file path.
    if lower.startswith("scaffold:") or lower.startswith("scaffold "):
        scaffold_text = stripped[9:].strip()
        return ScaffoldCommand(original_text=original, text=scaffold_text)

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
