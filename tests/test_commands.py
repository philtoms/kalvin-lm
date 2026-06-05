"""Tests for the shared command parser (HRNS-32).

Covers all command types, case insensitivity, whitespace handling,
and edge cases for parse_command and to_messages.
"""

from __future__ import annotations

from harness.constants import TRAINEE_ROLE, TRAINER_ROLE
from participants.commands import (
    FileGoalCommand,
    GoalCommand,
    GuidanceCommand,
    LoadCommand,
    PauseCommand,
    RatifyCommand,
    RestartCommand,
    ResumeCommand,
    SaveCommand,
    StartCommand,
    StopCommand,
    parse_command,
)

# ---------------------------------------------------------------------------
# Session control commands
# ---------------------------------------------------------------------------


class TestStartCommand:
    def test_parse(self):
        cmd = parse_command("start")
        assert isinstance(cmd, StartCommand)
        assert cmd.original_text == "start"

    def test_to_messages(self):
        cmd = parse_command("start")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "start")]


class TestStopCommand:
    def test_parse(self):
        cmd = parse_command("stop")
        assert isinstance(cmd, StopCommand)
        assert cmd.original_text == "stop"

    def test_to_messages(self):
        cmd = parse_command("stop")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "stop")]


class TestPauseCommand:
    def test_parse(self):
        cmd = parse_command("pause")
        assert isinstance(cmd, PauseCommand)
        assert cmd.original_text == "pause"

    def test_to_messages(self):
        cmd = parse_command("pause")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "pause")]


class TestResumeCommand:
    def test_parse(self):
        cmd = parse_command("resume")
        assert isinstance(cmd, ResumeCommand)
        assert cmd.original_text == "resume"

    def test_to_messages(self):
        cmd = parse_command("resume")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "resume")]


# ---------------------------------------------------------------------------
# Goal command
# ---------------------------------------------------------------------------


class TestGoalCommand:
    def test_parse_with_colon(self):
        cmd = parse_command("goal: learn SVO")
        assert isinstance(cmd, GoalCommand)
        assert cmd.text == "learn SVO"
        assert cmd.original_text == "goal: learn SVO"

    def test_parse_with_space(self):
        cmd = parse_command("goal learn SVO")
        assert isinstance(cmd, GoalCommand)
        assert cmd.text == "learn SVO"

    def test_to_messages(self):
        cmd = parse_command("goal: learn SVO")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "goal: learn SVO")]

    def test_preserves_original_text_in_to_messages(self):
        """to_messages carries the full original input, not just the goal text."""
        cmd = parse_command("goal: learn SVO")
        msgs = cmd.to_messages(None)
        assert len(msgs) == 1
        assert msgs[0][2] == "goal: learn SVO"


# ---------------------------------------------------------------------------
# Ratify command
# ---------------------------------------------------------------------------


class TestRatifyCommand:
    def test_parse(self):
        cmd = parse_command("ratify")
        assert isinstance(cmd, RatifyCommand)
        assert cmd.original_text == "ratify"

    def test_to_messages_with_proposal(self):
        cmd = parse_command("ratify")
        proposal = {"proposal": "MHALL = SVO"}
        msgs = cmd.to_messages(proposal)
        assert msgs == [(TRAINEE_ROLE, "countersign", {"proposal": "MHALL = SVO"})]

    def test_to_messages_without_proposal(self):
        cmd = parse_command("ratify")
        msgs = cmd.to_messages(None)
        assert msgs == []


# ---------------------------------------------------------------------------
# File goal command
# ---------------------------------------------------------------------------


class TestFileGoalCommand:
    def test_parse_md_extension(self):
        cmd = parse_command("lesson.md")
        assert isinstance(cmd, FileGoalCommand)
        assert cmd.path == "lesson.md"
        assert cmd.original_text == "lesson.md"

    def test_to_messages(self):
        cmd = parse_command("lesson.md")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "lesson.md")]

    def test_parse_absolute_path(self):
        cmd = parse_command("/path/to/lesson.md")
        assert isinstance(cmd, FileGoalCommand)
        assert cmd.path == "/path/to/lesson.md"

    def test_parse_relative_path(self):
        cmd = parse_command("./lessons/svo.md")
        assert isinstance(cmd, FileGoalCommand)
        assert cmd.path == "./lessons/svo.md"

    def test_parse_path_with_separator_and_extension(self):
        cmd = parse_command("lessons/svo.txt")
        assert isinstance(cmd, FileGoalCommand)


# ---------------------------------------------------------------------------
# Guidance command (default fallback)
# ---------------------------------------------------------------------------


class TestGuidanceCommand:
    def test_parse(self):
        cmd = parse_command("try a different approach")
        assert isinstance(cmd, GuidanceCommand)
        assert cmd.text == "try a different approach"
        assert cmd.original_text == "try a different approach"

    def test_to_messages(self):
        cmd = parse_command("try a different approach")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "try a different approach")]


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    def test_start_uppercase(self):
        assert isinstance(parse_command("START"), StartCommand)

    def test_stop_mixed(self):
        assert isinstance(parse_command("Stop"), StopCommand)

    def test_ratify_uppercase(self):
        assert isinstance(parse_command("RATIFY"), RatifyCommand)

    def test_goal_uppercase(self):
        cmd = parse_command("GOAL: something")
        assert isinstance(cmd, GoalCommand)
        assert cmd.text == "something"


# ---------------------------------------------------------------------------
# Whitespace handling
# ---------------------------------------------------------------------------


class TestWhitespace:
    def test_leading_trailing_whitespace(self):
        cmd = parse_command("  start  ")
        assert isinstance(cmd, StartCommand)

    def test_whitespace_preserved_in_original(self):
        cmd = parse_command("  start  ")
        assert cmd.original_text == "  start  "

    def test_goal_with_extra_whitespace(self):
        cmd = parse_command("  goal:   learn SVO  ")
        assert isinstance(cmd, GoalCommand)
        assert cmd.text == "learn SVO"


# ---------------------------------------------------------------------------
# Restart command
# ---------------------------------------------------------------------------


class TestRestartCommand:
    def test_parse(self):
        cmd = parse_command("restart")
        assert isinstance(cmd, RestartCommand)
        assert cmd.original_text == "restart"

    def test_to_messages(self):
        cmd = parse_command("restart")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINER_ROLE, "input", "restart")]

    def test_case_insensitive(self):
        assert isinstance(parse_command("RESTART"), RestartCommand)
        assert isinstance(parse_command("Restart"), RestartCommand)

    def test_whitespace(self):
        cmd = parse_command("  restart  ")
        assert isinstance(cmd, RestartCommand)


# ---------------------------------------------------------------------------
# Save command
# ---------------------------------------------------------------------------


class TestSaveCommand:
    def test_parse_bare(self):
        cmd = parse_command("save")
        assert isinstance(cmd, SaveCommand)
        assert cmd.path is None

    def test_parse_with_path(self):
        cmd = parse_command("save:data/my-model.bin")
        assert isinstance(cmd, SaveCommand)
        assert cmd.path == "data/my-model.bin"

    def test_parse_with_space(self):
        cmd = parse_command("save data/backup.json")
        assert isinstance(cmd, SaveCommand)
        assert cmd.path == "data/backup.json"

    def test_to_messages_routes_to_trainee(self):
        cmd = parse_command("save")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINEE_ROLE, "save", None)]

    def test_to_messages_with_path(self):
        cmd = parse_command("save:data/model.bin")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINEE_ROLE, "save", "data/model.bin")]

    def test_case_insensitive(self):
        assert isinstance(parse_command("SAVE"), SaveCommand)
        assert isinstance(parse_command("Save"), SaveCommand)


# ---------------------------------------------------------------------------
# Load command
# ---------------------------------------------------------------------------


class TestLoadCommand:
    def test_parse_bare(self):
        cmd = parse_command("load")
        assert isinstance(cmd, LoadCommand)
        assert cmd.path is None

    def test_parse_with_path(self):
        cmd = parse_command("load:data/my-model.bin")
        assert isinstance(cmd, LoadCommand)
        assert cmd.path == "data/my-model.bin"

    def test_parse_with_space(self):
        cmd = parse_command("load data/backup.json")
        assert isinstance(cmd, LoadCommand)
        assert cmd.path == "data/backup.json"

    def test_to_messages_routes_to_trainee(self):
        cmd = parse_command("load")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINEE_ROLE, "load", None)]

    def test_to_messages_with_path(self):
        cmd = parse_command("load:data/model.bin")
        msgs = cmd.to_messages(None)
        assert msgs == [(TRAINEE_ROLE, "load", "data/model.bin")]

    def test_case_insensitive(self):
        assert isinstance(parse_command("LOAD"), LoadCommand)
        assert isinstance(parse_command("Load"), LoadCommand)
