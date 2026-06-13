"""Tests for the auto-tune CLI (``python -m participants.auto_tune``).

Covers:
- build_parser() returns ArgumentParser with all 12 subcommands
- init subcommand parses required and optional arguments
- init handler calls SessionDir.init with correct arguments
- Non-init subcommands parse their arguments
- Non-init stubs raise NotImplementedError
- main() with no subcommand exits with error
- main() for non-init on missing session prints error
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from participants.auto_tune.cli import build_parser, main

# ---------------------------------------------------------------------------
# build_parser structure
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Verify parser structure — all 12 subcommands present."""

    EXPECTED_SUBCOMMANDS = [
        "init",
        "start-harness",
        "stop-harness",
        "start-supervisor",
        "stop-supervisor",
        "send",
        "events",
        "step",
        "status",
        "snapshot",
        "restore",
        "reset",
        "teardown",
    ]

    def test_returns_argument_parser(self) -> None:
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_has_all_thirteen_subcommands(self) -> None:
        parser = build_parser()
        # Parse the help output to find subcommand names
        # Or introspect the subparsers action
        subparsers_action = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
                break
        assert subparsers_action is not None
        names = set(subparsers_action._name_parser_map.keys())
        for name in self.EXPECTED_SUBCOMMANDS:
            assert name in names, f"Missing subcommand: {name}"
        assert len(names) == 13


# ---------------------------------------------------------------------------
# init subcommand parsing
# ---------------------------------------------------------------------------


class TestInitParsing:
    """init parses --session, --curriculum, and optional --host/--port."""

    def test_required_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["init", "--session", "s1", "--curriculum", "c.md"])
        assert args.command == "init"
        assert args.session == "s1"
        assert args.curriculum == "c.md"
        assert args.host is None
        assert args.port is None

    def test_all_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "init",
                "--session",
                "s1",
                "--curriculum",
                "c.md",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
            ]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9999

    def test_missing_session_fails(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["init", "--curriculum", "c.md"])

    def test_missing_curriculum_fails(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["init", "--session", "s1"])


# ---------------------------------------------------------------------------
# init handler calls SessionDir.init
# ---------------------------------------------------------------------------


class TestInitHandler:
    """init handler delegates to SessionDir.init."""

    @patch("participants.auto_tune.cli.SessionDir")
    def test_calls_init_with_correct_args(self, mock_session_dir: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_session_dir.init.return_value = mock_instance

        parser = build_parser()
        args = parser.parse_args(
            [
                "init",
                "--session",
                "s1",
                "--curriculum",
                "c.md",
                "--host",
                "myhost",
                "--port",
                "4321",
            ]
        )
        args.func(args)

        mock_session_dir.init.assert_called_once_with(
            session="s1",
            curriculum="c.md",
            host="myhost",
            port=4321,
        )

    @patch("participants.auto_tune.cli.SessionDir")
    def test_calls_init_with_defaults(self, mock_session_dir: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_session_dir.init.return_value = mock_instance

        parser = build_parser()
        args = parser.parse_args(["init", "--session", "s1", "--curriculum", "c.md"])
        args.func(args)

        mock_session_dir.init.assert_called_once_with(
            session="s1",
            curriculum="c.md",
            host=None,
            port=None,
        )


# ---------------------------------------------------------------------------
# Non-init subcommand argument parsing
# ---------------------------------------------------------------------------


class TestNonInitParsing:
    """Each non-init subcommand parses --session and any extra arguments."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "start-harness",
            "stop-harness",
            "start-supervisor",
            "stop-supervisor",
            "status",
            "snapshot",
        ],
    )
    def test_session_only_commands(self, cmd: str) -> None:
        parser = build_parser()
        args = parser.parse_args([cmd, "--session", "s1"])
        assert args.command == cmd
        assert args.session == "s1"

    def test_send_parses_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["send", "--session", "s1", "--command", '{"action":"start"}'])
        assert args.command == "send"
        assert args.session == "s1"
        assert args.command_json == '{"action":"start"}'

    def test_events_parses_after_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["events", "--session", "s1"])
        assert args.after == 0

    def test_events_parses_after_value(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["events", "--session", "s1", "--after", "42"])
        assert args.after == 42

    def test_step_parses_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["step", "--session", "s1", "--command", '{"action":"continue"}'])
        assert args.command == "step"
        assert args.session == "s1"
        assert args.command_json == '{"action":"continue"}'

    def test_restore_parses_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["restore", "--session", "s1", "--run", "3"])
        assert args.run == 3

    def test_reset_parses_fresh_model_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["reset", "--session", "s1", "--fresh-model"])
        assert args.fresh_model is True

    def test_reset_fresh_model_defaults_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["reset", "--session", "s1"])
        assert args.fresh_model is False


# ---------------------------------------------------------------------------
# Non-init handlers delegate to handler modules
# ---------------------------------------------------------------------------


class TestHandlerDelegation:
    """All non-init handlers delegate to the correct handler module."""

    @patch("participants.auto_tune.cli.lifecycle")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_start_harness_delegates(self, mock_sd: MagicMock, mock_lifecycle: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["start-harness", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_lifecycle.start_harness.assert_called_once_with("/auto-tune/s1")

    @patch("participants.auto_tune.cli.lifecycle")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_stop_harness_delegates(self, mock_sd: MagicMock, mock_lifecycle: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["stop-harness", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_lifecycle.stop_harness.assert_called_once_with("/auto-tune/s1")

    @patch("participants.auto_tune.cli.lifecycle")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_start_supervisor_delegates(
        self, mock_sd: MagicMock, mock_lifecycle: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["start-supervisor", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_lifecycle.start_supervisor.assert_called_once_with("/auto-tune/s1")

    @patch("participants.auto_tune.cli.lifecycle")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_stop_supervisor_delegates(self, mock_sd: MagicMock, mock_lifecycle: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/s1"
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["stop-supervisor", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_lifecycle.stop_supervisor.assert_called_once_with("/auto-tune/s1")

    @patch("participants.auto_tune.cli.orchestrate")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_send_delegates(self, mock_sd: MagicMock, mock_orch: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["send", "--session", "s1", "--command", '{"action":"start"}'])
        args._session_dir = mock_instance
        args.func(args)
        mock_orch.send_command.assert_called_once_with(mock_instance, {"action": "start"})

    @patch("participants.auto_tune.cli.orchestrate")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_events_delegates(self, mock_sd: MagicMock, mock_orch: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        mock_orch.read_events.return_value = [{"seq": 1, "type": "connected"}]
        parser = build_parser()
        args = parser.parse_args(["events", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_orch.read_events.assert_called_once_with(mock_instance, 0)

    @patch("participants.auto_tune.cli.orchestrate")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_step_delegates(self, mock_sd: MagicMock, mock_orch: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        mock_orch.step.return_value = [{"seq": 2, "type": "progress"}]
        parser = build_parser()
        args = parser.parse_args(["step", "--session", "s1", "--command", '{"action":"continue"}'])
        args._session_dir = mock_instance
        args.func(args)
        mock_orch.step.assert_called_once_with(mock_instance, {"action": "continue"})

    @patch("participants.auto_tune.cli.orchestrate")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_status_delegates(self, mock_sd: MagicMock, mock_orch: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        mock_orch.read_status.return_value = {"state": "waiting_for_event"}
        parser = build_parser()
        args = parser.parse_args(["status", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_orch.read_status.assert_called_once_with(mock_instance)

    @patch("participants.auto_tune.cli.snapshots")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_snapshot_delegates(self, mock_sd: MagicMock, mock_snap: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["snapshot", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_snap.snapshot.assert_called_once_with(mock_instance)

    @patch("participants.auto_tune.cli.snapshots")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_restore_delegates(self, mock_sd: MagicMock, mock_snap: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["restore", "--session", "s1", "--run", "3"])
        args._session_dir = mock_instance
        args.func(args)
        mock_snap.restore.assert_called_once_with(mock_instance, 3)

    @patch("participants.auto_tune.cli.snapshots")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_reset_delegates(self, mock_sd: MagicMock, mock_snap: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["reset", "--session", "s1"])
        args._session_dir = mock_instance
        args.func(args)
        mock_snap.reset.assert_called_once_with(mock_instance, fresh_model=False)

    @patch("participants.auto_tune.cli.snapshots")
    @patch("participants.auto_tune.cli.SessionDir")
    def test_reset_fresh_model_delegates(self, mock_sd: MagicMock, mock_snap: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_sd.load.return_value = mock_instance
        parser = build_parser()
        args = parser.parse_args(["reset", "--session", "s1", "--fresh-model"])
        args._session_dir = mock_instance
        args.func(args)
        mock_snap.reset.assert_called_once_with(mock_instance, fresh_model=True)


# ---------------------------------------------------------------------------
# main() behaviour
# ---------------------------------------------------------------------------


class TestMain:
    """main() dispatches and validates correctly."""

    def test_no_subcommand_exits_with_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    @patch("participants.auto_tune.cli.SessionDir")
    def test_non_init_missing_session_exits(self, mock_sd: MagicMock) -> None:
        mock_sd.load.side_effect = FileNotFoundError("nope")
        with pytest.raises(SystemExit) as exc_info:
            main(["status", "--session", "nonexistent"])
        assert exc_info.value.code == 1

    @patch("participants.auto_tune.cli.SessionDir")
    def test_init_calls_session_dir_init(self, mock_sd: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.config_path = MagicMock()
        mock_instance.config_path.parent = "/auto-tune/test"
        mock_sd.init.return_value = mock_instance

        main(["init", "--session", "test", "--curriculum", "foo.md"])

        mock_sd.init.assert_called_once_with(
            session="test",
            curriculum="foo.md",
            host=None,
            port=None,
        )


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------


class TestMainModule:
    """``python -m participants.auto_tune --help`` works."""

    def test_help_lists_all_subcommands(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")}
        result = subprocess.run(
            [sys.executable, "-m", "participants.auto_tune", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        for cmd in TestBuildParser.EXPECTED_SUBCOMMANDS:
            assert cmd in result.stdout, f"Subcommand {cmd!r} missing from --help"
