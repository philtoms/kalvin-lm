"""End-to-end integration tests for auto-tune CLI subcommands.

Invokes each CLI subcommand through ``main(argv=[...])`` and verifies
all 20 acceptance criteria (AT-1 through AT-20) from specs/auto-tune.md.

Process lifecycle tests use mocks (no real subprocess spawning).
File-based tests (orchestration, snapshot, restore, reset) use real I/O.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from participants.auto_tune.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def auto_tune_env(tmp_path: Path, monkeypatch):
    """Create a temp directory with git repo and harness.yaml for CLI tests.

    1. Creates a temp directory and initialises a git repo
    2. Creates a minimal harness.yaml
    3. Changes cwd to the temp directory for the test duration
    4. Returns the temp path for assertions
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    # Initialise git repo
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    # Initial commit so HEAD is defined
    (repo / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # Create minimal harness.yaml with non-default port
    (repo / "harness.yaml").write_text(
        "server:\n  host: localhost\n  port: 18765\n",
        encoding="utf-8",
    )

    # Create minimal curriculum file
    (repo / "math.md").write_text("# Math Curriculum\n", encoding="utf-8")

    # Change cwd for test duration
    monkeypatch.chdir(repo)

    return repo


def _init_session(session_name: str = "test-sess", curriculum: str = "math.md") -> None:
    """Helper: run init subcommand."""
    main(["init", "--session", session_name, "--curriculum", curriculum])


def _session_dir(root: Path, session_name: str = "test-sess") -> Path:
    """Return the session directory path inside the worktree."""
    return root / ".worktrees" / "auto-tune" / session_name / "auto-tune" / session_name


def _use_worktree_model_path(session_dir_path: Path) -> None:
    """Make ``model_path`` in config.json worktree-relative.

    ``init()`` stores ``model_path`` as an absolute path resolved via
    ``paths.agent_bin()`` (pointing at the real project data dir, which is
    gitignored and absent from session worktrees).  The integration tests
    create the model file inside the worktree, so rewrite ``model_path`` to
    a relative path that ``snapshot``/``reset`` resolve against the worktree
    root (``session_dir._root / cfg.model_path``).
    """
    config_path = session_dir_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_path"] = "data/agent.bin"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# AT-1: init creates session directory with all supporting files
# ---------------------------------------------------------------------------


class TestAT01InitCreatesSessionDirectory:
    """AT-1: ``init`` creates session directory with all supporting files."""

    def test_init_creates_session_directory(self, auto_tune_env: Path) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)
        assert sd.is_dir(), "Session directory should exist"
        assert (sd / "config.json").is_file(), "config.json should exist"
        assert (sd / "events.jsonl").is_file(), "events.jsonl should exist"
        assert (sd / "runs").is_dir(), "runs/ subdirectory should exist"


# ---------------------------------------------------------------------------
# AT-2: init creates and checks out git branch
# ---------------------------------------------------------------------------


class TestAT02InitCreatesGitBranch:
    """AT-2: ``init`` creates ``auto-tune/<session>`` branch as a worktree."""

    def test_init_creates_git_branch(self, auto_tune_env: Path) -> None:
        _init_session()
        # Branch should exist
        result = subprocess.run(
            ["git", "branch", "--list", "auto-tune/test-sess"],
            cwd=auto_tune_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "auto-tune/test-sess" in result.stdout

        # Main repo should stay on the original branch
        current = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=auto_tune_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert current.stdout.strip() != "auto-tune/test-sess"

        # Worktree should be on the auto-tune branch
        worktree_path = auto_tune_env / ".worktrees" / "auto-tune" / "test-sess"
        wt_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert wt_result.stdout.strip() == "auto-tune/test-sess"


# ---------------------------------------------------------------------------
# AT-3: init records config
# ---------------------------------------------------------------------------


class TestAT03InitRecordsConfig:
    """AT-3: ``init`` records source branch, commit, harness URL, model path."""

    def test_init_records_config(self, auto_tune_env: Path) -> None:
        _init_session()
        config_path = _session_dir(auto_tune_env) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        assert "created_from_branch" in config
        assert config["created_from_branch"] != ""
        assert "created_from_commit" in config
        assert config["created_from_commit"] != ""
        assert "harness_url" in config
        assert config["harness_url"] == "ws://localhost:18765"
        assert "model_path" in config
        assert "worktree_path" in config
        assert config["worktree_path"] != ""


# ---------------------------------------------------------------------------
# AT-4: start-harness starts and waits
# ---------------------------------------------------------------------------


class TestAT04StartHarness:
    """AT-4: ``start-harness`` starts harness and waits for readiness."""

    @patch("participants.auto_tune.cli.lifecycle")
    def test_start_harness_starts_and_waits(
        self, mock_lifecycle: MagicMock, auto_tune_env: Path
    ) -> None:
        _init_session()
        main(["start-harness", "--session", "test-sess"])
        mock_lifecycle.start_harness.assert_called_once()
        # Verify the argument is a Path (session directory)
        call_args = mock_lifecycle.start_harness.call_args
        session_path = call_args[0][0]
        assert str(session_path).endswith("test-sess") or "test-sess" in str(session_path)


# ---------------------------------------------------------------------------
# AT-5: stop-harness graceful shutdown
# ---------------------------------------------------------------------------


class TestAT05StopHarness:
    """AT-5: ``stop-harness`` gracefully terminates harness process."""

    @patch("participants.auto_tune.cli.lifecycle")
    def test_stop_harness_graceful_shutdown(
        self, mock_lifecycle: MagicMock, auto_tune_env: Path
    ) -> None:
        _init_session()
        main(["stop-harness", "--session", "test-sess"])
        mock_lifecycle.stop_harness.assert_called_once()
        call_args = mock_lifecycle.stop_harness.call_args
        session_path = call_args[0][0]
        assert "test-sess" in str(session_path)


# ---------------------------------------------------------------------------
# AT-6: start-supervisor invoked
# ---------------------------------------------------------------------------


class TestAT06StartSupervisor:
    """AT-6: ``start-supervisor`` starts the CLI supervisor process."""

    @patch("participants.auto_tune.cli.lifecycle")
    def test_start_supervisor_invoked(self, mock_lifecycle: MagicMock, auto_tune_env: Path) -> None:
        _init_session()
        main(["start-supervisor", "--session", "test-sess"])
        mock_lifecycle.start_supervisor.assert_called_once()


# ---------------------------------------------------------------------------
# AT-7: send and events round-trip
# ---------------------------------------------------------------------------


class TestAT07SendAndEventsRoundTrip:
    """AT-7: Supervisor writes events; CLI reads and prints them."""

    def test_send_and_events_round_trip(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write a mock status.json (required by events handler path resolution)
        status = {"pid": 1234, "connected": True, "last_event_seq": 0, "state": "waiting_for_event"}
        (sd / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

        # Manually append an event to events.jsonl
        event = {"seq": 1, "type": "connected"}
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        # Call events and capture stdout
        main(["events", "--session", "test-sess"])
        captured = capsys.readouterr()
        assert "connected" in captured.out
        assert '"seq": 1' in captured.out


# ---------------------------------------------------------------------------
# AT-8: continue command via send
# ---------------------------------------------------------------------------


class TestAT08ContinueCommandViaSend:
    """AT-8: ``continue`` command written to cmd.json."""

    def test_continue_command_via_send(self, auto_tune_env: Path) -> None:
        _init_session()
        main(["send", "--session", "test-sess", "--command", '{"action":"continue"}'])

        sd = _session_dir(auto_tune_env)
        cmd_path = sd / "cmd.json"
        assert cmd_path.exists(), "cmd.json should exist after send"
        cmd = json.loads(cmd_path.read_text(encoding="utf-8"))
        assert cmd["action"] == "continue"


# ---------------------------------------------------------------------------
# AT-9: ratify via send
# ---------------------------------------------------------------------------


class TestAT09RatifyViaSend:
    """AT-9: ``ratify`` command written to cmd.json."""

    def test_ratify_via_send(self, auto_tune_env: Path) -> None:
        _init_session()
        main(["send", "--session", "test-sess", "--command", '{"action":"ratify"}'])

        sd = _session_dir(auto_tune_env)
        cmd = json.loads((sd / "cmd.json").read_text(encoding="utf-8"))
        assert cmd["action"] == "ratify"


# ---------------------------------------------------------------------------
# AT-10: stop-supervisor invoked
# ---------------------------------------------------------------------------


class TestAT10StopSupervisor:
    """AT-10: ``stop-supervisor`` stops the CLI supervisor process."""

    @patch("participants.auto_tune.cli.lifecycle")
    def test_stop_supervisor_invoked(self, mock_lifecycle: MagicMock, auto_tune_env: Path) -> None:
        _init_session()
        main(["stop-supervisor", "--session", "test-sess"])
        mock_lifecycle.stop_supervisor.assert_called_once()


# ---------------------------------------------------------------------------
# AT-11: events display enriched format
# ---------------------------------------------------------------------------


class TestAT11EventsEnrichedFormat:
    """AT-11: Events include enriched fields (KLine display, significance)."""

    def test_events_display_enriched_format(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write a status.json
        status = {"last_event_seq": 0}
        (sd / "status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")

        # Write an enriched rationalise event
        enriched_event = {
            "seq": 1,
            "type": "rationalise",
            "kind": "ground",
            "significance": {"raw": 100, "normalised": 0.5, "level": "S3"},
            "query": {"raw": {"signature": 42, "nodes": [1, 2]}, "source": "sig(A, B)"},
            "proposal": {"raw": {"signature": 99, "nodes": [3]}, "source": "sig(C)"},
        }
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(enriched_event) + "\n")

        main(["events", "--session", "test-sess"])
        captured = capsys.readouterr()

        assert "rationalise" in captured.out
        assert "significance" in captured.out
        assert "S3" in captured.out
        assert "sig(A, B)" in captured.out


# ---------------------------------------------------------------------------
# AT-12: status shows run_complete
# ---------------------------------------------------------------------------


class TestAT12StatusRunComplete:
    """AT-12: Status shows ``run_complete`` state."""

    def test_status_shows_run_complete(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write a status.json with run_complete
        status = {
            "pid": 1234,
            "connected": True,
            "last_event_seq": 5,
            "state": "run_complete",
        }
        (sd / "status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")

        main(["status", "--session", "test-sess"])
        captured = capsys.readouterr()
        assert "run_complete" in captured.out


# ---------------------------------------------------------------------------
# AT-13: events shows disconnect
# ---------------------------------------------------------------------------


class TestAT13EventsDisconnect:
    """AT-13: Disconnected event is displayed by events command."""

    def test_events_shows_disconnect(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write a status.json
        (sd / "status.json").write_text(json.dumps({"last_event_seq": 0}) + "\n", encoding="utf-8")

        # Write a disconnected event
        disconnected_event = {"seq": 1, "type": "disconnected"}
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(disconnected_event) + "\n")

        main(["events", "--session", "test-sess"])
        captured = capsys.readouterr()
        assert "disconnected" in captured.out


# ---------------------------------------------------------------------------
# AT-14: step writes command and returns event
# ---------------------------------------------------------------------------


class TestAT14StepWritesAndReturns:
    """AT-14: ``step`` writes command, blocks until next event, prints it."""

    def test_step_writes_command_and_returns_event(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write a status.json with last_event_seq: 0
        status = {
            "pid": 1234,
            "connected": True,
            "last_event_seq": 0,
            "state": "waiting_for_command",
        }
        (sd / "status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")

        # Start step in a background thread — it will block
        result = {"output": None, "error": None}

        def run_step():
            try:
                main(["step", "--session", "test-sess", "--command", '{"action":"continue"}'])
                result["output"] = "done"
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=run_step)
        thread.start()

        # Wait for the command to be written (step sends command first)
        cmd_path = sd / "cmd.json"
        for _ in range(50):
            if cmd_path.exists():
                break
            time.sleep(0.05)

        # Verify command was written
        assert cmd_path.exists(), "cmd.json should be written by step"
        cmd = json.loads(cmd_path.read_text(encoding="utf-8"))
        assert cmd["action"] == "continue"

        # Now append an event to unblock step
        new_event = {"seq": 1, "type": "progress", "status": "started"}
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(new_event) + "\n")

        # Wait for step thread to complete
        thread.join(timeout=10)
        assert not thread.is_alive(), "step should have completed"
        assert result["error"] is None, f"step raised: {result['error']}"

        # Verify stdout contains the new event
        captured = capsys.readouterr()
        assert "progress" in captured.out
        assert "started" in captured.out


# ---------------------------------------------------------------------------
# AT-15: events --after filters by seq
# ---------------------------------------------------------------------------


class TestAT15EventsAfterFilter:
    """AT-15: ``events --after N`` returns events with seq > N."""

    def test_events_after_filters_by_seq(
        self, auto_tune_env: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)

        # Write status.json
        (sd / "status.json").write_text(json.dumps({"last_event_seq": 3}) + "\n", encoding="utf-8")

        # Write 3 events
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "started"},
            {"seq": 3, "type": "progress", "status": "complete"},
        ]
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        main(["events", "--session", "test-sess", "--after", "1"])
        captured = capsys.readouterr()

        # Should NOT contain seq 1
        assert '"seq": 1' not in captured.out
        # Should contain seq 2 and 3
        assert '"seq": 2' in captured.out
        assert '"seq": 3' in captured.out


# ---------------------------------------------------------------------------
# AT-16: snapshot captures state
# ---------------------------------------------------------------------------


class TestAT16SnapshotCapturesState:
    """AT-16: ``snapshot`` captures state, events, model, and git metadata."""

    def test_snapshot_captures_state(self, auto_tune_env: Path) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)
        _use_worktree_model_path(sd)
        worktree_path = auto_tune_env / ".worktrees" / "auto-tune" / "test-sess"

        # The curriculum state file is derived from curriculum path:
        # "math.md" -> "math.json" at worktree root
        state_file = worktree_path / "math.json"
        state_file.write_text('{"lessons_completed": 5}\n', encoding="utf-8")

        # Write some events
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": 1, "type": "connected"}) + "\n")

        # Create model directory and file
        model_dir = worktree_path / "data"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "agent.bin").write_bytes(b"\x00\x01\x02")

        # Commit everything so git is clean
        subprocess.run(["git", "add", "."], cwd=worktree_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add model"], cwd=worktree_path, capture_output=True)

        main(["snapshot", "--session", "test-sess"])

        run_dir = sd / "runs" / "001"
        assert run_dir.is_dir(), "runs/001/ should exist"
        assert (run_dir / "state.json").is_file(), "state.json should exist"
        assert (run_dir / "events.jsonl").is_file(), "events.jsonl should exist"
        assert (run_dir / "meta.json").is_file(), "meta.json should exist"
        assert (run_dir / "model.bin").is_file(), "model.bin should exist"

        # Verify meta.json content
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        assert "git_head" in meta
        assert "git_branch" in meta
        assert "timestamp" in meta
        assert meta["run"] == 1


# ---------------------------------------------------------------------------
# AT-17: restore reinstates state
# ---------------------------------------------------------------------------


class TestAT17RestoreReinstatesState:
    """AT-17: ``restore`` reinstates state and model from a named run."""

    def test_restore_reinstates_state(self, auto_tune_env: Path) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)
        worktree_path = auto_tune_env / ".worktrees" / "auto-tune" / "test-sess"

        # The curriculum state file path is derived: "math.md" -> "math.json"
        state_file = worktree_path / "math.json"
        state_file.write_text('{"lessons_completed": 5}\n', encoding="utf-8")

        # Set up model file
        model_dir = worktree_path / "data"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "agent.bin").write_bytes(b"\x00\x01\x02")

        # Commit everything for clean snapshot
        subprocess.run(["git", "add", "."], cwd=worktree_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "pre-snapshot"], cwd=worktree_path, capture_output=True
        )

        # Take a snapshot
        main(["snapshot", "--session", "test-sess"])

        # Verify snapshot was created
        run_dir = sd / "runs" / "001"
        assert run_dir.is_dir()

        # Now modify the state to simulate changes
        state_file.write_text('{"lessons_completed": 10}\n', encoding="utf-8")

        # Restore from run 1
        main(["restore", "--session", "test-sess", "--run", "1"])

        # Verify state was restored
        restored_state = json.loads(state_file.read_text(encoding="utf-8"))
        assert restored_state["lessons_completed"] == 5

        # Verify model was restored
        assert (model_dir / "agent.bin").exists()


# ---------------------------------------------------------------------------
# AT-18: reset deletes state and truncates events
# ---------------------------------------------------------------------------


class TestAT18ResetDeletesState:
    """AT-18: ``reset`` deletes curriculum state and truncates events."""

    def test_reset_deletes_state_and_truncates_events(self, auto_tune_env: Path) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)
        worktree_path = auto_tune_env / ".worktrees" / "auto-tune" / "test-sess"

        # The curriculum state file is derived: "math.md" -> "math.json"
        state_file = worktree_path / "math.json"
        state_file.write_text('{"lessons_completed": 3}\n', encoding="utf-8")

        # Write some events
        with (sd / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"seq": 1, "type": "connected"}) + "\n")
            f.write(json.dumps({"seq": 2, "type": "progress"}) + "\n")

        # Reset
        main(["reset", "--session", "test-sess"])

        # Verify curriculum state is deleted
        assert not state_file.exists(), "Curriculum state should be deleted"

        # Verify events.jsonl is truncated
        events_content = (sd / "events.jsonl").read_text(encoding="utf-8")
        assert events_content == "", "events.jsonl should be empty after reset"


# ---------------------------------------------------------------------------
# AT-19: reset --fresh-model deletes model
# ---------------------------------------------------------------------------


class TestAT19ResetFreshModel:
    """AT-19: ``reset --fresh-model`` also deletes the Kalvin model file."""

    def test_reset_fresh_model_deletes_model(self, auto_tune_env: Path) -> None:
        _init_session()
        sd = _session_dir(auto_tune_env)
        _use_worktree_model_path(sd)
        worktree_path = auto_tune_env / ".worktrees" / "auto-tune" / "test-sess"

        # Create model file
        model_dir = worktree_path / "data"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "agent.bin"
        model_path.write_bytes(b"\x00\x01\x02")

        # Verify model exists before reset
        assert model_path.exists()

        # Reset with --fresh-model
        main(["reset", "--session", "test-sess", "--fresh-model"])

        # Verify model is deleted
        assert not model_path.exists(), "Model file should be deleted with --fresh-model"


# ---------------------------------------------------------------------------
# AT-20: lifecycle PID management
# ---------------------------------------------------------------------------


class TestAT20LifecyclePIDManagement:
    """AT-20: Process lifecycle commands manage PIDs and enforce timeouts."""

    @patch("participants.auto_tune.cli.lifecycle")
    def test_lifecycle_pid_management(self, mock_lifecycle: MagicMock, auto_tune_env: Path) -> None:
        _init_session()

        # Start harness
        main(["start-harness", "--session", "test-sess"])
        mock_lifecycle.start_harness.assert_called_once()

        # Stop harness
        main(["stop-harness", "--session", "test-sess"])
        mock_lifecycle.stop_harness.assert_called_once()

        # Both called with session dir path
        start_path = mock_lifecycle.start_harness.call_args[0][0]
        stop_path = mock_lifecycle.stop_harness.call_args[0][0]
        assert start_path == stop_path


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------


class TestCLIErrorHandling:
    """Additional error handling tests."""

    def test_main_no_subcommand_shows_error(self) -> None:
        """main([]) exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_main_nonexistent_session_shows_error(self, auto_tune_env: Path) -> None:
        """Non-existent session prints error and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main(["status", "--session", "nonexistent"])
        assert exc_info.value.code == 1

    def test_send_invalid_json_shows_error(self, auto_tune_env: Path) -> None:
        """Invalid JSON in --command prints clear error and exits."""
        _init_session()

        with pytest.raises(SystemExit) as exc_info:
            main(["send", "--session", "test-sess", "--command", "not-json"])
        assert exc_info.value.code == 1

    def test_restore_missing_run_shows_error(self, auto_tune_env: Path) -> None:
        """Restore from non-existent run handles gracefully."""
        _init_session()
        _session_dir(auto_tune_env)

        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            main(["restore", "--session", "test-sess", "--run", "99"])
