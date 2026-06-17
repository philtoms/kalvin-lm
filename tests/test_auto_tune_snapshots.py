"""Tests for auto-tune snapshot and restore.

Covers AT-16 (snapshot captures state, events, model, git metadata) and
AT-17 (restore reinstates state and model from a named run).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from training.participants.auto_tune.session import SessionConfig, SessionDir
from training.participants.auto_tune.snapshots import restore, snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_tree(tmp_path: Path) -> SessionDir:
    """Create a minimal session directory tree with sample files."""
    session_dir = tmp_path / "auto-tune" / "test-session"
    session_dir.mkdir(parents=True)
    (session_dir / "runs").mkdir()

    # config.json
    cfg = SessionConfig(
        session="test-session",
        curriculum="curricula/first-steps.md",
        harness_url="ws://localhost:8765",
        model_path=str(tmp_path / "data" / "agent.bin"),
        run_counter=0,
    )
    (session_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    # events.jsonl
    (session_dir / "events.jsonl").write_text(
        '{"seq": 1, "type": "connected"}\n',
        encoding="utf-8",
    )

    # Curriculum state file (curricula/first-steps.json)
    curricula_dir = tmp_path / "curricula"
    curricula_dir.mkdir()
    (curricula_dir / "first-steps.md").write_text("# curriculum\n", encoding="utf-8")
    (curricula_dir / "first-steps.json").write_text(
        '{"lessons_completed": 3}\n',
        encoding="utf-8",
    )

    # Model file
    model_dir = tmp_path / "data"
    model_dir.mkdir()
    (model_dir / "agent.bin").write_bytes(b"\x00\x01\x02MODEL")

    sd = SessionDir(
        root=tmp_path,
        base_dir="auto-tune",
        _session="test-session",
        _config=cfg,
    )
    return sd


# ---------------------------------------------------------------------------
# AT-16: snapshot captures state, events, model, and git metadata
# ---------------------------------------------------------------------------


class TestSnapshot:
    """AT-16: snapshot captures state, events, model, git metadata."""

    def test_creates_run_directory(self, session_tree: SessionDir) -> None:
        run_number = snapshot(session_tree)
        assert run_number == 1
        run_dir = session_tree.runs_dir / "001"
        assert run_dir.is_dir()

    def test_copies_state_json(self, session_tree: SessionDir) -> None:
        snapshot(session_tree)
        state_file = session_tree.runs_dir / "001" / "state.json"
        assert state_file.exists()
        content = json.loads(state_file.read_text(encoding="utf-8"))
        assert content == {"lessons_completed": 3}

    def test_copies_events_jsonl(self, session_tree: SessionDir) -> None:
        snapshot(session_tree)
        events_file = session_tree.runs_dir / "001" / "events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["type"] == "connected"
        assert event["seq"] == 1

    def test_copies_model_bin(self, session_tree: SessionDir) -> None:
        snapshot(session_tree)
        model_file = session_tree.runs_dir / "001" / "model.bin"
        assert model_file.exists()
        assert model_file.read_bytes() == b"\x00\x01\x02MODEL"

    def test_writes_meta_json(self, session_tree: SessionDir) -> None:
        snapshot(session_tree)
        meta_file = session_tree.runs_dir / "001" / "meta.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        assert meta["run"] == 1
        # Valid ISO 8601 timestamp
        dt = datetime.fromisoformat(meta["timestamp"])
        assert dt.tzinfo is not None
        # git_head is 40-char hex
        assert re.match(r"^[0-9a-f]{40}$", meta["git_head"])
        # git_branch is a non-empty string
        assert isinstance(meta["git_branch"], str) and len(meta["git_branch"]) > 0
        # git_dirty is a bool
        assert isinstance(meta["git_dirty"], bool)

    def test_increments_run_counter(self, session_tree: SessionDir) -> None:
        run1 = snapshot(session_tree)
        run2 = snapshot(session_tree)
        assert run1 == 1
        assert run2 == 2

        # Both run directories exist
        assert (session_tree.runs_dir / "001").is_dir()
        assert (session_tree.runs_dir / "002").is_dir()

        # Config has the latest counter
        cfg_data = json.loads(session_tree.config_path.read_text(encoding="utf-8"))
        assert cfg_data["run_counter"] == 2

    def test_no_state_file_graceful(self, session_tree: SessionDir, tmp_path: Path) -> None:
        """Snapshot succeeds even when curriculum state file doesn't exist."""
        # Remove the curriculum state file
        (tmp_path / "curricula" / "first-steps.json").unlink()

        run_number = snapshot(session_tree)
        assert run_number == 1
        state_file = session_tree.runs_dir / "001" / "state.json"
        assert not state_file.exists()

    def test_no_model_file_graceful(self, session_tree: SessionDir, tmp_path: Path) -> None:
        """Snapshot succeeds even when model file doesn't exist."""
        (tmp_path / "data" / "agent.bin").unlink()

        run_number = snapshot(session_tree)
        assert run_number == 1
        model_file = session_tree.runs_dir / "001" / "model.bin"
        assert not model_file.exists()

    def test_no_model_path_configured(self, session_tree: SessionDir) -> None:
        """Snapshot succeeds when model_path is None in config."""
        # Update config to have None model_path
        session_tree._config.model_path = None
        session_tree.config_path.write_text(
            json.dumps(session_tree._config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        run_number = snapshot(session_tree)
        assert run_number == 1
        model_file = session_tree.runs_dir / "001" / "model.bin"
        assert not model_file.exists()

    def test_events_content_matches_original(self, session_tree: SessionDir) -> None:
        """Events snapshot content matches the original exactly."""
        snapshot(session_tree)
        original = session_tree.events_path.read_text(encoding="utf-8")
        copy = (session_tree.runs_dir / "001" / "events.jsonl").read_text(encoding="utf-8")
        assert original == copy


# ---------------------------------------------------------------------------
# AT-17: restore reinstates state and model from a named run
# ---------------------------------------------------------------------------


class TestRestore:
    """AT-17: restore reinstates state and model from a named run."""

    def test_restores_curriculum_state(self, session_tree: SessionDir, tmp_path: Path) -> None:
        # Snapshot the initial state
        snapshot(session_tree)

        # Modify the curriculum state
        state_path = tmp_path / "curricula" / "first-steps.json"
        state_path.write_text('{"lessons_completed": 99}\n', encoding="utf-8")

        # Restore
        restore(session_tree, 1)

        # Verify content matches the snapshot
        restored = json.loads(state_path.read_text(encoding="utf-8"))
        assert restored == {"lessons_completed": 3}

    def test_restores_model(self, session_tree: SessionDir, tmp_path: Path) -> None:
        # Snapshot
        snapshot(session_tree)

        # Delete the model
        model_path = tmp_path / "data" / "agent.bin"
        model_path.unlink()
        assert not model_path.exists()

        # Restore
        restore(session_tree, 1)

        # Verify model content matches
        assert model_path.exists()
        assert model_path.read_bytes() == b"\x00\x01\x02MODEL"

    def test_raises_file_not_found_for_missing_run(self, session_tree: SessionDir) -> None:
        with pytest.raises(FileNotFoundError, match="Run directory not found"):
            restore(session_tree, 999)

    def test_raises_runtime_error_for_running_process(self, session_tree: SessionDir) -> None:
        # Create status.json with the current process PID (guaranteed running)
        status = {"pid": os.getpid(), "state": "waiting_for_event"}
        session_tree.status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

        # Create a run directory so it exists
        run_dir = session_tree.runs_dir / "001"
        run_dir.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="supervisor is still running"):
            restore(session_tree, 1)

    def test_allows_restore_when_process_dead(self, session_tree: SessionDir) -> None:
        """Restore succeeds when status.json has a PID that is no longer running."""
        # Use a PID that definitely doesn't exist
        fake_pid = 99999999
        status = {"pid": fake_pid, "state": "errored"}
        session_tree.status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

        # Create a snapshot first
        snapshot(session_tree)

        # This should not raise
        restore(session_tree, 1)

    def test_restore_run_without_model(self, session_tree: SessionDir, tmp_path: Path) -> None:
        """Restore gracefully skips model when run has no model.bin."""
        # Remove the model before snapshot
        (tmp_path / "data" / "agent.bin").unlink()
        snapshot(session_tree)

        # Restore should not fail
        restore(session_tree, 1)

    def test_no_status_file_passes(self, session_tree: SessionDir) -> None:
        """Restore succeeds when status.json doesn't exist."""
        # Ensure no status file
        assert not session_tree.status_path.exists()
        snapshot(session_tree)
        # Should not raise
        restore(session_tree, 1)

    def test_empty_pid_passes(self, session_tree: SessionDir) -> None:
        """Restore succeeds when status.json has pid=0 or pid=null."""
        for pid_val in (0, None):
            status = {"pid": pid_val, "state": "waiting_for_event"}
            session_tree.status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")
            # Ensure run dir exists
            run_dir = session_tree.runs_dir / "001"
            if not run_dir.exists():
                snapshot(session_tree)
            restore(session_tree, 1)


# ---------------------------------------------------------------------------
# AT-42+: reset clears cmd.json and ensures correct git branch
# ---------------------------------------------------------------------------


class TestResetCmdCleanup:
    """reset() deletes stale cmd.json to prevent immediate shutdown."""

    def test_deletes_cmd_json(self, session_tree: SessionDir) -> None:
        """reset removes cmd.json if it exists."""
        # Write a stale cmd.json (e.g. from a previous stop-supervisor)
        session_tree.cmd_path.write_text('{"action": "shutdown"}', encoding="utf-8")
        assert session_tree.cmd_path.exists()

        from training.participants.auto_tune.snapshots import reset

        reset(session_tree)

        assert not session_tree.cmd_path.exists()

    def test_no_cmd_json_is_safe(self, session_tree: SessionDir) -> None:
        """reset succeeds when cmd.json does not exist."""
        assert not session_tree.cmd_path.exists()

        from training.participants.auto_tune.snapshots import reset

        reset(session_tree)  # Should not raise

    def test_truncates_events(self, session_tree: SessionDir) -> None:
        """reset truncates events.jsonl."""
        # Verify events has content
        assert session_tree.events_path.read_text(encoding="utf-8") != ""

        from training.participants.auto_tune.snapshots import reset

        reset(session_tree)

        assert session_tree.events_path.read_text(encoding="utf-8") == ""
