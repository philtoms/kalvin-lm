"""Tests for auto-tune process lifecycle functions.

All tests use mocking — no real harness or supervisor processes are started.
Covers spec criteria AT-4, AT-5, AT-20 (PID management, timeouts, graceful
and forced shutdown).
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a minimal session directory with config.json."""
    config = {
        "session": "test-session",
        "curriculum": "curricula/test.md",
        "harness_url": "ws://localhost:8765",
        "model_path": "data/agent.bin",
        "run_counter": 0,
        "created_from_branch": "main",
        "created_from_commit": "abc123",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def fake_process() -> MagicMock:
    """Return a mock subprocess.Popen result with PID 99999."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 99999
    return proc


# ---------------------------------------------------------------------------
# Tests: start_harness
# ---------------------------------------------------------------------------


class TestStartHarness:
    """AT-4: start-harness starts harness and waits for WebSocket readiness."""

    @patch("participants.auto_tune.lifecycle._generate_session_harness_config")
    @patch("participants.auto_tune.lifecycle.socket.socket")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_starts_harness_and_returns_pid(
        self,
        mock_popen: MagicMock,
        mock_socket_cls: MagicMock,
        mock_gen_config: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process
        session_config_path = session_dir / "harness.yaml"
        mock_gen_config.return_value = session_config_path

        # Socket connect_ex returns 0 (success) immediately
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)

        from participants.auto_tune.lifecycle import _resolve_python, start_harness

        pid = start_harness(session_dir)

        assert pid == 99999
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert call_args[0][0] == [
            _resolve_python(),
            "-m",
            "harness",
            "--config",
            str(session_config_path),
        ]
        # PID file written
        pid_path = session_dir / "harness.pid"
        assert pid_path.exists()
        assert int(pid_path.read_text()) == 99999

    @patch("participants.auto_tune.lifecycle.time.sleep")
    @patch("participants.auto_tune.lifecycle.socket.socket")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_timeout_when_port_never_ready(
        self,
        mock_popen: MagicMock,
        mock_socket_cls: MagicMock,
        mock_sleep: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process

        # Socket connect_ex always returns non-zero (refused)
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)

        from participants.auto_tune.lifecycle import start_harness

        with pytest.raises(TimeoutError, match="did not become ready"):
            start_harness(session_dir, poll_timeout=0.5)

    @patch("participants.auto_tune.lifecycle.socket.socket")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_reads_config_and_extracts_port(
        self,
        mock_popen: MagicMock,
        mock_socket_cls: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        # Override config with a custom port
        config = json.loads((session_dir / "config.json").read_text())
        config["harness_url"] = "ws://localhost:9999"
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        mock_popen.return_value = fake_process

        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)

        from participants.auto_tune.lifecycle import start_harness

        start_harness(session_dir)

        # Verify connect_ex was called with port 9999
        mock_sock.connect_ex.assert_called_with(("localhost", 9999))


# ---------------------------------------------------------------------------
# Tests: stop_harness
# ---------------------------------------------------------------------------


class TestStopHarness:
    """AT-5: stop-harness gracefully terminates harness process."""

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=True)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_sends_sigterm_and_waits(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "harness.pid"
        pid_path.write_text("12345", encoding="utf-8")

        from participants.auto_tune.lifecycle import stop_harness

        stop_harness(session_dir)

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        mock_wait.assert_called_once_with(12345, timeout=5.0)
        assert not pid_path.exists()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=False)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_sigkill_escalation(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "harness.pid"
        pid_path.write_text("12345", encoding="utf-8")

        from participants.auto_tune.lifecycle import stop_harness

        stop_harness(session_dir)

        # Should send SIGTERM first, then SIGKILL
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        mock_kill.assert_any_call(12345, signal.SIGKILL)
        assert not pid_path.exists()

    def test_missing_pid_file(self, session_dir: Path) -> None:
        from participants.auto_tune.lifecycle import stop_harness

        # Should return without error
        stop_harness(session_dir)

    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_already_dead_process(
        self,
        mock_kill: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "harness.pid"
        pid_path.write_text("12345", encoding="utf-8")

        # SIGTERM raises ProcessLookupError — process already dead
        mock_kill.side_effect = ProcessLookupError("No such process")

        from participants.auto_tune.lifecycle import stop_harness

        stop_harness(session_dir)

        # PID file should still be cleaned up
        assert not pid_path.exists()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=True)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_sigkill_ignores_process_lookup_error(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        """Process dies between SIGTERM and SIGKILL escalation."""
        pid_path = session_dir / "harness.pid"
        pid_path.write_text("12345", encoding="utf-8")

        # SIGTERM succeeds, but SIGKILL raises ProcessLookupError
        mock_kill.side_effect = [None, ProcessLookupError("No such process")]

        from participants.auto_tune.lifecycle import stop_harness

        stop_harness(session_dir)

        # PID file should still be cleaned up
        assert not pid_path.exists()


# ---------------------------------------------------------------------------
# Tests: start_supervisor
# ---------------------------------------------------------------------------


class TestStartSupervisor:
    """AT-20: Process lifecycle commands manage PIDs and enforce timeouts."""

    @patch("participants.auto_tune.lifecycle.time.sleep")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_starts_supervisor_and_waits_for_connected(
        self,
        mock_popen: MagicMock,
        mock_sleep: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process

        from participants.auto_tune.lifecycle import _resolve_python, start_supervisor

        # Simulate status.json appearing with connected=true on second check
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                status_path = session_dir / "status.json"
                status_path.write_text(
                    json.dumps({"connected": True, "pid": 99999}),
                    encoding="utf-8",
                )

        mock_sleep.side_effect = side_effect

        pid = start_supervisor(session_dir)

        assert pid == 99999
        mock_popen.assert_called_once_with(
            [
                _resolve_python(),
                "-m",
                "participants.auto_tune.supervisor",
                "--session-dir",
                str(session_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # PID file written
        pid_path = session_dir / "supervisor.pid"
        assert pid_path.exists()
        assert int(pid_path.read_text()) == 99999

    @patch("participants.auto_tune.lifecycle.time.sleep")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_timeout_when_status_never_connected(
        self,
        mock_popen: MagicMock,
        mock_sleep: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process

        from participants.auto_tune.lifecycle import start_supervisor

        with pytest.raises(TimeoutError, match="did not connect"):
            start_supervisor(session_dir, poll_timeout=0.5)

    @patch("participants.auto_tune.lifecycle.time.sleep")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_ignores_partial_status(
        self,
        mock_popen: MagicMock,
        mock_sleep: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        """status.json exists but connected=false until it becomes true."""
        mock_popen.return_value = fake_process

        from participants.auto_tune.lifecycle import start_supervisor

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            status_path = session_dir / "status.json"
            if call_count[0] < 3:
                status_path.write_text(
                    json.dumps({"connected": False, "state": "connecting"}),
                    encoding="utf-8",
                )
            else:
                status_path.write_text(
                    json.dumps({"connected": True, "state": "waiting_for_event"}),
                    encoding="utf-8",
                )

        mock_sleep.side_effect = side_effect

        pid = start_supervisor(session_dir)
        assert pid == 99999


# ---------------------------------------------------------------------------
# Tests: stop_supervisor
# ---------------------------------------------------------------------------


class TestStopSupervisor:
    """AT-20: Process lifecycle commands manage PIDs and enforce timeouts."""

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=True)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_writes_shutdown_command_and_cleans_up(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "supervisor.pid"
        pid_path.write_text("54321", encoding="utf-8")

        from participants.auto_tune.lifecycle import stop_supervisor

        stop_supervisor(session_dir)

        # cmd.json should contain shutdown command
        cmd_path = session_dir / "cmd.json"
        assert cmd_path.exists()
        cmd_data = json.loads(cmd_path.read_text(encoding="utf-8"))
        assert cmd_data == {"action": "shutdown"}

        # PID file should be deleted
        assert not pid_path.exists()

        # Should NOT send SIGKILL (graceful exit worked)
        mock_kill.assert_not_called()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=False)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_sigkill_escalation(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "supervisor.pid"
        pid_path.write_text("54321", encoding="utf-8")

        from participants.auto_tune.lifecycle import stop_supervisor

        stop_supervisor(session_dir)

        # Should send SIGKILL
        mock_kill.assert_called_once_with(54321, signal.SIGKILL)
        assert not pid_path.exists()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=False)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_sigkill_ignores_process_lookup_error(
        self,
        mock_kill: MagicMock,
        mock_wait: MagicMock,
        session_dir: Path,
    ) -> None:
        pid_path = session_dir / "supervisor.pid"
        pid_path.write_text("54321", encoding="utf-8")

        mock_kill.side_effect = ProcessLookupError("No such process")

        from participants.auto_tune.lifecycle import stop_supervisor

        stop_supervisor(session_dir)

        # PID file should still be cleaned up
        assert not pid_path.exists()

    def test_missing_pid_file(self, session_dir: Path) -> None:
        from participants.auto_tune.lifecycle import stop_supervisor

        # Should return without error
        stop_supervisor(session_dir)


# ---------------------------------------------------------------------------
# Tests: private helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for private helper functions."""

    def test_read_pid_returns_integer(self, tmp_path: Path) -> None:
        from participants.auto_tune.lifecycle import _read_pid

        pid_file = tmp_path / "test.pid"
        pid_file.write_text("12345", encoding="utf-8")
        assert _read_pid(pid_file) == 12345

    def test_read_pid_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        from participants.auto_tune.lifecycle import _read_pid

        assert _read_pid(tmp_path / "nonexistent.pid") is None

    def test_read_pid_returns_none_for_invalid_content(self, tmp_path: Path) -> None:
        from participants.auto_tune.lifecycle import _read_pid

        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number", encoding="utf-8")
        assert _read_pid(pid_file) is None

    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_wait_for_exit_returns_true_when_process_dies(self, mock_kill: MagicMock) -> None:
        from participants.auto_tune.lifecycle import _wait_for_exit

        # Process dies on second check
        call_count = [0]

        def side_effect(pid, sig):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise ProcessLookupError

        mock_kill.side_effect = side_effect

        with patch("participants.auto_tune.lifecycle.time.sleep"):
            result = _wait_for_exit(12345, timeout=1.0)

        assert result is True

    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_wait_for_exit_returns_false_on_timeout(self, mock_kill: MagicMock) -> None:
        from participants.auto_tune.lifecycle import _wait_for_exit

        # Process always alive (os.kill succeeds)
        mock_kill.return_value = None

        # Use monotonic mock to simulate timeout immediately
        call_count = [0]
        original_monotonic = time.monotonic

        def fake_monotonic():
            call_count[0] += 1
            if call_count[0] <= 2:
                return original_monotonic()
            return original_monotonic() + 100  # Far past deadline

        with (
            patch("participants.auto_tune.lifecycle.time.monotonic", side_effect=fake_monotonic),
            patch("participants.auto_tune.lifecycle.time.sleep"),
        ):
            result = _wait_for_exit(12345, timeout=5.0)

        assert result is False


# ---------------------------------------------------------------------------
# Tests: stale process detection
# ---------------------------------------------------------------------------


class TestKillStaleProcess:
    """_kill_stale_process detects and kills stale processes from PID files."""

    def test_no_pid_file_is_noop(self, tmp_path: Path) -> None:
        from participants.auto_tune.lifecycle import _kill_stale_process

        pid_path = tmp_path / "nonexistent.pid"
        _kill_stale_process(pid_path)  # Should not raise

    def test_dead_process_cleans_up_pid(self, tmp_path: Path) -> None:
        """PID file with a dead process is deleted."""
        from participants.auto_tune.lifecycle import _kill_stale_process

        pid_path = tmp_path / "test.pid"
        pid_path.write_text("99999999", encoding="utf-8")  # Non-existent PID

        _kill_stale_process(pid_path)

        assert not pid_path.exists()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=True)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_live_process_killed(
        self, mock_kill: MagicMock, mock_wait: MagicMock, tmp_path: Path
    ) -> None:
        """Stale running process is sent SIGTERM."""
        from participants.auto_tune.lifecycle import _kill_stale_process

        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345", encoding="utf-8")

        _kill_stale_process(pid_path)

        # First call: os.kill(pid, 0) probe, second: SIGTERM
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(12345, 0)
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        assert not pid_path.exists()

    @patch("participants.auto_tune.lifecycle._wait_for_exit", return_value=False)
    @patch("participants.auto_tune.lifecycle.os.kill")
    def test_live_process_sigkill_escalation(
        self, mock_kill: MagicMock, mock_wait: MagicMock, tmp_path: Path
    ) -> None:
        """Stale process that won't die gets SIGKILL."""
        from participants.auto_tune.lifecycle import _kill_stale_process

        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345", encoding="utf-8")

        _kill_stale_process(pid_path)

        # os.kill(pid, 0) probe + SIGTERM + SIGKILL
        assert mock_kill.call_count == 3
        mock_kill.assert_any_call(12345, 0)
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        mock_kill.assert_any_call(12345, signal.SIGKILL)
        assert not pid_path.exists()


# ---------------------------------------------------------------------------
# Tests: per-session harness config generation
# ---------------------------------------------------------------------------


class TestSessionHarnessConfig:
    """_generate_session_harness_config overrides curriculum from session config."""

    def test_overrides_curriculum_file(self, tmp_path: Path) -> None:
        import yaml

        from participants.auto_tune.lifecycle import _generate_session_harness_config
        from participants.auto_tune.session import SessionConfig

        # Create project harness.yaml
        project_yaml = tmp_path / "harness.yaml"
        project_yaml.write_text(
            "server:\n  host: localhost\n  port: 8765\n"
            "trainer:\n  curriculum_file: curricula/first-steps.md\n",
            encoding="utf-8",
        )

        session_dir = tmp_path / "auto-tune" / "test"
        session_dir.mkdir(parents=True)
        (session_dir / "runs").mkdir()

        cfg = SessionConfig(
            session="test",
            curriculum="curricula/custom.md",
            harness_url="ws://localhost:8765",
        )

        result = _generate_session_harness_config(session_dir, cfg)

        assert result == session_dir / "harness.yaml"
        data = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert data["trainer"]["curriculum_file"] == "curricula/custom.md"

    def test_preserves_other_config(self, tmp_path: Path) -> None:
        import yaml

        from participants.auto_tune.lifecycle import _generate_session_harness_config
        from participants.auto_tune.session import SessionConfig

        project_yaml = tmp_path / "harness.yaml"
        project_yaml.write_text(
            "server:\n  host: localhost\n  port: 8765\n"
            "trainer:\n  curriculum_file: curricula/old.md\n  max_reactive_rounds: 5\n",
            encoding="utf-8",
        )

        session_dir = tmp_path / "auto-tune" / "test"
        session_dir.mkdir(parents=True)
        (session_dir / "runs").mkdir()

        cfg = SessionConfig(
            session="test",
            curriculum="curricula/new.md",
            harness_url="ws://localhost:8765",
        )

        result = _generate_session_harness_config(session_dir, cfg)

        data = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert data["trainer"]["curriculum_file"] == "curricula/new.md"
        assert data["trainer"]["max_reactive_rounds"] == 5
        assert data["server"]["port"] == 8765
        # RD-12: enabled forced false even when not present in project config
        assert data["trainer"]["llm"]["enabled"] is False

    def test_sets_llm_enabled_false(self, tmp_path: Path) -> None:
        """RD-12: generated config forces trainer.llm.enabled=false.

        When the project harness.yaml has no ``trainer.llm`` section, the
        per-session config must still add it with ``enabled: false``.
        """
        import yaml

        from participants.auto_tune.lifecycle import _generate_session_harness_config
        from participants.auto_tune.session import SessionConfig

        project_yaml = tmp_path / "harness.yaml"
        project_yaml.write_text(
            "server:\n  host: localhost\n  port: 8765\n"
            "trainer:\n  curriculum_file: curricula/first-steps.md\n",
            encoding="utf-8",
        )

        session_dir = tmp_path / "auto-tune" / "test"
        session_dir.mkdir(parents=True)
        (session_dir / "runs").mkdir()

        cfg = SessionConfig(
            session="test",
            curriculum="curricula/custom.md",
            harness_url="ws://localhost:8765",
        )

        result = _generate_session_harness_config(session_dir, cfg)

        data = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert data["trainer"]["llm"]["enabled"] is False
        # curriculum_file still overridden from the session config
        assert data["trainer"]["curriculum_file"] == "curricula/custom.md"

    def test_preserves_llm_overrides(self, tmp_path: Path) -> None:
        """RD-12: existing llm.base_url / llm.model overrides are preserved.

        The merge must set ``enabled: false`` without clobbering sibling
        ``llm.*`` keys already present in the project config.
        """
        import yaml

        from participants.auto_tune.lifecycle import _generate_session_harness_config
        from participants.auto_tune.session import SessionConfig

        project_yaml = tmp_path / "harness.yaml"
        project_yaml.write_text(
            "server:\n  host: localhost\n  port: 8765\n"
            "trainer:\n  curriculum_file: curricula/old.md\n"
            "  llm:\n"
            "    base_url: https://open.bigmodel.cn/api/paas/v4\n"
            "    model: glm-5.1\n",
            encoding="utf-8",
        )

        session_dir = tmp_path / "auto-tune" / "test"
        session_dir.mkdir(parents=True)
        (session_dir / "runs").mkdir()

        cfg = SessionConfig(
            session="test",
            curriculum="curricula/new.md",
            harness_url="ws://localhost:8765",
        )

        result = _generate_session_harness_config(session_dir, cfg)

        data = yaml.safe_load(result.read_text(encoding="utf-8"))
        # enabled forced false
        assert data["trainer"]["llm"]["enabled"] is False
        # existing overrides preserved (not clobbered)
        assert data["trainer"]["llm"]["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert data["trainer"]["llm"]["model"] == "glm-5.1"

    def test_no_project_config_uses_minimal(self, tmp_path: Path) -> None:
        import yaml

        from participants.auto_tune.lifecycle import _generate_session_harness_config
        from participants.auto_tune.session import SessionConfig

        # No project harness.yaml
        session_dir = tmp_path / "auto-tune" / "test"
        session_dir.mkdir(parents=True)
        (session_dir / "runs").mkdir()

        cfg = SessionConfig(
            session="test",
            curriculum="curricula/custom.md",
            harness_url="ws://localhost:8765",
        )

        result = _generate_session_harness_config(session_dir, cfg)

        data = yaml.safe_load(result.read_text(encoding="utf-8"))
        assert data["trainer"]["curriculum_file"] == "curricula/custom.md"
        # RD-12: enabled forced false even with no project config
        assert data["trainer"]["llm"]["enabled"] is False


# ---------------------------------------------------------------------------
# Tests: start_harness and start_supervisor call _kill_stale_process
# ---------------------------------------------------------------------------


class TestStartKillsStaleProcesses:
    """start_harness and start_supervisor kill stale processes before starting."""

    @patch("participants.auto_tune.lifecycle._generate_session_harness_config")
    @patch("participants.auto_tune.lifecycle._kill_stale_process")
    @patch("participants.auto_tune.lifecycle.socket.socket")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_start_harness_kills_stale(
        self,
        mock_popen: MagicMock,
        mock_socket_cls: MagicMock,
        mock_kill_stale: MagicMock,
        mock_gen_config: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process
        mock_gen_config.return_value = session_dir / "harness.yaml"
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=mock_sock)
        mock_socket_cls.return_value.__exit__ = MagicMock(return_value=False)

        from participants.auto_tune.lifecycle import start_harness

        start_harness(session_dir)

        mock_kill_stale.assert_called_once_with(session_dir / "harness.pid")

    @patch("participants.auto_tune.lifecycle._kill_stale_process")
    @patch("participants.auto_tune.lifecycle.time.sleep")
    @patch("participants.auto_tune.lifecycle.subprocess.Popen")
    def test_start_supervisor_kills_stale(
        self,
        mock_popen: MagicMock,
        mock_sleep: MagicMock,
        mock_kill_stale: MagicMock,
        session_dir: Path,
        fake_process: MagicMock,
    ) -> None:
        mock_popen.return_value = fake_process

        from participants.auto_tune.lifecycle import start_supervisor

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                status_path = session_dir / "status.json"
                status_path.write_text(
                    json.dumps({"connected": True, "pid": 99999}),
                    encoding="utf-8",
                )

        mock_sleep.side_effect = side_effect

        start_supervisor(session_dir)

        mock_kill_stale.assert_called_once_with(session_dir / "supervisor.pid")
