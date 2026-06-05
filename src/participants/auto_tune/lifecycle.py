"""Process lifecycle management for auto-tune.

Provides functions to start and stop the harness server and CLI supervisor
as background processes, with PID tracking, readiness polling, graceful
shutdown with SIGTERM→SIGKILL escalation, and configurable timeouts.

Spec ref: specs/auto-tune.md §Harness Lifecycle (rules 7–10),
§Supervisor Lifecycle (rules 11–15), §Error Handling (rules 30–31)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from participants.auto_tune.session import SessionConfig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_pid(path: Path) -> int | None:
    """Read a PID file and return the integer, or ``None`` if missing."""
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _wait_for_exit(pid: int, timeout: float = 5.0) -> bool:
    """Poll process liveness using ``os.kill(pid, 0)``.

    Returns ``True`` if the process exited within *timeout* seconds.
    Uses ``os.kill`` rather than ``os.waitpid`` because the stop
    commands run in a separate CLI invocation from the start commands.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True  # Process has exited
        time.sleep(0.1)
    return False  # Still running after timeout


def _delete_pid_file(path: Path) -> None:
    """Remove a PID file, ignoring errors if it's already gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Harness lifecycle
# ---------------------------------------------------------------------------


def start_harness(session_dir: Path, *, poll_timeout: float = 30.0) -> int:
    """Start the harness server as a background process.

    1. Load ``SessionConfig`` from the session directory.
    2. Extract the WebSocket port from ``harness_url``.
    3. Start ``python -m harness`` as a background subprocess.
    4. Write the PID to ``harness.pid``.
    5. Poll the WebSocket port until it accepts connections.

    Args:
        session_dir: Path to the session directory.
        poll_timeout: Seconds to wait for the harness to become ready.

    Returns:
        The harness process PID.

    Raises:
        TimeoutError: If the harness doesn't accept connections within
            *poll_timeout* seconds.
    """
    # 1. Load config
    config_path = session_dir / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(data)

    # 2. Extract port from harness_url
    parsed = urllib.parse.urlparse(cfg.harness_url)
    port = parsed.port
    if port is None:
        raise ValueError(f"Cannot extract port from harness_url: {cfg.harness_url}")

    # 3. Start harness as background process
    proc = subprocess.Popen(
        [sys.executable, "-m", "harness", "--config", "harness.yaml"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 4. Write PID file
    pid_path = session_dir / "harness.pid"
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    # 5. Poll WebSocket port for readiness
    deadline = time.monotonic() + poll_timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                if sock.connect_ex(("localhost", port)) == 0:
                    return proc.pid
        except OSError:
            pass
        time.sleep(0.25)

    raise TimeoutError(
        f"Harness did not become ready on port {port} within {poll_timeout}s"
    )


def stop_harness(session_dir: Path) -> None:
    """Stop the harness server process.

    1. Read PID from ``harness.pid``.
    2. Send ``SIGTERM``.
    3. Wait up to 5 seconds for exit; send ``SIGKILL`` if still alive.
    4. Delete the PID file.

    Args:
        session_dir: Path to the session directory.
    """
    pid_path = session_dir / "harness.pid"
    pid = _read_pid(pid_path)
    if pid is None:
        return  # Nothing to stop

    # Send SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _delete_pid_file(pid_path)
        return

    # Wait for graceful exit
    if not _wait_for_exit(pid, timeout=5.0):
        # Escalate to SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _delete_pid_file(pid_path)


# ---------------------------------------------------------------------------
# Supervisor lifecycle
# ---------------------------------------------------------------------------


def start_supervisor(session_dir: Path, *, poll_timeout: float = 30.0) -> int:
    """Start the CLI supervisor as a background process.

    1. Start ``python -m participants.auto_tune.supervisor`` as a
       background subprocess.
    2. Write the PID to ``supervisor.pid``.
    3. Poll ``status.json`` until ``connected`` is ``true``.

    Args:
        session_dir: Path to the session directory.
        poll_timeout: Seconds to wait for the supervisor to connect.

    Returns:
        The supervisor process PID.

    Raises:
        TimeoutError: If the supervisor doesn't connect within
            *poll_timeout* seconds.
    """
    # 1. Start supervisor as background process
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "participants.auto_tune.supervisor",
            "--session-dir",
            str(session_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. Write PID file
    pid_path = session_dir / "supervisor.pid"
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    # 3. Poll status.json until connected
    status_path = session_dir / "status.json"
    deadline = time.monotonic() + poll_timeout
    while time.monotonic() < deadline:
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
                if data.get("connected") is True:
                    return proc.pid
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.25)

    raise TimeoutError(
        f"Supervisor did not connect within {poll_timeout}s"
    )


def stop_supervisor(session_dir: Path) -> None:
    """Stop the CLI supervisor process.

    1. Read PID from ``supervisor.pid``.
    2. Write ``{"action": "shutdown"}`` to ``cmd.json`` (atomic write).
    3. Wait up to 5 seconds for exit; send ``SIGKILL`` if still alive.
    4. Delete the PID file.

    Args:
        session_dir: Path to the session directory.
    """
    pid_path = session_dir / "supervisor.pid"
    pid = _read_pid(pid_path)
    if pid is None:
        return  # Nothing to stop

    # Write shutdown command atomically
    cmd_path = session_dir / "cmd.json"
    payload = json.dumps({"action": "shutdown"}) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(session_dir), prefix=".cmd-", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, str(cmd_path))
    except BaseException:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Wait for graceful exit
    if not _wait_for_exit(pid, timeout=5.0):
        # Escalate to SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _delete_pid_file(pid_path)
