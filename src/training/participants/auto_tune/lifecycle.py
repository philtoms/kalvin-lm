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

from training.participants.auto_tune.session import SessionConfig

# Private helpers


def _resolve_python() -> str:
    """Resolve the Python interpreter for subprocess launches.

    If the project has a ``.venv/bin/python``, use it.  This ensures the
    harness and supervisor always run with the venv's installed packages
    (e.g. ``openai``), even when the auto-tune CLI itself is invoked with
    a different Python (e.g. pyenv shim).

    Falls back to ``sys.executable`` when no venv is found.
    """
    # Walk upward to the project root (this file lives at
    # <project-root>/src/training/participants/auto_tune/lifecycle.py).
    venv_python = Path(__file__).resolve().parents[3] / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


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


def _kill_stale_process(pid_path: Path) -> None:
    """Check PID file for a stale process and kill it if found.

    If the PID file exists and the process is still running, sends
    SIGTERM and waits, escalating to SIGKILL if needed.  Removes the
    PID file regardless.
    """
    pid = _read_pid(pid_path)
    if pid is None:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        # Not running — just clean up PID file
        _delete_pid_file(pid_path)
        return
    except PermissionError:
        # Running but can't signal — warn and bail
        import warnings

        warnings.warn(
            f"Stale process {pid} still running (no permission to kill). PID file: {pid_path}"
        )
        return
    import warnings

    warnings.warn(f"Killing stale process {pid} from {pid_path}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _delete_pid_file(pid_path)
        return
    if not _wait_for_exit(pid, timeout=5.0):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _delete_pid_file(pid_path)


def _delete_pid_file(path: Path) -> None:
    """Remove a PID file, ignoring errors if it's already gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# Harness lifecycle

def start_harness(session_dir: Path, *, poll_timeout: float = 30.0) -> int:
    """Start the harness server as a background process.

    Kills any stale harness, loads the session config, generates a
    per-session ``training.harness.yaml``, launches ``python -m training.harness``, and polls
    the WebSocket port until it accepts connections.

    Args:
        session_dir: Path to the session directory.
        poll_timeout: Seconds to wait for the harness to become ready.

    Returns:
        The harness process PID.

    Raises:
        TimeoutError: If the harness doesn't accept connections within
            *poll_timeout* seconds.
    """
    pid_path = session_dir / "training.harness.pid"
    _kill_stale_process(pid_path)

    config_path = session_dir / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(data)

    parsed = urllib.parse.urlparse(cfg.harness_url)
    port = parsed.port
    if port is None:
        raise ValueError(f"Cannot extract port from harness_url: {cfg.harness_url}")

    harness_config_path = _generate_session_harness_config(session_dir, cfg)

    log_path = session_dir / "training.harness.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [_resolve_python(), "-m", "harness", "--config", str(harness_config_path)],
        stdout=subprocess.DEVNULL,
        stderr=log_file,
    )

    pid_path.write_text(str(proc.pid), encoding="utf-8")

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

    raise TimeoutError(f"Harness did not become ready on port {port} within {poll_timeout}s")


def stop_harness(session_dir: Path) -> None:
    """Stop the harness server: SIGTERM, wait up to 5s, SIGKILL if needed,
    then delete the PID file."""
    pid_path = session_dir / "training.harness.pid"
    pid = _read_pid(pid_path)
    if pid is None:
        return  # Nothing to stop

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _delete_pid_file(pid_path)
        return

    if not _wait_for_exit(pid, timeout=5.0):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _delete_pid_file(pid_path)


# Supervisor lifecycle

def start_supervisor(session_dir: Path, *, poll_timeout: float = 30.0) -> int:
    """Start the CLI supervisor as a background process.

    Kills any stale supervisor, launches
    ``python -m training.participants.auto_tune.supervisor``, and polls
    ``status.json`` until ``connected`` is ``true``.

    Args:
        session_dir: Path to the session directory.
        poll_timeout: Seconds to wait for the supervisor to connect.

    Returns:
        The supervisor process PID.

    Raises:
        TimeoutError: If the supervisor doesn't connect within
            *poll_timeout* seconds.
    """
    pid_path = session_dir / "supervisor.pid"
    _kill_stale_process(pid_path)

    proc = subprocess.Popen(
        [
            _resolve_python(),
            "-m",
            "training.participants.auto_tune.supervisor",
            "--session-dir",
            str(session_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pid_path = session_dir / "supervisor.pid"
    pid_path.write_text(str(proc.pid), encoding="utf-8")

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

    raise TimeoutError(f"Supervisor did not connect within {poll_timeout}s")


def stop_supervisor(session_dir: Path) -> None:
    """Stop the supervisor: write ``{"action": "shutdown"}`` to ``cmd.json``
    atomically, wait up to 5s, SIGKILL if still alive, then delete the
    PID file."""
    pid_path = session_dir / "supervisor.pid"
    pid = _read_pid(pid_path)
    if pid is None:
        return  # Nothing to stop

    cmd_path = session_dir / "cmd.json"
    payload = json.dumps({"action": "shutdown"}) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(session_dir), prefix=".cmd-", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, str(cmd_path))
    except BaseException:
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


# Session harness config generation

def _generate_session_harness_config(session_dir: Path, cfg: SessionConfig) -> Path:
    """Generate a per-session ``training.harness.yaml`` from the project's config.

    Reads the project's ``training.harness.yaml`` and overrides the ``curriculum_file``
    with the session's configured curriculum.  Writes the result to
    ``<session_dir>/training.harness.yaml`` so the harness loads the correct curriculum.

    Returns the path to the generated config.
    """
    import yaml

    project_config = session_dir.parent.parent / "training.harness.yaml"
    config_path = session_dir / "training.harness.yaml"

    # Read project training.harness.yaml if it exists, otherwise use minimal defaults
    if project_config.exists():
        data = yaml.safe_load(project_config.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    data.setdefault("trainer", {})["curriculum_file"] = cfg.curriculum

    # Force delegated reactive mode: pi (the CLI supervisor) is the sole
    # reactive decision-maker, not the Cogitator LLM agent.
    # setdefault preserves any existing llm.base_url / llm.model overrides.
    # Spec ref: specs/reactive-delegation.md RD-12, specs/auto-tune.md rule 7a.
    llm = data["trainer"].setdefault("llm", {})
    llm["enabled"] = False

    config_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    return config_path
