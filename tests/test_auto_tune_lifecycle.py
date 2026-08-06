"""Tests for auto-tune lifecycle — orphan process killing by port.

Covers the fix for the ``start-harness`` hang caused by orphaned harness
processes holding the WebSocket port after their PID file goes missing.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from training.auto_tune.lifecycle import (
    _kill_port_orphans,
    _port_pids,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _free_port() -> int:
    """Return a free TCP port (bind-and-release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hold_port(port: int) -> subprocess.Popen:
    """Spawn a child process that listens on *port* until killed.

    Returns the Popen handle (the holder process).
    """
    holder = subprocess.Popen(
        [
            "python3",
            "-c",
            (
                f"import socket, time; "
                f"s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                f"s.bind(('127.0.0.1', {port})); s.listen(1); "
                f"time.sleep(30)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until the port is actually held.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _port_pids(port):
            return holder
        time.sleep(0.05)
    holder.kill()
    pytest.fail(f"holder process did not bind port {port} in time")


# ── _port_pids ────────────────────────────────────────────────────────


def test_port_pids_empty_for_free_port():
    """A free port has no PIDs."""
    port = _free_port()
    assert _port_pids(port) == []


def test_port_pids_finds_holder():
    """A port with a listener returns its PID."""
    port = _free_port()
    holder = _hold_port(port)
    try:
        pids = _port_pids(port)
        assert holder.pid in pids
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_port_pids_no_lsof_returns_empty():
    """When lsof is missing, _port_pids returns [] (no crash)."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _port_pids(8765) == []


# ── _kill_port_orphans ────────────────────────────────────────────────


def test_kill_port_orphans_kills_holder():
    """An orphaned holder is killed."""
    port = _free_port()
    holder = _hold_port(port)
    try:
        assert holder.poll() is None  # alive
        _kill_port_orphans(port)
        # The holder should have exited.
        assert holder.wait(timeout=5) is not None
        assert _port_pids(port) == []
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


def test_kill_port_orphans_keeps_excluded_pid():
    """A PID passed as ``keep`` is left alive."""
    port = _free_port()
    holder = _hold_port(port)
    try:
        _kill_port_orphans(port, keep=holder.pid)
        assert holder.poll() is None  # still alive
        assert holder.pid in _port_pids(port)
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_kill_port_orphans_noop_on_free_port():
    """A free port is a no-op (no crash, no warning)."""
    port = _free_port()
    _kill_port_orphans(port)  # should not raise


# ── start_harness integration ─────────────────────────────────────────
# (covered indirectly; the unit tests above exercise the mechanism)


def test_lsof_available():
    """Sanity: lsof is on PATH (the orphan-kill depends on it)."""
    import shutil

    if shutil.which("lsof") is None:
        pytest.skip("lsof not installed — orphan-kill fallback unavailable")
