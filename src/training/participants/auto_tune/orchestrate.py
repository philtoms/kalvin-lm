"""Orchestration functions for driving an auto-tune session.

Pure functions that let pi drive an auto-tune session through its file-based
protocol: write commands to ``cmd.json``, read events from ``events.jsonl``,
and poll for new events.  These are the core operations behind the ``send``,
``events``, ``step``, and ``status`` CLI subcommands.

Spec ref: specs/auto-tune.md §CLI Subcommands, §Command Frame, §Event Frame,
          §Status Object
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from training.participants.auto_tune.session import SessionDir

# send_command


def send_command(session_dir: SessionDir, command_json: dict) -> None:
    """Write a command to the session's ``cmd.json`` file.

    Overwrites any existing file and returns immediately (no blocking).

    Args:
        session_dir: Bound session directory.
        command_json: Command payload to write (e.g. ``{"action": "start"}``).
    """
    session_dir.cmd_path.write_text(
        json.dumps(command_json, indent=2) + "\n",
        encoding="utf-8",
    )


# read_events


def read_events(session_dir: SessionDir, after_seq: int = -1) -> list[dict]:
    """Read events from the session's ``events.jsonl``, filtered by seq.

    Parses each line as JSON and returns entries where ``seq > after_seq``,
    preserving file order.  Blank and malformed lines are silently skipped.
    If ``events.jsonl`` does not exist, returns an empty list.

    Args:
        session_dir: Bound session directory.
        after_seq: Only return events with ``seq`` greater than this value.
            Defaults to ``-1`` (return all events).

    Returns:
        List of event dicts ordered by file position.
    """
    events_path: Path = session_dir.events_path
    if not events_path.exists():
        return []

    results: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("seq", -1) > after_seq:
            results.append(entry)
    return results


# read_status


def read_status(session_dir: SessionDir) -> dict:
    """Read and parse the session's ``status.json``.

    Args:
        session_dir: Bound session directory.

    Returns:
        The parsed status dict.

    Raises:
        FileNotFoundError: If ``status.json`` does not exist.
    """
    status_path: Path = session_dir.status_path
    return json.loads(status_path.read_text(encoding="utf-8"))


# step

_POLL_INTERVAL = 0.1  # seconds between polls


def step(
    session_dir: SessionDir,
    command_json: dict,
    *,
    timeout: float = 30.0,
) -> list[dict]:
    """Send a command and block until at least one new event appears.

    Writes *command_json* to ``cmd.json``, reads ``last_event_seq`` from
    ``status.json``, then polls ``events.jsonl`` until at least one event
    with ``seq > last_event_seq`` is found.

    Args:
        session_dir: Bound session directory.
        command_json: Command payload to send.
        timeout: Maximum seconds to wait for a new event (default 30).

    Returns:
        List of new event dicts (``seq > last_event_seq``).

    Raises:
        FileNotFoundError: If ``status.json`` does not exist on first read.
        TimeoutError: If no new event appears within *timeout* seconds.
    """
    send_command(session_dir, command_json)

    # Raises FileNotFoundError if status.json is missing.
    status = read_status(session_dir)
    last_event_seq: int = status.get("last_event_seq", -1)

    deadline = time.monotonic() + timeout
    while True:
        new_events = read_events(session_dir, after_seq=last_event_seq)
        if new_events:
            return new_events
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No new event after {timeout}s (waiting for seq > {last_event_seq})"
            )
        time.sleep(_POLL_INTERVAL)
