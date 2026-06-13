"""Tests for participants.auto_tune.orchestrate.

Covers send_command, read_events, step (AT-14), and read_status.
Uses tmp_path to create a fake session directory structure — no git needed,
just the file protocol.

Spec ref: specs/auto-tune.md §CLI Subcommands, AT-14, AT-15
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from participants.auto_tune.orchestrate import (
    read_events,
    read_status,
    send_command,
    step,
)
from participants.auto_tune.session import SessionDir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path, *, status: dict | None = None, events: list[dict] | None = None):
    """Create a minimal session directory structure under *tmp_path*.

    Returns a SessionDir bound to the fake session.
    """
    session_name = "test-session"
    session_dir = tmp_path / "auto-tune" / session_name
    session_dir.mkdir(parents=True)

    # Write config.json (required by SessionDir.load)
    config = {
        "session": session_name,
        "curriculum": "curricula/test.md",
        "harness_url": "ws://localhost:8765",
        "model_path": "data/agent.bin",
        "run_counter": 0,
        "created_from_branch": "main",
        "created_from_commit": "abc123",
    }
    (session_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Write events.jsonl if provided
    if events is not None:
        lines = [json.dumps(e) for e in events]
        (session_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write status.json if provided
    if status is not None:
        (session_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )

    return SessionDir.load(session_name, root=tmp_path)


def _append_event(session_dir: SessionDir, event: dict) -> None:
    """Append a single event to events.jsonl."""
    path = session_dir.events_path
    line = json.dumps(event) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# send_command
# ---------------------------------------------------------------------------


class TestSendCommand:
    def test_writes_command_json(self, tmp_path):
        sd = _make_session(tmp_path)
        cmd = {"action": "start"}
        send_command(sd, cmd)

        raw = sd.cmd_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed == cmd

    def test_overwrites_existing_command(self, tmp_path):
        sd = _make_session(tmp_path)
        send_command(sd, {"action": "start"})
        send_command(sd, {"action": "stop"})

        raw = sd.cmd_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed == {"action": "stop"}

    def test_pretty_printed(self, tmp_path):
        sd = _make_session(tmp_path)
        send_command(sd, {"action": "goal", "text": "improve accuracy"})

        raw = sd.cmd_path.read_text(encoding="utf-8")
        # Pretty-printed JSON has newlines and indentation
        assert "\n" in raw
        assert "  " in raw


# ---------------------------------------------------------------------------
# read_events
# ---------------------------------------------------------------------------


class TestReadEvents:
    def test_empty_events_file(self, tmp_path):
        sd = _make_session(tmp_path, events=[])
        assert read_events(sd) == []

    def test_missing_events_file(self, tmp_path):
        sd = _make_session(tmp_path)
        # No events.jsonl created
        assert read_events(sd) == []

    def test_returns_all_events_by_default(self, tmp_path):
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "started"},
            {"seq": 3, "type": "progress", "status": "complete"},
        ]
        sd = _make_session(tmp_path, events=events)
        result = read_events(sd)
        assert len(result) == 3
        assert result[0]["seq"] == 1
        assert result[2]["seq"] == 3

    def test_filters_by_after_seq(self, tmp_path):
        """AT-15: events --after N returns events with seq > N."""
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "started"},
            {"seq": 3, "type": "progress", "status": "complete"},
        ]
        sd = _make_session(tmp_path, events=events)

        result = read_events(sd, after_seq=1)
        assert len(result) == 2
        assert result[0]["seq"] == 2
        assert result[1]["seq"] == 3

    def test_after_seq_returns_empty_when_all_filtered(self, tmp_path):
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress"},
            {"seq": 3, "type": "progress"},
        ]
        sd = _make_session(tmp_path, events=events)
        assert read_events(sd, after_seq=3) == []

    def test_after_seq_zero_returns_all(self, tmp_path):
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress"},
            {"seq": 3, "type": "progress"},
        ]
        sd = _make_session(tmp_path, events=events)
        result = read_events(sd, after_seq=0)
        assert len(result) == 3

    def test_skips_blank_lines(self, tmp_path):
        sd = _make_session(tmp_path)
        # Write events with blank lines
        lines = [
            json.dumps({"seq": 1, "type": "connected"}),
            "",
            json.dumps({"seq": 2, "type": "progress"}),
            "   ",
            json.dumps({"seq": 3, "type": "progress"}),
            "",
        ]
        sd.events_path.write_text("\n".join(lines), encoding="utf-8")

        result = read_events(sd)
        assert len(result) == 3

    def test_skips_malformed_lines(self, tmp_path):
        sd = _make_session(tmp_path)
        lines = [
            json.dumps({"seq": 1, "type": "connected"}),
            "not json at all",
            json.dumps({"seq": 2, "type": "progress"}),
            "{broken json",
            json.dumps({"seq": 3, "type": "progress"}),
        ]
        sd.events_path.write_text("\n".join(lines), encoding="utf-8")

        result = read_events(sd)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# read_status
# ---------------------------------------------------------------------------


class TestReadStatus:
    def test_returns_status_dict(self, tmp_path):
        status = {
            "pid": 12345,
            "connected": True,
            "last_event_seq": 5,
            "last_command": None,
            "state": "waiting_for_command",
            "started_at": "2026-06-05T10:00:00Z",
        }
        sd = _make_session(tmp_path, status=status)
        result = read_status(sd)
        assert result == status

    def test_raises_file_not_found(self, tmp_path):
        sd = _make_session(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_status(sd)


# ---------------------------------------------------------------------------
# step (AT-14)
# ---------------------------------------------------------------------------


class TestStep:
    def test_blocks_until_event_appears(self, tmp_path):
        """AT-14: step writes command, blocks until next event, returns it."""
        status = {
            "pid": 12345,
            "connected": True,
            "last_event_seq": 2,
            "last_command": None,
            "state": "waiting_for_command",
            "started_at": "2026-06-05T10:00:00Z",
        }
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "started"},
        ]
        sd = _make_session(tmp_path, status=status, events=events)

        # Run step in a background thread; it should block
        result_holder: list[list[dict]] = []
        error_holder: list[Exception] = []

        def run_step():
            try:
                result = step(sd, {"action": "continue"}, timeout=5.0)
                result_holder.append(result)
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=run_step)
        t.start()

        # Give step() time to settle into its poll loop
        time.sleep(0.3)

        # Append a new event
        _append_event(sd, {"seq": 3, "type": "progress", "status": "complete"})

        t.join(timeout=5.0)

        assert not error_holder, f"step raised: {error_holder[0]}"
        assert len(result_holder) == 1
        new_events = result_holder[0]
        assert len(new_events) == 1
        assert new_events[0]["seq"] == 3

        # Verify command was written
        cmd_raw = sd.cmd_path.read_text(encoding="utf-8")
        assert json.loads(cmd_raw) == {"action": "continue"}

    def test_timeout_raises_timeout_error(self, tmp_path):
        """step with no new event raises TimeoutError within timeout."""
        status = {
            "pid": 12345,
            "connected": True,
            "last_event_seq": 2,
            "last_command": None,
            "state": "waiting_for_command",
            "started_at": "2026-06-05T10:00:00Z",
        }
        events = [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "started"},
        ]
        sd = _make_session(tmp_path, status=status, events=events)

        with pytest.raises(TimeoutError):
            step(sd, {"action": "continue"}, timeout=0.5)

    def test_raises_file_not_found_when_no_status(self, tmp_path):
        """step raises FileNotFoundError if status.json is missing."""
        sd = _make_session(tmp_path)
        with pytest.raises(FileNotFoundError):
            step(sd, {"action": "start"}, timeout=1.0)

    def test_returns_multiple_new_events(self, tmp_path):
        """If multiple events appear, step returns all of them."""
        status = {
            "pid": 12345,
            "connected": True,
            "last_event_seq": 1,
            "last_command": None,
            "state": "waiting_for_command",
            "started_at": "2026-06-05T10:00:00Z",
        }
        events = [
            {"seq": 1, "type": "connected"},
        ]
        sd = _make_session(tmp_path, status=status, events=events)

        def append_later():
            time.sleep(0.2)
            # Write both events in a single append so the poll loop never
            # observes an intermediate state with only seq 2.
            payload = (
                json.dumps({"seq": 2, "type": "progress", "status": "started"}) + "\n"
                + json.dumps({"seq": 3, "type": "progress", "status": "complete"}) + "\n"
            )
            with sd.events_path.open("a", encoding="utf-8") as f:
                f.write(payload)

        t = threading.Thread(target=append_later)
        t.start()

        result = step(sd, {"action": "continue"}, timeout=5.0)
        t.join(timeout=5.0)

        assert len(result) == 2
        assert result[0]["seq"] == 2
        assert result[1]["seq"] == 3
