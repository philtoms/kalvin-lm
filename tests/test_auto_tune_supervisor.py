"""Tests for CLISupervisor — headless WebSocket auto-tune client.

Uses a StubHarness (minimal WebSocket server) pattern adapted from
``tests/test_tui_client.py``.  Each test creates a temp directory as the
session dir with a ``config.json`` pointing to the stub server's URL.

``enrich_event`` is mocked so tests are independent of KB-131's
implementation details.

Acceptance criteria mapping:
  AT-6  — test_connect_registers_and_writes_connected_event
  AT-7  — test_per_event_blocking
  AT-8  — test_continue_is_noop
  AT-9  — test_ratify_sends_countersign
  AT-10 — test_shutdown_cleans_up
  AT-12 — test_run_complete_does_not_exit
  AT-13 — test_unexpected_disconnect_sets_errored
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import websockets

from participants.auto_tune.supervisor import CLISupervisor

# Short alias to avoid E501 on repeated patch() calls
_ENRICH_PATCH = "participants.auto_tune.supervisor.enrich_event"


# ---------------------------------------------------------------------------
# Stub harness WebSocket server
# ---------------------------------------------------------------------------


class StubHarness:
    """Minimal WebSocket server for testing.

    Records received frames and allows sending frames to the client.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._server: websockets.asyncio.server.Server | None = None
        self._client_ws: websockets.asyncio.server.ServerConnection | None = None
        self.received_frames: list[dict[str, Any]] = []
        self._connected = asyncio.Event()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1] if self._server else 0
        return f"ws://{self._host}:{port}"

    async def __aenter__(self) -> StubHarness:
        self._server = await websockets.serve(self._handle, self._host, self._port)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client_ws is not None:
            try:
                await self._client_ws.close()
            except websockets.ConnectionClosed:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        """Handle a single client connection."""
        self._client_ws = ws
        self._connected.set()
        try:
            async for raw in ws:
                frame = json.loads(raw)
                self.received_frames.append(frame)
        except websockets.ConnectionClosed:
            pass

    async def send_to_client(self, frame: dict[str, Any]) -> None:
        """Send a JSON frame to the connected client."""
        await self._connected.wait()
        assert self._client_ws is not None
        await self._client_ws.send(json.dumps(frame))

    async def wait_for_frames(self, n: int, timeout: float = 5.0) -> None:
        """Wait until *n* frames have been received."""
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.received_frames) < n:
            await asyncio.sleep(0.05)
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Expected {n} frames, got {len(self.received_frames)}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a temp session directory with config.json."""
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_enrich_event(raw_frame: dict, seq: int) -> dict:
    """Passthrough enrich: copy the frame, add seq, set type from action.

    Mimics the real enrich_event by promoting key fields to the top level.
    """
    action = raw_frame.get("action", "unknown")
    message = raw_frame.get("message", {})
    event: dict[str, Any] = {"seq": seq, "type": action}

    # Promote fields the supervisor inspects at the top level
    if isinstance(message, dict):
        if action == "progress":
            event["status"] = message.get("status")
        if action == "ratify_request":
            event["proposal"] = message.get("proposal")

    return event


def _read_events(session_dir: Path) -> list[dict]:
    """Read all events from events.jsonl."""
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        return []
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_status(session_dir: Path) -> dict:
    """Read status.json."""
    status_path = session_dir / "status.json"
    return json.loads(status_path.read_text(encoding="utf-8"))


async def _wait_for_state(session_dir: Path, state: str, timeout: float = 5.0) -> dict:
    """Poll status.json until the state matches. Returns the status dict."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        status = _read_status(session_dir)
        if status.get("state") == state:
            return status
        await asyncio.sleep(0.05)
    status = _read_status(session_dir)
    raise TimeoutError(f"Expected state {state!r}, got {status.get('state')!r}")


async def _write_cmd(session_dir: Path, cmd: dict) -> None:
    """Write a command to cmd.json."""
    cmd_path = session_dir / "cmd.json"
    cmd_path.write_text(json.dumps(cmd) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# AT-6: Supervisor connects, registers as supervisor role, writes connected event
# ---------------------------------------------------------------------------


async def test_connect_registers_and_writes_connected_event(session_dir: Path) -> None:
    """AT-6: connect() sends registration, writes connected event and status."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            await sv.connect()

            # Verify stub received registration
            await stub.wait_for_frames(1)
            assert stub.received_frames[0] == {"register": "supervisor"}

            # Verify events.jsonl has connected event
            events = _read_events(session_dir)
            assert len(events) == 1
            assert events[0] == {"seq": 1, "type": "connected"}

            # Verify status.json
            status = _read_status(session_dir)
            assert status["connected"] is True
            assert status["state"] == "waiting_for_event"
            assert status["pid"] == os.getpid()
            assert status["started_at"] is not None

            await sv.disconnect()


# ---------------------------------------------------------------------------
# AT-7: Per-event blocking
# ---------------------------------------------------------------------------


async def test_per_event_blocking(session_dir: Path) -> None:
    """AT-7: Supervisor writes one event per WebSocket message and blocks for command."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                # Wait for connection
                await stub.wait_for_frames(1)

                # The supervisor polls for an initial command before reading
                # any WebSocket frames. Unblock that poll first, then send the
                # event once the supervisor is in the receive loop.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Send one event frame
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "started",
                            "lessons_total": 1,
                            "lessons_completed": 0,
                        },
                    }
                )

                # Wait for waiting_for_command state
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write continue command
                await _write_cmd(session_dir, {"action": "continue"})

                # Wait for waiting_for_event state
                await _wait_for_state(session_dir, "waiting_for_event")

                # Verify events.jsonl has connected + enriched event
                events = _read_events(session_dir)
                assert len(events) == 2
                assert events[0]["type"] == "connected"
                assert events[1]["type"] == "progress"
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass


# ---------------------------------------------------------------------------
# AT-8: continue is no-op
# ---------------------------------------------------------------------------


async def test_continue_is_noop(session_dir: Path) -> None:
    """AT-8: continue command produces no harness message."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll before sending any frames.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Record frame count (should be 1 — just registration)
                assert len(stub.received_frames) == 1

                # Send one frame
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "started",
                            "lessons_total": 1,
                            "lessons_completed": 0,
                        },
                    }
                )
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write continue command (the no-op under test)
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Assert no new frames received
                assert len(stub.received_frames) == 1
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass


# ---------------------------------------------------------------------------
# AT-9: ratify sends countersign
# ---------------------------------------------------------------------------


async def test_ratify_sends_countersign(session_dir: Path) -> None:
    """AT-9: ratify command sends countersign for latest buffered proposal."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll before sending any frames.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Send ratify_request with a proposal
                proposal_data = {"sig": 42, "nodes": [1, 2, 3]}
                await stub.send_to_client(
                    {
                        "action": "ratify_request",
                        "message": {"proposal": proposal_data},
                    }
                )
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write ratify command
                await _write_cmd(session_dir, {"action": "ratify"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Wait for countersign frame
                await stub.wait_for_frames(2, timeout=3.0)

                msg = stub.received_frames[1]
                assert msg["role"] == "trainee"
                assert msg["action"] == "countersign"
                assert msg["message"] == proposal_data
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass


# ---------------------------------------------------------------------------
# AT-10: shutdown cleans up
# ---------------------------------------------------------------------------


async def test_shutdown_cleans_up(session_dir: Path) -> None:
    """AT-10: shutdown command disconnects and exits cleanly."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Send one frame to get to waiting_for_command
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "started",
                            "lessons_total": 1,
                            "lessons_completed": 0,
                        },
                    }
                )
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write shutdown command
                await _write_cmd(session_dir, {"action": "shutdown"})

                # Wait for task to complete
                await asyncio.wait_for(task, timeout=5.0)

                # Assert status
                status = _read_status(session_dir)
                assert status["state"] == "shutting_down"

                # Assert events.jsonl has disconnected event
                events = _read_events(session_dir)
                disconnected_events = [e for e in events if e.get("type") == "disconnected"]
                assert len(disconnected_events) == 1

                # Assert _connected is False
                assert sv._connected is False
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass
                raise


# ---------------------------------------------------------------------------
# AT-12: run complete does not exit
# ---------------------------------------------------------------------------


async def test_run_complete_does_not_exit(session_dir: Path) -> None:
    """AT-12: progress complete sets run_complete state without exiting."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll before sending any frames.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Send progress complete frame
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "complete",
                            "lessons_total": 1,
                            "lessons_completed": 1,
                        },
                    }
                )

                # Wait for run_complete state
                await _wait_for_state(session_dir, "run_complete")

                # Write continue command
                await _write_cmd(session_dir, {"action": "continue"})

                # Wait for waiting_for_event state
                await _wait_for_state(session_dir, "waiting_for_event")

                # Assert task is still running
                assert not task.done()
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass


# ---------------------------------------------------------------------------
# AT-13: unexpected disconnect sets errored
# ---------------------------------------------------------------------------


async def test_unexpected_disconnect_sets_errored(session_dir: Path) -> None:
    """AT-13: Unexpected disconnect writes disconnected event and sets errored state."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll so the supervisor enters
                # the WebSocket receive loop before we close the connection.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Close the stub server's client WebSocket
                assert stub._client_ws is not None
                await stub._client_ws.close()

                # Wait for task to complete
                await asyncio.wait_for(task, timeout=5.0)

                # Assert status
                status = _read_status(session_dir)
                assert status["state"] == "errored"

                # Assert events.jsonl has disconnected event
                events = _read_events(session_dir)
                disconnected_events = [e for e in events if e.get("type") == "disconnected"]
                assert len(disconnected_events) == 1
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass
                raise


# ---------------------------------------------------------------------------
# Additional: malformed frame skipped
# ---------------------------------------------------------------------------


async def test_malformed_frame_skipped(session_dir: Path) -> None:
    """Malformed JSON from harness is skipped without crash."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll before sending any frames.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Send invalid JSON
                assert stub._client_ws is not None
                await stub._client_ws.send("not valid json{{{}}}")

                # Give a moment for processing
                await asyncio.sleep(0.2)

                # Send a valid frame — should be processed normally
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "started",
                            "lessons_total": 1,
                            "lessons_completed": 0,
                        },
                    }
                )
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write continue to finish the loop
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Verify the valid event was processed
                events = _read_events(session_dir)
                # connected + progress (malformed skipped)
                assert len(events) == 2
                assert events[0]["type"] == "connected"
                assert events[1]["type"] == "progress"
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass


# ---------------------------------------------------------------------------
# Additional: goal command preserves text
# ---------------------------------------------------------------------------


async def test_goal_command_preserves_text(session_dir: Path) -> None:
    """goal command reconstructs 'goal: <text>' before parse_command."""
    async with StubHarness() as stub:
        config = {"session": "test", "harness_url": stub.url}
        (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (session_dir / "events.jsonl").write_text("", encoding="utf-8")

        with patch(_ENRICH_PATCH, side_effect=_mock_enrich_event):
            sv = CLISupervisor(str(session_dir))
            task = asyncio.create_task(sv.run())
            try:
                await stub.wait_for_frames(1)

                # Unblock the initial-command poll before sending any frames.
                await _wait_for_state(session_dir, "waiting_for_command")
                await _write_cmd(session_dir, {"action": "continue"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Send one frame to get to waiting_for_command
                await stub.send_to_client(
                    {
                        "action": "progress",
                        "message": {
                            "status": "started",
                            "lessons_total": 1,
                            "lessons_completed": 0,
                        },
                    }
                )
                await _wait_for_state(session_dir, "waiting_for_command")

                # Write goal command
                await _write_cmd(session_dir, {"action": "goal", "text": "improve accuracy"})
                await _wait_for_state(session_dir, "waiting_for_event")

                # Wait for the frame at stub
                await stub.wait_for_frames(2, timeout=3.0)

                msg = stub.received_frames[1]
                assert msg["role"] == "trainer"
                assert msg["action"] == "input"
                # parse_command("goal: improve accuracy") → GoalCommand
                # → to_messages sends original_text
                assert "improve accuracy" in msg["message"]
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass
