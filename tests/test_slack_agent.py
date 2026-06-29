"""Tests for SlackParticipant — HRNS-17, HRNS-18, HRNS-31, HRNS-34.

Uses a stub WebSocket server to avoid requiring a running harness.
Mocks the Slack SDK to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import websockets

from training.supervisors.slack_agent import SlackParticipant

# ---------------------------------------------------------------------------
# Stub harness WebSocket server
# ---------------------------------------------------------------------------


class StubHarness:
    """Minimal WebSocket server that records received frames and can send
    frames back to the connected client.

    Usage::

        async with StubHarness() as stub:
            participant = SlackParticipant(stub.url, ...)
            await participant.start()
            # inspect stub.received_frames
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
            await self._client_ws.close()
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

    async def wait_for_frames(self, n: int, timeout: float = 2.0) -> None:
        """Wait until *n* frames have been received."""
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.received_frames) < n:
            await asyncio.sleep(0.05)
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Expected {n} frames, got {len(self.received_frames)}")


# ---------------------------------------------------------------------------
# Helper to set up a connected participant with mocked Slack
# ---------------------------------------------------------------------------


async def _make_participant(stub: StubHarness) -> SlackParticipant:
    """Create a SlackParticipant connected to *stub* with mocked Slack SDK."""
    participant = SlackParticipant(
        harness_url=stub.url,
        slack_token="xoxb-fake",
        channel_id="C123",
    )
    participant._start_slack_listener = AsyncMock()  # type: ignore[method-assign]

    mock_client = MagicMock()
    mock_client.chat_postMessage = MagicMock()
    participant._slack_web_client = mock_client

    await participant.start()
    await stub.wait_for_frames(1)  # registration frame
    return participant


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_slack_registers_on_connect():
    """HRNS-31: SlackParticipant sends ``{"register": "supervisor"}`` on connect."""
    async with StubHarness() as stub:
        participant = SlackParticipant(
            harness_url=stub.url,
            slack_token="xoxb-fake",
            channel_id="C123",
        )
        # Override Slack listener to avoid Socket Mode startup
        participant._start_slack_listener = AsyncMock()  # type: ignore[method-assign]

        await participant.start()
        await stub.wait_for_frames(1)

        assert len(stub.received_frames) >= 1
        reg = stub.received_frames[0]
        assert reg == {"register": "supervisor"}

        await participant.stop()


async def test_slack_renders_progress():
    """HRNS-18: SlackParticipant renders ``progress`` messages to Slack."""
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        # Harness sends a progress message
        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "progress",
                "message": "Training paused",
            }
        )
        await asyncio.sleep(0.2)

        participant._slack_web_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="📊 Training paused",
        )

        await participant.stop()


async def test_slack_renders_event():
    """HRNS-18: SlackParticipant renders ``event`` messages to Slack."""
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "event",
                "message": "S2 proposal",
            }
        )
        await asyncio.sleep(0.2)

        participant._slack_web_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="🔬 S2 proposal",
        )

        await participant.stop()


async def test_slack_renders_escalation():
    """HRNS-18: SlackParticipant renders ``escalation`` messages to Slack."""
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "escalation",
                "message": "budget exhausted",
            }
        )
        await asyncio.sleep(0.2)

        participant._slack_web_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="🚨 budget exhausted",
        )

        await participant.stop()


async def test_slack_renders_ratify_request():
    """HRNS-18: SlackParticipant renders ``ratify_request`` with hint and stores proposal.

    The harness sends a full ``{proposal, query, significance}`` envelope.
    Slack rendering must still receive the **full** message (proving the
    whole envelope is rendered), while only the canonical KLine proposal
    wire dict is buffered for the ratify command.
    """
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        proposal = {"signature": 42, "nodes": [1, 2]}
        message = {
            "proposal": proposal,
            "query": "is this consistent?",
            "significance": "high",
        }
        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "ratify_request",
                "message": message,
            }
        )
        await asyncio.sleep(0.2)

        # Should have posted to Slack with ratify hint, rendering the FULL
        # message envelope (not just the proposal).
        participant._slack_web_client.chat_postMessage.assert_called_once()
        call_args = participant._slack_web_client.chat_postMessage.call_args
        assert call_args.kwargs["channel"] == "C123"
        # Full-envelope keys appear in the rendered text, proving the whole
        # ratify_request message (not just the proposal) was rendered.
        assert "signature" in call_args.kwargs["text"]
        assert "query" in call_args.kwargs["text"]
        assert "significance" in call_args.kwargs["text"]
        assert "ratify" in call_args.kwargs["text"]

        # Should have buffered only the KLine proposal wire dict
        assert participant._latest_ratify_proposal == proposal

        await participant.stop()


async def test_slack_forwards_human_input():
    """HRNS-17: SlackParticipant forwards human input via command parser.

    "hello" is parsed as a GuidanceCommand, which sends input to trainer role.
    """
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        # Simulate a human message dispatched through the command parser
        await participant._dispatch_command("hello")
        await stub.wait_for_frames(2)

        # Second frame should be dispatched via command parser
        msg_frame = stub.received_frames[1]
        assert msg_frame == {
            "role": "trainer",
            "action": "input",
            "message": "hello",
        }

        await participant.stop()


async def test_slack_ratify_command_routes_supervisor_decision():
    """HRNS-34: ``ratify`` command routes a ``supervisor_decision`` to the trainer.

    The buffered ``_latest_ratify_proposal`` (canonical KLine wire dict) is
    carried verbatim inside the decision payload (`@specs/supervisor-decision.md` SD-9).
    """
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        # Set a pending ratify proposal (canonical KLine wire dict)
        proposal = {"signature": 42, "nodes": [1, 2, 3]}
        participant._latest_ratify_proposal = proposal

        # Dispatch ratify command
        await participant._dispatch_command("ratify")
        await stub.wait_for_frames(2)

        # Second frame should be a supervisor_decision to the trainer carrying
        # the wire dict as the proposal.
        msg_frame = stub.received_frames[1]
        assert msg_frame == {
            "role": "trainer",
            "action": "supervisor_decision",
            "message": {"decision": "ratify", "proposal": proposal},
        }

        await participant.stop()


async def test_slack_ratify_request_to_supervisor_decision_wire_shape():
    """Full round-trip regression: ratify_request → "ratify" → supervisor_decision.

    The critical guard for the buffering fix: a properly shaped
    ``ratify_request`` (full ``{proposal, query, significance}`` envelope)
    followed by a ``"ratify"`` dispatch must produce a ``supervisor_decision``
    frame whose ``proposal`` is the raw KLine wire dict — NOT the
    ``{proposal, query, significance}`` envelope (`@specs/supervisor-decision.md` SD-9).
    """
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        proposal = {"signature": 0xBEEF, "nodes": [7]}
        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "ratify_request",
                "message": {
                    "proposal": proposal,
                    "query": "consistent?",
                    "significance": "high",
                },
            }
        )
        await asyncio.sleep(0.2)

        # Dispatch the ratify command via the Slack command path
        await participant._dispatch_command("ratify")
        await stub.wait_for_frames(2)

        # The supervisor_decision frame must carry the raw KLine wire dict as
        # its proposal, not the {proposal, query, significance} envelope.
        msg_frame = stub.received_frames[1]
        assert msg_frame == {
            "role": "trainer",
            "action": "supervisor_decision",
            "message": {"decision": "ratify", "proposal": proposal},
        }

        await participant.stop()


async def test_slack_ratify_without_request_sends_nothing():
    """Ratify with no pending proposal sends no frame."""
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        # No ratify request pending (default None)
        assert participant._latest_ratify_proposal is None

        # Dispatch ratify command — should produce no messages
        await participant._dispatch_command("ratify")
        await asyncio.sleep(0.2)

        # Only the registration frame should exist
        assert len(stub.received_frames) == 1
        assert stub.received_frames[0] == {"register": "supervisor"}

        await participant.stop()


async def test_slack_ignores_unknown_action():
    """Unknown actions are logged/ignored — no Slack API call, no crash."""
    async with StubHarness() as stub:
        participant = await _make_participant(stub)

        # Send a frame with an unknown action
        await stub.send_to_client(
            {
                "role": "supervisor",
                "action": "unknown",
                "message": "something",
            }
        )
        await asyncio.sleep(0.2)

        # No Slack API call should have been made
        participant._slack_web_client.chat_postMessage.assert_not_called()

        await participant.stop()
