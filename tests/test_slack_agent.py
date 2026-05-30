"""Tests for SlackParticipant — HRNS-17, HRNS-18.

Uses a stub WebSocket server to avoid requiring a running harness.
Mocks the Slack SDK to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from participants.slack_agent import SlackParticipant


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
        self._server = await websockets.serve(
            self._handle, self._host, self._port
        )
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
                raise TimeoutError(
                    f"Expected {n} frames, got {len(self.received_frames)}"
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_slack_registers_on_connect():
    """SlackParticipant sends ``{"register": "slack"}`` on connect."""
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
        assert reg == {"register": "slack"}

        await participant.stop()


async def test_slack_renders_notify():
    """HRNS-18: SlackParticipant renders ``notify`` messages to Slack."""
    async with StubHarness() as stub:
        participant = SlackParticipant(
            harness_url=stub.url,
            slack_token="xoxb-fake",
            channel_id="C123",
        )
        participant._start_slack_listener = AsyncMock()  # type: ignore[method-assign]

        # Mock the Slack WebClient
        mock_client = MagicMock()
        mock_client.chat_postMessage = MagicMock()
        participant._slack_web_client = mock_client

        await participant.start()
        await stub.wait_for_frames(1)  # registration frame

        # Harness sends a notify message
        await stub.send_to_client({
            "address": "slack",
            "action": "notify",
            "message": "Training paused",
        })
        await asyncio.sleep(0.2)

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123",
            text="Training paused",
        )

        await participant.stop()


async def test_slack_forwards_human_input():
    """HRNS-17: SlackParticipant forwards human input to Trainer."""
    async with StubHarness() as stub:
        participant = SlackParticipant(
            harness_url=stub.url,
            slack_token="xoxb-fake",
            channel_id="C123",
        )
        participant._start_slack_listener = AsyncMock()  # type: ignore[method-assign]

        await participant.start()
        await stub.wait_for_frames(1)  # registration frame

        # Simulate a human message from Slack
        await participant._send_to_trainer("hello")
        await stub.wait_for_frames(2)

        # Second frame should be the trainer message
        msg_frame = stub.received_frames[1]
        assert msg_frame == {
            "address": "trainer",
            "action": "input",
            "message": "hello",
        }

        await participant.stop()


async def test_slack_ignores_unknown_action():
    """Unknown actions are logged/ignored — no Slack API call, no crash."""
    async with StubHarness() as stub:
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
        await stub.wait_for_frames(1)

        # Send a frame with an unknown action
        await stub.send_to_client({
            "address": "slack",
            "action": "unknown",
            "message": "something",
        })
        await asyncio.sleep(0.2)

        # No Slack API call should have been made
        mock_client.chat_postMessage.assert_not_called()

        await participant.stop()
