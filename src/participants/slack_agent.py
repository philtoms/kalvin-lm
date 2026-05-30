"""Slack Participant — bridges Slack API and the harness message bus.

Registers as ``"slack"`` on connect to the harness WebSocket server.
Renders ``notify`` messages to a Slack channel and forwards human Slack
input to the Trainer as ``{address: "trainer", action: "input", message: <text>}``.

Spec reference: specs/harness-server.md §Slack Participant
Test mapping: HRNS-17 (forward human input), HRNS-18 (render notify)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class SlackParticipant:
    """WebSocket client participant that translates between Slack and harness messages.

    Parameters
    ----------
    harness_url:
        WebSocket URL of the harness server (e.g. ``"ws://localhost:8765"``).
    slack_token:
        Slack Bot OAuth token (``xoxb-...``).  Falls back to ``SLACK_BOT_TOKEN``
        environment variable if not provided.
    channel_id:
        Slack channel ID to post notifications to.
    app_token:
        Slack App-Level token (``xapp-...``) for Socket Mode.  Falls back to
        ``SLACK_APP_TOKEN`` environment variable if not provided.
    """

    def __init__(
        self,
        harness_url: str,
        slack_token: str | None = None,
        channel_id: str = "",
        app_token: str | None = None,
    ) -> None:
        self._harness_url = harness_url
        self._slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._channel_id = channel_id
        self._app_token = app_token or os.environ.get("SLACK_APP_TOKEN", "")

        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._slack_task: asyncio.Task[None] | None = None
        self._running = False

        # Lazy imports — only needed when actually talking to Slack.
        self._slack_web_client: Any = None

    # -- public API ----------------------------------------------------------

    async def start(self) -> None:
        """Connect to the harness, register as ``"slack"``, and start loops."""
        self._ws = await websockets.connect(self._harness_url)

        # Send registration frame
        await self._ws.send(json.dumps({"register": "slack"}))
        logger.info("SlackParticipant registered on harness")

        self._running = True

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Start Slack event listener (Socket Mode)
        self._slack_task = asyncio.create_task(self._start_slack_listener())

    async def stop(self) -> None:
        """Close WebSocket and clean up Slack listeners."""
        self._running = False

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._slack_task is not None:
            self._slack_task.cancel()
            try:
                await self._slack_task
            except asyncio.CancelledError:
                pass
            self._slack_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # -- harness receive loop -----------------------------------------------

    async def _receive_loop(self) -> None:
        """Read frames from harness WebSocket and dispatch by action."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed frame from harness: %s", raw[:200])
                    continue

                action = frame.get("action", "")
                message = frame.get("message")

                if action == "notify":
                    await self._render_to_slack(message)
                else:
                    logger.debug("Ignoring action %r from harness", action)

        except websockets.ConnectionClosed:
            logger.info("Harness WebSocket connection closed")
        except asyncio.CancelledError:
            pass

    # -- Slack rendering (HRNS-18) ------------------------------------------

    async def _render_to_slack(self, content: Any) -> None:
        """Post *content* to the configured Slack channel.

        The content is stringified if not already a string.
        """
        text = str(content) if not isinstance(content, str) else content
        try:
            client = self._get_slack_web_client()
            client.chat_postMessage(
                channel=self._channel_id,
                text=text,
            )
            logger.debug("Posted to Slack: %s", text[:100])
        except Exception:
            logger.exception("Failed to post message to Slack")

    def _get_slack_web_client(self) -> Any:
        """Lazily create the Slack WebClient."""
        if self._slack_web_client is None:
            from slack_sdk.web import WebClient

            self._slack_web_client = WebClient(token=self._slack_token)
        return self._slack_web_client

    # -- Slack event listener (HRNS-17) -------------------------------------

    async def _start_slack_listener(self) -> None:
        """Listen for human messages in the training channel via Socket Mode.

        On a human message, sends ``{address: "trainer", action: "input",
        message: <text>}`` to the harness.
        """
        if not self._app_token:
            logger.warning("No SLACK_APP_TOKEN set; Slack listener disabled")
            return

        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient
            from slack_sdk.socket_mode.response import SocketModeResponse

            client = SocketModeClient(
                app_token=self._app_token,
                web_client=self._get_slack_web_client(),
            )

            @client.event  # type: ignore[misc]
            async def handle_message(event: dict[str, Any]) -> None:
                # Only handle messages from humans (skip bot messages)
                if event.get("type") != "message":
                    return
                if event.get("subtype") is not None:
                    return
                if event.get("bot_id") is not None:
                    return

                text = event.get("text", "")
                if not text:
                    return

                await self._send_to_trainer(text)

            await client.connect()  # type: ignore[union-attr]

            # Keep running until cancelled
            while self._running:
                await asyncio.sleep(1)

            await client.disconnect()  # type: ignore[union-attr]

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Slack Socket Mode listener failed")

    async def _send_to_trainer(self, text: str) -> None:
        """Send a message to the Trainer via the harness WebSocket."""
        if self._ws is None:
            logger.warning("Cannot send to trainer: not connected")
            return

        frame = json.dumps({
            "address": "trainer",
            "action": "input",
            "message": text,
        })
        await self._ws.send(frame)
        logger.debug("Forwarded to trainer: %s", text[:100])
