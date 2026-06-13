"""Slack Participant — bridges Slack API and the harness message bus.

Registers as ``"supervisor"`` on connect to the harness WebSocket server.
Renders all supervisor actions (``progress``, ``event``, ``escalation``,
``ratify_request``) to a Slack channel and forwards supervisor Slack input
through the shared command parser to the appropriate harness role.

Spec reference: specs/harness-server.md §Slack Participant, §Supervisor Participant
Test mapping: HRNS-17 (forward supervisor input), HRNS-18 (render supervisor actions),
              HRNS-31 (supervisor registration), HRNS-34 (ratify countersign)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import websockets

from participants.commands import parse_command

logger = logging.getLogger(__name__)


class SlackParticipant:
    """WebSocket client participant that translates between Slack and harness messages.

    Registers as ``"supervisor"`` on connect.  Receives supervisor actions
    (progress, event, escalation, ratify_request) and renders them to Slack.
    supervisor Slack input is parsed through the shared command parser and dispatched
    to the appropriate harness role.

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
        self._latest_ratify_request: Any = None

        # Lazy imports — only needed when actually talking to Slack.
        self._slack_web_client: Any = None

    # -- public API ----------------------------------------------------------

    async def start(self) -> None:
        """Connect to the harness, register as ``"supervisor"``, and start loops."""
        self._ws = await websockets.connect(self._harness_url)

        # Send registration frame
        await self._ws.send(json.dumps({"register": "supervisor"}))
        logger.info("SlackParticipant registered for role 'supervisor'")

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

                if action in ("progress", "event", "escalation", "ratify_request"):
                    await self._render_to_slack(message, action)
                    if action == "ratify_request":
                        self._latest_ratify_request = message
                else:
                    logger.debug("Ignoring action %r from harness", action)

        except websockets.ConnectionClosed:
            logger.info("Harness WebSocket connection closed")
        except asyncio.CancelledError:
            pass

    # -- Slack rendering (HRNS-18) ------------------------------------------

    async def _render_to_slack(self, content: Any, action: str = "notify") -> None:
        """Post *content* to the configured Slack channel.

        Formats the message with an action-specific prefix. For
        ``ratify_request``, appends a hint about the ``ratify`` command.
        """
        prefixes = {
            "progress": "📊 ",
            "event": "🔬 ",
            "escalation": "🚨 ",
            "ratify_request": "✋ ",
        }
        prefix = prefixes.get(action, "")
        text = f"{prefix}{content}"
        if action == "ratify_request":
            text += "\n→ Reply `ratify` to approve"
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
        """Listen for supervisor messages in the training channel via Socket Mode.

        On a supervisor message, dispatches through the shared command parser via
        ``_dispatch_command``.
        """
        if not self._app_token:
            logger.warning("No SLACK_APP_TOKEN set; Slack listener disabled")
            return

        try:
            from slack_sdk.socket_mode.aiohttp import SocketModeClient

            client = SocketModeClient(
                app_token=self._app_token,
                web_client=self._get_slack_web_client(),
            )

            @client.event  # type: ignore[misc]
            async def handle_message(event: dict[str, Any]) -> None:
                # Only handle messages from supervisors (skip bot messages)
                if event.get("type") != "message":
                    return
                if event.get("subtype") is not None:
                    return
                if event.get("bot_id") is not None:
                    return

                text = event.get("text", "")
                if not text:
                    return

                await self._dispatch_command(text)

            await client.connect()  # type: ignore[union-attr]

            # Keep running until cancelled
            while self._running:
                await asyncio.sleep(1)

            await client.disconnect()  # type: ignore[union-attr]

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Slack Socket Mode listener failed")

    async def _dispatch_command(self, text: str) -> None:
        """Parse supervisor input via shared command parser and send resulting messages."""
        if self._ws is None:
            logger.warning("Cannot send: not connected")
            return

        command = parse_command(text)
        messages = command.to_messages(self._latest_ratify_request)

        for target_role, action, payload in messages:
            frame = json.dumps(
                {
                    "role": target_role,
                    "action": action,
                    "message": payload,
                }
            )
            await self._ws.send(frame)
            logger.debug("Dispatched %s → %s: %s", action, target_role, str(payload)[:100])
