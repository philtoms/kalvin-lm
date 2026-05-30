"""WebSocket protocol for client participants in the multi-agent harness.

Handles registration, bidirectional JSON frame routing, and silent-drop
disconnect semantics (HRNS-4, HRNS-21).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from harness.bus import MessageBus
from harness.message import Message

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Tracks a connected WebSocket client."""

    ws: ServerConnection
    address: str | None = None


class _ClientParticipant:
    """Bus-facing wrapper for a connected WebSocket client.

    Implements the Participant protocol (``address``, ``on_message``).
    ``on_message`` is called synchronously on the bus dispatch thread and
    bridges to the async WebSocket via ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        address: str,
        protocol: WebSocketProtocol,
    ) -> None:
        self.address = address
        self._protocol = protocol

    def on_message(self, msg: Message) -> None:
        """Handle a bus message by sending it to the WebSocket client.

        Silently drops if the client has disconnected.
        """
        try:
            self._protocol._send_to_client_sync(self.address, msg)
        except Exception:
            # Silent drop — client may have disconnected between check and send.
            logger.debug("Silent drop for %s: client gone", self.address)


class WebSocketProtocol:
    """Manages WebSocket client connections and routes messages to/from the bus.

    Wire protocol (JSON frames):
        Registration:  ``{"register": "<address>"}``
        Message:       ``{"address": "...", "action": "...", "message": ...}``
        Error:         ``{"error": "description"}``

    After registration, inbound frames are treated as messages with the
    registered address as the implicit ``sender``.  Outbound messages
    addressed to a client are serialised and sent over the WebSocket.

    Disconnect semantics (HRNS-21): the bus subscription is *not* removed on
    disconnect.  Messages to a disconnected client are silently dropped until
    the client reconnects with the same address.
    """

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._loop: asyncio.AbstractEventLoop | None = None

        # address → ClientConnection
        self._connections: dict[str, ClientConnection] = {}
        # ws → address  (for disconnect cleanup)
        self._reverse: dict[int, str] = {}

        # address → _ClientParticipant (bus subscription wrapper)
        self._participants: dict[str, _ClientParticipant] = {}

    # -- connection handler (called by websockets.serve) ---------------------

    async def handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection for its lifetime.

        Called once per client by ``websockets.serve``.
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        conn = ClientConnection(ws=ws)
        address: str | None = None

        try:
            async for raw_frame in ws:
                try:
                    frame = json.loads(raw_frame)
                except (json.JSONDecodeError, TypeError):
                    await self._send_error(ws, "malformed frame: not valid JSON")
                    continue

                if address is None:
                    # First frame must be registration.
                    address = frame.get("register")
                    if not address or not isinstance(address, str):
                        await self._send_error(ws, "first frame must be registration")
                        await ws.close(
                            code=4001,
                            reason="missing registration",
                        )
                        return

                    if address in self._connections:
                        # Check if the existing connection is stale (same ws or
                        # already closed).  For a fresh duplicate, reject.
                        existing = self._connections[address]
                        if existing.ws != ws:
                            await self._send_error(
                                ws,
                                f"address '{address}' already registered",
                            )
                            await ws.close(
                                code=4002,
                                reason="duplicate address",
                            )
                            return

                    # Register.
                    conn.address = address
                    self._connections[address] = conn
                    self._reverse[id(ws)] = address
                    logger.info("Client registered: %s", address)

                    # Subscribe to bus (or reuse existing participant on
                    # reconnect).
                    if address not in self._participants:
                        participant = _ClientParticipant(address, self)
                        self._participants[address] = participant
                        self._bus.subscribe(address, participant.on_message)
                    else:
                        # Reconnect: participant already subscribed.  Nothing
                        # extra to do — _send_to_client_sync will pick up the
                        # new connection automatically.
                        pass
                else:
                    # Subsequent frames: message with implicit sender.
                    msg = self._parse_message_frame(frame, sender=address)
                    if msg is None:
                        await self._send_error(ws, "malformed message frame")
                        continue
                    self._bus.send(msg)

        except websockets.ConnectionClosed:
            logger.debug("Connection closed for %s", address)
        finally:
            # Clean up mappings but keep the bus subscription alive.
            if address is not None:
                self._connections.pop(address, None)
                self._reverse.pop(id(ws), None)
                logger.info("Client disconnected: %s", address)

    # -- public API ----------------------------------------------------------

    async def send_to_client(self, address: str, msg: Message) -> None:
        """Send *msg* to the WebSocket client registered as *address*.

        Silently drops if the client is not connected.
        """
        conn = self._connections.get(address)
        if conn is None:
            return
        frame = self._serialise_message(msg)
        try:
            await conn.ws.send(frame)
        except websockets.ConnectionClosed:
            logger.debug("Silent drop: %s disconnected during send", address)

    # -- internal helpers ----------------------------------------------------

    def _send_to_client_sync(self, address: str, msg: Message) -> None:
        """Thread-safe bridge: schedule an async send from a sync context.

        Called by ``_ClientParticipant.on_message`` which runs on the bus
        dispatch thread.
        """
        conn = self._connections.get(address)
        if conn is None:
            # Silently drop — client disconnected.
            return
        if self._loop is None or self._loop.is_closed():
            return
        frame = self._serialise_message(msg)
        asyncio.run_coroutine_threadsafe(self._do_send(conn.ws, frame), self._loop)

    @staticmethod
    async def _do_send(ws: ServerConnection, frame: str) -> None:
        """Low-level async send with connection-closed guard."""
        try:
            await ws.send(frame)
        except websockets.ConnectionClosed:
            pass

    @staticmethod
    def _serialise_message(msg: Message) -> str:
        """Serialise a ``Message`` to a JSON frame."""
        payload: dict[str, Any] = {
            "address": msg.address,
            "action": msg.action,
            "message": msg.message,
        }
        if msg.sender is not None:
            payload["sender"] = msg.sender
        return json.dumps(payload)

    @staticmethod
    def _parse_message_frame(
        frame: dict[str, Any], sender: str
    ) -> Message | None:
        """Parse a JSON frame dict into a ``Message``.  Returns *None* on
        malformed input.
        """
        address = frame.get("address")
        action = frame.get("action")
        if not isinstance(address, str) or not isinstance(action, str):
            return None
        return Message(
            address=address,
            action=action,
            message=frame.get("message"),
            sender=sender,
        )

    @staticmethod
    async def _send_error(ws: ServerConnection, description: str) -> None:
        """Send an error frame to a client."""
        frame = json.dumps({"error": description})
        try:
            await ws.send(frame)
        except websockets.ConnectionClosed:
            pass
