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

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from training.harness.bus import MessageBus
from training.harness.message import Message

logger = logging.getLogger(__name__)


def _domain_json_default(obj: Any) -> Any:
    """JSON ``default`` hook encoding harness domain objects to wire dicts.

    Called by ``json.dumps`` (via ``WebSocketProtocol._serialise_message``) for
    objects it cannot natively serialise. Produces exactly the dict shapes that
    ``enrich_event`` (the auto-tune supervisor) consumes, so a bus ``Message``
    carrying a domain object becomes a valid WebSocket JSON frame at the wire
    boundary. See specs/auto-tune.md §Event Frame and §KLine Display Object.

    - ``KLine`` → ``{"signature": int, "nodes": list[int]}``
    - ``RationaliseEvent`` → ``{"kind", "query", "proposal", "significance"}``.
      The nested ``KLine``\\s are encoded recursively by ``json.dumps``.
      ``candidate`` is intentionally omitted: ``enrich_event`` does not read it
      and the spec's ``rationalise`` frame lists only kind/significance/query/
      proposal.

    Any other non-serialisable type raises ``TypeError`` (json's default
    behaviour) so future unknown payloads fail loudly rather than being coerced
    into something meaningless.
    """
    if isinstance(obj, KLine):
        return {"signature": obj.signature, "nodes": list(obj.nodes)}
    if isinstance(obj, RationaliseEvent):
        return {
            "kind": obj.kind,
            "query": obj.query,
            "proposal": obj.proposal,
            "significance": obj.significance,
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class ClientConnection:
    """Tracks a connected WebSocket client."""

    ws: ServerConnection
    role: str | None = None


class _ClientParticipant:
    """Bus-facing wrapper for a connected WebSocket client.

    Implements the Participant protocol (``role``, ``on_message``).
    ``on_message`` is called synchronously on the bus dispatch thread and
    bridges to the async WebSocket via ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        role: str,
        protocol: WebSocketProtocol,
    ) -> None:
        self.role = role
        self._protocol = protocol

    def on_message(self, msg: Message) -> None:
        """Handle a bus message by sending it to the WebSocket client.

        Silently drops if the client has disconnected.
        """
        try:
            self._protocol._send_to_client_sync(self.role, msg)
        except (TypeError, ValueError) as exc:
            # Serialisation failure — a real bug, not a disconnect. Log at
            # WARNING so silent drops of domain-object payloads are
            # visible rather than mislabelled as "client gone". Drop behaviour
            # is unchanged: no exception escapes on_message, the bus stays
            # stable (HRNS-21).
            logger.warning(
                "Serialisation failed for %s action=%r: %s",
                self.role,
                msg.action,
                exc,
            )
        except Exception:
            # Silent drop — client may have disconnected between check and send.
            logger.debug("Silent drop for %s: client gone", self.role)


class WebSocketProtocol:
    """Manages WebSocket client connections and routes messages to/from the bus.

    Wire protocol (JSON frames):
        Registration:  ``{"register": "<role>"}``
        Message:       ``{"role": "...", "action": "...", "message": ...}``
        Error:         ``{"error": "description"}``

    After registration, inbound frames are treated as messages with the
    registered role as the implicit ``sender``.  Outbound messages
    addressed to a client are serialised and sent over the WebSocket.

    Outbound ``Message`` payloads may carry harness domain objects — ``KLine``
    and ``RationaliseEvent`` — which ``_serialise_message`` encodes to their
    wire dicts via a ``json.dumps(default=...)`` hook (``_domain_json_default``),
    so ``action="event"``/``"ratify_request"`` frames reach every client
    participant. Wire shapes match specs/auto-tune.md §Event Frame and
    §KLine Display Object.

    Disconnect semantics (HRNS-21): the bus subscription is *not* removed on
    disconnect.  Messages to a disconnected client are silently dropped until
    the client reconnects with the same role.  (Serialisation failures are a
    distinct, louder case — logged at WARNING, never re-raised.)
    """

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._loop: asyncio.AbstractEventLoop | None = None

        # role → list of ClientConnection
        self._connections: dict[str, list[ClientConnection]] = {}
        # ws → role  (for disconnect cleanup)
        self._reverse: dict[int, str] = {}

        # role → _ClientParticipant (bus subscription wrapper)
        self._participants: dict[str, _ClientParticipant] = {}

    # -- connection handler (called by websockets.serve) ---------------------

    async def handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection for its lifetime.

        Called once per client by ``websockets.serve``.
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        conn = ClientConnection(ws=ws)
        role: str | None = None

        try:
            async for raw_frame in ws:
                try:
                    frame = json.loads(raw_frame)
                except (json.JSONDecodeError, TypeError):
                    await self._send_error(ws, "malformed frame: not valid JSON")
                    continue

                if role is None:
                    # First frame must be registration.
                    role = frame.get("register")
                    if not role or not isinstance(role, str):
                        await self._send_error(ws, "first frame must be registration")
                        await ws.close(
                            code=4001,
                            reason="missing registration",
                        )
                        return

                    # Register (multiple clients per role allowed).
                    conn.role = role
                    self._connections.setdefault(role, []).append(conn)
                    self._reverse[id(ws)] = role
                    logger.info("Client registered: %s", role)

                    # Subscribe to bus (one participant per role, even with
                    # multiple connections).
                    if role not in self._participants:
                        participant = _ClientParticipant(role, self)
                        self._participants[role] = participant
                        self._bus.subscribe(role, participant.on_message)
                else:
                    # Subsequent frames: message with implicit sender.
                    msg = self._parse_message_frame(frame, sender=role)
                    if msg is None:
                        await self._send_error(ws, "malformed message frame")
                        continue
                    self._bus.send(msg)

        except websockets.ConnectionClosed:
            logger.debug("Connection closed for %s", role)
        finally:
            # Clean up mappings but keep the bus subscription alive.
            if role is not None:
                conns = self._connections.get(role)
                if conns is not None:
                    conns = [c for c in conns if c.ws is not ws]
                    if conns:
                        self._connections[role] = conns
                    else:
                        del self._connections[role]
                self._reverse.pop(id(ws), None)
                logger.info("Client disconnected: %s", role)

    # -- public API ----------------------------------------------------------

    async def send_to_client(self, role: str, msg: Message) -> None:
        """Send *msg* to all WebSocket clients registered as *role*.

        Silently drops if no clients are connected.
        """
        conns = self._connections.get(role)
        if not conns:
            return
        frame = self._serialise_message(msg)
        for conn in conns:
            try:
                await conn.ws.send(frame)
            except websockets.ConnectionClosed:
                logger.debug("Silent drop: %s client disconnected during send", role)

    # -- internal helpers ----------------------------------------------------

    def _send_to_client_sync(self, role: str, msg: Message) -> None:
        """Thread-safe bridge: schedule an async send from a sync context.

        Called by ``_ClientParticipant.on_message`` which runs on the bus
        dispatch thread.
        """
        conns = self._connections.get(role)
        if not conns:
            return
        if self._loop is None or self._loop.is_closed():
            return
        frame = self._serialise_message(msg)
        for conn in conns:
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
        """Serialise a ``Message`` to a JSON frame.

        Domain objects carried in the payload (``KLine``, ``RationaliseEvent``)
        are encoded to their wire dicts via the ``default=`` hook
        (see ``_domain_json_default``). This is the single outbound
        serialisation site for domain payloads.
        """
        payload: dict[str, Any] = {
            "role": msg.role,
            "action": msg.action,
            "message": msg.message,
        }
        if msg.sender is not None:
            payload["sender"] = msg.sender
        return json.dumps(payload, default=_domain_json_default)

    @staticmethod
    def _parse_message_frame(frame: dict[str, Any], sender: str) -> Message | None:
        """Parse a JSON frame dict into a ``Message``.  Returns *None* on
        malformed input.
        """
        role = frame.get("role")
        action = frame.get("action")
        if not isinstance(role, str) or not isinstance(action, str):
            return None
        return Message(
            role=role,
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
