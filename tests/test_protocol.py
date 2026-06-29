"""Tests for the WebSocket protocol (HRNS-4, HRNS-21).

Uses ``pytest-asyncio`` with ``websockets`` client for test connections.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets
import websockets.asyncio.client

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.harness.bus import MessageBus
from training.harness.message import Message
from training.harness.protocol import WebSocketProtocol
from training.supervisors.auto_tune.events import enrich_event

# -- helpers ---------------------------------------------------------------

# Use a fixed port range for tests to avoid conflicts.
_TEST_PORT = 18765


class _BusRecorder:
    """Records messages sent to the bus from protocol handlers."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self.messages: list[Message] = []
        bus.subscribe("*", self._on_message)

    def _on_message(self, msg: Message) -> None:
        self.messages.append(msg)


async def _start_server(bus: MessageBus, port: int = _TEST_PORT) -> tuple[WebSocketProtocol, Any]:
    """Start a test WebSocket server and return (protocol, server)."""
    protocol = WebSocketProtocol(bus)
    server = await websockets.serve(
        protocol.handle_connection,
        "localhost",
        port,
    )
    return protocol, server


async def _stop_server(server: Any) -> None:
    """Stop a test WebSocket server."""
    server.close()
    await server.wait_closed()


# -- tests -----------------------------------------------------------------


class TestClientRegistration:
    """HRNS-4: WebSocket client registers role; subsequent frames have
    implicit sender."""

    @pytest.mark.asyncio
    async def test_client_registration(self) -> None:
        bus = MessageBus()
        recorder = _BusRecorder(bus)
        protocol, server = await _start_server(bus)

        # Start bus dispatch in a background thread.
        import threading

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            async with websockets.connect("ws://localhost:18765") as ws:
                # Register.
                await ws.send(json.dumps({"register": "trainee"}))
                # Give the server a moment to process.
                await asyncio.sleep(0.05)

                # Send a message frame.
                await ws.send(
                    json.dumps(
                        {
                            "role": "supervisor",
                            "action": "submit",
                            "message": "MHALL = SVO",
                        }
                    )
                )
                await asyncio.sleep(0.05)

        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)

        # The bus should have received the message with sender="trainee".
        submit_msgs = [m for m in recorder.messages if m.action == "submit"]
        assert len(submit_msgs) == 1
        msg = submit_msgs[0]
        assert msg.role == "supervisor"
        assert msg.action == "submit"
        assert msg.message == "MHALL = SVO"
        assert msg.sender == "trainee"


class TestDisconnectSilentDrop:
    """HRNS-21: Disconnected client — messages silently dropped."""

    @pytest.mark.asyncio
    async def test_disconnect_silent_drop(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18766)

        # Subscribe a "still alive" handler for a different role.
        alive_messages: list[Message] = []
        bus.subscribe("trainer", lambda m: alive_messages.append(m))

        # Start bus in a background thread.
        import threading

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Connect and register as "slack".
            ws = await websockets.connect("ws://localhost:18766")
            await ws.send(json.dumps({"register": "slack"}))
            await asyncio.sleep(0.05)

            # Disconnect the client.
            await ws.close()
            await asyncio.sleep(0.05)

            # Send a message to "slack" via the bus — should be silently
            # dropped (no exception, no crash).
            bus.send(
                Message(
                    role="slack",
                    action="notify",
                    message="hello",
                    sender="trainer",
                )
            )

            # Send to "trainer" — should still work.
            bus.send(
                Message(
                    role="trainer",
                    action="ping",
                    message="are you there?",
                    sender="kalvin",
                )
            )
            await asyncio.sleep(0.1)

        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)

        # No crash occurred (we got here).  Trainer should have received its
        # message.
        assert len(alive_messages) == 1
        assert alive_messages[0].role == "trainer"


class TestMultipleClientsSameRole:
    """Multiple clients registering for the same role both receive messages."""

    @pytest.mark.asyncio
    async def test_multiple_clients_same_role(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18767)

        import threading

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Two clients register for the same role.
            ws1 = await websockets.connect("ws://localhost:18767")
            ws2 = await websockets.connect("ws://localhost:18767")

            await ws1.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)
            await ws2.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)

            # Send a message to "supervisor" via the bus.
            bus.send(
                Message(
                    role="supervisor",
                    action="progress",
                    message="50%",
                    sender="trainer",
                )
            )
            await asyncio.sleep(0.1)

            # Both clients should receive the message.
            frame1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=2.0))
            frame2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=2.0))

            assert frame1["role"] == "supervisor"
            assert frame1["action"] == "progress"
            assert frame2["role"] == "supervisor"
            assert frame2["action"] == "progress"

            await ws1.close()
            await ws2.close()
        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)


class TestMalformedFrameError:
    """Non-JSON frame before registration produces an error frame."""

    @pytest.mark.asyncio
    async def test_malformed_frame_error(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18768)

        try:
            ws = await websockets.connect("ws://localhost:18768")
            # Send non-JSON.
            await ws.send("this is not json")
            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frame = json.loads(response)

            assert "error" in frame
            assert "not valid JSON" in frame["error"]

            await ws.close()
        finally:
            await _stop_server(server)


class TestSendToClient:
    """``send_to_client`` delivers a message to the registered client."""

    @pytest.mark.asyncio
    async def test_send_to_client(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18769)

        try:
            ws = await websockets.connect("ws://localhost:18769")
            await ws.send(json.dumps({"register": "ui"}))
            await asyncio.sleep(0.05)

            # Send via the protocol.
            msg = Message(
                role="ui",
                action="event",
                message={"type": "s1"},
                sender="kalvin",
            )
            await protocol.send_to_client("ui", msg)

            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frame = json.loads(response)

            assert frame["role"] == "ui"
            assert frame["action"] == "event"
            assert frame["message"] == {"type": "s1"}
            assert frame["sender"] == "kalvin"

            await ws.close()
        finally:
            await _stop_server(server)


class TestSendToClientSilentDrop:
    """``send_to_client`` silently drops when client is not connected."""

    @pytest.mark.asyncio
    async def test_send_to_client_no_connection(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18770)

        try:
            # Send to a non-existent client — should not raise.
            msg = Message(
                role="nobody",
                action="ping",
                message="hello",
            )
            await protocol.send_to_client("nobody", msg)
            # If we get here without exception, the test passes.
        finally:
            await _stop_server(server)


class TestMultipleClientsSameRoleFanOut:
    """HRNS-30: two WebSocket clients register for the same role and both
    receive messages sent to that role."""

    @pytest.mark.asyncio
    async def test_multiple_clients_same_role_both_receive(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18771)

        import threading

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Two clients connect and register for the same role.
            ws1 = await websockets.connect("ws://localhost:18771")
            ws2 = await websockets.connect("ws://localhost:18771")

            await ws1.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)
            await ws2.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)

            # Send a message to "supervisor" via the bus.
            bus.send(
                Message(
                    role="supervisor",
                    action="progress",
                    message="50%",
                    sender="trainer",
                )
            )
            await asyncio.sleep(0.1)

            # Both clients should receive the message.
            frame1 = json.loads(await asyncio.wait_for(ws1.recv(), timeout=2.0))
            frame2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=2.0))

            assert frame1["role"] == "supervisor"
            assert frame1["action"] == "progress"
            assert frame2["role"] == "supervisor"
            assert frame2["action"] == "progress"

            await ws1.close()
            await ws2.close()
        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)


class TestDisconnectOneOfTwoSameRole:
    """When one of two clients for the same role disconnects, the remaining
    client still receives messages."""

    @pytest.mark.asyncio
    async def test_disconnect_one_of_two_same_role(self) -> None:
        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18772)

        import threading

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            ws1 = await websockets.connect("ws://localhost:18772")
            ws2 = await websockets.connect("ws://localhost:18772")
            await ws1.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)
            await ws2.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)

            # Disconnect ws1.
            await ws1.close()
            await asyncio.sleep(0.05)

            # Send a message to "supervisor" via the bus.
            bus.send(
                Message(
                    role="supervisor",
                    action="event",
                    message="still alive",
                    sender="trainer",
                )
            )
            await asyncio.sleep(0.1)

            # ws2 should still receive.
            frame = json.loads(await asyncio.wait_for(ws2.recv(), timeout=2.0))
            assert frame["role"] == "supervisor"
            assert frame["action"] == "event"

            await ws2.close()
        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)


class TestDomainObjectPayloadSerialisation:
    """bus Messages carrying domain objects (``KLine``,
    ``RationaliseEvent``) must reach a connected WebSocket client.

    Regression for the silent drop at the wire boundary: before the fix,
    ``WebSocketProtocol._serialise_message`` called ``json.dumps(payload)``
    with no domain-object encoder, so ``action="event"`` (carrying a
    ``RationaliseEvent``) and ``action="ratify_request"`` (carrying a dict of
    ``KLine``\\s) raised ``TypeError``, which ``_ClientParticipant.on_message``
    swallowed as a "client gone" DEBUG log. Only plain-dict actions
    (``progress``, ``notify``) serialised — matching the observed symptom
    where ``events.jsonl`` only ever recorded ``connected``/``progress``.
    """

    @pytest.mark.asyncio
    async def test_event_with_rationalise_event_reaches_client(self) -> None:
        import threading

        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18773)

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            ws = await websockets.connect("ws://localhost:18773")
            await ws.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)

            # Relay a ground/frame event exactly as the Trainer does
            # (HRNS-33): message is the RationaliseEvent object itself.
            bus.send(
                Message(
                    role="supervisor",
                    action="event",
                    message=RationaliseEvent(
                        kind="frame",
                        query=KValue(KLine(0xAA, [0xAA]), significance=0),
                        proposal=KValue(KLine(0xBB, [0xBB, 0xCC]), significance=99),
                    ),
                    sender="trainer",
                )
            )
            await asyncio.sleep(0.1)

            # Frame must reach the client (times out before the fix).
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frame = json.loads(raw)

            assert frame["role"] == "supervisor"
            assert frame["action"] == "event"
            message = frame["message"]
            assert message["kind"] == "frame"
            assert "significance" not in message  # no top-level significance (KE-3)
            assert message["query"] == {"signature": 0xAA, "nodes": [0xAA], "significance": 0}
            assert message["proposal"] == {
                "signature": 0xBB,
                "nodes": [0xBB, 0xCC],
                "significance": 99,
            }

            await ws.close()
        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)

    @pytest.mark.asyncio
    async def test_ratify_request_with_kline_payload_reaches_client(self) -> None:
        import threading

        bus = MessageBus()
        protocol, server = await _start_server(bus, port=18774)

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            ws = await websockets.connect("ws://localhost:18774")
            await ws.send(json.dumps({"register": "supervisor"}))
            await asyncio.sleep(0.05)

            # Relay an S2/S3 ratify request exactly as the Trainer does
            # (HRNS-33): message is a dict whose query/proposal are KValues
            # (the trainer wraps event.query/event.proposal). The payload
            # retains its top-level "significance" key (a trainer-constructed
            # dict, not a RationaliseEvent).
            bus.send(
                Message(
                    role="supervisor",
                    action="ratify_request",
                    message={
                        "proposal": KValue(KLine(0xBB, [0xBB]), significance=99),
                        "query": KValue(KLine(0xAA, [0xAA]), significance=0),
                        "significance": 99,
                    },
                    sender="trainer",
                )
            )
            await asyncio.sleep(0.1)

            # Frame must reach the client (times out before the fix).
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            frame = json.loads(raw)

            assert frame["role"] == "supervisor"
            assert frame["action"] == "ratify_request"
            assert frame["message"] == {
                "proposal": {"signature": 0xBB, "nodes": [0xBB], "significance": 99},
                "query": {"signature": 0xAA, "nodes": [0xAA], "significance": 0},
                "significance": 99,
            }

            await ws.close()
        finally:
            bus.stop()
            bus_thread.join(timeout=5)
            await _stop_server(server)


class TestWireFrameRoundTrip:
    """wire frames produced by ``_serialise_message``'s domain-object
    encoder must be accepted by ``enrich_event`` and yield the documented
    enriched events.

    This is a pure unit round-trip (no bus / WebSocket / threading): it nails
    down that the encoder's *output shape* and ``enrich_event``'s *input shape*
    agree, so the auto-tune observable model is restored end-to-end. It
    complements ``TestDomainObjectPayloadSerialisation`` (which proves
    *delivery* through the live bus + WebSocket) by proving *shape agreement*.
    """

    def test_event_frame_enriches_to_rationalise(self) -> None:
        # Build the exact Message the Trainer relays, serialise it at the wire
        # boundary, parse the JSON back, and enrich it — the supervisor's path.
        msg = Message(
            role="supervisor",
            action="event",
            message=RationaliseEvent(
                kind="frame",
                query=KValue(KLine(0xAA, [0xAA]), significance=0),
                proposal=KValue(KLine(0xBB, [0xBB, 0xCC]), significance=99),
            ),
            sender="trainer",
        )
        frame = json.loads(WebSocketProtocol._serialise_message(msg))

        event = enrich_event(frame, seq=5)

        assert event["type"] == "rationalise"
        assert event["seq"] == 5
        assert event["kind"] == "frame"
        # Significance 99 (well below D_MAX) classifies as S3.
        assert event["significance"]["raw"] == 99
        assert event["significance"]["level"] == "S3"
        assert "normalised" in event["significance"]
        # KLines round-trip to their KLine Display Object raw shape.
        assert event["query"]["raw"] == {"signature": 0xAA, "nodes": [0xAA]}
        assert event["proposal"]["raw"] == {"signature": 0xBB, "nodes": [0xBB, 0xCC]}
        assert isinstance(event["query"]["source"], str)
        assert isinstance(event["proposal"]["source"], str)

    def test_ratify_request_frame_enriches_to_ratify_request(self) -> None:
        msg = Message(
            role="supervisor",
            action="ratify_request",
            message={
                "proposal": KValue(KLine(0xBB, [0xBB]), significance=99),
                "query": KValue(KLine(0xAA, [0xAA]), significance=0),
                "significance": 99,
            },
            sender="trainer",
        )
        frame = json.loads(WebSocketProtocol._serialise_message(msg))

        event = enrich_event(frame, seq=7)

        assert event["type"] == "ratify_request"
        assert event["seq"] == 7
        assert event["significance"]["raw"] == 99
        assert event["significance"]["level"] == "S3"
        assert event["query"]["raw"] == {"signature": 0xAA, "nodes": [0xAA]}
        assert event["proposal"]["raw"] == {"signature": 0xBB, "nodes": [0xBB]}
        assert isinstance(event["query"]["source"], str)
        assert isinstance(event["proposal"]["source"], str)
