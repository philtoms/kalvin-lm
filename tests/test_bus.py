"""Tests for Message dataclass and MessageBus.

Covers spec criteria HRNS-1, HRNS-2, HRNS-3, HRNS-11, HRNS-23
and additional behavioural tests.
"""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError

import pytest

from harness.bus import MessageBus
from harness.message import Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for(event: threading.Event, timeout: float = 2.0) -> None:
    """Block until *event* is set, failing the test on timeout."""
    assert event.wait(timeout), "Timed out waiting for event"


# ---------------------------------------------------------------------------
# HRNS-1: Message bus routes by address to correct subscriber
# ---------------------------------------------------------------------------

class TestRouteByAddress:
    """HRNS-1: route messages to the handler subscribed for the address."""

    def test_route_by_address(self) -> None:
        bus = MessageBus()
        kalvin_msgs: list[Message] = []

        def kalvin_handler(msg: Message) -> None:
            kalvin_msgs.append(msg)

        bus.subscribe("kalvin", kalvin_handler)
        msg = Message(address="kalvin", action="submit", message="MHALL = SVO")
        bus.send(msg)

        # Run inline (no thread needed for single-message test)
        bus._dispatch(bus._queue.get())  # noqa: SLF001

        assert len(kalvin_msgs) == 1
        assert kalvin_msgs[0] is msg

    def test_other_address_does_not_receive(self) -> None:
        bus = MessageBus()
        kalvin_msgs: list[Message] = []
        trainer_msgs: list[Message] = []

        bus.subscribe("kalvin", lambda m: kalvin_msgs.append(m))
        bus.subscribe("trainer", lambda m: trainer_msgs.append(m))

        bus.send(Message(address="kalvin", action="submit", message="x"))

        bus._dispatch(bus._queue.get())  # noqa: SLF001

        assert len(kalvin_msgs) == 1
        assert len(trainer_msgs) == 0


# ---------------------------------------------------------------------------
# HRNS-2: Thread-safe send from another thread
# ---------------------------------------------------------------------------

class TestThreadsafeSend:
    """HRNS-2: message sent from a different thread arrives correctly."""

    def test_threadsafe_send(self) -> None:
        bus = MessageBus()
        received: list[Message] = []
        event = threading.Event()

        def handler(msg: Message) -> None:
            received.append(msg)
            event.set()

        bus.subscribe("kalvin", handler)

        # Start run() in a background thread.
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        # Send from the main thread.
        msg = Message(address="kalvin", action="submit", message="thread test")
        bus.send(msg)

        _wait_for(event)
        bus.stop()
        bus_thread.join(timeout=2)

        assert len(received) == 1
        assert received[0].message == "thread test"


# ---------------------------------------------------------------------------
# HRNS-3: Unknown address produces error response to sender
# ---------------------------------------------------------------------------

class TestUnknownAddressError:
    """HRNS-3: unknown address sends error back to sender."""

    def test_unknown_address_error(self) -> None:
        bus = MessageBus()
        origin_msgs: list[Message] = []
        event = threading.Event()

        def origin_handler(msg: Message) -> None:
            origin_msgs.append(msg)
            event.set()

        # Subscribe "origin" to receive error responses.
        bus.subscribe("origin", origin_handler)

        # Send to an unknown address with sender set.
        bus.send(Message(address="nonexistent", action="ping", message="hi", sender="origin"))

        # Run the bus loop in a background thread.
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        _wait_for(event, timeout=2)
        bus.stop()
        bus_thread.join(timeout=2)

        assert len(origin_msgs) == 1
        error = origin_msgs[0]
        assert error.address == "origin"
        assert error.action == "error"
        assert "nonexistent" in error.message


# ---------------------------------------------------------------------------
# HRNS-11: Wildcard diagnostic listener receives all messages
# ---------------------------------------------------------------------------

class TestWildcardDiagnosticListener:
    """HRNS-11: wildcard subscribers receive every dispatched message."""

    def test_wildcard_receives_addressed_message(self) -> None:
        bus = MessageBus()
        handler_a_msgs: list[Message] = []
        handler_wild_msgs: list[Message] = []
        event = threading.Event()

        def handler_a(msg: Message) -> None:
            handler_a_msgs.append(msg)

        def handler_wild(msg: Message) -> None:
            handler_wild_msgs.append(msg)
            event.set()

        bus.subscribe("kalvin", handler_a)
        bus.subscribe("*", handler_wild)

        msg = Message(address="kalvin", action="submit", message="wild test")
        bus.send(msg)

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        _wait_for(event)
        bus.stop()
        bus_thread.join(timeout=2)

        assert len(handler_a_msgs) == 1
        assert len(handler_wild_msgs) == 1
        assert handler_wild_msgs[0] is msg

    def test_wildcard_receives_unknown_address_message(self) -> None:
        bus = MessageBus()
        wild_msgs: list[Message] = []
        origin_msgs: list[Message] = []

        wild_event = threading.Event()
        origin_event = threading.Event()

        def wild_handler(msg: Message) -> None:
            wild_msgs.append(msg)
            wild_event.set()

        def origin_handler(msg: Message) -> None:
            origin_msgs.append(msg)
            origin_event.set()

        bus.subscribe("*", wild_handler)
        bus.subscribe("origin", origin_handler)

        # Send to unknown address with a sender.
        bus.send(Message(address="trainer", action="ping", message="?", sender="origin"))

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        # Wait for both: the wildcard sees the original message AND the error.
        # The wildcard will be called for the original message first, then for
        # the error message that gets re-enqueued.
        wild_event.wait(timeout=2)
        origin_event.wait(timeout=2)
        bus.stop()
        bus_thread.join(timeout=2)

        # Wildcard should have received the original message.
        assert any(m.address == "trainer" for m in wild_msgs)
        # Origin handler should have received the error.
        assert len(origin_msgs) == 1
        assert origin_msgs[0].action == "error"


# ---------------------------------------------------------------------------
# HRNS-23: Single dispatch thread
# ---------------------------------------------------------------------------

class TestSingleDispatchThread:
    """HRNS-23: all handlers execute on the same event-loop thread."""

    def test_single_dispatch_thread(self) -> None:
        bus = MessageBus()
        dispatch_threads: list[threading.Thread] = []
        event = threading.Event()
        message_count = 5

        def handler(msg: Message) -> None:
            dispatch_threads.append(threading.current_thread())
            if len(dispatch_threads) == message_count:
                event.set()

        bus.subscribe("test", handler)

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        for i in range(message_count):
            bus.send(Message(address="test", action="tick", message=i))

        _wait_for(event)
        bus.stop()
        bus_thread.join(timeout=2)

        assert len(dispatch_threads) == message_count
        # All dispatches happened on the same thread.
        assert len(set(t.ident for t in dispatch_threads)) == 1
        # And that thread is not the main thread.
        assert dispatch_threads[0].ident != threading.current_thread().ident


# ---------------------------------------------------------------------------
# Additional: Message dataclass properties
# ---------------------------------------------------------------------------

class TestMessageDataclass:
    """Message immutability and repr."""

    def test_frozen(self) -> None:
        msg = Message(address="a", action="b", message="c")
        with pytest.raises(FrozenInstanceError):
            msg.address = "z"  # type: ignore[misc]

    def test_repr(self) -> None:
        msg = Message(address="kalvin", action="submit", message="data", sender="trainer")
        r = repr(msg)
        assert "kalvin" in r
        assert "submit" in r
        assert "trainer" in r
        # repr should NOT include the message payload (for clarity)
        assert "data" not in r

    def test_sender_default_none(self) -> None:
        msg = Message(address="a", action="b", message="c")
        assert msg.sender is None

    def test_sender_set(self) -> None:
        msg = Message(address="a", action="b", message="c", sender="origin")
        assert msg.sender == "origin"


# ---------------------------------------------------------------------------
# Additional: Graceful shutdown
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    """stop() causes run() to return."""

    def test_stop_graceful_shutdown(self) -> None:
        bus = MessageBus()
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        bus.stop()
        bus_thread.join(timeout=2)

        assert not bus_thread.is_alive()


# ---------------------------------------------------------------------------
# Additional: Multiple handlers per address
# ---------------------------------------------------------------------------

class TestMultipleHandlers:
    """Multiple handlers registered for the same address all receive messages."""

    def test_multiple_handlers_per_address(self) -> None:
        bus = MessageBus()
        received_a: list[Message] = []
        received_b: list[Message] = []
        event = threading.Event()

        def handler_a(msg: Message) -> None:
            received_a.append(msg)

        def handler_b(msg: Message) -> None:
            received_b.append(msg)
            event.set()

        bus.subscribe("kalvin", handler_a)
        bus.subscribe("kalvin", handler_b)

        bus.send(Message(address="kalvin", action="submit", message="multi"))

        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        _wait_for(event)
        bus.stop()
        bus_thread.join(timeout=2)

        assert len(received_a) == 1
        assert len(received_b) == 1
