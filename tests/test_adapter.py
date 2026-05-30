"""Tests for KAgent adapter — HRNS-7, HRNS-8, HRNS-9, HRNS-10, HRNS-22.

The adapter bridges the KAgent rationalisation pipeline and the addressed
message bus.  These tests verify compilation, sender-map addressing,
countersign forwarding, error handling, and direct KAgent→adapter callbacks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from harness.adapter import KAgentAdapter
from harness.bus import MessageBus
from harness.message import Message
from harness.protocols import Participant
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.signature import make_signature


# ── Helpers ──────────────────────────────────────────────────────────────


class FakeKAgent:
    """Minimal KAgent stub for unit-testing the adapter."""

    def __init__(self) -> None:
        self.rationalise = MagicMock(return_value=True)
        self.countersign = MagicMock(return_value=True)


class BusCapture:
    """Wraps a MessageBus and captures all messages sent via send()."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.messages: list[Message] = []
        self._original_send = bus.send

        def capturing_send(msg: Message) -> None:
            self.messages.append(msg)
            self._original_send(msg)

        bus.send = capturing_send  # type: ignore[assignment]

    def for_address(self, address: str) -> list[Message]:
        """Return captured messages addressed to *address*."""
        return [m for m in self.messages if m.address == address]

    def with_action(self, action: str) -> list[Message]:
        """Return captured messages with the given action."""
        return [m for m in self.messages if m.action == action]


# ── HRNS-7: Submit compiles and submits ──────────────────────────────────


class TestHRNS7SubmitCompilesAndSubmits:
    """HRNS-7: KAgent adapter compiles KScript and submits entries one at a time."""

    def test_submit_compiles_and_submits(self) -> None:
        """Compile KScript source and call rationalise for each entry."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        adapter.on_message(
            Message(address="kalvin", action="submit", message="MHALL = SVO", sender="trainer")
        )

        # compile_source("MHALL = SVO") produces multiple entries
        assert kagent.rationalise.call_count > 0, "rationalise should be called at least once"

    def test_submit_passes_compiled_entries(self) -> None:
        """Each call to rationalise receives a CompiledEntry (KLine subclass)."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        adapter.on_message(
            Message(address="kalvin", action="submit", message="MHALL = SVO", sender="trainer")
        )

        for call in kagent.rationalise.call_args_list:
            entry = call[0][0]
            assert isinstance(entry, KLine), f"Expected KLine, got {type(entry)}"


# ── HRNS-8: Compilation error response ──────────────────────────────────


class TestHRNS8CompilationErrorResponse:
    """HRNS-8: KAgent adapter sends compilation errors back to sender."""

    def test_compilation_error_response(self) -> None:
        """Invalid KScript triggers an error message to the sender."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        adapter.on_message(
            Message(address="kalvin", action="submit", message="!!! invalid !!", sender="trainer")
        )

        errors = capture.for_address("trainer")
        assert len(errors) == 1, f"Expected 1 error message, got {len(errors)}"
        assert errors[0].action == "error"
        assert "invalid" in str(errors[0].message).lower() or errors[0].message != ""

    def test_compilation_error_does_not_rationalise(self) -> None:
        """After a compilation error, rationalise is never called."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        adapter.on_message(
            Message(address="kalvin", action="submit", message="!!! bad !!", sender="trainer")
        )

        kagent.rationalise.assert_not_called()


# ── HRNS-9: Sender map response addressing ──────────────────────────────


class TestHRNS9SenderMapResponseAddressing:
    """HRNS-9: KAgent adapter maintains sender map; responses addressed to sender."""

    def test_sender_map_response_addressing(self) -> None:
        """Callback event is addressed to the original sender."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        # Submit valid KScript from "trainer"
        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )

        # The rationalise mock captured the entries
        assert kagent.rationalise.call_count > 0
        first_entry = kagent.rationalise.call_args_list[0][0][0]

        # Simulate KAgent callback for this entry
        event = RationaliseEvent("frame", first_entry, first_entry, 0)
        adapter.on_event(event)

        # Response should be addressed to "trainer"
        responses = capture.for_address("trainer")
        assert len(responses) == 1
        assert responses[0].action == "frame"
        assert responses[0].message is event

    def test_sender_map_records_entry_key(self) -> None:
        """Sender map correctly maps (sig, nodes) → sender."""
        bus = MessageBus()
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )

        first_entry = kagent.rationalise.call_args_list[0][0][0]
        key = (first_entry.signature, tuple(first_entry.nodes))
        assert adapter._sender_map[key] == "trainer"

    def test_different_senders_tracked_separately(self) -> None:
        """Entries from different senders map to their respective senders."""
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        # Submit from trainer
        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )
        entry_a = kagent.rationalise.call_args_list[0][0][0]

        # Submit from ui (reset mock to track separately)
        kagent.rationalise.reset_mock()
        adapter.on_message(
            Message(address="kalvin", action="submit", message="X = Y", sender="ui")
        )
        entry_x = kagent.rationalise.call_args_list[0][0][0]

        # Callback for entry_a → trainer
        adapter.on_event(RationaliseEvent("frame", entry_a, entry_a, 0))
        # Callback for entry_x → ui
        adapter.on_event(RationaliseEvent("frame", entry_x, entry_x, 0))

        trainer_msgs = capture.for_address("trainer")
        ui_msgs = capture.for_address("ui")

        assert len(trainer_msgs) == 1
        assert len(ui_msgs) == 1


# ── HRNS-10: Countersign action ─────────────────────────────────────────


class TestHRNS10CountersignAction:
    """HRNS-10: KAgent adapter handles countersign action."""

    def test_countersign_action(self) -> None:
        """Countersign message triggers kagent.countersign with the kline."""
        bus = MessageBus()
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        kline = KLine(0xABCD, [0x1234])
        adapter.on_message(
            Message(address="kalvin", action="countersign", message=kline, sender="trainer")
        )

        kagent.countersign.assert_called_once_with(kline)

    def test_countersign_does_not_rationalise(self) -> None:
        """Countersign action does not call rationalise."""
        bus = MessageBus()
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        kline = KLine(0xABCD, [0x1234])
        adapter.on_message(
            Message(address="kalvin", action="countersign", message=kline, sender="trainer")
        )

        kagent.rationalise.assert_not_called()


# ── HRNS-22: KAgent calls adapter directly ───────────────────────────────


class TestHRNS22KAgentCallsAdapterDirectly:
    """HRNS-22: KAgent calls adapter directly (no internal EventBus)."""

    def test_kagent_calls_adapter_directly(self) -> None:
        """Real KAgent with adapter as callback produces events via on_event."""
        from kalvin.agent import KAgent

        bus = MessageBus()
        capture = BusCapture(bus)
        adapter = KAgentAdapter(bus)

        # Create real KAgent with adapter as its adapter callback
        kagent = KAgent(adapter=adapter)
        adapter.bind(kagent)

        # Subscribe a handler on the bus to capture messages for "trainer"
        received: list[Message] = []
        bus.subscribe("trainer", lambda m: received.append(m))

        # Submit a simple KScript line via the adapter
        # "A = B" produces entries with tokenised nodes
        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )

        # The KAgent may produce fast-path events synchronously (S4/S1)
        # or slow-path events via Cogitator thread.
        # Stop the cogitator to flush any pending work.
        kagent.cogitate_join(timeout=5.0)

        # At least one event should have been routed back to "trainer"
        # through the adapter's on_event → bus.send path.
        # Note: if all entries hit S1 fast-path, events are published
        # synchronously during rationalise. S4 entries also publish synchronously.
        assert len(received) > 0 or len(capture.for_address("trainer")) > 0, (
            "Expected at least one event routed back to trainer via adapter.on_event"
        )

    def test_no_eventbus_in_pipeline(self) -> None:
        """Verify the adapter's on_event is called (not an internal EventBus)."""
        from kalvin.agent import KAgent

        bus = MessageBus()
        adapter = KAgentAdapter(bus)

        # Spy on adapter.on_event
        original_on_event = adapter.on_event
        events_received: list[RationaliseEvent] = []
        call_lock = threading.Lock()

        def spying_on_event(event: RationaliseEvent) -> None:
            with call_lock:
                events_received.append(event)
            original_on_event(event)

        adapter.on_event = spying_on_event  # type: ignore[assignment]

        kagent = KAgent(adapter=adapter)
        adapter.bind(kagent)

        # Submit a KScript line — should trigger events
        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )

        kagent.cogitate_join(timeout=5.0)

        with call_lock:
            assert len(events_received) > 0, (
                "adapter.on_event should be called by KAgent (no internal EventBus)"
            )


# ── Additional coverage ─────────────────────────────────────────────────


class TestUnknownAction:
    """Unknown actions are silently ignored."""

    def test_unknown_action_ignored(self) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        # Should not raise
        adapter.on_message(
            Message(address="kalvin", action="unknown_action", message="data", sender="trainer")
        )

        kagent.rationalise.assert_not_called()
        kagent.countersign.assert_not_called()

        # No error sent back
        errors = capture.with_action("error")
        assert len(errors) == 0


class TestOrphanEvent:
    """Events with no sender in the sender map are silently dropped."""

    def test_orphan_event_dropped(self) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        kagent = FakeKAgent()
        adapter = KAgentAdapter(bus, kagent=kagent)

        # Simulate a callback with no matching sender
        orphan_kline = KLine(0xDEAD, [0xBEEF])
        event = RationaliseEvent("done", orphan_kline, orphan_kline, 0)
        adapter.on_event(event)

        # No message should be sent to the bus
        assert len(capture.messages) == 0


class TestAdapterRegistersOnBus:
    """Adapter subscribes to the bus at construction time."""

    def test_registers_on_construction(self) -> None:
        bus = MessageBus()
        adapter = KAgentAdapter(bus, kagent=FakeKAgent())

        # The adapter should be registered for "kalvin" address
        assert "kalvin" in bus._handlers
        assert bus._handlers["kalvin"][-1] == adapter.on_message

    def test_custom_address(self) -> None:
        bus = MessageBus()
        adapter = KAgentAdapter(bus, address="custom", kagent=FakeKAgent())

        assert adapter.address == "custom"
        assert "custom" in bus._handlers


class TestNoKAgentBound:
    """Operations without a bound KAgent are handled gracefully."""

    def test_submit_without_kagent(self) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        adapter = KAgentAdapter(bus)  # No kagent

        # Should not raise
        adapter.on_message(
            Message(address="kalvin", action="submit", message="A = B", sender="trainer")
        )

        # No error sent for submit (only compilation attempted if kagent exists)
        # Actually, with no kagent, the adapter returns early

    def test_countersign_without_kagent(self) -> None:
        bus = MessageBus()
        adapter = KAgentAdapter(bus)  # No kagent

        kline = KLine(0xABCD, [0x1234])
        # Should not raise
        adapter.on_message(
            Message(address="kalvin", action="countersign", message=kline, sender="trainer")
        )
