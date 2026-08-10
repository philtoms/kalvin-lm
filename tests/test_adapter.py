"""Tests for Rationaliser adapter.
The adapter bridges the Rationaliser rationalisation pipeline and the role-based
message bus.  These tests verify compilation, sender-map routing,
countersign forwarding, error handling, and direct Rationaliser→adapter callbacks.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from kalvin.events import RationaliseEvent
from kalvin.significance import SIG_S1, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.model import Model
from tests.conftest import requires_tokenizer_data
from training.harness.adapter import RationaliserAdapter, _materialise_kvalue
from training.harness.bus import MessageBus
from training.harness.constants import TRAINEE_ROLE
from training.harness.message import Message

# ── Helpers ──────────────────────────────────────────────────────────────


class FakeRationaliser:
    """Minimal Rationaliser stub for unit-testing the adapter."""

    def __init__(self) -> None:
        self.rationalise = MagicMock(return_value=True)
        self.countersign = MagicMock(return_value=True)
        self.save = MagicMock()
        self._model = Model()
        self._activity = {}
        self._cogitator = MagicMock()


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

    def for_role(self, role: str) -> list[Message]:
        """Return captured messages for *role*."""
        return [m for m in self.messages if m.role == role]

    def with_action(self, action: str) -> list[Message]:
        """Return captured messages with the given action."""
        return [m for m in self.messages if m.action == action]


# ── _materialise_kvalue unit tests ────────────────────────────────────────


class TestMaterialiseKValue:
    """``_materialise_kvalue`` accepts a live KValue, a wire dict, and a
    legacy KLine (wrapped at SIG_S1); raises TypeError otherwise.

    These exercise the materialisation boundary directly and require no
    tokenizer data, so they run on a fresh clone.
    """

    def test_live_kvalue_passthrough(self) -> None:
        """A live KValue is returned unchanged (in-process auto-countersign)."""
        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S1)
        result = _materialise_kvalue(kv)
        assert result is kv

    def test_wire_dict_construction(self) -> None:
        """A wire dict {signature, nodes, significance} builds a KValue."""
        sig = 0xCAFEBABE
        wire = {"signature": 0xABCD, "nodes": [0x1234, 0x5678], "significance": sig}
        result = _materialise_kvalue(wire)
        assert isinstance(result, KValue)
        assert result.kline == KLine(0xABCD, [0x1234, 0x5678])
        assert result.significance == sig

    def test_wire_dict_missing_significance_raises(self) -> None:
        """A wire dict without 'significance' raises TypeError (fail-loud).

        This is the legacy two-key wire shape ``{signature, nodes}`` — now
        malformed because significance rides on the KValue.
        """
        wire = {"signature": 0xABCD, "nodes": [0x1234]}
        with pytest.raises(TypeError):
            _materialise_kvalue(wire)

    def test_legacy_kline_wrapped_at_s1(self) -> None:
        """A legacy bare KLine is wrapped at SIG_S1 (: countersign is S1)."""
        kline = KLine(0xABCD, [0x1234])
        result = _materialise_kvalue(kline)
        assert isinstance(result, KValue)
        assert result.kline is kline
        assert result.significance == SIG_S1

    def test_malformed_input_raises_typeerror(self) -> None:
        """Any other type raises TypeError (fail-loud)."""
        for bad in (42, "not a kvalue", None, [1, 2, 3]):
            with pytest.raises(TypeError):
                _materialise_kvalue(bad)


# ── Submit compiles and submits ──────────────────────────────────


class TestHRNS7SubmitCompilesAndSubmits:
    """Rationaliser adapter compiles KScript and submits entries one at a time."""

    @requires_tokenizer_data
    def test_submit_compiles_and_submits(self) -> None:
        """Compile KScript source and call rationalise for each entry."""
        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="MHALL = SVO", sender="trainer")
        )

        # compile_source("MHALL = SVO") produces multiple entries
        assert rationaliser.rationalise.call_count > 0, "rationalise should be called at least once"

    @requires_tokenizer_data
    def test_submit_passes_compiled_entries(self) -> None:
        """Each call to rationalise receives a KValue (compile_source returns KValues)."""
        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="MHALL = SVO", sender="trainer")
        )

        for call in rationaliser.rationalise.call_args_list:
            entry = call[0][0]
            assert isinstance(entry, KValue), f"Expected KValue, got {type(entry)}"

    def test_submit_passes_kvalue_to_rationalise(self, monkeypatch) -> None:
        """rationalise receives a KValue; sender-map key uses entry.kline.

        compile_source is patched so this runs without tokenizer data.
        Verifies the bus-side KValue boundary: submit consumes list[KValue]
        and forwards the KValue (not the bare KLine) to rationalise.
        """
        import training.harness.adapter as adapter_mod

        kline_a = KLine(0xAA, [0x11])
        kline_b = KLine(0xBB, [0x22])
        fake_entries = [KValue(kline_a, SIG_S1), KValue(kline_b, SIG_S1)]
        monkeypatch.setattr(adapter_mod, "compile_source", lambda *a, **k: fake_entries)

        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(
                role=TRAINEE_ROLE,
                action="submit",
                message="irrelevant; patched",
                sender="trainer",
            )
        )

        assert rationaliser.rationalise.call_count == 2
        for call in rationaliser.rationalise.call_args_list:
            value = call[0][0]
            assert isinstance(value, KValue), f"Expected KValue, got {type(value)}"

        # Sender-map key built from entry.kline (signature, frozen nodes).
        key_a = (kline_a.signature, tuple(kline_a.nodes))
        key_b = (kline_b.signature, tuple(kline_b.nodes))
        assert adapter._sender_map[key_a] == "trainer"
        assert adapter._sender_map[key_b] == "trainer"


# ── Compilation error response ──────────────────────────────────


class TestHRNS8CompilationErrorResponse:
    """Rationaliser adapter sends compilation errors back to sender."""

    def test_compilation_error_response(self) -> None:
        """Invalid KScript triggers an error message to the sender."""
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="!!! invalid !!", sender="trainer")
        )

        errors = capture.for_role("trainer")
        assert len(errors) == 1, f"Expected 1 error message, got {len(errors)}"
        assert errors[0].action == "error"
        assert "invalid" in str(errors[0].message).lower() or errors[0].message != ""

    def test_compilation_error_does_not_rationalise(self) -> None:
        """After a compilation error, rationalise is never called."""
        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="!!! bad !!", sender="trainer")
        )

        rationaliser.rationalise.assert_not_called()


# ── Sender map response routing ──────────────────────────────


@requires_tokenizer_data
class TestHRNS9SenderMapResponseRouting:
    """Rationaliser adapter maintains sender map; responses routed to sender."""

    def test_sender_map_response_routing(self) -> None:
        """Callback event is routed to the original sender."""
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        # Submit valid KScript from "trainer"
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )

        # The rationalise mock captured the entries
        assert rationaliser.rationalise.call_count > 0
        first_entry = rationaliser.rationalise.call_args_list[0][0][0]

        # Simulate Rationaliser callback for this entry (query/proposal are KValues).
        event = RationaliseEvent("frame", first_entry, first_entry)
        adapter.on_event(event)

        # Response should be routed to "trainer"
        responses = capture.for_role("trainer")
        assert len(responses) == 1
        assert responses[0].action == "frame"
        assert responses[0].message is event

    def test_sender_map_records_entry_key(self) -> None:
        """Sender map correctly maps (sig, nodes) → sender."""
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )

        first_entry = rationaliser.rationalise.call_args_list[0][0][0]
        key = (first_entry.kline.signature, tuple(first_entry.kline.nodes))
        assert adapter._sender_map[key] == "trainer"

    def test_different_senders_tracked_separately(self) -> None:
        """Entries from different senders map to their respective senders."""
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        # Submit from trainer
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )
        entry_a = rationaliser.rationalise.call_args_list[0][0][0]

        # Submit from ui (reset mock to track separately)
        rationaliser.rationalise.reset_mock()
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="X = Y", sender="ui")
        )
        entry_x = rationaliser.rationalise.call_args_list[0][0][0]

        # Callback for entry_a → trainer (query/proposal are KValues)
        adapter.on_event(RationaliseEvent("frame", entry_a, entry_a))
        # Callback for entry_x → ui
        adapter.on_event(RationaliseEvent("frame", entry_x, entry_x))

        trainer_msgs = capture.for_role("trainer")
        ui_msgs = capture.for_role("ui")

        assert len(trainer_msgs) == 1
        assert len(ui_msgs) == 1


# ── Countersign action ─────────────────────────────────────────


class TestHRNS10CountersignAction:
    """Rationaliser adapter handles countersign action."""

    def test_countersign_live_kvalue(self) -> None:
        """A live KValue payload is passed through unchanged to countersign.

        This is the in-process auto-countersign path (the reactor posts the
        proposal KValue directly on the bus). The KValue must arrive at
        ``countersign`` as the same object — no wrapping or reconstruction.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S1)
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="countersign", message=kv, sender="trainer")
        )

        rationaliser.countersign.assert_called_once_with(kv)

    def test_countersign_action(self) -> None:
        """Countersign message triggers rationaliser.countersign with the payload.

        A legacy bare KLine is wrapped at SIG_S1 (: countersign is an S1
        ratification) before being handed to ``countersign`` as a KValue.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kline = KLine(0xABCD, [0x1234])
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="countersign", message=kline, sender="trainer")
        )

        rationaliser.countersign.assert_called_once()
        call_arg = rationaliser.countersign.call_args[0][0]
        assert isinstance(call_arg, KValue), f"Expected KValue, got {type(call_arg)}"
        assert call_arg.kline == kline
        assert call_arg.significance == SIG_S1

    def test_countersign_does_not_rationalise(self) -> None:
        """Countersign action does not call rationalise."""
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kline = KLine(0xABCD, [0x1234])
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="countersign", message=kline, sender="trainer")
        )

        rationaliser.rationalise.assert_not_called()

    def test_countersign_materialises_wire_dict(self) -> None:
        """A wire-dict payload is materialised to a KValue before countersign.

        Regression guard: a countersign frame that traversed the WebSocket
        arrives as a plain dict (the canonical KValue wire shape produced by
        the harness's outbound encoder). Without materialisation
        ``Rationaliser.countersign`` would touch ``.kline`` on a ``dict`` and raise
        ``AttributeError``, killing the bus-dispatch thread and stalling the
        training run.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        sig = 0xCAFEBABE
        wire = {"signature": 0xABCD, "nodes": [0x1234, 0x5678], "significance": sig}
        adapter.on_message(
            Message(
                role=TRAINEE_ROLE,
                action="countersign",
                message=wire,
                sender="supervisor",
            )
        )

        rationaliser.countersign.assert_called_once()
        call_arg = rationaliser.countersign.call_args[0][0]
        assert isinstance(call_arg, KValue)
        # Compared via KLine.__eq__ (signature + nodes).
        assert call_arg.kline == KLine(0xABCD, [0x1234, 0x5678])
        assert call_arg.significance == sig
        # No AttributeError raised — the crash is gone.


# ── Rationalise action (two-way significance dialog) ───────────────────────


class TestRationaliseAction:
    """The ``rationalise`` action delivers a participant-constructed KValue
    straight to ``rationaliser.rationalise`` — the path a participant uses to hand
    Kalvin a KValue with its own declared significance (the MVP uses it for a
    declared-S4 drop signal). Mirrors 's countersign shape but routes
    to ``rationalise`` instead of ``countersign``.
    """

    def test_rationalise_live_kvalue(self) -> None:
        """A live KValue payload is passed through unchanged to rationalise.

        The KValue's significance is the sender's declared assessment and must
        arrive intact — unlike ``countersign`` (which forces SIG_S1) and
        ``submit`` (which re-derives significance from structure), this action
        preserves the sender's declared value.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S4)
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender="trainer")
        )

        rationaliser.rationalise.assert_called_once_with(kv)
        rationaliser.countersign.assert_not_called()

    def test_rationalise_materialises_wire_dict(self) -> None:
        """A wire-dict payload is materialised to a KValue before rationalise.

        A declared-S4 drop signal may arrive from a remote participant as a
        plain dict (the canonical KValue wire shape). Materialisation must
        preserve the declared significance.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        wire = {"signature": 0xABCD, "nodes": [0x1234, 0x5678], "significance": SIG_S4}
        adapter.on_message(
            Message(
                role=TRAINEE_ROLE,
                action="rationalise",
                message=wire,
                sender="trainer",
            )
        )

        rationaliser.rationalise.assert_called_once()
        call_arg = rationaliser.rationalise.call_args[0][0]
        assert isinstance(call_arg, KValue)
        assert call_arg.kline == KLine(0xABCD, [0x1234, 0x5678])
        assert call_arg.significance == SIG_S4

    def test_rationalise_wire_dict_missing_significance_raises(self) -> None:
        """A wire dict missing ``significance`` fails loud (fail-loud contract).

        Reuses ``_materialise_kvalue``: a declared significance is mandatory,
        so a malformed payload cannot silently become an S0/S4 drop.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        wire = {"signature": 0xABCD, "nodes": [0x1234]}
        with pytest.raises(TypeError):
            adapter.on_message(
                Message(
                    role=TRAINEE_ROLE,
                    action="rationalise",
                    message=wire,
                    sender="trainer",
                )
            )

    def test_rationalise_does_not_countersign(self) -> None:
        """Rationalise action does not call countersign."""
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S4)
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender="trainer")
        )

        rationaliser.countersign.assert_not_called()

    def test_rationalise_no_rationaliser_is_safe(self) -> None:
        """No bound Rationaliser → logs and returns, no crash."""
        bus = MessageBus()
        adapter = RationaliserAdapter(bus, rationaliser=None)

        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S4)
        # Must not raise.
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender="trainer")
        )

    def test_rationalise_records_sender_for_routing(self) -> None:
        """A rationalise submission records the sender so its events route back.

        Mirrors ``submit``: ``on_event`` looks the query kline up in the
        sender map and orphan-drops anything absent. Without this record, a
        paced KValue submission (e.g. a dialogue runner or the future paced
        Trainer handing Kalvin withheld klines one at a time) would never
        receive the events Kalvin emits about it.
        """
        bus = MessageBus()
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        kv = KValue(KLine(0xABCD, [0x1234]), SIG_S4)
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender="trainer")
        )

        key = (0xABCD, (0x1234,))
        assert adapter._sender_map[key] == "trainer"


# ── Rationaliser calls adapter directly ───────────────────────────────


@requires_tokenizer_data
class TestHRNS22RationaliserCallsAdapterDirectly:
    """Rationaliser calls adapter directly (no internal EventBus)."""

    def test_rationaliser_calls_adapter_directly(self) -> None:
        """Real Rationaliser with adapter as callback produces events via on_event."""
        from kalvin.rationaliser import Rationaliser

        bus = MessageBus()
        capture = BusCapture(bus)
        adapter = RationaliserAdapter(bus)

        # Create real Rationaliser with adapter as its adapter callback
        rationaliser = Rationaliser(adapter=adapter)
        adapter.bind(rationaliser)

        # Subscribe a handler on the bus to capture messages for "trainer"
        received: list[Message] = []
        bus.subscribe("trainer", lambda m: received.append(m))

        # Submit a simple KScript line via the adapter
        # "A = B" produces entries with tokenised nodes
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )

        # The Rationaliser may produce fast-path events synchronously (S4/S1)
        # or slow-path events via Cogitator thread.
        # Stop the cogitator to flush any pending work.
        rationaliser.cogitate_join(timeout=5.0)

        # At least one event should have been routed back to "trainer"
        # through the adapter's on_event → bus.send path.
        # Note: if all entries hit S1 fast-path, events are published
        # synchronously during rationalise. S4 entries also publish synchronously.
        assert len(received) > 0 or len(capture.for_role("trainer")) > 0, (
            "Expected at least one event routed back to trainer via adapter.on_event"
        )

    def test_no_eventbus_in_pipeline(self) -> None:
        """Verify the adapter's on_event is called (not an internal EventBus)."""
        from kalvin.rationaliser import Rationaliser

        bus = MessageBus()
        adapter = RationaliserAdapter(bus)

        # Spy on adapter.on_event
        original_on_event = adapter.on_event
        events_received: list[RationaliseEvent] = []
        call_lock = threading.Lock()

        def spying_on_event(event: RationaliseEvent) -> None:
            with call_lock:
                events_received.append(event)
            original_on_event(event)

        adapter.on_event = spying_on_event  # type: ignore[assignment]

        rationaliser = Rationaliser(adapter=adapter)
        adapter.bind(rationaliser)

        # Submit a KScript line — should trigger events
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )

        rationaliser.cogitate_join(timeout=5.0)

        with call_lock:
            assert len(events_received) > 0, (
                "adapter.on_event should be called by Rationaliser (no internal EventBus)"
            )


# ── Additional coverage ─────────────────────────────────────────────────


class TestUnknownAction:
    """Unknown actions are silently ignored."""

    def test_unknown_action_ignored(self) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        # Should not raise
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="unknown_action", message="data", sender="trainer")
        )

        rationaliser.rationalise.assert_not_called()
        rationaliser.countersign.assert_not_called()

        # No error sent back
        errors = capture.with_action("error")
        assert len(errors) == 0


class TestOrphanEvent:
    """Events with no sender in the sender map are silently dropped."""

    def test_orphan_event_dropped(self) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        # Simulate a callback with no matching sender
        orphan_kline = KLine(0xDEAD, [0xBEEF])
        orphan_value = KValue(orphan_kline, SIG_S1)
        event = RationaliseEvent("done", orphan_value, orphan_value)
        adapter.on_event(event)

        # No message should be sent to the bus
        assert len(capture.messages) == 0


class TestAdapterRegistersOnBus:
    """Adapter subscribes to the bus at construction time."""

    def test_registers_on_construction(self) -> None:
        bus = MessageBus()
        adapter = RationaliserAdapter(bus, rationaliser=FakeRationaliser())

        # The adapter should be registered for the trainee role
        assert TRAINEE_ROLE in bus._handlers
        assert bus._handlers[TRAINEE_ROLE][-1] == adapter.on_message

    def test_custom_role(self) -> None:
        bus = MessageBus()
        adapter = RationaliserAdapter(bus, role="custom", rationaliser=FakeRationaliser())

        assert adapter.role == "custom"
        assert "custom" in bus._handlers


class TestNoRationaliserBound:
    """Operations without a bound Rationaliser are handled gracefully."""

    def test_submit_without_rationaliser(self) -> None:
        bus = MessageBus()
        BusCapture(bus)
        adapter = RationaliserAdapter(bus)  # No rationaliser

        # Should not raise
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="submit", message="A = B", sender="trainer")
        )

        # No error sent for submit (only compilation attempted if rationaliser exists)
        # Actually, with no rationaliser, the adapter returns early

    def test_countersign_without_rationaliser(self) -> None:
        bus = MessageBus()
        adapter = RationaliserAdapter(bus)  # No rationaliser

        kline = KLine(0xABCD, [0x1234])
        # Should not raise
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="countersign", message=kline, sender="trainer")
        )


# ── Save / Load actions ──────────────────────────────────────────────


class TestSaveAction:
    """Save action persists Kalvin's model via agent_codec."""

    def test_save_calls_rationaliser_save(self, tmp_path) -> None:
        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        save_path = str(tmp_path / "model.bin")
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="save", message=save_path, sender="supervisor")
        )

        rationaliser.save.assert_called_once_with(save_path)

    def test_save_sends_confirmation(self, tmp_path) -> None:
        bus = MessageBus()
        capture = BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        save_path = str(tmp_path / "model.bin")
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="save", message=save_path, sender="supervisor")
        )

        saved_msgs = capture.with_action("saved")
        assert len(saved_msgs) == 1
        assert saved_msgs[0].message["path"] == save_path

    def test_save_uses_default_path(self) -> None:
        bus = MessageBus()
        BusCapture(bus)
        rationaliser = FakeRationaliser()
        adapter = RationaliserAdapter(bus, rationaliser=rationaliser)

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="save", message=None, sender="supervisor")
        )

        from kalvin.paths import agent_bin

        rationaliser.save.assert_called_once_with(str(agent_bin()))

    def test_save_without_rationaliser(self) -> None:
        bus = MessageBus()
        adapter = RationaliserAdapter(bus)  # No rationaliser

        # Should not raise
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="save", message="test.bin", sender="supervisor")
        )


class TestLoadAction:
    """Load action replaces Kalvin's model via agent_codec."""

    @requires_tokenizer_data
    def test_load_replaces_model(self, tmp_path) -> None:
        from kalvin.rationaliser import Rationaliser

        bus = MessageBus()
        BusCapture(bus)
        adapter = RationaliserAdapter(bus)
        rationaliser = Rationaliser(adapter=adapter)
        adapter.bind(rationaliser)

        # Teach the agent something so the model is non-empty.
        # rationalise takes a KValue; an empty-nodes KLine is an
        # identity/S4 entry — wrap it at SIG_S1.
        rationaliser.rationalise(KValue(KLine(0xFF, []), SIG_S1))
        old_model = rationaliser._model
        assert len(old_model) > 0

        # Save to file
        save_path = tmp_path / "model.bin"
        rationaliser.save(str(save_path))

        # Clear model to prove load restores it
        rationaliser._model = Model()
        assert len(rationaliser._model) == 0

        # Load via adapter
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="load", message=str(save_path), sender="supervisor")
        )

        # Model should be restored
        assert len(rationaliser._model) > 0

    @requires_tokenizer_data
    def test_load_sends_confirmation(self, tmp_path) -> None:
        from kalvin.rationaliser import Rationaliser

        bus = MessageBus()
        capture = BusCapture(bus)
        adapter = RationaliserAdapter(bus)
        rationaliser = Rationaliser(adapter=adapter)
        adapter.bind(rationaliser)

        # Save a model first
        save_path = tmp_path / "model.bin"
        rationaliser.save(str(save_path))

        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="load", message=str(save_path), sender="supervisor")
        )

        loaded_msgs = capture.with_action("loaded")
        assert len(loaded_msgs) == 1
        assert loaded_msgs[0].message["path"] == str(save_path)
        assert "frame_size" in loaded_msgs[0].message

    @requires_tokenizer_data
    def test_load_sends_error_on_bad_path(self) -> None:
        from kalvin.rationaliser import Rationaliser

        bus = MessageBus()
        capture = BusCapture(bus)
        adapter = RationaliserAdapter(bus)
        rationaliser = Rationaliser(adapter=adapter)
        adapter.bind(rationaliser)

        adapter.on_message(
            Message(
                role=TRAINEE_ROLE,
                action="load",
                message="/nonexistent/path/model.bin",
                sender="supervisor",
            )
        )

        error_msgs = capture.with_action("error")
        assert len(error_msgs) == 1
        assert "Load failed" in error_msgs[0].message

    def test_load_without_rationaliser(self) -> None:
        bus = MessageBus()
        adapter = RationaliserAdapter(bus)  # No rationaliser

        # Should not raise
        adapter.on_message(
            Message(role=TRAINEE_ROLE, action="load", message="test.bin", sender="supervisor")
        )
