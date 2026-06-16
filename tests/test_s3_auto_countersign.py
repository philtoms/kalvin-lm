"""Tests for S3 auto-countersign — ratify_request suppression.

Covers: process_s2_s3 return value, conditional ratify_request suppression,
event relay regardless of auto-countersign outcome.

See specs/s3-auto-countersign.md for the specification.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from harness.bus import MessageBus
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KDbg, KLine
from tests.conftest import requires_nlp_data
from trainer.curriculum import Curriculum, CurriculumState
from trainer.reactor import Reactor

# ── Significance constants ────────────────────────────────────────────

_S2_SIGNIFICANCE = 100


# ── Helpers ───────────────────────────────────────────────────────────


def _make_entry(sig: int, nodes: list[int]) -> KLine:
    """Create a KLine with the given signature and nodes."""
    return KLine(signature=sig, nodes=nodes, dbg=KDbg(label=f"test-{sig:#x}"))


def _make_event(
    kind: str,
    query: KLine,
    proposal: KLine,
    significance: int,
) -> RationaliseEvent:
    """Create a RationaliseEvent."""
    return RationaliseEvent(
        kind=kind,
        query=query,
        proposal=proposal,
        significance=significance,
    )


class BusCapture:
    """Captures messages sent via bus.send() for test assertions."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.messages: list[Message] = []
        self._original_send = bus.send

    def install(self) -> None:
        capture = self

        def capturing_send(msg: Message) -> None:
            capture.messages.append(msg)

        self._bus.send = capturing_send  # type: ignore[assignment]

    def find_all(self, role: str, action: str) -> list[Message]:
        return [m for m in self.messages if m.role == role and m.action == action]

    def reset(self) -> None:
        self.messages.clear()


def _make_reactor(
    *,
    entries: list[KLine] | None = None,
    max_reactive_rounds: int = 5,
) -> tuple[Reactor, BusCapture]:
    """Create a Reactor with BusCapture installed."""
    bus = MessageBus()
    curriculum = Curriculum([])
    state = CurriculumState(curriculum)
    capture = BusCapture(bus)
    capture.install()
    reactor = Reactor(
        bus,
        state,
        role="trainer",
        max_reactive_rounds=max_reactive_rounds,
    )
    if entries:
        reactor.load_lesson(entries)
    return reactor, capture


# ── SAC-1: process_s2_s3 returns True on auto-countersign ────────────


class TestProcessS2S3ReturnTrue:
    """SAC-1: process_s2_s3 returns True when auto-countersign succeeds."""

    def test_returns_true_on_auto_countersign(self) -> None:
        """When proposal matches an entry, returns True."""
        entry = _make_entry(100, [10, 20])
        reactor, _ = _make_reactor(entries=[entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)
        assert result is True

    def test_countersign_sent_on_match(self) -> None:
        """When auto-countersign succeeds, countersign message is sent to trainee."""
        entry = _make_entry(100, [10, 20])
        reactor, capture = _make_reactor(entries=[entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].message == proposal


# ── SAC-2: process_s2_s3 returns False on no match ───────────────────


class TestProcessS2S3ReturnFalse:
    """SAC-2: process_s2_s3 returns False when auto-countersign fails."""

    def test_returns_false_on_no_match(self) -> None:
        """When proposal doesn't match any entry, returns False."""
        entry = _make_entry(100, [10, 20])
        reactor, _ = _make_reactor(entries=[entry])

        proposal = KLine(signature=999, nodes=[88])  # no match
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)
        assert result is False


# ── SAC-3: _handle_reactive not called on auto-countersign ────────────


class TestHandleReactiveNotCalledOnAutoCountersign:
    """SAC-3: _handle_reactive is not called when auto-countersign succeeds."""

    def test_no_escalation_on_auto_countersign(self) -> None:
        """When auto-countersign succeeds, no escalation or notify messages."""
        entry = _make_entry(100, [10, 20])
        reactor, capture = _make_reactor(entries=[entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # No supervisor notifications (escalation, notify, etc.)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) == 0


# ── SAC-4 & SAC-5: Trainer-level ratify_request conditional ──────────


@requires_nlp_data
class TestTrainerRatifySuppression:
    """SAC-4, SAC-5: Trainer suppresses ratify_request on auto-countersign."""

    @patch("trainer.trainer.compile_source")
    def test_ratify_suppressed_on_auto_countersign(self, mock_compile: MagicMock) -> None:
        """SAC-4: No ratify_request when auto-countersign matches."""
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        from trainer.trainer import Trainer

        trainer = Trainer(bus, curriculum)
        capture = BusCapture(bus)
        capture.install()

        trainer.start_session()
        # Respond to the drain message so _do_submit_lesson fires
        # and entries are compiled + loaded into the reactor
        trainer.on_message(Message(role=TRAINEE_ROLE, action="drained", message=None))
        capture.reset()

        # Send S2/S3 frame event matching the entry → auto-countersign succeeds
        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        ratify_msgs = capture.find_all(SUPERVISOR_ROLE, "ratify_request")
        assert len(ratify_msgs) == 0

    @patch("trainer.trainer.compile_source")
    def test_ratify_sent_when_auto_countersign_fails(self, mock_compile: MagicMock) -> None:
        """SAC-5: ratify_request sent when auto-countersign does NOT match."""
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        from trainer.trainer import Trainer

        trainer = Trainer(bus, curriculum)
        capture = BusCapture(bus)
        capture.install()

        trainer.start_session()
        # Respond to drain so entries are loaded into the reactor
        trainer.on_message(Message(role=TRAINEE_ROLE, action="drained", message=None))
        capture.reset()

        # Send S2/S3 frame event NOT matching → auto-countersign fails
        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        ratify_msgs = capture.find_all(SUPERVISOR_ROLE, "ratify_request")
        assert len(ratify_msgs) == 1
        payload = ratify_msgs[0].message
        assert payload["proposal"] is event.proposal
        assert payload["query"] is event.query


# ── SAC-6: Event relay regardless of auto-countersign ─────────────────


@requires_nlp_data
class TestEventRelayRegardless:
    """SAC-6: Event relay sent to supervisor regardless of auto-countersign."""

    @patch("trainer.trainer.compile_source")
    def test_relay_on_auto_countersign(self, mock_compile: MagicMock) -> None:
        """Event relay sent even when auto-countersign succeeds."""
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        from trainer.trainer import Trainer

        trainer = Trainer(bus, curriculum)
        capture = BusCapture(bus)
        capture.install()

        trainer.start_session()
        # Respond to drain so entries are loaded into the reactor
        trainer.on_message(Message(role=TRAINEE_ROLE, action="drained", message=None))
        capture.reset()

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1
        assert relay_msgs[0].message is event

    @patch("trainer.trainer.compile_source")
    def test_relay_on_no_auto_countersign(self, mock_compile: MagicMock) -> None:
        """Event relay sent even when auto-countersign fails."""
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        from trainer.trainer import Trainer

        trainer = Trainer(bus, curriculum)
        capture = BusCapture(bus)
        capture.install()

        trainer.start_session()
        # Respond to drain so entries are loaded into the reactor
        trainer.on_message(Message(role=TRAINEE_ROLE, action="drained", message=None))
        capture.reset()

        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1
        assert relay_msgs[0].message is event
