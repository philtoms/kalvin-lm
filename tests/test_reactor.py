"""Tests for the Reactor — S2/S3 event processing in isolation.

Covers: auto-countersign matching, reactive scaffolding, budget
exhaustion, low-confidence escalation, lesson-complete tracking,
and state reset on lesson load.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from harness.bus import MessageBus
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KDbg, KLine
from ks import KLine
from trainer.curriculum import Curriculum, CurriculumState
from trainer.reactor import Reactor

# ── Significance constants ────────────────────────────────────────────

_S2_SIGNIFICANCE = 100


# ── Test helpers ──────────────────────────────────────────────────────


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
        """Replace bus.send with our capturing wrapper."""
        capture = self

        def capturing_send(msg: Message) -> None:
            capture.messages.append(msg)

        self._bus.send = capturing_send  # type: ignore[assignment]

    def find_all(self, role: str, action: str) -> list[Message]:
        """Return all captured messages matching role and action."""
        return [
            m
            for m in self.messages
            if m.role == role and m.action == action
        ]

    def find_one(self, role: str, action: str) -> Message | None:
        """Return the first captured message matching role and action, or None."""
        matches = self.find_all(role, action)
        return matches[0] if matches else None

    def reset(self) -> None:
        """Clear captured messages."""
        self.messages.clear()


def _make_reactor(
    *,
    max_reactive_rounds: int = 5,
    cogitate_fn=None,
) -> tuple[Reactor, BusCapture]:
    """Create a Reactor with a fresh bus and curriculum state."""
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
        cogitate_fn=cogitate_fn,
    )
    return reactor, capture


# ── Tests ─────────────────────────────────────────────────────────────


# ── Integration tests (via Trainer → Reactor delegation) ──────────────

# These tests exercise reactive behaviour through the full Trainer
# stack, verifying that the Trainer correctly delegates to Reactor.
# They use the same helpers from test_trainer.py (imported below)
# but target reactive paths specifically.


def _entry_key(kline: KLine):
    """Create an EntryKey from a KLine."""
    return (kline.signature, tuple(kline.nodes))


_S1_SIGNIFICANCE = 0xFFFF_FFFF_FFFF_FFFE


def _make_trainer(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path=None,
    max_reactive_rounds: int = 5,
    cogitate_fn=None,
    curriculum_file=None,
    curricula_dir=None,
    llm_client=None,
):
    """Create a Trainer with BusCapture for integration tests."""
    from trainer.trainer import Trainer

    trainer = Trainer(
        bus,
        curriculum,
        save_path=save_path,
        max_reactive_rounds=max_reactive_rounds,
        cogitate_fn=cogitate_fn,
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
        llm_client=llm_client,
    )
    capture = BusCapture(bus)
    capture.install()
    return trainer, capture


class TestAutoCountersignStructuralMatch:
    """HRNS-12: Trainer auto-countersigns structurally matching proposals."""

    @patch("trainer.trainer.compile_source")
    def test_auto_countersign_structural_match(self, mock_compile: MagicMock) -> None:
        bus = MessageBus()
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        curriculum = Curriculum(["MHALL = SVO"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Clear startup messages
        capture.reset()

        # Simulate KAgent frame event with matching proposal (non-S1)
        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(role="trainer", action="frame", message=event)
        )

        # Verify countersign was sent to kalvin
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].sender == "trainer"
        # The countersign message contains the proposal KLine
        assert cs_msgs[0].message == proposal

        # Verify entry is marked satisfied
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)


class TestReactiveModeOnS2S3:
    """HRNS-13: Trainer enters reactive mode on S2/S3 events."""

    @patch("trainer.trainer.compile_source")
    def test_reactive_mode_on_s2_s3(self, mock_compile: MagicMock) -> None:
        bus = MessageBus()
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        curriculum = Curriculum(["MHALL = SVO", "S = M / V = H"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Clear startup messages
        capture.reset()

        # Simulate KAgent frame event with non-matching proposal
        proposal = KLine(signature=999, nodes=[88])  # doesn't match entry
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(role="trainer", action="frame", message=event)
        )

        # Verify NO countersign was sent
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 0

        # Verify escalation to slack (low_confidence since no cogitate_fn)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) >= 1
        escalation = notify_msgs[0].message
        assert escalation["reason"] == "low_confidence"


class TestEscalationOnBudgetExhaustion:
    """HRNS-14: Trainer escalates to Slack on budget exhaustion."""

    @patch("trainer.trainer.compile_source")
    def test_escalation_on_budget_exhaustion(self, mock_compile: MagicMock) -> None:
        # Use max_reactive_rounds=3 and a lesson with 3 entries
        # so reactive_rounds can reach 3 within a single lesson
        entries = [
            _make_entry(100 + i, [10 + i])
            for i in range(3)
        ]
        mock_compile.return_value = entries

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(
            bus, curriculum, max_reactive_rounds=3
        )
        trainer.start_session()
        capture.reset()

        # Send 3 non-matching S2/S3 frame events
        for i in range(3):
            proposal = KLine(signature=900 + i, nodes=[99 + i])
            query = KLine(signature=800 + i, nodes=[i])
            event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
            trainer.on_message(
                Message(role="trainer", action="frame", message=event)
            )

        # Verify budget_exhaustion escalation
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_esc = [
            m for m in notify_msgs if m.message["reason"] == "budget_exhaustion"
        ]
        assert len(budget_esc) >= 1


class TestCogitateFnInjection:
    """Provide a cogitate_fn that returns scaffolding — reactive mode uses it."""

    @patch("trainer.trainer.compile_source")
    def test_cogitate_fn_injection(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        # Mock cogitate function that returns scaffolding
        mock_cogitate = MagicMock(
            return_value=("S = X / V = Y", 0.85)
        )

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(
            bus, curriculum, cogitate_fn=mock_cogitate
        )
        trainer.start_session()
        capture.reset()

        # Non-matching S2/S3 event
        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(role="trainer", action="frame", message=event)
        )

        # Cogitate was called
        mock_cogitate.assert_called_once_with(event)

        # Reactive scaffolding was submitted to kalvin
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        scaffolding_msgs = [m for m in submit_msgs if m.message == "S = X / V = Y"]
        assert len(scaffolding_msgs) == 1
        assert scaffolding_msgs[0].sender == "trainer"

        # No escalation (low_confidence) should have occurred
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        escalation_msgs = [
            m for m in notify_msgs
            if m.message.get("reason") in ("low_confidence", "budget_exhaustion")
        ]
        assert len(escalation_msgs) == 0


# ── Unit tests (Reactor in isolation) ────────────────────────────────


class TestAutoCountersign:
    """Auto-countersign sends bus message and marks entry satisfied."""

    def test_matching_proposal_countersigns(self) -> None:
        """Matching proposal → countersign message sent, entry marked satisfied."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # Countersign sent
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].sender == "trainer"
        assert cs_msgs[0].message == proposal

        # Entry marked satisfied
        from trainer.reactor import _entry_key

        key = _entry_key(entry)
        assert reactor._state.is_satisfied(key)

    def test_no_escalation_on_match(self) -> None:
        """When auto-countersign matches, no escalation occurs."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) == 0


class TestAutoCountersignNoMatch:
    """Non-matching proposal triggers escalation (no cogitate_fn)."""

    def test_no_match_triggers_low_confidence_escalation(self) -> None:
        """Non-matching proposal → no countersign, low_confidence escalation."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[88])  # doesn't match
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # No countersign sent
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 0

        # Escalation to slack (low_confidence since no cogitate_fn)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) >= 1
        assert notify_msgs[0].message["reason"] == "low_confidence"


class TestReactiveScaffolding:
    """cogitate_fn returning scaffolding → submit message, no escalation."""

    def test_cogitate_fn_returns_scaffolding(self) -> None:
        """cogitate_fn returns (source, confidence) → submit message sent."""
        mock_cogitate = MagicMock(return_value=("S = X / V = Y", 0.85))
        reactor, capture = _make_reactor(cogitate_fn=mock_cogitate)
        entry = _make_entry(100, [10])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # Cogitate was called
        mock_cogitate.assert_called_once_with(event)

        # Reactive scaffolding submitted
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        scaffolding_msgs = [m for m in submit_msgs if m.message == "S = X / V = Y"]
        assert len(scaffolding_msgs) == 1
        assert scaffolding_msgs[0].sender == "trainer"

        # No escalation
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        escalation_msgs = [
            m
            for m in notify_msgs
            if m.message.get("reason") in ("low_confidence", "budget_exhaustion")
        ]
        assert len(escalation_msgs) == 0


class TestReactiveLowConfidence:
    """cogitate_fn returning None → low_confidence escalation."""

    def test_cogitate_fn_returns_none(self) -> None:
        """cogitate_fn returns None → escalation with low_confidence."""
        mock_cogitate = MagicMock(return_value=None)
        reactor, capture = _make_reactor(cogitate_fn=mock_cogitate)
        entry = _make_entry(100, [10])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # Escalation with low_confidence
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) >= 1
        assert notify_msgs[0].message["reason"] == "low_confidence"

        # No scaffolding submitted
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_msgs) == 0


class TestBudgetExhaustion:
    """max_reactive_rounds exceeded → budget_exhaustion escalation."""

    def test_budget_exhaustion_after_max_rounds(self) -> None:
        """After max_reactive_rounds non-matching events, budget_exhaustion fires."""
        entries = [_make_entry(100 + i, [10 + i]) for i in range(5)]
        reactor, capture = _make_reactor(max_reactive_rounds=3)
        reactor.load_lesson(entries)

        # Send 3 non-matching events
        for i in range(3):
            proposal = KLine(signature=900 + i, nodes=[99 + i])
            query = KLine(signature=800 + i, nodes=[i])
            event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
            reactor.process_s2_s3(event)

        # Verify budget_exhaustion escalation
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_esc = [
            m for m in notify_msgs if m.message["reason"] == "budget_exhaustion"
        ]
        assert len(budget_esc) >= 1


class TestLessonCompleteTracking:
    """is_lesson_complete tracks received vs expected count."""

    def test_complete_after_all_responses(self) -> None:
        """After expected_count responses, is_lesson_complete is True."""
        reactor, _ = _make_reactor()
        entries = [_make_entry(100, [10]), _make_entry(200, [20]), _make_entry(300, [30])]
        reactor.load_lesson(entries)

        assert reactor.received_count == 0
        assert reactor.expected_count == 3
        assert not reactor.is_lesson_complete

        reactor.record_response()
        reactor.record_response()
        assert not reactor.is_lesson_complete

        reactor.record_response()
        assert reactor.is_lesson_complete

    def test_not_complete_with_fewer_responses(self) -> None:
        """With fewer responses than expected, is_lesson_complete is False."""
        reactor, _ = _make_reactor()
        entries = [_make_entry(100, [10]), _make_entry(200, [20])]
        reactor.load_lesson(entries)

        reactor.record_response()
        assert not reactor.is_lesson_complete


class TestErrorRecording:
    """record_response() increments received_count."""

    def test_record_response_increments(self) -> None:
        reactor, _ = _make_reactor()
        entries = [_make_entry(100, [10]), _make_entry(200, [20])]
        reactor.load_lesson(entries)

        assert reactor.received_count == 0

        reactor.record_response()
        assert reactor.received_count == 1

        reactor.record_response()
        assert reactor.received_count == 2


class TestLoadLessonResetsState:
    """Loading a new lesson resets all per-lesson counters."""

    def test_load_resets_counters(self) -> None:
        reactor, _ = _make_reactor()

        # Load lesson A with 3 entries
        entries_a = [_make_entry(100 + i, [10 + i]) for i in range(3)]
        reactor.load_lesson(entries_a)
        reactor.record_response()
        reactor.record_response()

        assert reactor.received_count == 2
        assert reactor.expected_count == 3

        # Load lesson B with 2 entries → counters reset
        entries_b = [_make_entry(200, [20]), _make_entry(300, [30])]
        reactor.load_lesson(entries_b)

        assert reactor.received_count == 0
        assert reactor.expected_count == 2
        assert not reactor.is_lesson_complete
        assert reactor.current_entries == entries_b
