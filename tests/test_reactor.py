"""Tests for the Reactor — S2/S3 event processing in isolation.

Covers: auto-countersign matching, reactive scaffolding, budget
exhaustion, low-confidence escalation, lesson-complete tracking,
and state reset on lesson load.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S4
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from tests.conftest import requires_tokenizer_data
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from training.harness.message import Message
from training.trainer.curriculum import Curriculum, CurriculumState
from training.trainer.reactor import Reactor

# ── Significance constants ────────────────────────────────────────────

_S2_SIGNIFICANCE = 100


# ── Test helpers ──────────────────────────────────────────────────────


def _make_entry(sig: int, nodes: list[int]) -> KValue:
    """Create a KValue entry (a compiled expectation) for a lesson."""
    return KValue(
        KLine(signature=sig, nodes=nodes, dbg=KDbg(label=f"test-{sig:#x}")),
        _S2_SIGNIFICANCE,
    )


def _make_event(
    kind: str,
    query: KLine,
    proposal: KLine,
    significance: int = _S2_SIGNIFICANCE,
) -> RationaliseEvent:
    """Create a RationaliseEvent with KValue query/proposal (KB-354 shape).

    Both query and proposal are wrapped in KValues carrying ``significance``
    (Kalvin's assessment for the proposal voice). For the two-voice KV-15
    case — where query and proposal carry *different* significances —
    construct the KValues directly.
    """
    return RationaliseEvent(
        kind=kind,
        query=KValue(query, significance),
        proposal=KValue(proposal, significance),
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
        return [m for m in self.messages if m.role == role and m.action == action]

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
    delegate_reactive: bool = False,
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
        delegate_reactive=delegate_reactive,
    )
    return reactor, capture


# ── Tests ─────────────────────────────────────────────────────────────


# ── Integration tests (via Trainer → Reactor delegation) ──────────────

# These tests exercise reactive behaviour through the full Trainer
# stack, verifying that the Trainer correctly delegates to Reactor.
# They use the same helpers from test_trainer.py (imported below)
# but target reactive paths specifically.


def _entry_key(value: KValue):
    """Create an EntryKey from a KValue (via its kline)."""
    return (value.kline.signature, tuple(value.kline.nodes))


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
    from training.trainer.trainer import Trainer

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


@requires_tokenizer_data
class TestAutoCountersignStructuralMatch:
    """HRNS-12: Trainer auto-countersigns structurally matching proposals."""

    @patch("training.trainer.trainer.compile_source")
    def test_auto_countersign_structural_match(self, mock_compile: MagicMock) -> None:
        bus = MessageBus()
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        curriculum = Curriculum(["MHALL = SVO"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        # Simulate the cogitator drain so the lesson is compiled and loaded
        # into the reactor (otherwise _auto_countersign finds no entries).
        trainer.on_message(Message(role="adapter", action="drained", message=None))

        # Clear startup messages
        capture.reset()

        # Simulate KAgent frame event with matching proposal (non-S1)
        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        # Verify countersign was sent to kalvin
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].sender == "trainer"
        # The countersign message carries the proposal KValue
        assert cs_msgs[0].message == event.proposal

        # Verify entry is marked satisfied
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)


@requires_tokenizer_data
class TestReactiveModeOnS2S3:
    """HRNS-13: Trainer enters reactive mode on S2/S3 events."""

    @patch("training.trainer.trainer.compile_source")
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

        trainer.on_message(Message(role="trainer", action="frame", message=event))

        # Verify NO countersign was sent
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 0

        # Verify escalation to slack (low_confidence since no cogitate_fn)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) >= 1
        escalation = notify_msgs[0].message
        assert escalation["reason"] == "low_confidence"


@requires_tokenizer_data
class TestEscalationOnBudgetExhaustion:
    """HRNS-14: Trainer escalates to Slack on budget exhaustion."""

    @patch("training.trainer.trainer.compile_source")
    def test_escalation_on_budget_exhaustion(self, mock_compile: MagicMock) -> None:
        # Use max_reactive_rounds=3 and a lesson with 3 entries
        # so reactive_rounds can reach 3 within a single lesson
        entries = [_make_entry(100 + i, [10 + i]) for i in range(3)]
        mock_compile.return_value = entries

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(bus, curriculum, max_reactive_rounds=3)
        trainer.start_session()
        capture.reset()

        # Send 3 non-matching S2/S3 frame events
        for i in range(3):
            proposal = KLine(signature=900 + i, nodes=[99 + i])
            query = KLine(signature=800 + i, nodes=[i])
            event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
            trainer.on_message(Message(role="trainer", action="frame", message=event))

        # Verify budget_exhaustion escalation
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_esc = [m for m in notify_msgs if m.message["reason"] == "budget_exhaustion"]
        assert len(budget_esc) >= 1


@requires_tokenizer_data
class TestCogitateFnInjection:
    """Provide a cogitate_fn that returns scaffolding — reactive mode uses it."""

    @patch("training.trainer.trainer.compile_source")
    def test_cogitate_fn_injection(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        # Mock cogitate function that returns scaffolding
        mock_cogitate = MagicMock(return_value=("S = X / V = Y", 0.85))

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum, cogitate_fn=mock_cogitate)
        trainer.start_session()
        capture.reset()

        # Non-matching S2/S3 event
        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(Message(role="trainer", action="frame", message=event))

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
            m
            for m in notify_msgs
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
        assert cs_msgs[0].message == event.proposal

        # Entry marked satisfied
        from training.trainer.reactor import _entry_key

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

    def test_structural_match_ignores_significance(self) -> None:
        """KV-2: a proposal matches an entry with the same kline regardless of significance.

        The entry is loaded at ``_S2_SIGNIFICANCE``; the proposal wraps the
        same kline at ``SIG_S1`` (a different assessment). KValue structural
        equality ignores significance, so the match succeeds and the proposal
        KValue is posted on the countersign bus.
        """
        from kalvin.expand import SIG_S1

        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])  # KValue at _S2_SIGNIFICANCE
        reactor.load_lesson([entry])

        # Same kline as the entry, but a different significance (SIG_S1).
        proposal = KValue(KLine(signature=100, nodes=[10, 20]), SIG_S1)
        query = KValue(KLine(signature=999, nodes=[1]), _S2_SIGNIFICANCE)
        event = RationaliseEvent("frame", query, proposal)

        result = reactor.process_s2_s3(event)
        assert result is True

        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        # The proposal KValue (with its own significance) is carried on the bus.
        assert cs_msgs[0].message == proposal


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
        budget_esc = [m for m in notify_msgs if m.message["reason"] == "budget_exhaustion"]
        assert len(budget_esc) >= 1


# ── Recurrence → declared-S4 drop signal ──────────────────────────────────


class TestRecurrenceDeclaresS4:
    """Intra-expectation fan-out: the same proposal reappears across two
    events against one expectation. First sighting scaffolds; second sighting
    re-submits the proposal at a declared S4 so Kalvin's rationalise drops it.
    Recurrence counts toward the reactive budget so the escalation net survives.
    """

    def test_first_sighting_scaffolds_not_drops(self) -> None:
        """First sighting of a non-matching proposal goes to reactive
        handling — no rationalise/drop message is sent, the proposal is
        recorded in the seen-set."""
        reactor, capture = _make_reactor(cogitate_fn=lambda e: ("A = B", 0.9))
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is False  # reactive path invoked (not handled)
        # Scaffolding submitted, not a rationalise drop signal.
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []
        assert len(capture.find_all(TRAINEE_ROLE, "submit")) == 1
        # Proposal recorded for recurrence detection.
        from training.trainer.reactor import _entry_key

        assert _entry_key(event.proposal) in reactor._seen_proposals

    def test_second_sighting_sends_declared_s4(self) -> None:
        """Second sighting of the same proposal kline re-submits it at declared
        S4 and returns True (handled, no supervisor)."""
        reactor, capture = _make_reactor(cogitate_fn=lambda e: ("A = B", 0.9))
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        # First sighting
        reactor.process_s2_s3(event)
        capture.reset()

        # Second sighting — same proposal kline
        result = reactor.process_s2_s3(event)

        assert result is True  # recurrence handled; no supervisor needed
        rmsgs = capture.find_all(TRAINEE_ROLE, "rationalise")
        assert len(rmsgs) == 1
        sent: KValue = rmsgs[0].message
        assert sent.kline == event.proposal.kline
        assert sent.significance == SIG_S4

    def test_second_sighting_counts_toward_budget(self) -> None:
        """Recurrence increments the reactive round (not just scaffolding)."""
        reactor, capture = _make_reactor(max_reactive_rounds=5, cogitate_fn=lambda e: None)
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)  # first sighting → reactive (round 1)
        assert reactor._reactive_rounds == 1
        reactor.process_s2_s3(event)  # second sighting → recurrence (round 2)
        assert reactor._reactive_rounds == 2

    def test_pure_recurrence_at_cliff_escalates(self) -> None:
        """A lesson whose proposals all recur still escalates budget_exhaustion
        — the escalation safety net is preserved for pure-recurrence stalls."""
        reactor, capture = _make_reactor(max_reactive_rounds=2, cogitate_fn=lambda e: None)
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)   # 1st sighting → reactive, round 1
        reactor.process_s2_s3(event)   # 2nd sighting → recurrence, round 2 (cliff)
        reactor.process_s2_s3(event)   # 3rd sighting → recurrence, round 3 (over)

        budget_esc = [
            m
            for m in capture.find_all(SUPERVISOR_ROLE, "notify")
            if m.message["reason"] == "budget_exhaustion"
        ]
        assert len(budget_esc) >= 1

    def test_refined_proposal_is_not_recurrence(self) -> None:
        """A structurally-different proposal is genuinely new — it gets its
        own first-sighting treatment, not a recurrence drop."""
        reactor, capture = _make_reactor(cogitate_fn=lambda e: ("A = B", 0.9))
        reactor.load_lesson([_make_entry(100, [10, 20])])

        p1 = KLine(signature=900, nodes=[99])
        p2 = KLine(signature=901, nodes=[98])  # different kline
        q = KLine(signature=800, nodes=[1])

        reactor.process_s2_s3(_make_event("frame", q, p1, _S2_SIGNIFICANCE))
        capture.reset()
        result = reactor.process_s2_s3(_make_event("frame", q, p2, _S2_SIGNIFICANCE))

        # p2 is a distinct first sighting → reactive path, not a drop.
        assert result is False
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []

    def test_load_lesson_clears_seen_proposals(self) -> None:
        """load_lesson resets the seen-set: a proposal that recurred in lesson 1
        gets a fresh first sighting in lesson 2."""
        reactor, capture = _make_reactor(cogitate_fn=lambda e: ("A = B", 0.9))
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)  # record in lesson 1's seen-set

        reactor.load_lesson([_make_entry(200, [20, 30])])  # new lesson
        assert reactor._seen_proposals == set()

        capture.reset()
        # Same proposal in lesson 2 → first sighting, not recurrence.
        result = reactor.process_s2_s3(event)
        assert result is False
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []


class TestLoadLessonResetsState:
    """Loading a new lesson resets per-lesson state."""

    def test_load_sets_entries_and_resets_rounds(self) -> None:
        reactor, _ = _make_reactor()

        # Load lesson A with 3 entries
        entries_a = [_make_entry(100 + i, [10 + i]) for i in range(3)]
        reactor.load_lesson(entries_a)
        assert reactor.current_entries == entries_a

        # Load lesson B with 2 entries → entries replaced
        entries_b = [_make_entry(200, [20]), _make_entry(300, [30])]
        reactor.load_lesson(entries_b)

        assert reactor.current_entries == entries_b
        assert reactor._reactive_rounds == 0


class TestDelegatedMode:
    """RD-4/RD-5/RD-6: Reactor delegated mode defers to the supervisor.

    When ``delegate_reactive=True`` the auto-countersign fast path is
    unaffected, but any non-matching proposal produces zero side effects
    (no cogitate, no submit, no escalation, no round increment).
    """

    def test_auto_countersign_still_runs(self) -> None:
        """RD-4: a structurally matching proposal still auto-countersigns."""
        reactor, capture = _make_reactor(delegate_reactive=True)
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is True

        # Countersign sent to trainee
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].message == event.proposal

        # Entry marked satisfied
        from training.trainer.reactor import _entry_key

        key = _entry_key(entry)
        assert reactor._state.is_satisfied(key)

        # No supervisor notifications
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) == 0

    def test_no_cogitate_no_submit_no_escalate(self) -> None:
        """RD-5: a non-matching proposal produces no side effects."""
        mock_cogitate = MagicMock(return_value=("S = X", 0.9))
        reactor, capture = _make_reactor(delegate_reactive=True, cogitate_fn=mock_cogitate)
        entry = _make_entry(100, [10])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[88])  # no match
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is False

        # cogitate_fn never invoked
        mock_cogitate.assert_not_called()

        # No scaffolding submitted to trainee
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_msgs) == 0

        # No supervisor notifications
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) == 0

    def test_no_round_increment_no_budget_escalation(self) -> None:
        """RD-6: the reactive-round budget is never consulted in delegated mode."""
        entries = [_make_entry(100 + i, [10 + i]) for i in range(5)]
        reactor, capture = _make_reactor(delegate_reactive=True, max_reactive_rounds=2)
        reactor.load_lesson(entries)

        # Send more than max_reactive_rounds non-matching events
        for i in range(4):
            proposal = KLine(signature=900 + i, nodes=[99 + i])
            query = KLine(signature=800 + i, nodes=[i])
            event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
            reactor.process_s2_s3(event)

        # Reactive round counter never incremented
        assert reactor._reactive_rounds == 0

        # No budget_exhaustion escalation (nor any notify)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        assert len(notify_msgs) == 0
        budget_esc = [m for m in notify_msgs if m.message.get("reason") == "budget_exhaustion"]
        assert len(budget_esc) == 0
