"""Tests for the Reactor — S2/S3 event processing in isolation.

The Reactor owns the Trainer's *mechanical* S2/S3 handling: auto-countersign
of structurally matching proposals and within-lesson recurrence dedup. Every
proposal it cannot resolve itself returns ``False`` so the Trainer can
escalate it to the supervisor as a decision (`@specs/supervisor-decision.md`).

Covers: auto-countersign matching (SD-13), recurrence dedup (SD-14), and the
``False``→escalation contract (SD-1).
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


def _make_reactor() -> tuple[Reactor, BusCapture]:
    """Create a Reactor with a fresh bus and curriculum state."""
    bus = MessageBus()
    curriculum = Curriculum([])
    state = CurriculumState(curriculum)
    capture = BusCapture(bus)
    capture.install()
    reactor = Reactor(bus, state, role="trainer")
    return reactor, capture


# ── Auto-countersign (SD-13) ──────────────────────────────────────────


class TestAutoCountersign:
    """SD-13: a structurally matching proposal is auto-countersigned and
    never reaches a decider."""

    def test_matching_proposal_countersigns(self) -> None:
        """Matching proposal → countersign message sent, entry marked satisfied."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is True  # auto-resolved; no decider needed

        # Countersign sent
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].sender == "trainer"
        assert cs_msgs[0].message == event.proposal

        # Entry marked satisfied
        from training.trainer.reactor import _entry_key

        key = _entry_key(entry)
        assert reactor._state.is_satisfied(key)

    def test_match_produces_no_supervisor_messages(self) -> None:
        """When auto-countersign matches, no supervisor interaction occurs."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        reactor.process_s2_s3(event)

        # No ratify_request (escalation), no notify, nothing to the supervisor.
        assert capture.find_all(SUPERVISOR_ROLE, "ratify_request") == []
        assert capture.find_all(SUPERVISOR_ROLE, "notify") == []

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

    def test_already_satisfied_match_is_absorbed(self) -> None:
        """A structural match for an already-satisfied entry is absorbed —
        no duplicate countersign is sent, and the event is treated as
        resolved (returns True)."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        # First sighting — countersigns.
        assert reactor.process_s2_s3(event) is True
        assert len(capture.find_all(TRAINEE_ROLE, "countersign")) == 1

        # Second sighting — absorbed, no second countersign.
        assert reactor.process_s2_s3(event) is True
        assert len(capture.find_all(TRAINEE_ROLE, "countersign")) == 1


# ── No-match → escalation (SD-1) ──────────────────────────────────────


class TestNoMatchEscalates:
    """SD-1: a proposal the Reactor cannot resolve returns False, producing
    no side effects. The Trainer surfaces it as a ratify_request."""

    def test_no_match_returns_false_with_no_side_effects(self) -> None:
        """A non-matching first sighting returns False — no countersign, no
        rationalise, no supervisor messages. The Trainer escalates it."""
        reactor, capture = _make_reactor()
        entry = _make_entry(100, [10, 20])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[88])  # doesn't match
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is False  # surfaced for escalation

        # No countersign, no rationalise drop, no supervisor messages.
        assert capture.find_all(TRAINEE_ROLE, "countersign") == []
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []
        assert capture.find_all(SUPERVISOR_ROLE, "ratify_request") == []
        assert capture.find_all(SUPERVISOR_ROLE, "notify") == []

        # Proposal recorded in the seen-set for recurrence detection.
        from training.trainer.reactor import _entry_key

        assert _entry_key(event.proposal) in reactor._seen_proposals


# ── Recurrence dedup (SD-14) ──────────────────────────────────────────


class TestRecurrenceDeclaresS4:
    """SD-14: intra-expectation fan-out — the same proposal reappears across
    two events against one expectation. The first sighting is surfaced for
    escalation; the second sighting is re-submitted at a declared S4 so
    Kalvin's rationalise drops it instead of re-cogitating indefinitely."""

    def test_first_sighting_is_surfaced_not_dropped(self) -> None:
        """First sighting of a non-matching proposal is surfaced (returns
        False) — no rationalise/drop message is sent, the proposal is
        recorded in the seen-set."""
        reactor, capture = _make_reactor()
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        result = reactor.process_s2_s3(event)

        assert result is False  # surfaced for escalation
        # No rationalise drop signal — the Trainer escalates this.
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []
        # Proposal recorded for recurrence detection.
        from training.trainer.reactor import _entry_key

        assert _entry_key(event.proposal) in reactor._seen_proposals

    def test_second_sighting_sends_declared_s4(self) -> None:
        """Second sighting of the same proposal kline re-submits it at
        declared S4 and returns True (resolved; no supervisor needed)."""
        reactor, capture = _make_reactor()
        reactor.load_lesson([_make_entry(100, [10, 20])])

        proposal = KLine(signature=900, nodes=[99])
        query = KLine(signature=800, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        # First sighting — surfaced.
        reactor.process_s2_s3(event)
        capture.reset()

        # Second sighting — same proposal kline.
        result = reactor.process_s2_s3(event)

        assert result is True  # recurrence handled; no supervisor needed
        rmsgs = capture.find_all(TRAINEE_ROLE, "rationalise")
        assert len(rmsgs) == 1
        sent: KValue = rmsgs[0].message
        assert sent.kline == event.proposal.kline
        assert sent.significance == SIG_S4

    def test_refined_proposal_is_not_recurrence(self) -> None:
        """A structurally-different proposal is genuinely new — it gets its
        own first-sighting treatment (surfaced), not a recurrence drop."""
        reactor, capture = _make_reactor()
        reactor.load_lesson([_make_entry(100, [10, 20])])

        p1 = KLine(signature=900, nodes=[99])
        p2 = KLine(signature=901, nodes=[98])  # different kline
        q = KLine(signature=800, nodes=[1])

        reactor.process_s2_s3(_make_event("frame", q, p1, _S2_SIGNIFICANCE))
        capture.reset()
        result = reactor.process_s2_s3(_make_event("frame", q, p2, _S2_SIGNIFICANCE))

        # p2 is a distinct first sighting → surfaced, not a drop.
        assert result is False
        assert capture.find_all(TRAINEE_ROLE, "rationalise") == []

    def test_load_lesson_clears_seen_proposals(self) -> None:
        """load_lesson resets the seen-set: a proposal that recurred in
        lesson 1 gets a fresh first sighting in lesson 2."""
        reactor, capture = _make_reactor()
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


# ── Lesson lifecycle ──────────────────────────────────────────────────


class TestLoadLessonResetsState:
    """Loading a new lesson resets per-lesson state."""

    def test_load_sets_entries_and_clears_seen_set(self) -> None:
        reactor, _ = _make_reactor()

        # Load lesson A with 3 entries
        entries_a = [_make_entry(100 + i, [10 + i]) for i in range(3)]
        reactor.load_lesson(entries_a)
        assert reactor.current_entries == entries_a

        # Load lesson B with 2 entries → entries replaced, seen-set cleared
        entries_b = [_make_entry(200, [20]), _make_entry(300, [30])]
        reactor.load_lesson(entries_b)

        assert reactor.current_entries == entries_b
        assert reactor._seen_proposals == set()


# ── Integration: Trainer → Reactor (auto-countersign) ─────────────────


def _entry_key(value: KValue):
    """Create an EntryKey from a KValue (via its kline)."""
    return (value.kline.signature, tuple(value.kline.nodes))


_S1_SIGNIFICANCE = 0xFFFF_FFFF_FFFF_FFFE


def _make_trainer(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path=None,
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
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
        llm_client=llm_client,
    )
    capture = BusCapture(bus)
    capture.install()
    return trainer, capture


@requires_tokenizer_data
class TestAutoCountersignStructuralMatch:
    """SD-13: a structurally matching proposal is auto-countersigned
    end-to-end through the Trainer → Reactor stack."""

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
