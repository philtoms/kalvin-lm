"""Tests for cascade control — candidate cap and reactor silent-drop.

Spec: specs/agent.md §Candidate Cap;
specs/reactive-delegation.md §Reactive-Round Budget (Default Mode).
"""

from kalvin.agent import KAgent
from kalvin.cogitator import WorkItem
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from tests.conftest import requires_nlp_data
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE
from training.harness.message import Message
from training.trainer.curriculum import Curriculum, CurriculumState
from training.trainer.reactor import Reactor

# ── Helpers ────────────────────────────────────────────────────────────


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


class _CaptureAdapter:
    """Minimal adapter that captures published events."""

    def __init__(self):
        self.events: list[RationaliseEvent] = []

    def on_event(self, event: RationaliseEvent) -> None:
        self.events.append(event)


def _make_kline(sig: int, nodes: list[int]) -> KLine:
    """Create a KLine with given signature and nodes."""
    return KLine(sig, nodes)


# ── Candidate cap tests ───────────────────────────────────────────────


@requires_nlp_data
class TestCandidateCap:
    """CC-1 through CC-3, CC-6 through CC-8."""

    def test_default_max_candidates(self):
        """CC-8: default max_candidates is 8."""
        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter)
        assert agent._max_candidates == 8

    def test_rationalise_caps_candidates(self):
        """CC-1: rationalise caps slow candidates at max_candidates."""
        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter, max_candidates=3)

        # Add identity atoms to the model
        for i in range(10):
            sig = 1 << i
            kline = KLine(sig, [])
            agent.model.add_to_ltm(kline)

        # Create a query with signature overlapping all identities
        # This should generate many candidates
        query_sig = sum(1 << i for i in range(10))  # all bits set
        query = KLine(query_sig, [1 << 0, 1 << 1])  # nodes don't matter for candidate retrieval

        # Track how many work items are submitted to the cogitator
        submitted = []

        def capture_submit(item: WorkItem):
            submitted.append(item)

        agent._cogitator.submit = capture_submit

        # rationalise should cap at max_candidates
        agent.rationalise(query)

        # All submitted work items should be within the cap
        assert len(submitted) <= 3

    def test_s2_prioritised_over_s3(self):
        """CC-2: S2 (overlap) candidates prioritised over S3 (no overlap)."""

        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter, max_candidates=2)

        # Add candidates that will produce both S2 and S3 matches.
        # Under the S2/S3-only routing model, overlap (full or partial)
        # routes S2; no overlap routes S3.
        # A kline with signature 0x3 and nodes [0x1, 0x2]
        cand_s2 = KLine(0x3, [0x1, 0x2])
        agent.model.add_to_ltm(cand_s2)

        # A kline with signature 0xC and nodes [0x8, 0x10] — no overlap with query nodes
        cand_s3 = KLine(0xC, [0x8, 0x10])
        agent.model.add_to_ltm(cand_s3)

        # Query with signature overlapping both candidates
        # Nodes [0x1] overlap with cand_s2 → S2, no overlap with cand_s3 → S3
        query = KLine(0x3 | 0xC, [0x1])

        submitted = []

        def capture_submit(item: WorkItem):
            submitted.append(item)

        agent._cogitator.submit = capture_submit
        agent.rationalise(query)

        # Both candidates should be submitted
        assert len(submitted) == 2
        levels = {item.level for item in submitted}
        assert "S2" in levels
        assert "S3" in levels
        assert "S1" not in levels

    def test_s1_fast_path_unaffected(self):
        """CC-6: S1 fast-path is unaffected by cap."""
        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter, max_candidates=1)  # very low cap

        # Add a countersign pair
        a = KLine(0x1, [])
        b = KLine(0x2, [])
        ab = KLine(0x3, [0x1, 0x2])
        agent.model.add_to_ltm(a)
        agent.model.add_to_ltm(b)
        agent.model.add_to_ltm(ab)

        # Query that matches the countersign → S1 fast path
        query = KLine(0x3, [0x1, 0x2])
        result = agent.rationalise(query)

        # S1 fast path returns True
        assert result is True
        assert len(adapter.events) >= 1
        # Either grounded or framed — both indicate S1 fast path
        assert adapter.events[0].kind in ("ground", "frame")
        assert adapter.events[0].significance > 0  # S1 significance

    def test_s4_routing_unaffected(self):
        """CC-7: S4 routing (no candidates) is unaffected by cap."""
        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter, max_candidates=1)

        # Query with a signature not in the model → no candidates → S4
        query = KLine(0xFF00, [0x100, 0x200])
        result = agent.rationalise(query)

        # S4 returns True
        assert result is True
        assert len(adapter.events) == 1
        assert adapter.events[0].significance == 0  # S4 significance

    def test_higher_overlap_prioritised(self):
        """CC-3: Higher overlap candidates prioritised in truncation."""

        adapter = _CaptureAdapter()
        agent = KAgent(adapter=adapter, max_candidates=2)

        # Add candidates with different overlap levels.
        # All overlap candidates route S2 under the S2/S3-only routing model;
        # priority among them is by node-overlap count.
        # Candidate with 3 overlapping nodes (full overlap)
        high_overlap = KLine(0x7, [0x1, 0x2, 0x4])
        agent.model.add_to_ltm(high_overlap)

        # Candidate with 1 overlapping node (partial overlap)
        low_overlap = KLine(0x3, [0x1])
        agent.model.add_to_ltm(low_overlap)

        # Candidate with no overlapping nodes (S3)
        no_overlap = KLine(0x18, [0x8, 0x10])
        agent.model.add_to_ltm(no_overlap)

        # Query with nodes that overlap differently with each candidate
        query = KLine(0x7 | 0x3 | 0x18, [0x1, 0x2, 0x4])

        submitted = []

        def capture_submit(item: WorkItem):
            submitted.append(item)

        agent._cogitator.submit = capture_submit
        agent.rationalise(query)

        # With cap=2, the two highest priority (most overlap) should be submitted
        assert len(submitted) <= 2
        # The highest-overlap candidate should be included
        candidate_sigs = {item.candidate.signature for item in submitted}
        assert high_overlap.signature in candidate_sigs
        # All submitted overlap candidates route S2 (no S1 routing)
        levels = {item.level for item in submitted}
        assert "S1" not in levels


# ── Reactor silent-drop tests ─────────────────────────────────────────


class TestReactorSilentDrop:
    """CC-4, CC-5."""

    def test_first_budget_exhaustion_escalates(self):
        """CC-4: First budget exhaustion event escalates."""
        bus = MessageBus()
        capture = BusCapture(bus)
        capture.install()
        state = CurriculumState(Curriculum(["A", "B"]))

        reactor = Reactor(
            bus,
            state,
            role="trainer",
            max_reactive_rounds=3,
            cogitate_fn=lambda e: None,
        )
        reactor.load_lesson([])

        # Process events up to budget
        for i in range(3):
            event = RationaliseEvent(
                "frame",
                KLine(1, [2]),
                KLine(3, [4]),
                100,
            )
            reactor.process_s2_s3(event)

        # The third event should have triggered escalation
        escalation_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_msgs = [m for m in escalation_msgs if m.message.get("reason") == "budget_exhaustion"]
        assert len(budget_msgs) >= 1

    def test_subsequent_events_silently_dropped(self):
        """CC-5: Subsequent events after budget exhaustion are silently dropped."""
        bus = MessageBus()
        capture = BusCapture(bus)
        capture.install()
        state = CurriculumState(Curriculum(["A", "B"]))

        call_count = 0

        def counting_cogitate(event):
            nonlocal call_count
            call_count += 1
            return None

        reactor = Reactor(
            bus,
            state,
            role="trainer",
            max_reactive_rounds=3,
            cogitate_fn=counting_cogitate,
        )
        reactor.load_lesson([])

        # Process 10 events — only first 3 should trigger cogitate
        for i in range(10):
            event = RationaliseEvent(
                "frame",
                KLine(1, [2]),
                KLine(3, [4]),
                100,
            )
            reactor.process_s2_s3(event)

        # Cogitate should have been called at most max_reactive_rounds times
        assert call_count <= 3

        # Only one budget_exhaustion escalation (not 7)
        escalation_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_msgs = [m for m in escalation_msgs if m.message.get("reason") == "budget_exhaustion"]
        assert len(budget_msgs) == 1
