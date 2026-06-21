"""Cogitator handler-seam integration tests.

These tests exercise the ``Cogitator`` → ``CogitationHandler`` dispatch seam
(``on_s1`` / ``on_expansion``) using a recording fake handler wired to a bare
``Model()`` + ``Cogitator()``.  They do **not** instantiate ``KAgent`` and
therefore require **no tokenizer data** — they run in standard CI.
"""

from kalvin.cogitator import Cogitator, WorkItem
from kalvin.events import EventBus
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.signature import make_signature


class RecordingCogitationHandler:
    """Test fake: records all cogitation callbacks for assertion."""

    def __init__(self):
        self.s1_calls: list[tuple[KLine, KLine]] = []
        self.expansion_calls: list[tuple[KLine, KLine, int]] = []

    def on_s1(self, query: KLine, candidate: KLine) -> None:
        self.s1_calls.append((query, candidate))

    def on_expansion(
        self,
        query: KLine,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        self.expansion_calls.append((query, proposal, significance))


# ── Fake-Handler Integration Tests ────────────────────────────────────


class TestCogitatorWithFakeHandler:
    """Cogitator wired to RecordingCogitationHandler — proves the seam works."""

    def test_fake_handler_receives_s1(self):
        """Cogitator calls handler.on_s1 when expand yields an S1-classified result.

        Two empty-node identity klines with the same signature yield
        total_distance=0 from expand() → significance D_MAX → classified S1.
        (Previously used KLine(10, [10]) which is identity, not canonical, since
        commit 040bc0c — expand penalised the self-referential matched node and
        yielded S2, so on_s1 was never called.)
        """
        m = Model()
        # Identity kline (empty nodes) — expand yields S1 (total_distance=0,
        # significance=D_MAX). KLine(10, [10]) would be identity too but the
        # self-referential node is penalised in expand(); empty nodes are not.
        c = KLine(10, [])  # identity: empty nodes
        m.add_to_ltm(c)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        # Identity query with the same signature — matches c trivially (no nodes
        # to resolve), so m.add_to_frame(q) is unnecessary.
        q = KLine(0, [])  # identity: empty nodes
        q.signature = 10

        cogitator.submit(WorkItem(q, c, "S2"))
        cogitator.join(timeout=2.0)

        assert len(recorder.s1_calls) >= 1
        assert recorder.s1_calls[0][0] is q
        assert recorder.s1_calls[0][1] is c

    def test_fake_handler_receives_expansion(self):
        """Cogitator calls handler.on_expansion when a misfit kline expands."""
        m = Model()
        # Build model with identity klines that resolve nodes
        k1 = KLine(0b100, [0b100])  # identity (self-referential since 040bc0c)
        m.add_to_ltm(k1)
        k2 = KLine(0b010, [0b010])  # identity — resolves as a contributor
        m.add_to_ltm(k2)
        # A misfit kline — underfitting: sig=0b110 but nodes only give 0b100
        k3 = KLine(0b110, [0b100])
        m.add_to_ltm(k3)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        # Query with no overlapping nodes to k3 → S3 after expand,
        # and k3 is misfit so propose_expansions triggers generate_expansions.
        q = KLine(0, [0b010])
        q.signature = make_signature([0b010])
        m.add_to_frame(q)

        cogitator.submit(WorkItem(q, k3, "S3"))
        cogitator.join(timeout=2.0)

        assert len(recorder.expansion_calls) >= 1

    def test_cogitator_stops_on_s1(self):
        """Cogitator._run_work_item breaks after finding S1 — no more handler calls.

        Once S1 is discovered during expansion, the work item should stop
        processing. No further on_s1 or on_expansion calls should happen
        for that work item.

        Uses the same empty-node identity topology as
        test_fake_handler_receives_s1.
        """
        m = Model()
        # Identity kline (empty nodes) — expand yields S1 (total_distance=0).
        c = KLine(10, [])  # identity: empty nodes
        m.add_to_ltm(c)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        q = KLine(0, [])  # identity: empty nodes
        q.signature = 10

        cogitator.submit(WorkItem(q, c, "S2"))
        cogitator.join(timeout=2.0)

        # S1 should be called exactly once (not multiple times)
        assert len(recorder.s1_calls) == 1
        # No expansion proposals should be generated after S1
        assert len(recorder.expansion_calls) == 0
