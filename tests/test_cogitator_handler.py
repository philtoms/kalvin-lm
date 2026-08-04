"""Cogitator handler-seam integration tests.

These tests exercise the ``Cogitator`` → ``CogitationHandler`` dispatch seam
(``on_s1`` / ``on_expansion``) using a recording fake handler wired to a bare
``Model()`` + ``Cogitator()``.  They do **not** instantiate ``Rationaliser`` and
therefore require **no tokenizer data** — they run in standard CI.
"""

from kalvin.cogitator import Cogitator, WorkItem
from kalvin.events import EventBus
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.model import Model
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def t(bits: int) -> int:
    """Place type-word bits in the upper 32 bits of a uint64 (NLP layout)."""
    return bits << 32


class RecordingCogitationHandler:
    """Test fake: records all cogitation callbacks for assertion."""

    def __init__(self):
        self.s1_calls: list[tuple[KValue, KLine]] = []
        self.expansion_calls: list[tuple[KValue, KLine, int]] = []

    def on_s1(self, query: KValue, candidate: KLine) -> None:
        self.s1_calls.append((query, candidate))

    def on_expansion(
        self,
        query: KValue,
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
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder, signifier=signifier)

        # Identity query with the same signature — matches c trivially (no nodes
        # to resolve), so m.add_to_frame(q) is unnecessary.
        q = KLine(0, [])  # identity: empty nodes
        q.signature = 10
        q_value = KValue(q, 0x1234)  # declared significance rides the slow path

        cogitator.submit(WorkItem(q_value, c, "S2"))
        cogitator.join(timeout=2.0)

        assert len(recorder.s1_calls) >= 1
        assert recorder.s1_calls[0][0] is q_value
        assert recorder.s1_calls[0][1] is c

    def test_fake_handler_receives_expansion(self):
        """Cogitator calls handler.on_expansion when a misfit pair expands.

        The candidate is a genuine underfit misfit (sig t(0b110) promises
        more than nodes [t(0b100)] deliver → classify_misfit (True, False)).
        For ``on_expansion`` to fire, ``expand`` must route the pair as S2/S3,
        *not* S1: a query that fully resolves against the candidate yields S1
        (see ``test_fake_handler_receives_s1``) and breaks before proposals.
        So the query carries an over-claimed signature (0b111) over the same
        node (t(0b100)) — the pair is accounted but not exact, expand yields
        S2, and ``propose_expansions`` reshapes the misfit candidate into
        proposals that reach ``on_expansion``.
        """
        m = Model(signifier=signifier)
        # Underfit misfit candidate: sig t(0b110) vs nodes [t(0b100)].
        k3 = KLine(t(0b110), [t(0b100)])
        m.add_to_ltm(k3)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder, signifier=signifier)

        # Query over the candidate's node but with an over-claimed signature
        # (0b111) so the pair is non-exact → expand yields S2, not S1.
        q = KLine(0, [t(0b100)])
        q.signature = t(0b100) | t(0b010) | t(0b001)
        m.add_to_frame(q)

        cogitator.submit(WorkItem(KValue(q, 0x5678), k3, "S3"))
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
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder, signifier=signifier)

        q = KLine(0, [])  # identity: empty nodes
        q.signature = 10

        cogitator.submit(WorkItem(KValue(q, 0x9ABC), c, "S2"))
        cogitator.join(timeout=2.0)

        # S1 should be called exactly once (not multiple times)
        assert len(recorder.s1_calls) == 1
        # No expansion proposals should be generated after S1
        assert len(recorder.expansion_calls) == 0
