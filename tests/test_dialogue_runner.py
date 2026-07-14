"""Dialogue actor tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-9/10/16 (actor shape) and
``@specs/dialogue-runner.md`` (run). The actors publish their turns as a
**burst** via an injected ``EventSink`` (``accept``), and never inspect the
incoming burst's content to decide what to emit. The runner drives them over
the harness :class:`~training.harness.bus.MessageBus` and validates emissions
against the authored table.

The decisive acceptance test is the canonical end-to-end run (the "Mary had
a little lamb" reference dialogue, frozen in :mod:`tests._fixtures`): runs to
completion through the runner with the default actors, every emission
validated by content match.
"""

from __future__ import annotations

import pytest

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1, SIG_S2
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_table
from training.dialogue import (
    TableTrainee,
    TableTrainer,
    decode,
    load_table,
    run,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _sigf() -> NLPSignifier:
    return NLPSignifier()


@pytest.fixture(scope="module")
def _decoded_mhall(_sigf: NLPSignifier):
    tok = NLPTokenizer()
    table = load_table(_run_mhall())
    return decode(table, tokenizer=tok, signifier=_sigf)


@pytest.fixture(scope="module")
def _compiled_mhall(_sigf: NLPSignifier):
    """The MHALL compiled script + the signifier it was built with."""
    from ks.compiler import compile_source

    tok = NLPTokenizer()
    table = load_table(_run_mhall())
    compiled = compile_source(table.script, tokenizer=tok, signifier=_sigf, dev=True)
    return compiled, _sigf


def _run_mhall() -> dict:
    """The MHALL table in dialogue mode (a ``run`` section opts into run-validation)."""
    table = mhall_table()
    table["run"] = {}
    return table


class _CapturingSink:
    """An ``EventSink`` that records every published event (for actor unit tests)."""

    def __init__(self) -> None:
        self.events: list[RationaliseEvent] = []

    def on_burst(self, events: list[RationaliseEvent]) -> None:
        self.events.extend(events)


def _run_default(decoded):
    """Drive ``decoded`` through the runner with the default table-reading actors."""
    return run(
        decoded,
        lambda sink: TableTrainer(decoded, sink=sink),
        lambda sink: TableTrainee(decoded, sink=sink),
    ).run()


# ── DDT-9/10/16: an actor yields its rows one at a time, never inspecting incoming ─


def test_actor_yields_rows_one_at_a_time(_decoded_mhall):
    """DDT-9/16: each accept() publishes the next row's event; a PASS once
    exhausted (``burst >= 1``, DDT-22 — an actor never replies zero). The actor
    holds no cursor it returns to the runner."""
    from training.dialogue.runner import is_pass

    sink = _CapturingSink()
    actor = TableTrainer(_decoded_mhall, sink=sink)
    t_rows = [t for t in _decoded_mhall if t.role == "T"]
    for _ in t_rows:
        before = len(sink.events)
        actor.accept([])
        assert len(sink.events) == before + 1  # exactly one event published
    actor.accept([])  # exhausted → a PASS (burst >= 1), not silence
    assert len(sink.events) == len(t_rows) + 1
    assert is_pass(sink.events[-1])


def test_actor_does_not_inspect_incoming(_decoded_mhall):
    """DDT-10: the emitted row is independent of the incoming burst."""
    first_t = next(t for t in _decoded_mhall if t.role == "T")
    a = _CapturingSink()
    b = _CapturingSink()
    TableTrainer(_decoded_mhall, sink=a).accept(
        [RationaliseEvent(kind="frame", query=KValue(KLine(0x1, ()), 0), proposal=KValue(KLine(0x1, ()), 0), role="K")]
    )
    TableTrainer(_decoded_mhall, sink=b).accept([])
    assert a.events[0].proposal.kline == b.events[0].proposal.kline == first_t.value.kline


def test_actor_query_is_incoming_proposal(_decoded_mhall):
    """DDT-9: the event's query is the incoming burst's last proposal (own turn
    on opening)."""
    sink = _CapturingSink()
    incoming = RationaliseEvent(
        kind="frame",
        query=KValue(KLine(0x1, ()), 0),
        proposal=KValue(KLine(0x7, ()), 0),
        role="K",
    )
    TableTrainer(_decoded_mhall, sink=sink).accept([incoming])
    assert sink.events[0].query.kline == incoming.proposal.kline


def test_actor_yields_first_row_on_first_accept(_decoded_mhall):
    """DDT-16: the actor yields its first row on the first accept."""
    sink = _CapturingSink()
    first_k = next(t for t in _decoded_mhall if t.role == "K")
    TableTrainee(_decoded_mhall, sink=sink).accept([])
    assert sink.events[0].proposal.kline == first_k.value.kline


def test_table_actor_emits_contiguous_run_as_one_burst():
    """On the opening seed (an empty burst), when the authored table has
    **consecutive same-role rows** at the start, the actor emits them all as
    one burst (not one row per accept). In an interleaved T/K table each run is
    one row, so this only shows up with a genuine same-role run."""
    from training.dialogue.decoder import DecodedTurn

    # A table with a two-row T run up front, then K, then close.
    def _t(role: str, sig: int) -> DecodedTurn:
        return DecodedTurn(role=role, op="IDENTITY", value=KValue(KLine(sig), SIG_S1))

    table = [
        _t("T", 0x11),
        _t("T", 0x12),  # consecutive same-role row → same opening run
        _t("K", 0x21),
        _t("T", 0x13),  # close (last row)
    ]
    sink = _CapturingSink()
    trainer = TableTrainer(table, sink=sink)
    trainer.accept([])  # opening seed
    # The whole T-run (0x11, 0x12) is emitted in one burst — two events.
    assert [e.proposal.kline.signature for e in sink.events] == [0x11, 0x12]
    # The next accept resumes after the run: the close row 0x13.
    sink.events.clear()
    trainer.accept([])
    assert [e.proposal.kline.signature for e in sink.events] == [0x13]


def test_table_actor_answers_each_event_in_a_received_batch():
    """When the incoming burst carries N events, the table actor emits N
    response rows — one per entry — not a single row for the whole batch. Each
    response row's ``query`` is the corresponding incoming event's proposal.
    This is the intermediate step between R2 (one row per accept, only the last
    entry's query) and R1 (content-lookup realignment): the actor matches and
    answers every entry in the batch by cardinality, advancing its own cursor
    in table order (it stays content-blind — DDT-10)."""
    from training.dialogue.decoder import DecodedTurn

    def _t(role: str, sig: int) -> DecodedTurn:
        return DecodedTurn(role=role, op="IDENTITY", value=KValue(KLine(sig), SIG_S1))

    # Interleaved table: T opens, then three T/K pairs. The trainer has three
    # T rows after the opener to answer with.
    table = [
        _t("T", 0x11),  # opener
        _t("K", 0x21), _t("T", 0x12),
        _t("K", 0x22), _t("T", 0x13),
        _t("K", 0x23), _t("T", 0x14),  # close (last row)
    ]
    sink = _CapturingSink()
    trainer = TableTrainer(table, sink=sink)
    trainer.accept([])  # opening seed → the opener 0x11
    assert [e.proposal.kline.signature for e in sink.events] == [0x11]

    # A trainee burst of THREE events arrives in one accept. The trainer
    # answers each with its next T-row: 0x12, 0x13, 0x14 — three responses.
    sink.events.clear()
    burst = [
        RationaliseEvent(kind="frame", query=KValue(KLine(0x21), SIG_S1),
                         proposal=KValue(KLine(0x21), SIG_S1), role="K"),
        RationaliseEvent(kind="frame", query=KValue(KLine(0x22), SIG_S1),
                         proposal=KValue(KLine(0x22), SIG_S1), role="K"),
        RationaliseEvent(kind="frame", query=KValue(KLine(0x23), SIG_S1),
                         proposal=KValue(KLine(0x23), SIG_S1), role="K"),
    ]
    trainer.accept(burst)
    assert [e.proposal.kline.signature for e in sink.events] == [0x12, 0x13, 0x14]
    # Each response row's query is the corresponding incoming event's proposal.
    assert [e.query.kline.signature for e in sink.events] == [0x21, 0x22, 0x23]


# ── DDT-8: trainer and trainee are symmetric ──────────────────────────────


def test_trainer_and_trainee_are_symmetric_readers(_decoded_mhall):
    """Both yield their own rows in order; together they cover the table (zero
    displacement)."""
    res = _run_default(_decoded_mhall)
    assert res.uncovered == []


def test_event_carries_emitting_role(_decoded_mhall):
    """The default actors self-declare their role on every emitted event."""
    res = _run_default(_decoded_mhall)
    roles = [e.role for e in res.events]
    assert roles[0] == "T"  # the MHALL table opens with a trainer turn
    assert set(roles) == {"T", "K"}


# ── Canonical end-to-end (decisive acceptance) ────────────────────────────


def test_canonical_run_closes_and_covers(_decoded_mhall):
    """The full MHALL dialogue runs through the runner with the default actors
    and covers the whole exchange (zero displacement). The runner is not a
    judge: the run may terminate by entry exhaustion (full coverage) before the
    close content is itself emitted — that is not distinguished from a close."""
    res = _run_default(_decoded_mhall)
    assert res.uncovered == []  # full coverage — zero displacement


def test_canonical_run_emits_every_coverage_row(_decoded_mhall):
    """Every distinct coverage row's content is emitted at least once by the
    default actors — zero displacement (``uncovered`` is empty). Duplicate
    coverage rows collapse to one distinct content; the arrival log may also
    carry no-op PASSes."""
    from training.dialogue.runner import is_pass

    res = _run_default(_decoded_mhall)
    assert res.uncovered == []  # every distinct coverage row was traversed
    # Sanity: the non-PASS emissions are a subset of the table's content.
    from training.dialogue.decoder import turn_content_key
    close_idx = next((i for i, t in enumerate(_decoded_mhall) if t.close), len(_decoded_mhall) - 1)
    coverage_set = {turn_content_key(t) for i, t in enumerate(_decoded_mhall) if i != close_idx}
    emitted_set = {
        (e.role, e.proposal.kline.signature, tuple(e.proposal.kline.nodes), e.proposal.significance)
        for e in res.events if not is_pass(e)
    }
    assert coverage_set <= emitted_set


# ── SynthesizingTrainer integration (plan §Phase 4.2) ────────────────────


def test_synthesizing_trainer_runs_mhall_to_exhaustion(
    _decoded_mhall, _compiled_mhall
):
    """The SynthesizingTrainer (a real, script-deriving trainer) runs MHALL to
    completion with zero divergence against the golden master, driven through the
    runner. The trainee stays a ``TableTrainee`` (the deterministic oracle).
    This is the canonical end-to-end proof that a synthesizing trainer is a
    drop-in replacement for ``TableTrainer`` — the symmetric counterpart to the
    Rationaliser test above."""
    from training.dialogue.decoder import primaries_from_source
    from training.dialogue.actors import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    primaries = primaries_from_source(
        load_table(_run_mhall()).script, tokenizer=NLPTokenizer(), signifier=sigf
    )
    res = run(
        _decoded_mhall,
        lambda sink: SynthesizingTrainer(compiled, sigf, primaries, sink=sink),
        lambda sink: TableTrainee(_decoded_mhall, sink=sink),
        on_divergence="accept",
    ).run()
    assert res.uncovered == []  # zero displacement
    assert res.unmatched == []  # zero divergence


# ── R4 — open-dialog-close (plan @plans/implement-synthesizing-trainer.md) ─


def test_synthesizing_trainer_replies_to_any_incoming(_compiled_mhall):
    """The trainer does not detect script closes and never withholds on its
    own — it synthesises a reply for whatever ``incoming`` it is handed.
    Close-detection is the runner's job (it owns the table and routes
    accordingly). This replaces the former R4 trainer-side withhold."""
    from training.dialogue.actors import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    sink = _CapturingSink()
    trainer = SynthesizingTrainer(compiled, sigf, [compiled[0].kline], sink=sink)
    # A primary-shaped incoming (formerly the close): the trainer still replies
    # (R3 echoes the matching compiled kline) — it does not withhold.
    trainer.accept(
        [RationaliseEvent(
            kind="frame",
            query=KValue(KLine(compiled[0].kline.signature, ()), 0),
            proposal=KValue(compiled[0].kline, SIG_S1),
            role="K",
        )]
    )
    assert len(sink.events) == 1


def test_synthesizing_trainer_advances_primary_on_each_open():
    """On the opening seed (an empty incoming burst) the trainer emits the
    current primary at S2 (R1) and advances, so a multi-script trainer opens
    each script's own primary in turn: primary 0 on the first open, primary 1
    after the first close, and so on. The runner drives the opens."""
    from pathlib import Path

    from training.dialogue.decoder import primaries_from_source
    from training.dialogue.actors import SynthesizingTrainer

    tok, sigf = NLPTokenizer(), NLPSignifier()
    source = Path("data/scripts/mhall.ks").read_text()
    primaries = primaries_from_source(source, tokenizer=tok, signifier=sigf)
    p0, p1 = primaries
    # ``compiled`` only feeds R2/R3 structure lookups; any non-empty list works
    # here since both opens take the empty-burst branch.
    sink = _CapturingSink()
    trainer = SynthesizingTrainer([KValue(p0, 0)], sigf, [p0, p1], sink=sink)
    trainer.accept([])            # open script 1
    assert sink.events[0].proposal.kline == p0
    assert sink.events[0].proposal.significance == SIG_S2
    trainer.accept([])            # open script 2 (after a close)
    assert sink.events[1].proposal.kline == p1
    assert sink.events[1].proposal.significance == SIG_S2


def test_rationalising_trainee_reacts_to_all_events_not_one_at_a_time(_decoded_mhall, _sigf):
    """The RationalisingTrainee passes every received event to the engine in
    one reaction, so it does not re-emit an unresolved countersignature once
    per event. Driven through the runner against MHALL, the run progresses
    past the Mary:[Subject] S3 countersignature — the divergence (if any) is
    not that S3 pairing.

    (Re-emitting per event would exhaust the Mary:[Subject] coverage budget on
    the second event and diverge on an S3 emission. Reacting to all events at
    once emits it once, so the run traverses it cleanly. A separate re-emission
    of the ALL residual may still surface later; this test pins only this fix.)"""
    from kalvin.expand import SIG_S3
    from training.dialogue.actors import RationalisingTrainee

    decoded = _decoded_mhall
    res = run(
        decoded,
        lambda sink: TableTrainer(decoded, sink=sink),
        lambda sink: RationalisingTrainee(_sigf, sink=sink),
        on_divergence="accept",
    ).run()

    for e in res.unmatched:
        assert e.proposal.significance != SIG_S3, (
            "trainee diverged on an S3 countersignature pairing — it re-emitted "
            "the pairing once per received event instead of reacting to all at once"
        )
