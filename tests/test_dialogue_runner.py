"""Dialogue actor tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-9/10/16 (actor shape) and
``@specs/peer-dialogue.md`` (peer run). The actors publish their turns via an
injected ``EventSink`` (``accept``), one row per call, and never inspect the
incoming event to decide what to emit. The peer runner drives them over the
harness :class:`~training.harness.bus.MessageBus` and validates emissions
against the authored table.

The decisive acceptance test is the canonical end-to-end run (the "Mary had
a little lamb" reference dialogue, frozen in :mod:`tests._fixtures`): runs to
completion through the peer runner with the default actors, every emission
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
    run_peer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _sigf() -> NLPSignifier:
    return NLPSignifier()


@pytest.fixture(scope="module")
def _decoded_mhall(_sigf: NLPSignifier):
    tok = NLPTokenizer()
    table = load_table(_peer_mhall())
    return decode(table, tokenizer=tok, signifier=_sigf)


@pytest.fixture(scope="module")
def _compiled_mhall(_sigf: NLPSignifier):
    """The MHALL compiled script + the signifier it was built with."""
    from ks.compiler import compile_source

    tok = NLPTokenizer()
    table = load_table(_peer_mhall())
    compiled = compile_source(table.script, tokenizer=tok, signifier=_sigf, dev=True)
    return compiled, _sigf


def _peer_mhall() -> dict:
    """The MHALL table in peer mode (a ``peer`` section selects the regime)."""
    table = mhall_table()
    table["peer"] = {}
    return table


class _CapturingSink:
    """An ``EventSink`` that records every published event (for actor unit tests)."""

    def __init__(self) -> None:
        self.events: list[RationaliseEvent] = []

    def on_event(self, event: RationaliseEvent) -> None:
        self.events.append(event)


def _run_peer_default(decoded):
    """Drive ``decoded`` through the peer runner with the default table-reading actors."""
    return run_peer(
        decoded,
        lambda sink: TableTrainer(decoded, sink=sink),
        lambda sink: TableTrainee(decoded, sink=sink),
    ).run()


# ── DDT-9/10/16: an actor yields its rows one at a time, never inspecting incoming ─


def test_actor_yields_rows_one_at_a_time(_decoded_mhall):
    """DDT-9/16: each accept() publishes the next row's event; nothing once
    exhausted. The actor holds no cursor it returns to the runner."""
    sink = _CapturingSink()
    actor = TableTrainer(_decoded_mhall, sink=sink)
    t_rows = [t for t in _decoded_mhall if t.role == "T"]
    for _ in t_rows:
        before = len(sink.events)
        actor.accept(None)
        assert len(sink.events) == before + 1  # exactly one event published
    actor.accept(None)  # exhausted → nothing published
    assert len(sink.events) == len(t_rows)


def test_actor_does_not_inspect_incoming(_decoded_mhall):
    """DDT-10: the emitted row is independent of the incoming event."""
    first_t = next(t for t in _decoded_mhall if t.role == "T")
    a = _CapturingSink()
    b = _CapturingSink()
    TableTrainer(_decoded_mhall, sink=a).accept(
        RationaliseEvent(kind="frame", query=KValue(KLine(0x1, ()), 0), proposal=KValue(KLine(0x1, ()), 0), role="K")
    )
    TableTrainer(_decoded_mhall, sink=b).accept(None)
    assert a.events[0].proposal.kline == b.events[0].proposal.kline == first_t.value.kline


def test_actor_query_is_incoming_proposal(_decoded_mhall):
    """DDT-9: the event's query is the incoming proposal (own turn on opening)."""
    sink = _CapturingSink()
    incoming = RationaliseEvent(
        kind="frame",
        query=KValue(KLine(0x1, ()), 0),
        proposal=KValue(KLine(0x7, ()), 0),
        role="K",
    )
    TableTrainer(_decoded_mhall, sink=sink).accept(incoming)
    assert sink.events[0].query.kline == incoming.proposal.kline


def test_actor_yields_first_row_on_first_accept(_decoded_mhall):
    """DDT-16: the actor yields its first row on the first accept."""
    sink = _CapturingSink()
    first_k = next(t for t in _decoded_mhall if t.role == "K")
    TableTrainee(_decoded_mhall, sink=sink).accept(None)
    assert sink.events[0].proposal.kline == first_k.value.kline


# ── DDT-8: trainer and trainee are symmetric ──────────────────────────────


def test_trainer_and_trainee_are_symmetric_readers(_decoded_mhall):
    """DDT-8: both yield their own rows in order; together they cover the table."""
    res = _run_peer_default(_decoded_mhall)
    assert res.complete
    assert res.covered


def test_event_carries_emitting_role(_decoded_mhall):
    """The default actors self-declare their role on every emitted event."""
    res = _run_peer_default(_decoded_mhall)
    roles = [e.role for e in res.events]
    assert roles[0] == "T"  # the MHALL table opens with a trainer turn
    assert set(roles) == {"T", "K"}


# ── Canonical end-to-end (decisive acceptance) ────────────────────────────


def test_canonical_run_completes(_decoded_mhall):
    """The full MHALL dialogue runs to completion through the peer runner with the
    default actors, the closing S1 countersign of the primary delivered last."""
    res = _run_peer_default(_decoded_mhall)
    assert res.complete
    assert res.events[-1].proposal.significance == SIG_S1
    assert res.events[-1].proposal.kline == res.events[0].proposal.kline


def test_canonical_run_emits_every_table_row(_decoded_mhall):
    """Every decoded turn's content is emitted exactly once by the default actors."""
    from training.dialogue.decoder import turn_content_key

    res = _run_peer_default(_decoded_mhall)
    emitted_keys = [
        (e.role, e.proposal.kline.signature, tuple(e.proposal.kline.nodes), e.proposal.significance)
        for e in res.events
    ]
    table_keys = [turn_content_key(t) for t in _decoded_mhall]
    assert sorted(emitted_keys) == sorted(table_keys)


# ── Rationaliser integration (plan §Phase 2.2) ────────────────────────────


def test_rationaliser_runs_mhall_to_exhaustion(_decoded_mhall, _sigf: NLPSignifier):
    """The RationalisingTrainee (a real, stateful trainee) runs MHALL to
    completion with zero divergence against the golden master, driven through
    the peer runner. The trainer stays a ``TableTrainer`` (the deterministic
    oracle). This is the canonical end-to-end proof that a rationalising trainee
    is a drop-in replacement for ``TableTrainee``."""
    from training.dialogue.runner import RationalisingTrainee

    res = run_peer(
        _decoded_mhall,
        lambda sink: TableTrainer(_decoded_mhall, sink=sink),
        lambda sink: RationalisingTrainee(_sigf, sink=sink),
        on_divergence="accept",
    ).run()
    assert res.complete
    # The rationaliser eventually delivers the closing S1 countersign of the
    # primary (it may also emit unmatched off-table content along the way).
    closing = res.events[-1]
    assert closing.proposal.significance == SIG_S1
    assert closing.proposal.kline == res.events[0].proposal.kline


# ── SynthesizingTrainer integration (plan §Phase 4.2) ────────────────────


def test_synthesizing_trainer_runs_mhall_to_exhaustion(
    _decoded_mhall, _compiled_mhall
):
    """The SynthesizingTrainer (a real, script-deriving trainer) runs MHALL to
    completion with zero divergence against the golden master, driven through the
    peer runner. The trainee stays a ``TableTrainee`` (the deterministic oracle).
    This is the canonical end-to-end proof that a synthesizing trainer is a
    drop-in replacement for ``TableTrainer`` — the symmetric counterpart to the
    Rationaliser test above."""
    from training.dialogue.decoder import primaries_from_source
    from training.dialogue.runner import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    primaries = primaries_from_source(
        load_table(_peer_mhall()).script, tokenizer=NLPTokenizer(), signifier=sigf
    )
    res = run_peer(
        _decoded_mhall,
        lambda sink: SynthesizingTrainer(compiled, sigf, primaries, sink=sink),
        lambda sink: TableTrainee(_decoded_mhall, sink=sink),
        on_divergence="accept",
    ).run()
    assert res.complete
    closing = res.events[-1]
    assert closing.proposal.significance == SIG_S1
    assert closing.proposal.kline == res.events[0].proposal.kline


# ── R4 — open-dialog-close (plan @plans/implement-synthesizing-trainer.md) ─


def test_synthesizing_trainer_replies_to_any_incoming(_compiled_mhall):
    """The trainer does not detect script closes and never withholds on its
    own — it synthesises a reply for whatever ``incoming`` it is handed.
    Close-detection is the peer runner's job (it owns the table and routes
    accordingly). This replaces the former R4 trainer-side withhold."""
    from training.dialogue.runner import SynthesizingTrainer

    compiled, sigf = _compiled_mhall
    sink = _CapturingSink()
    trainer = SynthesizingTrainer(compiled, sigf, [compiled[0].kline], sink=sink)
    # A primary-shaped incoming (formerly the close): the trainer still replies
    # (R3 echoes the matching compiled kline) — it does not withhold.
    trainer.accept(
        RationaliseEvent(
            kind="frame",
            query=KValue(KLine(compiled[0].kline.signature, ()), 0),
            proposal=KValue(compiled[0].kline, SIG_S1),
            role="K",
        )
    )
    assert len(sink.events) == 1


def test_synthesizing_trainer_advances_primary_on_each_open():
    """On the opening seed (``incoming=None``) the trainer emits the current
    primary at S2 (R1) and advances, so a multi-script trainer opens each
    script's own primary in turn: primary 0 on the first open, primary 1 after
    the first close, and so on. The peer runner drives the opens."""
    from pathlib import Path

    from training.dialogue.decoder import primaries_from_source
    from training.dialogue.runner import SynthesizingTrainer

    tok, sigf = NLPTokenizer(), NLPSignifier()
    source = Path("data/scripts/mhall.ks").read_text()
    primaries = primaries_from_source(source, tokenizer=tok, signifier=sigf)
    p0, p1 = primaries
    # ``compiled`` only feeds R2/R3 structure lookups; any non-empty list works
    # here since both opens take the ``incoming=None`` branch.
    sink = _CapturingSink()
    trainer = SynthesizingTrainer([KValue(p0, 0)], sigf, [p0, p1], sink=sink)
    trainer.accept(None)            # open script 1
    assert sink.events[0].proposal.kline == p0
    assert sink.events[0].proposal.significance == SIG_S2
    trainer.accept(None)            # open script 2 (after a close)
    assert sink.events[1].proposal.kline == p1
    assert sink.events[1].proposal.significance == SIG_S2
