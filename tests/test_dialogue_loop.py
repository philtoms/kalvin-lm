"""Phase 3 — self-cursored StubKAgent + dispatch loop tests.

Spec coverage:
- ``@specs/stub-kagent.md`` ST-1..13 (the self-cursored table-reader stub).
- ``@specs/dialogue-driven-training.md`` DDT-7, DDT-18, DDT-22..27 (the
  dispatch-driven loop: greedy cursors, two-sided Model A validation,
  dual-exhaustion termination).

The decisive acceptance test is the canonical end-to-end run
(``scripts/dialogue-mhall.json``): every turn validated two-sided, terminating
on the primary's closing S1 countersign via dual-exhaustion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1, SIG_S2, SIG_S4
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.dialogue import (
    LoopError,
    StubKAgent,
    decode,
    load_table,
    run_session,
)
from training.dialogue.decoder import DecodedTurn
from training.harness.adapter import KAgentAdapter
from training.harness.bus import MessageBus
from training.harness.constants import TRAINEE_ROLE

MHALL = Path(__file__).resolve().parent.parent / "scripts" / "dialogue-mhall.json"


class _RecordingAdapter:
    """Minimal adapter capturing on_event calls (for stub unit tests)."""

    def __init__(self) -> None:
        self.events: list[RationaliseEvent] = []

    def on_event(self, event: RationaliseEvent) -> None:
        self.events.append(event)


@pytest.fixture(scope="module")
def kit():
    tok, sigf = NLPTokenizer(), NLPSignifier()
    table = load_table(json.loads(MHALL.read_text()))
    decoded = decode(table, tokenizer=tok, signifier=sigf)
    return tok, sigf, table, decoded


def _kv(sig: int, nodes=(), significance: int = SIG_S4) -> KValue:
    from kalvin.kline import KLine

    return KValue(KLine(sig, list(nodes)), significance)


# ── ST-1..13: the self-cursored stub ──────────────────────────────────────


def test_stub_satisfies_kagentlike_surface():
    """ST-1: StubKAgent satisfies _KAgentLike (rationalise/countersign/save/codec)."""
    stub = StubKAgent(_RecordingAdapter(), [])
    assert callable(stub.rationalise)
    assert callable(stub.countersign)
    assert callable(stub.save)
    assert callable(stub.codec)
    assert stub.countersign(_kv(0x1)) is True  # ST-8: no-op returning True
    assert stub.save("/tmp/none") is None
    assert stub.codec() is None


def test_stub_emits_next_krow_per_rationalise_and_advances_cursor(kit):
    """ST-2/3/4/5: each rationalise() emits the next K-row and advances the cursor;
    the cursor never moves backward and never inspects value."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, k_rows)

    # The submitted value is irrelevant to what the stub emits (ST-5): pass a
    # sentinel KValue; the stub must still emit the K-rows in order.
    sentinel = _kv(0xDEAD, significance=SIG_S4)
    first = stub.rationalise(sentinel)
    assert first is True
    assert len(adapter.events) == 1
    assert adapter.events[0].proposal.kline == k_rows[0].value.kline
    assert adapter.events[0].proposal.significance == k_rows[0].value.significance
    assert stub.cursor == 1

    # The query voice of every event is the submitted value (ST-13).
    assert adapter.events[0].query is sentinel


def test_stub_at_end_returns_true_with_no_events(kit):
    """ST-6: rationalise() with the cursor at end returns True with no events."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, k_rows)
    # Burn through every K-row.
    for _ in k_rows:
        stub.rationalise(_kv(0x1))
    assert stub.exhausted
    before = len(adapter.events)
    assert stub.rationalise(_kv(0x1)) is True
    assert len(adapter.events) == before  # no new events


def test_stub_emits_in_authored_order(kit):
    """ST-7: events are emitted in authored (cursor) order."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, k_rows)
    for i in range(len(k_rows)):
        stub.rationalise(_kv(0x1))
    emitted = [ev.proposal.kline for ev in adapter.events]
    assert emitted == [t.value.kline for t in k_rows]


def test_stub_event_significance_on_proposal_not_kind(kit):
    """ST-11: significance is carried on proposal.significance, not on event kind."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, k_rows)
    stub.rationalise(_kv(0x1))
    ev = adapter.events[0]
    assert ev.kind == "frame"  # every stub emission is a frame event
    # The significance is the K-row's declared band (read off proposal).
    assert ev.proposal.significance == k_rows[0].value.significance


def test_stub_grounded_is_observational(kit):
    """ST-10: grounded is observational — populated but never consulted for emission."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    stub = StubKAgent(_RecordingAdapter(), k_rows)
    stub.rationalise(_kv(0x1))
    assert k_rows[0].value.kline.signature in stub.grounded


def test_stub_consumes_shared_table_krows(kit):
    """ST-12: the stub consumes the K-rows of the shared pre-decoded table (no
    separate response table; no trigger-matching)."""
    _, _, _, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    stub = StubKAgent(_RecordingAdapter(), k_rows)
    assert stub.cursor == 0
    assert not hasattr(stub, "_pending_rows")  # no trigger-keyed index
    assert not hasattr(stub, "_initial_row")  # no INITIAL sentinel


# ── DDT-22..27: the dispatch loop ─────────────────────────────────────────


def _run_canonical(kit):
    tok, sigf, table, decoded = kit
    k_rows = [t for t in decoded if t.actor == "K"]
    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE, tokenizer=tok, signifier=sigf)
    adapter.bind(StubKAgent(adapter, k_rows))
    return run_session(table, adapter=adapter, tokenizer=tok, signifier=sigf)


def test_canonical_run_completes_on_primary_closing_s1(kit):
    """DDT-18/27: the canonical run terminates on dual-exhaustion; the closing K
    is the primary's S1 countersign, verified by construction (not detected)."""
    result = _run_canonical(kit)
    assert result.dual_exhaustion
    closing = result.closing
    assert closing is not None
    assert closing.value.significance == SIG_S1
    # The closing K is the primary's kline (the MHALL countersign).
    primary = result.t_submissions[0]
    assert closing.value.kline == primary.value.kline


def test_loop_is_dispatch_driven_every_T_validated(kit):
    """DDT-22/25: every T turn is computed by the supply function and validated
    against the table; every K is validated against the table. Counts match."""
    _, _, _, decoded = kit
    result = _run_canonical(kit)
    t_rows = [t for t in decoded if t.actor == "T"]
    k_rows = [t for t in decoded if t.actor == "K"]
    assert len(result.t_submissions) == len(t_rows)
    assert len(result.k_emissions) == len(k_rows)


def test_truncated_table_fails_loudly(kit):
    """DDT-26: a table whose final K-run is never consumed fails with k_unemitted
    (non-empty stub cursor at trainer exhaustion)."""
    tok, sigf, table, decoded = kit
    # Append a duplicate trailing K-row: the trainer stops at the real closing S1
    # (terminal no-op) but the stub has a K-row it never emits.
    extra = decoded + [decoded[-1]]
    k_rows = [t for t in extra if t.actor == "K"]
    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE, tokenizer=tok, signifier=sigf)
    adapter.bind(StubKAgent(adapter, k_rows))

    from training.dialogue import build_held_index, run_with_held
    from ks.compiler import compile_source

    held = build_held_index(
        compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
    )
    import queue
    from training.harness.constants import SUPERVISOR_ROLE
    from training.harness.message import Message

    captured: list[RationaliseEvent] = []
    bus.subscribe(SUPERVISOR_ROLE, lambda m: captured.append(m.message))

    def submit(kv):
        before = len(captured)
        bus.send(Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender=SUPERVISOR_ROLE))
        while True:
            try:
                m = bus._queue.get_nowait()  # noqa: SLF001
            except queue.Empty:
                break
            bus._dispatch(m)  # noqa: SLF001
        return captured[before:]

    with pytest.raises(LoopError) as exc:
        run_with_held(extra, held, submit=submit)
    assert exc.value.kind == "k_unemitted"


def test_stub_divergence_is_attributed(kit):
    """DDT-25: a stub that emits a K != table K fails as stub_divergence."""
    tok, sigf, table, decoded = kit
    # Give the stub a corrupted first K-row (wrong signature) while the table is intact.
    from kalvin.kline import KLine

    k_rows = [t for t in decoded if t.actor == "K"]
    bad_first = DecodedTurn(
        actor="K",
        op=k_rows[0].op,
        value=KValue(KLine(0xBAD, []), k_rows[0].value.significance),
    )
    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE, tokenizer=tok, signifier=sigf)
    adapter.bind(StubKAgent(adapter, [bad_first] + k_rows[1:]))
    with pytest.raises(LoopError) as exc:
        run_session(table, adapter=adapter, tokenizer=tok, signifier=sigf)
    assert exc.value.kind == "stub_divergence"


def test_supply_bug_is_attributed(kit):
    """DDT-25: a table whose T diverges from the supply function's output fails as
    supply_bug. (Corrupt a T-row so the computed T no longer matches the table.)"""
    tok, sigf, table, decoded = kit
    # Rebuild the table with a corrupted T-row (wrong band) so the supply function
    # computes the correct T while the table expects a different one.
    t_rows = [t for t in decoded if t.actor == "T"]
    bad_t = DecodedTurn(
        actor="T",
        op=t_rows[1].op,
        value=KValue(t_rows[1].value.kline, SIG_S1),  # wrong band
    )
    corrupted = [decoded[0], bad_t] + decoded[2:]
    # Reload as a fresh DialogueTable-shaped run via run_with_held with a stub fed
    # the corrupted table's K-rows.
    k_rows = [t for t in corrupted if t.actor == "K"]
    bus = MessageBus()
    adapter = KAgentAdapter(bus, role=TRAINEE_ROLE, tokenizer=tok, signifier=sigf)
    adapter.bind(StubKAgent(adapter, k_rows))

    from training.dialogue import build_held_index, run_with_held
    from ks.compiler import compile_source

    held = build_held_index(
        compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
    )
    import queue
    from training.harness.constants import SUPERVISOR_ROLE
    from training.harness.message import Message

    captured: list[RationaliseEvent] = []
    bus.subscribe(SUPERVISOR_ROLE, lambda m: captured.append(m.message))

    def submit(kv):
        before = len(captured)
        bus.send(Message(role=TRAINEE_ROLE, action="rationalise", message=kv, sender=SUPERVISOR_ROLE))
        while True:
            try:
                m = bus._queue.get_nowait()  # noqa: SLF001
            except queue.Empty:
                break
            bus._dispatch(m)  # noqa: SLF001
        return captured[before:]

    with pytest.raises(LoopError) as exc:
        run_with_held(corrupted, held, submit=submit)
    assert exc.value.kind == "supply_bug"


# ── DDT-7: opening is computed and table-validated ────────────────────────


def test_opening_is_computed_and_matches_table(kit):
    """DDT-7: the opening T (primary half at S2) is computed by the supply
    function and validated against the table — not read from the table."""
    _, _, _, decoded = kit
    result = _run_canonical(kit)
    opening = result.t_submissions[0]
    assert opening.value.significance == SIG_S2
    assert opening.value.kline == decoded[0].value.kline  # the primary
