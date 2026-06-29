"""Tests for the table-driven StubKAgent — specs/stub-kagent.md ST-1..ST-11.

The stub is a deterministic contract double for ``KAgent``: it implements
``_KAgentLike`` and emits ``RationaliseEvent``s from an authored Response Table
instead of rationalising. These tests exercise the table-matching, single-cascade,
and band-carrying rules directly, using a tiny recording adapter so no real bus is
needed.
"""

from __future__ import annotations

import pytest

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.stub_kagent import INITIAL, ResponseRow, StubKAgent

# ── Helpers ──────────────────────────────────────────────────────────────


class _RecordingAdapter:
    """Minimal adapter: records every ``on_event`` call. Stands in for KAgentAdapter."""

    def __init__(self) -> None:
        self.events: list[RationaliseEvent] = []

    def on_event(self, event: RationaliseEvent) -> None:
        self.events.append(event)


def _kl(sig: int, nodes: list[int]) -> KLine:
    return KLine(sig, nodes)


def _kv(sig_val: int, nodes: list[int], significance: int) -> KValue:
    return KValue(_kl(sig_val, nodes), significance)


def _row(
    trigger: KValue | str,
    *,
    requests: tuple[KValue, ...] = (),
    grounds: tuple[KValue, ...] = (),
    countersigns: tuple[KValue, ...] = (),
) -> ResponseRow:
    return ResponseRow(
        trigger=trigger,
        requests=requests,
        grounds=grounds,
        countersigns=countersigns,
    )


# ── ST-1: satisfies _KAgentLike ──────────────────────────────────────────


def test_stub_satisfies_kagent_like():
    """ST-1: rationalise/countersign/save/codec are present with the specified returns."""
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, [])
    trigger = _kv(0xAA, [1, 2], SIG_S4)

    assert stub.rationalise(trigger) is True        # no rows → silent True
    assert stub.countersign(trigger) is True        # no-op True
    assert stub.save("/tmp/ignored") is None        # no-op
    assert stub.codec() is None                     # placeholder
    assert stub.cogitate_drain() is True            # adapter-compat no-op


# ── ST-2: requests emitted as frame at SIG_S4 ────────────────────────────


def test_requests_emitted_as_frame_at_s4():
    """ST-2: a row's requests fire as `frame` events with proposal at SIG_S4."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    missing = _kv(0x20, [2], SIG_S4)
    stub = StubKAgent(adapter, [_row(trigger, requests=(missing,))])

    assert stub.rationalise(trigger) is True

    assert len(adapter.events) == 1
    ev = adapter.events[0]
    assert ev.kind == "frame"
    assert ev.query == trigger
    assert ev.proposal.kline == missing.kline
    assert ev.proposal.significance == SIG_S4


# ── ST-3: grounds emitted at their structural band ───────────────────────


def test_grounds_emitted_at_structural_band():
    """ST-3: each ground fires at the band authored on its KValue (S2/S3/S4)."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    canon = _kv(0x21, [2, 3], SIG_S2)      # canon → S2
    relation = _kv(0x31, [4], SIG_S3)      # relation → S3
    atom = _kv(0x41, [5], SIG_S4)          # atom → S4
    stub = StubKAgent(adapter, [_row(trigger, grounds=(canon, relation, atom))])

    stub.rationalise(trigger)

    proposals = [ev.proposal for ev in adapter.events]
    assert proposals[0].significance == SIG_S2
    assert proposals[1].significance == SIG_S3
    assert proposals[2].significance == SIG_S4
    # all carried as frame events
    assert all(ev.kind == "frame" for ev in adapter.events)
    # grounded records each ground's signature (observational)
    expected = {
        canon.kline.signature,
        relation.kline.signature,
        atom.kline.signature,
    }
    assert stub.grounded == frozenset(expected)


# ── ST-4: countersigns emitted at SIG_S1 ─────────────────────────────────


def test_countersigns_emitted_at_s1():
    """ST-4: a row's countersigns fire as frame events with proposal at SIG_S1."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    primary = _kv(0x50, [9], SIG_S1)
    stub = StubKAgent(adapter, [_row(trigger, countersigns=(primary,))])

    stub.rationalise(trigger)

    assert len(adapter.events) == 1
    ev = adapter.events[0]
    assert ev.kind == "frame"
    assert ev.proposal.significance == SIG_S1
    assert ev.proposal.kline == primary.kline
    # a countersign ratifies → recorded as grounded
    assert primary.kline.signature in stub.grounded


# ── ST-5: a row fires at most once ───────────────────────────────────────


def test_row_fires_at_most_once():
    """ST-5: a second submission of the same trigger finds no pending row → silent."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    req = _kv(0x20, [2], SIG_S4)
    stub = StubKAgent(adapter, [_row(trigger, requests=(req,))])

    stub.rationalise(trigger)
    assert len(adapter.events) == 1

    # Second submission of the same trigger kline — row already consumed.
    stub.rationalise(trigger)
    assert len(adapter.events) == 1  # nothing new emitted
    assert trigger in stub.fired


# ── ST-6: no match returns True, silent ──────────────────────────────────


def test_no_match_returns_true_silent():
    """ST-6: a submission matching no pending row returns True with no events."""
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, [_row(_kv(0x10, [1], SIG_S4), requests=(_kv(0x20, [2], SIG_S4),))])

    unrelated = _kv(0x99, [7], SIG_S4)
    assert stub.rationalise(unrelated) is True
    assert adapter.events == []


# ── ST-7: authored order (requests, then grounds, then countersigns) ─────


def test_emission_order_requests_then_grounds_then_countersigns():
    """ST-7: within a row, emissions are ordered requests → grounds → countersigns."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    req = _kv(0x20, [2], SIG_S4)
    ground = _kv(0x30, [3], SIG_S2)
    cs = _kv(0x40, [4], SIG_S1)
    stub = StubKAgent(
        adapter,
        [_row(trigger, requests=(req,), grounds=(ground,), countersigns=(cs,))],
    )

    stub.rationalise(trigger)

    assert [ev.proposal.kline for ev in adapter.events] == [req.kline, ground.kline, cs.kline]
    assert [ev.proposal.significance for ev in adapter.events] == [SIG_S4, SIG_S2, SIG_S1]


# ── ST-8: countersign is a no-op returning True ──────────────────────────


def test_countersign_is_noop_true():
    """ST-8: countersign never emits and always returns True."""
    adapter = _RecordingAdapter()
    stub = StubKAgent(adapter, [])

    payload = _kv(0x10, [1], SIG_S1)
    assert stub.countersign(payload) is True
    assert adapter.events == []


# ── ST-9: single cascade (initial row, then a matched chain) ─────────────


def test_single_cascade_initial_row_then_chain():
    """ST-9: the stub drives one cascade — initial row kicks off, chain advances per submission."""
    adapter = _RecordingAdapter()
    # The cascade: trainer submits the primary first → initial row requests its operands.
    primary_kl = _kl(0x01, [10, 11])
    primary = KValue(primary_kl, SIG_S4)            # the trainer's submission
    primary_cs = KValue(primary_kl, SIG_S1)         # the stub's S1 ratification
    operand_s = _kv(0x02, [20], SIG_S4)
    operand_o = _kv(0x03, [30], SIG_S4)
    # initial: respond to the first submission by requesting S and O.
    initial = _row(INITIAL, requests=(operand_s, operand_o))
    # when S is submitted, ground it as an atom (S4)
    row_s = _row(operand_s, grounds=(_kv(0x12, [20], SIG_S4),))
    # when O is submitted, countersign the primary at S1 → lesson completes
    row_o = _row(operand_o, countersigns=(primary_cs,))
    stub = StubKAgent(adapter, [initial, row_s, row_o])

    # First submission (the primary) fires the initial row — single cascade begins.
    stub.rationalise(primary)
    assert len(adapter.events) == 2  # the two operand requests
    assert [ev.proposal for ev in adapter.events] == [operand_s, operand_o]
    # No proactive re-prompt needed mid-cascade: the stub is not idle, it is driven.

    # Trainer submits S → row_s grounds the atom.
    stub.rationalise(operand_s)
    assert adapter.events[-1].proposal.significance == SIG_S4

    # Trainer submits O → row_o countersigns the primary at S1 (cascade complete).
    stub.rationalise(operand_o)
    assert adapter.events[-1].proposal.significance == SIG_S1
    assert adapter.events[-1].proposal.kline == primary_kl

    # The cascade is done; resubmitting the primary finds no row (initial consumed).
    stub.rationalise(primary)
    assert len(adapter.events) == 4  # no new emission


# ── ST-10: atom reuse is table-prescribed ────────────────────────────────


def test_atom_reuse_table_prescribed():
    """ST-10: a Canon whose operands are grounded emits its ground with zero new requests.

    The table authors the reuse (omits requests); the stub does not infer it.
    """
    adapter = _RecordingAdapter()
    # Operands already grounded earlier in the cascade (by prior rows, elided here).
    canon = _kv(0x22, [2, 3], SIG_S2)
    # The Canon's row prescribes the ground at S2 with NO requests — operands reused.
    canon_row = _row(canon, grounds=(canon,))
    stub = StubKAgent(adapter, [canon_row])

    stub.rationalise(canon)

    # Exactly one event: the canon ground at S2. Zero requests emitted.
    assert len(adapter.events) == 1
    assert adapter.events[0].proposal.significance == SIG_S2
    assert all(ev.kind == "frame" for ev in adapter.events)


# ── ST-11: significance on proposal, not on the event ────────────────────


def test_significance_carried_on_proposal_not_event():
    """ST-11: RationaliseEvent carries no significance field; the band is on proposal."""
    adapter = _RecordingAdapter()
    trigger = _kv(0x10, [1], SIG_S4)
    proposal = _kv(0x30, [3], SIG_S2)
    stub = StubKAgent(adapter, [_row(trigger, grounds=(proposal,))])

    stub.rationalise(trigger)

    ev = adapter.events[0]
    # The event has kind/query/proposal only — no top-level significance attribute.
    assert not hasattr(ev, "significance")
    assert ev.kind == "frame"
    assert ev.proposal.significance == SIG_S2


# ── §Definition: the "initial" row ───────────────────────────────────────


def test_initial_row_fires_on_first_call_only():
    """§Definitions: an `initial` row fires on the first rationalise() call and is consumed."""
    adapter = _RecordingAdapter()
    first_req = _kv(0x20, [2], SIG_S4)
    stub = StubKAgent(adapter, [_row(INITIAL, requests=(first_req,))])

    # First call — any submission fires the initial row.
    any_submission = _kv(0xAB, [99], SIG_S4)
    stub.rationalise(any_submission)
    assert len(adapter.events) == 1
    assert adapter.events[0].proposal == first_req

    # Second call — the initial row is gone; no kline rows either → silent.
    stub.rationalise(any_submission)
    assert len(adapter.events) == 1


def test_two_initial_rows_rejected():
    """Validation: at most one `initial` row is permitted."""
    adapter = _RecordingAdapter()
    with pytest.raises(ValueError, match="initial"):
        StubKAgent(adapter, [_row(INITIAL), _row(INITIAL)])


# ── adapter-compat (decision 3) ─────────────────────────────────────────


def test_cogitate_drain_is_noop_true():
    """The harness adapter drains cogitation unconditionally; the stub is synchronous."""
    stub = StubKAgent(_RecordingAdapter(), [])
    assert stub.cogitate_drain() is True
    assert stub.cogitate_drain(timeout=5.0) is True
