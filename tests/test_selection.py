"""Selection by occurrence — Def 16 in the engine (D6).

A candidate is selectable when its signature occurs in one of the entry's
nodes (containment in bit space). Content overlap is not selection.
"""

from __future__ import annotations

from dialogue.engine_state import EngineState
from dialogue.expand_fit import ExpandFit
from dialogue.pivot_fill import PivotFill
from kalvin.kline import KLine, KNode
from kalvin.signifier import NLPSignifier


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


W, D, M, H = bit(0), bit(1), bit(2), bit(3)
A, L = bit(4), bit(5)
DH = D | H
O = bit(6)
ALL = A | L
MHALL = M | H | ALL
WDMH = W | D | M | H


def _state() -> EngineState:
    state = EngineState(NLPSignifier())
    for kl in (
        KLine(M, [M]),  # identity — inert
        KLine(DH, [D, H]),  # canon
        KLine(MHALL, [M, H, A, L, L]),  # canon
        KLine(W, [O]),  # denotation
    ):
        state.ground(kl, state.ltm)
    return state


def _sigs(klines):
    return {int(k.signature) for k in klines}


def test_entry_nodes_select_their_references():
    entry = KLine(WDMH, [W, DH, M])
    candidates = ExpandFit(_state())._candidates(entry)
    sigs = _sigs(candidates)
    assert int(W) in sigs  # w ⊆ node w — w:[o] selectable
    assert int(DH) in sigs  # dh ⊆ node dh — the canon selectable
    assert int(MHALL) not in sigs  # mhall occurs in no node — never selectable


def test_identity_is_inert():
    entry = KLine(WDMH, [W, DH, M])
    sigs = _sigs(ExpandFit(_state())._candidates(entry))
    assert int(M) not in sigs


def test_overlap_is_not_selection():
    # mhall word-overlaps wdmh, but occurs in no node of the entry.
    entry = KLine(WDMH, [W, DH, M])
    state = _state()
    assert state.signifier.signifies(WDMH, MHALL)
    assert int(MHALL) not in _sigs(PivotFill(state)._candidates(entry))


def test_empty_entry_selects_nothing():
    ask = KLine(WDMH, [])
    assert ExpandFit(_state())._candidates(ask) == []
