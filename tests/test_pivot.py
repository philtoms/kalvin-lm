"""The pivot arm as Def 17 slot walk — consume licence and acquisition depth.

The WDMH→MHALL alignment: M shared (cost 0), DH→H through the connotation
(cost 1), the lone gap W grouped-filled from the pivot's leftovers priced
by its real chain W→O→Q→ALL (cost 3). The proposal must strictly shrink
the misfit mass against the pivot.
"""

from __future__ import annotations

from dialogue.engine_state import EngineState
from dialogue.pivot_fill import PivotFill
from kalvin.kline import KLine, KNode
from kalvin.significance import misfit_mass
from kalvin.signifier import NLPSignifier


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


W, D, M, H = bit(0), bit(1), bit(2), bit(3)
A, L, O, Q = bit(4), bit(5), bit(6), bit(7)
DH = D | H
ALL = A | L
MHALL = M | H | ALL
WDMH = W | D | M | H


def _state() -> EngineState:
    state = EngineState(NLPSignifier())
    for kl in (
        KLine(WDMH, [W, DH, M]),  # the entry canon
        KLine(MHALL, [M, H, A, L, L]),  # the pivot
        KLine(DH, [D, H]),  # did-have canon
        KLine(DH, [H]),  # connotation: did have -> had
        KLine(W, [O]),  # what -> Object
        KLine(O, [Q]),  # Object -> Query
        KLine(Q, [ALL]),  # Query -> the object phrase
    ):
        state.ground(kl, state.ltm)
    return state


def test_wdmh_alignment_priced_and_licensed():
    entry = KLine(WDMH, [W, DH, M])
    proposals = PivotFill(_state())._pivot_proposals(entry)
    answer = [kv for kv in proposals if set(kv.kline.nodes) == {M, H, A, L}]
    assert answer, "expected the full alignment proposal"
    kv = answer[0]
    assert sorted(kv.kline.nodes, key=int) == sorted([H, M, A, L, L], key=int)
    assert kv.kline.acq_depth == 3  # W->O->Q->ALL
    # The consume licence: the misfit mass against the pivot strictly shrank.
    sig = NLPSignifier()
    assert misfit_mass(sig.signature_of(kv.kline.nodes), MHALL) < misfit_mass(
        sig.signature_of(entry.nodes), MHALL
    )


def test_misfit_mass_is_the_symmetric_difference():
    assert misfit_mass(MHALL, MHALL) == 0
    assert misfit_mass(WDMH, MHALL) == 4  # {W, D} vs {A, L}
    assert misfit_mass(0, 0) == 0
