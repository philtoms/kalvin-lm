"""Derivation model golden master — the §9 worked example, plus the
stuck-at-entry path (Def 15)."""

from __future__ import annotations

from collections import Counter

from kalvin.derivation import Derivation
from kalvin.kline import KLine, KNode
from kalvin.signifier import NLPSignifier


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


M, H, A, L = bit(0), bit(1), bit(2), bit(3)
W, D, O = bit(4), bit(5), bit(6)
DH = D | H
ALL = A | L
MHALL = M | H | ALL
WDMH = W | D | M | H

SIG = NLPSignifier()


def _memory() -> list[KLine]:
    return [
        KLine(MHALL, [M, H, A, L, L]),  # B — held canon
        KLine(DH, [D, H]),  # canon
        KLine(ALL, [A, L, L]),  # canon
        KLine(DH, [H]),  # connotation (S2)
        KLine(W, [O]),  # denotation (S3)
        KLine(ALL, [O]),  # denotation (S3)
        KLine(M, [M]),  # identity — inert
    ]


def _run() -> object:
    return Derivation(
        _memory(), KLine(WDMH, [W, D, M, H]), KLine(MHALL, [M, H, A, L, L]), SIG
    ).run()


def test_done_with_documented_trace():
    r = _run()
    assert r.ending == "done"
    # Step 1: {d,h} ⇉ [dh] under dh:[d,h]
    assert r.trace[1] == [W, DH, M]
    # Step 2: dh ⇉ [h] under dh:[h] (connotation), misfit mass 4→3
    assert r.trace[2] == [W, H, M]
    # Step 4 consumes the composed witness; final ≡ [h,m,a,l,l] multiset-wise
    assert Counter(map(int, r.trace[-1])) == Counter(map(int, [H, M, A, L, L]))


def test_walk_writes_composed_correspondence():
    r = _run()
    assert len(r.composed) == 1
    assert int(r.composed[0].signature) == int(W)
    assert r.composed[0].nodes == [A, L, L]
    assert r.composed[0].acq_depth == 3


def test_subject_identity_is_never_replaced():
    r = _run()
    assert all(int(M) in map(int, state) for state in r.trace)


def test_measurement_matches_section_11():
    r = _run()
    assert abs(r.j0 - 1 / 3) < 1e-9
    assert abs(r.j1 - 1.0) < 1e-9
    assert abs(r.hbar - 9 / 5) < 1e-9
    assert abs(r.gamma - 2 ** (-9 / 5)) < 1e-9


def test_stuck_at_entry_without_a_bridge():
    r = Derivation(
        [KLine(O, [O])], KLine(ALL, [A, L]), KLine(O, [O]), SIG
    ).run()
    assert r.ending == "stuck"
    assert len(r.trace) == 1


def _overfit_memory() -> list[KLine]:
    return [
        KLine(MHALL, [M, H, ALL]),  # B — held canon, overfit sealed in [all]
        KLine(ALL, [O]),  # denotation
        KLine(ALL, [A, L, L]),  # canon
        KLine(O, [M]),  # denotation — the bridge edge
        KLine(M, [M]),  # identity — inert
    ]


def _overfit_run(**kwargs) -> object:
    return Derivation(
        _overfit_memory(),
        KLine(M | H, [M, H]),
        KLine(MHALL, [M, H, ALL]),
        SIG,
        **kwargs,
    ).run()


def test_overfit_walk_anchors_and_adopts():
    r = _overfit_run()
    assert r.ending == "done"
    # pure overfit: no A-side slot exists; the walk departs the goal's node
    assert len(r.composed) == 1
    assert int(r.composed[0].signature) == int(M)  # head = the anchor
    assert r.composed[0].nodes == [M, ALL]
    assert r.composed[0].acq_depth == 2
    assert Counter(map(int, r.trace[-1])) == Counter(map(int, [M, ALL, H]))


def test_overfit_stuck_at_entry_without_b_walks():
    r = _overfit_run(b_walks=False)
    assert r.ending == "stuck"
    assert len(r.trace) == 1


def test_b_walk_arrival_without_resolution_does_not_ground():
    """Arrival on content overlap at a node ν_A does not hold, with no
    canon contracting it, writes no bridge and ends stuck (Def 17 anchor
    refinement; the pre-fix loop grounded duplicates to the bound)."""
    X = bit(7)
    MX = M | X
    memory = [
        KLine(MHALL, [M, H, ALL]),
        KLine(ALL, [O]),
        KLine(O, [MX]),  # denotation — arrives at a compound sharing M
        KLine(M, [M]),
    ]
    r = Derivation(
        memory, KLine(M | H, [M, H]), KLine(MHALL, [M, H, ALL]), SIG
    ).run()
    assert r.ending == "stuck"
    assert r.composed == []
    assert len(r.trace) == 1


def test_two_ended_guard_licenses_adoption():
    memory = [
        KLine(MHALL, [M, H, ALL]),
        KLine(M, [M, ALL]),  # overfit — adopt fwd at shared content
    ]
    r = Derivation(
        memory, KLine(M | H, [M, H]), KLine(MHALL, [M, H, ALL]), SIG
    ).run()
    assert r.ending == "done"
    assert r.composed == []
    assert Counter(map(int, r.trace[-1])) == Counter(map(int, [M, ALL, H]))
