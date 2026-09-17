"""Derivation model golden master — the §9 worked example, plus the
stuck-at-entry path (Def 15)."""

from __future__ import annotations

from collections import Counter

from dialogue.engine_state import EngineState
from kalvin.derivation import Derivation
from kalvin.hop import run_hops
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


def _state(*memory: KLine) -> EngineState:
    st = EngineState(SIG)
    st.work_list.extend(memory or _memory())
    return st


def _run() -> object:
    return Derivation(
        _memory(), KLine(WDMH, [W, D, M, H]), KLine(MHALL, [M, H, A, L, L]), SIG
    ).run()


def test_single_derivation_freezes_scope_and_writes_bridge():
    r = _run()
    # The scope never sees the walk's write: the derivation ends stuck at
    # the state the bridge applies at — a later hop consumes it (Def 23).
    assert r.ending == "stuck"
    # Step 1: {d,h} ⇉ [dh] under dh:[d,h]
    assert r.trace[1] == [W, DH, M]
    # Step 2: dh ⇉ [h] under dh:[h] (connotation), misfit mass 4→3
    assert r.trace[2] == [W, H, M]


def test_hop_derives_documented_example():
    h = run_hops(_state(), KLine(WDMH, [W, D, M, H]), SIG)
    assert h.ending == "done"
    done = h.results[-1]
    assert Counter(map(int, done.trace[-1])) == Counter(map(int, [H, M, A, L, L]))
    # the goal is taken from the top of the list: mhall leads the done hop
    assert int(h.goals[-1].signature) == int(MHALL)


def test_walk_writes_composed_correspondence():
    r = _run()
    assert len(r.composed) == 1
    assert int(r.composed[0].signature) == int(W)
    assert r.composed[0].nodes == [A, L, L]
    assert r.composed[0].acq_depth == 3


def test_subject_identity_is_never_replaced():
    r = _run()
    assert all(int(M) in map(int, state) for state in r.trace)


def test_measurement_matches_section_10_example():
    # The done derivation inside the re-entry chain reproduces §10's
    # example: Ĥ = 9/5 on the adopted overfit, γ = 2^(-9/5).
    h = run_hops(_state(), KLine(WDMH, [W, D, M, H]), SIG)
    assert h.ending == "done"
    done = h.results[-1]
    assert abs(done.j0 - 2 / 5) < 1e-9  # [w,h,m] against mhall
    assert abs(done.j1 - 1.0) < 1e-9
    assert abs(done.hbar - 9 / 5) < 1e-9
    assert abs(done.gamma - 2 ** (-9 / 5)) < 1e-9


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


def test_overfit_walk_writes_and_hop_adopts():
    r = _overfit_run()
    # pure overfit: no A-side slot exists; the walk departs the goal's node
    assert r.ending == "stuck"
    assert len(r.composed) == 1
    assert int(r.composed[0].signature) == int(M)  # head = the anchor
    assert r.composed[0].nodes == [M, ALL]
    assert r.composed[0].acq_depth == 2
    h = run_hops(
        _state(
            KLine(MHALL, [M, H, ALL]),  # B — held canon, overfit sealed in [all]
            KLine(ALL, [O]),  # denotation
            KLine(ALL, [A, L, L]),  # canon
            KLine(O, [M]),  # denotation — the bridge edge
            KLine(M, [M]),  # identity — inert
        ),
        KLine(M | H, [M, H]),
        SIG,
    )
    assert h.ending == "done"
    assert Counter(map(int, h.results[-1].trace[-1])) == Counter(
        map(int, [M, ALL, H])
    )


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
