"""Hop layer — Defs 21–23 in the engine (§11).

Selection assembles candidate goals (Def 22); the scope is a dual-rooted
trawl (Def 23); a hop runs derivations down the goal list (Def 21);
re-entry changes A and reselects.
"""

from __future__ import annotations

from kalvin.hop import Hop, candidate_goals, run_hops, trawl
from kalvin.kline import KLine, KNode
from kalvin.memory import Memory
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
        KLine(MHALL, [M, H, A, L, L]),
        KLine(DH, [D, H]),
        KLine(ALL, [A, L, L]),
        KLine(DH, [H]),
        KLine(W, [O]),
        KLine(ALL, [O]),
        KLine(M, [M]),  # identity — inert
    ]


def _state(*memory: KLine) -> Memory:
    st = Memory(SIG)
    st.work_list.extend(memory or _memory())
    return st


def test_selection_orders_goals_by_overlap():
    goals = candidate_goals(_state(), KLine(WDMH, [W, D, M, H]), SIG)
    sigs = [int(k.signature) for k in goals]
    # J(wdhm, dh) = 1/2 leads; mhall 1/3 follows; m:[m] covers node m
    assert sigs[:2] == [int(DH), int(MHALL)]
    assert int(MHALL) in sigs
    # all:[a,l,l] covers no node of [w,d,h,m] — never a candidate (Def 8)
    assert int(ALL) not in sigs


def test_trawl_is_dual_rooted_and_depth_bounded():
    mem = _state()
    # roots w (A-side) and m,h,a,l,l (B-side): the o-bridge is reached in
    # one round — w:[o] touches w, all:[o] touches a,l,l
    scope = trawl(mem, [W, D, M, H], [M, H, A, L, L], SIG, max_depth=1)
    sigs = {int(k.signature) for k in scope}
    assert {int(W), int(ALL)} <= sigs
    # depth 0 trawls nothing
    assert trawl(mem, [W, D, M, H], [M, H, A, L, L], SIG, max_depth=0) == []


def test_trawl_excludes_terminals():
    scope = trawl(_state(), [M, H], [M, H, A, L, L], SIG)
    assert int(M) not in {int(k.signature) for k in scope}


def test_trawl_touches_at_content():
    # A connoted compound shares content with the roots — it scopes,
    # even though its whole value never equals a root's value.
    ws = bit(7)  # w|s compound, root w
    scope = trawl(_state(KLine(ws, [W])), [W, D, M, H], [M, H, A, L, L], SIG, max_depth=1)
    assert int(ws) in {int(k.signature) for k in scope}
    # Token-id bits alone carry no correspondence: values sharing only
    # low bits never touch.
    root = (1 << (32 + 9)) | 0xFF
    sig = (1 << (32 + 8)) | 0xFF
    node = (1 << (32 + 10)) | 0xFF  # shares 0xFF with root, no word bit
    assert trawl(_state(KLine(sig, [node])), [root], [root], SIG, max_depth=5) == []


def test_hop_ends_abandoned_at_the_goal_bound():
    hop = Hop(_state(), KLine(WDMH, [W, D, M, H]), SIG, max_goals=1).run()
    assert hop.ending == "abandoned"
    assert len(hop.results) == 1


def test_reentry_reselects_and_runs_to_done():
    h = run_hops(_state(), KLine(WDMH, [W, D, M, H]), SIG)
    assert h.ending == "done"
    # hop 1 wrote the bridge hop 2 consumed
    assert any(int(k.signature) == int(W) for k in h.writes)
    # re-entry queues the done derivation's ending state
    assert [int(n) for n in h.reentry.nodes] == [
        int(n) for n in h.results[-1].trace[-1]
    ]


def test_ask_when_nothing_progresses():
    # only an identity: no candidate covers a node, the hop has no goals
    h = run_hops(_state(KLine(O, [O])), KLine(ALL, [A, L]), SIG)
    assert h.ending == "stuck"
    assert h.results == []
