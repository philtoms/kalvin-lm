"""The cogitator — the cogitation pass over the memory work list.

The rationaliser feeds memory: directly on the fast path, or indirectly by
queuing on the work list. :func:`cogitate` is the second half of the turn —
a single oldest-first pass over that attention. A pass that changes the
work list can unblock further entries; callers re-enter cogitate until a
pass changes nothing.
"""

from __future__ import annotations

from kalvin.hop import run_hops
from kalvin.kline import KLine, canon_key
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.significance import gamma_to_byte

__all__ = ["cogitate"]


def cogitate(state: Memory) -> list[KValue]:
    """One oldest-first pass over the work list: ask, propose, or ground.

    Per entry, in priority order: a groundable entry grounds; a misfit
    entry draws proposals from the strategy; a grounded entry leaves
    attention. Entries that match no path persist for a later turn.
    A pass that changes the work list can unblock further entries —
    the caller re-enters until a pass changes nothing.
    """
    batch: list[KValue] = []

    idx = 0
    while idx < len(state.work_list):
        # Re-check the index each iteration: the ground cascade (via the
        # S2 strategy's ground callback, or the countersign/groundable
        # arms) can remove arbitrary work-list entries, shrinking the list
        # below the index this loop intends to visit.
        if idx >= len(state.work_list):
            break

        kline = state.work_list[idx]
        if state.is_groundable(kline):
            state.ground_cascade(kline)
        if state.is_answered(kline):
            # The ask's content form is grounded — the question has
            # its answer; attention leaves.
            state.remove_work_at(idx)
            continue

        batch.extend(_propose(state, kline))

        if state.is_grounded(kline):
            state.remove_work_at(idx)
            continue

        idx += 1

    return batch


def _propose(state: Memory, kline: KLine) -> list[KValue]:
    """The re-entry chain over the held memory (Defs 21–23): goals
    from the selection list in order, each scoped and derived to an
    ending; a hop that ends without done re-enters at the ending
    state of the derivation that wrote — the evidence-building
    route — so a composed correspondence is consumed by a later
    derivation of the same queued kline. Done derivations propose
    at their significance — J of the final content against the goal
    (Defs 16, 20); γ — significance net of complexity — grades
    effort and never selects the band. The chain's writes extend
    the STM tier later hops trawl."""
    hop = run_hops(state, kline, state.signifier)
    batch: list[KValue] = []
    original = [int(n) for n in kline.nodes]
    for result in hop.results:
        if result.ending != "done":
            # Stuck and abandoned ask; done at entry is the ground
            # path's, not a proposal.
            continue
        if result.trace[-1] == original:
            # Done without moving — the queued kline as held: the
            # ground path's done, not a derivation's answer.
            continue
        proposal = KLine(
            canon_key(kline.signature), result.trace[-1]
        )
        if not state.is_refused(proposal):
            batch.append(KValue(proposal, gamma_to_byte(result.j1)))
    return batch
