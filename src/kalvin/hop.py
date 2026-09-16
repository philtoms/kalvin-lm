"""Hop — the strategy unit over the derivation model (kalvin-algebra.md §11).

Def 22 Selection — candidate goals for a queued kline: held klines whose
content covers a node of the queued witness (Def 8), ordered by descending
γ(A, K). γ's depth factor is constant in K, so the order is content
overlap. The goal is taken from the top of the list.

Def 23 Scope — a depth-bounded trawl of the correspondence graph, rooted
at both parties' nodes. The scope is a derivation's memory for its
duration; writes go to the reservoir, and only later trawls reach them.

Def 21 Hop — the strategy unit of one queued kline: goals down the list in
order, each scoped and derived to an ending. A derivation that ends
without done yields the next goal; the hop ends at done, at the list's
exhaustion, or at a bound.

Re-entry changes A: the next hop queues the ending state of the
derivation that wrote — the evidence-building route — or the last
derivation's ending state, and reselects candidates for B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.derivation import Derivation, DerivationResult
from kalvin.kline import KLine, is_terminal
from kalvin.significance import word_atom_count

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier

#: Def 21 bound — goals a hop takes from its list.
MAX_GOALS = 8
#: Def 23 bound — trawl depth, in correspondence-graph edges.
TRAWL_DEPTH = 4
#: Re-entry bound — the hop ceiling.
MAX_HOPS = 8


def candidate_goals(
    memory: list[KLine], queued: KLine, signifier: KSignifier
) -> list[KLine]:
    """Def 22 — the goal list: coverage pool (Def 8), γ(A, K) order.

    The queued kline itself is never a candidate: C(A, A) is Canon at
    entry and proves nothing — a hop that breaks on it never reaches the
    real goals down the list.
    """
    content = int(signifier.signature_of(queued.nodes))
    scored: list[tuple[float, int, KLine]] = []
    for i, k in enumerate(memory):
        if k.signature == queued.signature and k.nodes == queued.nodes:
            continue  # the queued kline is not its own goal
        kc = int(signifier.signature_of(k.nodes))
        if not any(int(n) & kc for n in queued.nodes):
            continue  # covers no node of ν_A — not a candidate
        union = word_atom_count(content | kc)
        scored.append((word_atom_count(content & kc) / union, i, k))
    return [k for _, _, k in sorted(scored, key=lambda t: (-t[0], t[1]))]


def trawl(
    memory: list[KLine],
    a_nodes: list[int],
    b_nodes: list[int],
    *,
    max_depth: int = TRAWL_DEPTH,
) -> list[KLine]:
    """Def 23 — dual-rooted, depth-bounded correspondence-graph trawl."""
    reach = {int(n) for n in a_nodes} | {int(n) for n in b_nodes}
    pending = [k for k in memory if not is_terminal(k)]
    scope: list[KLine] = []
    for _ in range(max_depth):
        hit = [
            k
            for k in pending
            if int(k.signature) in reach
            or any(int(n) in reach for n in k.nodes)
        ]
        if not hit:
            break
        taken = {id(k) for k in hit}
        for k in hit:
            scope.append(k)
            reach.add(int(k.signature))
            reach.update(int(n) for n in k.nodes)
        pending = [k for k in pending if id(k) not in taken]
    return scope


@dataclass
class HopResult:
    """A hop's outcome: ending, per-goal results, writes, re-entry kline."""

    ending: str  # "done" | "stuck" | "abandoned"
    goals: list[KLine] = field(default_factory=list)
    results: list[DerivationResult] = field(default_factory=list)
    writes: list[KLine] = field(default_factory=list)
    reentry: KLine | None = None


class Hop:
    """Def 21 — one queued kline, derivations down the goal list."""

    def __init__(
        self,
        memory: list[KLine],
        queued: KLine,
        signifier: KSignifier,
        *,
        max_goals: int = MAX_GOALS,
        trawl_depth: int = TRAWL_DEPTH,
        **derivation_kwargs,
    ) -> None:
        self.memory = memory  # the reservoir: writes extend it in place
        self.queued = queued
        self.signifier = signifier
        self.max_goals = max_goals
        self.trawl_depth = trawl_depth
        self.derivation_kwargs = derivation_kwargs

    def run(self) -> HopResult:
        goals = candidate_goals(self.memory, self.queued, self.signifier)
        res = HopResult(ending="stuck")
        for goal in goals[: self.max_goals]:
            scope = trawl(
                self.memory,
                self.queued.nodes,
                goal.nodes,
                max_depth=self.trawl_depth,
            )
            r = Derivation(
                scope, self.queued, goal, self.signifier, **self.derivation_kwargs
            ).run()
            res.goals.append(goal)
            res.results.append(r)
            self.memory.extend(r.composed)  # writes go to memory, not the scope
            res.writes.extend(r.composed)
            if r.ending == "done":
                res.ending = "done"
                break
        else:
            # done | list exhausted | the goal bound cut the list
            res.ending = "abandoned" if len(goals) > self.max_goals else "stuck"
        res.reentry = self._reentry(res)
        return res

    def _reentry(self, res: HopResult) -> KLine | None:
        if not res.results:
            return None
        writer = next(
            (r for r in reversed(res.results) if r.composed), res.results[-1]
        )
        return KLine(self.queued.signature, writer.trace[-1])


def run_hops(
    memory: list[KLine],
    queued: KLine,
    signifier: KSignifier,
    *,
    max_hops: int = MAX_HOPS,
    **hop_kwargs,
) -> HopResult:
    """The re-entry chain: hop k's ending state queues as hop k+1's input.

    Progress is a moved A or a grown memory; a hop that achieves neither
    is the ask. The hop ceiling bounds the chain.
    """
    reservoir = list(memory)
    a = queued
    results: list[DerivationResult] = []
    writes: list[KLine] = []
    goals: list[KLine] = []
    hop: HopResult | None = None

    def chain(h: HopResult) -> HopResult:
        """The returned result carries the whole chain, not just the last hop."""
        h.results = results + h.results
        h.writes = writes + h.writes
        h.goals = goals + h.goals
        return h

    for _ in range(max_hops):
        hop = Hop(reservoir, a, signifier, **hop_kwargs).run()
        results += hop.results
        writes += hop.writes
        goals += hop.goals
        if hop.ending == "done":
            return chain(hop)
        stalled = not hop.writes and (
            hop.reentry is None
            or [int(n) for n in hop.reentry.nodes] == [int(n) for n in a.nodes]
        )
        if stalled:
            return chain(hop)
        if hop.reentry is not None:
            a = hop.reentry
    assert hop is not None
    hop.ending = "abandoned"  # the hop ceiling
    return chain(hop)
