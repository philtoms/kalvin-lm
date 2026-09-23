"""Hop — the strategy unit over the derivation model (kalvin-algebra.md §11).

Def 22 Selection — candidate goals for a queued kline: held klines each
of whose nodes lies in ν_A's walk (Def 15) — a node of ν_A, a witness
reached from one, or the end of a witness path over held klines —
ordered by descending γ(A, K). γ's depth factor is constant in K, so the
order is content overlap. The goal is taken from the top of the list.

Def 23 Scope — a depth-bounded trawl of the correspondence graph, rooted
at both parties' nodes. The scope is a derivation's memory for its
duration; writes go to STM, and only later trawls reach them.

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
from typing import TYPE_CHECKING, Sequence

from kalvin.derivation import (
    Derivation, DerivationResult, MAX_WALK_EDGES,
)
from kalvin.kline import KLine, canon_key, is_ask, is_terminal

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.memory import Memory

#: Def 21 bound — goals a hop takes from its list.
MAX_GOALS = 8
#: Def 23 bound — trawl depth, in correspondence-graph edges.
TRAWL_DEPTH = 5
#: Re-entry bound — the hop ceiling.
MAX_HOPS = 8


def walk_closure(
    klines: list[KLine], nodes: Sequence[int], signifier: KSignifier,
    *, bound: int = MAX_WALK_EDGES,
) -> set[int]:
    """ν_A's witness-path closure over held klines (§7's graph, any
    direction, depth-bounded): a value is covered when it is a node of
    ν_A, or shares a held kline with a covered value — as witness
    (the kline's ν carries it) or as content (σ of the head contains
    it). Selection coverage, not a derivation licence: the walk's step
    restrictions (Def 15) bind derivations, not the reachability a
    candidate is judged by."""
    reached: set[int] = {int(v) for v in nodes}
    frontier = list(reached)
    depth = 0
    while frontier and depth < bound:
        nxt: list[int] = []
        for v in frontier:
            for k in klines:
                if is_terminal(k):
                    continue
                w = [int(n) for n in k.nodes]
                head = int(k.signature)
                touches = (
                    v in w
                    or (head != v and int(signifier.residual(v, head)) == 0)
                )
                if not touches:
                    continue
                for value in w + [head]:
                    if value not in reached and value != v:
                        reached.add(value)
                        nxt.append(value)
        frontier = nxt
        depth += 1
    return reached


def candidate_goals(
    state: Memory, queued: KLine, signifier: KSignifier
) -> list[KLine]:
    """Def 22 — the goal list: held klines each of whose nodes lies in
    ν_A's walk (directly, or by a witness path), γ(A, K) order.

    The queued kline itself is never a candidate: C(A, A) is Canon at
    entry and proves nothing — a hop that breaks on it never reaches the
    real goals down the list. An ask's canon is that same vacuous case
    one step removed: the marker is not an atom, so the ask and its
    canon share content — the canon is excluded too, by the ask step.
    """
    content = int(signifier.signature_of(queued.nodes))
    ask_base = (
        canon_key(queued.signature) if is_ask(queued.signature) else None
    )
    pool = state.where(lambda k: not is_terminal(k))
    closure = walk_closure(pool, queued.nodes, signifier)
    scored: list[tuple[float, int, KLine]] = []
    # Goals are held content (frame/ltm), never STM: bridges are scratch
    # evidence for replacements, not klines to derive toward.
    for i, k in enumerate(pool):
        if k.signature == queued.signature and k.nodes == queued.nodes:
            continue  # the queued kline is not its own goal
        if is_ask(k.signature):
            continue  # a question is never a goal
        if ask_base is not None and canon_key(k.signature) == ask_base:
            continue  # an ask never heads its own goal list — nor its canon
        if not all(int(n) in closure for n in k.nodes):
            continue  # some node uncovered from ν_A — not a candidate
        kc = int(signifier.signature_of(k.nodes))
        union = signifier.measure(content | kc)
        scored.append((signifier.measure(content & kc) / union, i, k))
    return [k for _, _, k in sorted(scored, key=lambda t: (-t[0], t[1]))]


def trawl(
    state: Memory,
    a_nodes: list[int],
    b_nodes: list[int],
    signifier: KSignifier,
    *,
    max_depth: int = TRAWL_DEPTH,
) -> list[KLine]:
    """Def 23 — dual-rooted, depth-bounded correspondence-graph trawl.

    A kline touches the reached set when its signature or any node
    shares content with it — the same content-level coverage Def 22
    reads (token-id bits carry no correspondence). The reached set is
    the composition of every root's and scoped kline's values.
    """
    reach = 0
    for n in list(a_nodes) + list(b_nodes):
        reach |= int(n)
    pending = state.where(lambda k: not is_terminal(k), True)
    scope: list[KLine] = []
    for _ in range(max_depth):
        hit = [
            k
            for k in pending
            if signifier.signifies(k.signature, reach)
            or any(signifier.signifies(n, reach) for n in k.nodes)
        ]
        if not hit:
            break
        taken = {id(k) for k in hit}
        for k in hit:
            scope.append(k)
            reach |= int(k.signature)
            for n in k.nodes:
                reach |= int(n)
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
        state: Memory,
        queued: KLine,
        signifier: KSignifier,
        *,
        max_goals: int = MAX_GOALS,
        trawl_depth: int = TRAWL_DEPTH,
        **derivation_kwargs,
    ) -> None:
        self.state = state  # writes land in its STM tier
        self.queued = queued
        self.signifier = signifier
        self.max_goals = max_goals
        self.trawl_depth = trawl_depth
        self.derivation_kwargs = derivation_kwargs

    def run(self) -> HopResult:
        res = HopResult(ending="stuck")
        for goal in candidate_goals(
            self.state, self.queued, self.signifier
        )[: self.max_goals]:
            scope = trawl(
                self.state,
                self.queued.nodes,
                goal.nodes,
                self.signifier,
                max_depth=self.trawl_depth,
            )
            r = Derivation(
                scope, self.queued, goal, self.signifier, **self.derivation_kwargs
            ).run()
            r.goal = goal
            res.goals.append(goal)
            res.results.append(r)
            self.state.extend_stm(r.composed)  # writes go to STM, not the scope
            res.writes.extend(r.composed)
            if r.ending == "done":
                res.ending = "done"
                break
        else:
            # done | list exhausted | the goal bound cut the list
            n = len(res.goals)
            res.ending = (
                "abandoned"
                if n == self.max_goals
                and len(candidate_goals(
                    self.state, self.queued, self.signifier
                ))
                > n
                else "stuck"
            )
        res.reentry = self._reentry(res)
        return res

    def _reentry(self, res: HopResult) -> KLine | None:
        if not res.results:
            return None
        writer = next(
            (r for r in reversed(res.results) if r.composed), res.results[-1]
        )
        # The ask's goal declaration rides the re-entry — selection keeps it.
        return KLine(
            self.queued.signature, writer.trace[-1], dbg=self.queued.dbg
        )


def run_hops(
    state: Memory,
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
        hop = Hop(state, a, signifier, **hop_kwargs).run()
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
