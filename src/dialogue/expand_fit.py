"""The expand S2 strategy — emit true proposals for a pending misfit.

Edge-hop connotations are gathered from the entry's nodes and its underfit
gap's covering bridges. Grounded canon klines whose own chains cross one of
those connotations are the crossing candidates; their nodes and signatures
that reach a connotation (at ``connotation_hops + crossing_hops``) are the
fills. A fill is proposed under the entry's own signature — added to the
nodes (underfit), swapped for the excess nodes (badfit) — graded by the
fill's crossover distance. No proposal is invented: every node comes from
the entry or a grounded contributor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    classify_misfit,
    is_canon,
    is_identity,
    is_misfit,
    is_terminal,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    DEFAULT_AGGREGATOR,
    SIG8_MAX,
    SIG_MASK,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from kalvin.abstract import KSignifier

__all__ = ["ExpandFit"]

# Upper bound on edge hop chain depth (_edge_hops's traversal bound).
_MAX_HOP = 100

# The BPE token ID occupies the lower 32 bits of a node value; distance
# calculations match word identity on this half (see NLPSignifier's packing).
_BPE_MASK = 0xFFFF_FFFF


class ExpandFit:
    """The S2 strategy: grade every candidate, emit the most significant proposal."""

    def __init__(
        self,
        state: EngineState,
    ) -> None:
        self._state: EngineState = state

    @property
    def signifier(self) -> KSignifier:
        return self._state.signifier

    @property
    def state(self) -> EngineState:
        return self._state

    def propose(
        self,
        entry: KLine,
        _depth: int = 2,
    ) -> list[KValue]:
        signifier = self._state.signifier
        underfit, overfit = classify_misfit(entry, signifier)
        if not underfit and not overfit:
            return []
        nodes_sig = signifier.signature_of(entry.nodes)
        gap = signifier.residual(entry.signature, nodes_sig)
        excess = signifier.residual(nodes_sig, entry.signature)

        conns = self._crossover_connotations(entry)
        if not conns:
            return []

        if underfit and overfit:
            base_nodes = [
                n for n in entry.nodes if not signifier.signifies(n, excess)
            ]
            fills = self._crossing_fills(entry, conns)
        elif underfit:
            base_nodes = list(entry.nodes)
            fills = self._crossing_fills(entry, conns)
        else:
            base_nodes = [
                n for n in entry.nodes if not signifier.signifies(n, excess)
            ]
            fills = {}

        graded: list[KValue] = []
        for sig in fills:
            if gap and signifier.residual(gap, sig) == 0:
                # A gap-covering fill is the query word itself, not an answer.
                continue
            if sig in base_nodes:
                continue
            expanded = base_nodes + [sig]
            if sorted(expanded) == sorted(entry.nodes):
                # A self-fill reconstructs the entry: nothing new is said.
                continue
            if not signifier.signifies(
                signifier.signature_of(expanded), entry.signature
            ):
                continue
            kline = KLine(entry.signature, expanded, entry.dbg)
            if is_terminal(kline):
                continue
            graded.append(KValue(kline, self._grade(entry, kline)))
        pivot_out = self._pivot_proposals(entry)
        graded.extend(pivot_out)
        graded.sort(key=lambda kv: kv.significance & SIG_MASK, reverse=True)
        graded = [
            kv for kv in graded
            if not self._state.is_refused(kv.kline)
        ]
        if _depth > 0:
            # Reentry: each proposal's own nodes widen the connotation set;
            # propose from them to reach fills one hop further out.
            for kv in list(graded):
                best = kv.kline
                if is_misfit(best, signifier) and not self._state.is_grounded(best):
                    graded.extend(self.propose(best, _depth=_depth - 1))
        # Fill-derived proposals drop when their gap could not be filled
        # (uncovered bits = unassigned work). Pivot proposals are slot-accounted
        # by construction and survive.
        pivot_set = {(kv.kline.signature, tuple(kv.kline.nodes)) for kv in pivot_out}
        return [
            kv for kv in graded
            if (kv.kline.signature, tuple(kv.kline.nodes)) in pivot_set
            or signifier.residual(
                entry.signature, signifier.signature_of(kv.kline.nodes)
            ) == 0
        ]

    def _grade(self, entry: KLine, kline: KLine) -> int:
        """Post-hoc significance of a proposal for ``entry``: K's understanding
        of the proposal, per proposal node.

        1. The proposal itself is grounded (its exact shape is ratified) — S1.
        2. A node of the entry's canon — S2 (hop 1).
        3. A node crossover-reachable from a canon node in either direction —
           S3 at the crossover hops.
        4. Otherwise — S4: work assigned but unaccounted (0.0).
        """
        if self._state.is_grounded(kline):
            return SIG8_MAX
        canon = self._state.canon_nodes(entry.signature) or []
        aggregator = DEFAULT_AGGREGATOR
        slots: list[float] = []
        for n in kline.nodes:
            if n in canon:
                slots.append(1.0)
                continue
            hops = self._crossover_hops(n, canon)
            slots.append(aggregator.decay(hops) if hops is not None else 0.0)
        return aggregator.compose_terminal(slots)

    def _crossover_hops(self, node: int, canon: list[int]) -> int | None:
        """Minimum hops between ``node`` and a canon node, in either
        direction of the edge-hop chain; None when unreachable (S4)."""
        canon_set = set(canon)
        best: int | None = None
        for hops, sig in self._edge_hops(node):
            if sig in canon_set:
                best = hops if best is None else min(best, hops)
        for c in canon:
            for hops, sig in self._edge_hops(c):
                if sig == node:
                    best = hops if best is None else min(best, hops)
        return best

    def _pivot_proposals(self, entry: KLine) -> list[KValue]:
        """Align the entry's canon against each pivot canon that shares a node
        with it, and graft the pivot's word form onto the entry.

        Per entry-canon node: a shared node resolves to itself; a node with
        an edge-hop path into the pivot's nodes resolves to its pivot
        counterpart; anything else is a gap slot to be filled from the
        pivot's surplus. Significance is graded post-hoc by ``_grade``.
        """
        signifier = self._state.signifier
        state = self._state
        canon = state.canon_nodes(entry.signature)
        if not canon or len(canon) < 2:
            return []
        canon_set = set(canon)
        out: list[KValue] = []
        for pivot in state.where(
            lambda k: is_canon(k, signifier) and k.signature != entry.signature
        ):
            pnodes = list(pivot.nodes)
            pnode_set = set(pnodes)
            if not (canon_set & pnode_set):
                continue  # no S1 anchor: not a pivot
            resolved: list[int] = []
            gaps: list[int] = []
            # Grouped resolution first: canon nodes forming a grounded
            # sub-canon resolve as a unit through its signature's path.
            grouped: set[int] = set()
            for sub in state.where(lambda k: is_canon(k, signifier)):
                sub_set = set(sub.nodes)
                if sub_set and sub_set < canon_set and sub.signature != pivot.signature:
                    hit = next(
                        ((h, s) for h, s in self._edge_hops(sub.signature) if s in pnode_set),
                        None,
                    )
                    if hit is not None:
                        resolved.extend([hit[1]] * len(sub.nodes))
                        grouped |= sub_set
            for n in canon:
                if n in grouped:
                    continue
                if n in pnode_set:
                    resolved.append(n)
                    continue
                hit = next(
                    ((h, s) for h, s in self._edge_hops(n) if s in pnode_set), None
                )
                if hit is not None:
                    # The pivot node does this node's work: replace it.
                    resolved.append(hit[1])
                else:
                    gaps.append(n)
            # Fill the gap slots with the pivot's unassigned nodes (S4 fill:
            # work is assigned, if only by adjacency); a gap left with no node
            # at all is unfilled work — drop the proposal. A leftover with no
            # open gap has no work to do. A lone gap takes the whole leftover
            # residual as a grouped fill.
            leftovers = [n for n in pnodes if n not in resolved]
            if not gaps:
                pass
            elif len(gaps) == 1:
                resolved.extend(leftovers)
            elif len(gaps) <= len(leftovers):
                resolved.extend(leftovers[: len(gaps)])
            else:
                continue
            nodes: list[int] = []
            for n in resolved:
                if n not in nodes:
                    nodes.append(n)
            if sorted(nodes) == sorted(entry.nodes):
                continue
            kline = KLine(entry.signature, nodes, entry.dbg)
            if is_terminal(kline):
                continue
            if not signifier.signifies(
                signifier.signature_of(nodes), entry.signature
            ):
                continue
            out.append(KValue(kline, self._grade(entry, kline)))
        return out

    def _crossover_connotations(self, entry: KLine) -> dict[int, int]:
        """``sig -> min hops`` over edge-hop chains from the entry's nodes and
        from its underfit gap's covering bridges."""
        signifier = self._state.signifier
        conns: dict[int, int] = {}
        for node in entry.nodes:
            for hops, sig in self._edge_hops(node):
                if sig not in conns or hops < conns[sig]:
                    conns[sig] = hops
        underfit, _ = classify_misfit(entry, signifier)
        if underfit:
            nodes_sig = signifier.signature_of(entry.nodes)
            gap = signifier.residual(entry.signature, nodes_sig)
            for bridge in self._state.where(
                lambda k: not is_identity(k)
                and signifier.residual(gap, k.signature) == 0
                and signifier.signature_of(entry.nodes + [k.signature]) == entry.signature
            ):
                for node in bridge.nodes:
                    for hops, sig in self._edge_hops(node):
                        if sig not in conns or hops < conns[sig]:
                            conns[sig] = hops
        return conns

    def _crossing_candidates(
        self, entry: KLine, conns: dict[int, int]
    ) -> list[KLine]:
        """Grounded canon klines whose edge-hop chains cross a connotation."""
        signifier = self._state.signifier
        out: list[KLine] = []
        for candidate in self._state.where(
            lambda k: not is_identity(k) and is_canon(k, signifier)
        ):
            if candidate.signature == entry.signature:
                continue
            hop_sigs = [s for _, s in self._edge_hops(candidate.signature)]
            hop_sigs += [s for node in candidate.nodes for _, s in self._edge_hops(node)]
            if any(s in conns for s in hop_sigs) or any(
                n in conns for n in candidate.nodes
            ):
                out.append(candidate)
        return out

    def _crossing_fills(
        self, entry: KLine, conns: dict[int, int]
    ) -> dict[int, int]:
        """``fill sig -> total hops`` for candidate values crossing a connotation.

        A candidate's signature or node is a fill when it reaches a connotation
        through its own edge-hop chain; the distance is the connotation's hops
        plus the crossing hops.
        """
        fills: dict[int, int] = {}
        for candidate in self._crossing_candidates(entry, conns):
            for sig in (candidate.signature, *candidate.nodes):
                if sig in conns and (sig not in fills or conns[sig] < fills[sig]):
                    fills[sig] = conns[sig]
                for hops, reached in self._edge_hops(sig):
                    if reached in conns:
                        total = conns[reached] + hops
                        if sig not in fills or total < fills[sig]:
                            fills[sig] = total
        return fills

    def _edge_hops(
        self, sig: int
    ) -> Iterator[tuple[int, int]]:
        """Yield ``(hop_count, next_sig)`` for each resolution step.

        Follows: resolve sig → kline → signifier.signature_of(kline.nodes) → repeat.
        Stops at a dead end, an identity kline, or a cycle.
        """
        state = self._state
        signifier = self._state.signifier
        hop_count = 0
        visited: set[int] = set()
        while hop_count < _MAX_HOP:
            if sig in visited:
                break  # cycle detected
            visited.add(sig)
            for kline in state.find_bucket(sig):
                if kline is None or is_terminal(kline) or is_identity(kline):
                    break
                hop_count += 1
                sig = signifier.signature_of(kline.nodes)
                yield hop_count, sig
