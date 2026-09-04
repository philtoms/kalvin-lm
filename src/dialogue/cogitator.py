from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import KLine, KNode, KNodes, classify_misfit, is_identity, is_terminal
from kalvin.kvalue import KValue
from kalvin.significance import (
    PROPOSAL_AGGREGATOR,
)

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier

# Upper bound on edge hop chain depth (edge_hops's traversal bound).
MAX_HOP = 100


class Cogitator:
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


    def cogitate(self, entry: KLine) -> Iterator[KValue]:
        candidates = self._candidates(entry)
        queries = [entry] if self.signifier.is_ask(entry.signature) else self.state.find_canons(entry.signature)
        for query in queries:
            for candidate in candidates:
                q_set = set(query.nodes)
                c_set = set(candidate.nodes)
                underfit = list(q_set - c_set)
                overfit = list(c_set - q_set)
                fit = list(q_set & c_set)

                proposal, distance = self.expand(underfit, overfit, fit, exclude=query.signature)
                yield KValue(KLine(entry.signature, proposal), distance)
        return

    def expand(self,
            underfit: list[KNode],
            overfit: list[KNode],
            fit: list[KNode],
            exclude: KNode | None = None,
    ) -> tuple[list[KNode], int]:
        proposal: list[KNode] = []
        remainder: list[KNode] = []
        distance = 0

        for m1, m2 in [(underfit, overfit), (overfit, remainder)]:
            while len(m1) > 0:
                n=m1.pop(0)
                reserve=True if m1 is underfit else False
                for kl, hops in self.connotateY(n, exclude=exclude):
                    for m_nodes in [m2, fit]:
                        if kl.signature in m_nodes:
                            distance += hops
                            proposal.append(kl.signature)
                            m_nodes.remove(kl.signature)
                            reserve=False
                            break
                        if all(n in m_nodes for n in kl.nodes):
                            proposal.append(kl.signature)
                            for n in kl.nodes:
                                distance += hops
                                m_nodes.remove(n)
                                reserve=False
                            break

                if reserve:
                    remainder.append(n)

        if underfit or overfit or remainder:
            return [], 0

        proposal.extend(fit)
        return proposal, distance


    def connotateY(self, sig: KNode, depth: int = MAX_HOP, exclude: KNode | None = None) -> Iterator[tuple[KLine, int]]:
        """Yield ``(kline, hops)`` breadth-first over *every* non-terminal,
        non-identity connotation edge — not one deterministic path.

        Forward edges: klines resolving ``cur``'s signature. Reverse edges:
        klines whose signature shares a word bit with ``cur``, or that
        contain ``cur`` as a node — the bridges from word-level gaps to
        compound-signature klines. Klines resolving ``exclude`` (the
        query's own signature) are neither yielded nor traversed — the
        query must not act as a hub between its gap nodes' bits.

        BFS order is min-hops-first, so consumers halt at their k nearest
        results and never explore past them.
        """
        state = self._state
        signifier = self._state.signifier
        frontier: list[KLine] = [KLine(sig, [])]
        visited: set[KLine] = set()
        hop_count = 0
        while frontier and hop_count < depth:
            hop_count += 1
            next_frontier: list[KLine] = []
            for cur in frontier:
                edges = list(state.find_sig(cur.signature))
                edges.extend(
                    k
                    for k in state.where(
                        lambda k: k.signature != cur.signature
                        and not is_terminal(k)
                        and not is_identity(k)
                        and (
                            signifier.signifies(cur.signature, k.signature)
                            or cur.signature in k.nodes
                        )
                    )
                )
                for kline in edges:
                    if (
                        kline is None
                        or is_terminal(kline)
                        or is_identity(kline)
                    ):
                        continue
                    if kline in visited:
                        continue
                    reached = KLine(signifier.signature_of(kline.nodes), kline.nodes)
                    if exclude is not None and (
                        kline.signature == exclude or reached.signature == exclude
                    ):
                        continue
                    visited.add(kline)
                    yield reached, hop_count
                    next_frontier.append(reached)
            frontier = next_frontier


    def canonise(
        self,
        underfit: list[KNode],
        overfit: list[KNode],
        proposal: list[KNode],
    ) -> Iterator[KValue]:
        """Bridge the gap with connotation crossovers, one proposal per encoding.

        Every underfit connotation path may cross an overfit path where two
        elements signify each other; the klines under those elements are the
        bricks. A proposal is the shared ``proposal`` nodes prepended to a
        unique set of bricks whose signatures completely encode
        ``proposal_sig`` — every gap bit sits inside their OR-reduction.
        Brick sets are enumerated smallest-first and the search stops at the
        first size that encodes, so the leanest encodings come out first.
        Each slot (gap node) is graded by the hop depth of the nearest brick
        sharing its bits, through the proposal aggregator.
        """
        state = self._state
        signifier = self.signifier
        aggregator = PROPOSAL_AGGREGATOR
        decay = aggregator.decay

        proposal_sig = signifier.signature_of(underfit + overfit)
        if not proposal_sig or not underfit or not overfit:
            return  # a crossover needs both sides and something to encode

        # Every path element tagged with its hop depth; roots at depth 0.
        u_elements = [
            (e, hops)
            for paths in (self.connotate(n) for n in underfit)
            for path in paths
            for hops, e in enumerate(path)
        ]
        o_elements = [
            (e, hops)
            for paths in (self.connotate(n) for n in overfit)
            for path in paths
            for hops, e in enumerate(path)
        ]

        # Crossovers: (u, o) pairs whose type-words overlap. Each element
        # accumulates its root, so a pair encodes at least u_root | o_root.
        crossovers: list[tuple[KNode, int, KNode, int]] = [
            (u, u_hops, o, o_hops)
            for u, u_hops in u_elements
            for o, o_hops in o_elements
            if signifier.signifies(u, o)
        ]
        crossovers = list(dict.fromkeys(crossovers))
        if not crossovers:
            return

        slots = underfit + overfit
        seen: set[frozenset[KNode]] = set()
        for size in range(1, len(crossovers) + 1):
            produced = False
            for combo in combinations(crossovers, size):
                # Min hop depth per distinct element across the combo.
                hop_of: dict[KNode, int] = {}
                for u, u_hops, o, o_hops in combo:
                    for e, h in ((u, u_hops), (o, o_hops)):
                        if e not in hop_of or h < hop_of[e]:
                            hop_of[e] = h
                # Bricks: the klines heading the crossed elements. Elements
                # with no kline (bare roots) contribute nothing.
                brick_hops: dict[KNode, int] = {}
                for e, h in hop_of.items():
                    kline = state.find(e)
                    if kline is None:
                        continue
                    s = kline.signature
                    if s not in brick_hops or h < brick_hops[s]:
                        brick_hops[s] = h
                if not brick_hops:
                    continue
                brick_sigs = frozenset(brick_hops)
                if (
                    brick_sigs in seen
                    or not signifier.bit_in(
                        proposal_sig, signifier.signature_of(list(brick_sigs))
                    )
                ):
                    continue
                seen.add(brick_sigs)
                produced = True
                slot_values = [
                    decay(min(h for s, h in brick_hops.items() if s & n != 0))
                    if any(s & n != 0 for s in brick_hops)
                    else 0.0
                    for n in slots
                ]
                proposal_kline = KLine(proposal_sig, list(proposal) + list(brick_hops))
                yield KValue(proposal_kline, aggregator.compose_terminal(slot_values))
            if produced:
                break  # leanest encodings only — stop at the first size that works


    def connotate(self, sig: KNode, depth: int = MAX_HOP) -> list[list[KNode]]:
        """Root-to-leaf connotation paths from *sig*, to *depth* hops.

        Breadth-first over non-terminal, non-identity resolution edges.
        Each path is a chain of OR-accumulated signatures starting at *sig*;
        a node reached by an earlier hop appears in only that path prefix,
        so paths are loop-free. A path ends at a leaf: no further edges, or
        the *depth* bound.
        """
        state = self._state
        signifier = self._state.signifier
        paths: list[list[KNode]] = [[sig]]
        visited: set[KNode] = {sig}
        for _ in range(depth):
            next_paths: list[list[KNode]] = []
            extended: set[KNode] = set()
            for path in paths:
                cur = path[-1]
                branched = False
                for kline in state.find_sig(cur):
                    if (
                        kline is None
                        or is_terminal(kline)
                        or is_identity(kline)
                    ):
                        continue
                    reached = cur.merge(signifier.signature_of(kline.nodes))
                    if reached in visited:
                        continue
                    branched = True
                    visited.add(reached)
                    next_paths.append(path + [reached])
                if not branched:
                    extended.add(cur)
            paths = next_paths + [p for p in paths if p[-1] in extended]
            if not next_paths:
                break
        return paths

    def _candidates(self, entry: KLine) -> list[KLine]:
        signifier = self._state.signifier
        conns: list[KLine] = []

        for sig in self._state.where(
            lambda k: entry.signature != k.signature
            and not is_identity(k)
            and signifier.signifies(entry.signature, k.signature)
        ):
            conns.append(sig)
        return conns
