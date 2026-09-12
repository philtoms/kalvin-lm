from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import KLine, KNode, KSig, is_identity, is_terminal, is_relationship
from kalvin.kpath import KPath
from kalvin.kvalue import KValue
from kalvin.significance import (
    PROPOSAL_AGGREGATOR,
    SlotRecord,
    gamma_aggregate,
    gamma_to_byte,
    word_atom_count,
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
        # The held canon under the signature is the known decomposition;
        # without one, the entry's own nodes are the only decomposition in
        # hand. An ask-marked signature finds no canon under its key (the
        # atom is part of the value), so asks fall to their own nodes —
        # the same structural rule, no mark read.
        query = self.state.find_canon(entry.signature) or entry
        q_set = self.reduce(query)
        for candidate in self._candidates(entry.signature):
            c_set = self.reduce(candidate)
            underfit = list(q_set - c_set)
            overfit = list(c_set - q_set)
            fit = list(q_set & c_set)

            proposal, slots = self.expand(underfit, overfit, fit)
            kline = KLine(entry.signature, proposal)
            if not self._state.is_refused(kline):
                gamma = gamma_aggregate(
                    slots,
                    self.signifier.signature_of(query.nodes),
                    self.signifier.signature_of(candidate.nodes),
                )
                yield KValue(kline, gamma_to_byte(gamma))

    def reduce(self, entry: KLine) -> set[KNode]:
        reduced: list[KNode] = []
        remaining = list(entry.nodes)
        idx=0
        while idx < len(remaining):
            node = remaining[idx]
            for overlap in self._candidates(node):
                if overlap.signature!=entry.signature:
                    if all(o in remaining for o in overlap.nodes):
                        reduced.append(overlap.signature)
                        for o in overlap.nodes:
                            remaining.remove(o)
                        continue
            idx += 1
        reduced.extend(remaining)
        return set(reduced)

    def expand(
            self,
            underfit: list[KNode],
            overfit: list[KNode],
            fit: list[KNode],
    ) -> tuple[list[KNode], list[SlotRecord]]:
        """The bridging fill and its per-slot depth records (ks2 §11).

        Each proposal atom is priced by the edges crossed to arrive: a fit
        node is free (depth 0), a bridged node carries its hop count, a
        crossover carries both directions' hops. A slot with no bridge
        empties the proposal — the misfit asks.
        """
        proposal: list[KNode] = []
        remainder: list[KNode] = []
        rev_paths: dict[KSig, KPath] = {}
        slots: list[SlotRecord] = []

        while len(underfit) > 0:
            n=underfit.pop(0)
            preserve=True
            for fwd_path in self.connotateY(n):
                right, hops = fwd_path.right, fwd_path.hops
                for m_nodes in [overfit, fit]:
                    if right in m_nodes:
                        m_nodes.remove(right)
                        proposal.append(right)
                        slots.append((word_atom_count(right), hops))
                        preserve = False
                        break

                if preserve:
                    rev_paths[right] = fwd_path
                else:
                    break

            if preserve:
                remainder.append(n)

        if remainder:
            underfit = remainder
            while len(overfit) > 0:
                n=overfit.pop(0)
                for fwd_path in self.connotateY(n):
                    right, hops = fwd_path.right, fwd_path.hops
                    if right in rev_paths:
                        rev = rev_paths[right]
                        if rev.left in underfit:
                            underfit.remove(rev.left)
                            proposal.append(n)
                            slots.append((word_atom_count(n), hops + rev.hops))
                        elif rev.left in fit:
                            fit.remove(rev.left)
                            proposal.append(n)
                            slots.append((word_atom_count(n), hops + rev.hops))
                    else:
                        for m_nodes in [underfit, fit]:
                            if right in m_nodes:
                                m_nodes.remove(right)
                                proposal.append(n)
                                slots.append((word_atom_count(n), hops))
                                break

        if underfit or overfit:
            return [], []

        proposal.extend(fit)
        slots.extend((word_atom_count(n), 0) for n in fit)
        return proposal, slots


    def connotateY(self, left: KNode, depth: int = MAX_HOP) -> Iterator[KPath]:
        state = self._state
        signifier = self._state.signifier
        frontier: list[KNode] = [left]
        visited: set[KNode] = set()
        hops = 0
        while frontier and hops < depth:
            hops += 1
            next_frontier: list[KNode] = []
            for cur in frontier:
                for kline in state.where(lambda k: signifier.node_in(cur, k.signature)
                                         and is_relationship(k)):
                    right = signifier.signature_of(kline.nodes)
                    if right != left and right not in visited:
                        visited.add(right)
                        yield KPath(left, right, hops)
                        next_frontier.append(right)
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
                    or not signifier.node_in(
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

    def _candidates(self, sig: KSig) -> Iterator[KLine]:
        signifier = self._state.signifier

        for kline in self._state.where(
            lambda k: sig != k.signature
            and not is_identity(k)
            and not is_relationship(k)
            and signifier.signifies(sig, k.signature)
        ):
            yield kline
