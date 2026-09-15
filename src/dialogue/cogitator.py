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
        for candidate in self._selectable(entry):
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
        """The bridging fill and its per-slot depth records (kalvin-algebra §11).

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
            for fwd_path in self.denotate(n):
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
                for fwd_path in self.denotate(n):
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


    def denotate(self, left: KNode, depth: int = MAX_HOP) -> Iterator[KPath]:
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


    def _selectable(self, entry: KLine) -> Iterator[KLine]:
        """Held non-terminal klines whose signature occurs as a node of
        ``entry`` — Def 16: selection is occurrence, not content overlap;
        terminals (unknowns, identities) offer no second side."""
        for kline in self._state.where(
            lambda k: k.signature != entry.signature
            and not is_terminal(k)
            and k.signature in entry.nodes
        ):
            yield kline

    def _candidates(self, sig: KSig) -> Iterator[KLine]:
        """Held klines whose signature overlaps ``sig`` — the search index
        for a reverse-occurrence licence (reduce's contraction); the exact
        licence is witness membership, checked at the use site."""
        signifier = self._state.signifier

        for kline in self._state.where(
            lambda k: sig != k.signature
            and not is_identity(k)
            and not is_relationship(k)
            and signifier.signifies(sig, k.signature)
        ):
            yield kline
