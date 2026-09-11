"""Expand — graph expansion.

This module owns the *graph* layer: traversing the model to expand a
query|candidate pair into connotations and a terminal significance byte
(``expand``).

It builds on the significance topology layer (``kalvin.significance``), which
owns the 8-bit distance algebra, the band layout, the decay/compose seams and
the ``Aggregator``, and the structural-grounding predicates. The dependency
is strictly one-way: expand → significance, never the reverse.

The module reads from the Model (storage) but is a separate responsibility:
Model indexes and retrieves; Expand walks the graph.

Misfit-comprehension (generating expansion proposals for candidates whose
signature and nodes' signature disagree) lives in its own module,
:mod:`kalvin.proposals`. Promotion of structurally-participating klines
after ratification is the Rationaliser's responsibility (see
:attr:`Rationaliser._promote_participating`).

Module-level constants and types:
  MAX_HOP, edge_hops, expand
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import KLine, KNode, classify_misfit, is_identity, is_terminal
from kalvin.kvalue import KValue
from kalvin.significance import (
    PROPOSAL_AGGREGATOR,
)

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier

# Upper bound on edge hop chain depth (edge_hops's traversal bound).
MAX_HOP = 100


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


    def propose(self, entry: KLine) -> Iterator[KValue]:
        candidates = self._candidates(entry)
        queries = [entry] if self.signifier.is_ask(entry.signature) else self.state.findCanons(entry.signature)
        for query in queries:
            for candidate in candidates:
                yield from self.expand(query, candidate, _visited=set())
        return

    def expand(
        self,
        query: KLine,
        candidate: KLine,
        *,
        _visited: set[tuple[int, KNode]],
        _top: bool = True,
    ) -> Iterator[KValue]:
        """Expand a query-candidate pair, yielding connotations and terminal byte.

        Compose-on-return aggregation: topology is captured on descent (per-node
        accountedness retained as a float), and composition is applied on the
        return phase.

        Per-node accountedness:
        matched & grounded       -> 1.0
        matched but ungrounded   -> decay(1)   (one hop of doubt)
        resolvable in h hops      -> decay(h)
        unresolvable              -> 0.0

        Yield asymmetry: exact opposing matches and S3 connotation bridges
        recurse; signifies matches emit a side-candidate and do not recurse.
        The final yield is always the terminal KValue for the original
        pair.

        ``aggregator`` bundles the layout (S2_S3_BOUNDARY) and the two pluggable
        seams (DecayFunction, ComposeFunction).
        """
        key = (query.signature, candidate.signature)
        if key in _visited:
            return  # cycle detected
        _visited.add(key)

        state = self._state
        signifier = state.signifier

        q_set = set(query.nodes)
        c_set = set(candidate.nodes)
        underfit = q_set - c_set
        overfit = c_set - q_set
        s2_fit = q_set & c_set

        s2_target: list[KNode] = list(s2_fit)
        s3_connotations: dict[KNode, int] = {}  # sig -> min hops from any query node

        # Per-node accountedness, in slot order. One float per slot.
        slot_values: list[float] = []

        aggregator = PROPOSAL_AGGREGATOR
        decay = aggregator.decay

        for n in underfit:
            accounted = 0.0  # unresolvable default (case F)
            q_kline = state.find(n)
            if q_kline is not None:
                s3_connotations[n] = 1
                for hops, hop_sig in self._edge_hops(n):
                    c_kline = state.find(hop_sig)
                    if c_kline is None:
                        continue
                    if hop_sig in overfit:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        yield from self.expand(
                            q_kline, c_kline, _visited=_visited, _top=False
                        )
                        break
                    elif signifier.signifies(n, hop_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        if self._sayable(c_kline):
                            yield KValue(c_kline, sig_byte)
                        break
                    if hop_sig not in s3_connotations or hops < s3_connotations[hop_sig]:
                        s3_connotations[hop_sig] = hops
            slot_values.append(accounted)

        for n in overfit:
            accounted = 0.0
            q_kline = state.find(n)
            if q_kline is not None:
                for hops, hop_sig in self._edge_hops(n):
                    c_kline = state.find(hop_sig)
                    if c_kline is None:
                        continue
                    underfit_sig = signifier.signature_of(list(underfit))
                    if signifier.sig_in(hop_sig, underfit_sig):
                        accounted = decay(hops)
                        s2_target.append(n)
                        break
                    if hop_sig in underfit:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        yield from self.expand(
                            q_kline, c_kline, _visited=_visited, _top=False
                        )
                        break
                    elif signifier.signifies(n, hop_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        if self._sayable(c_kline):
                            yield KValue(c_kline, sig_byte)
                        break
                    elif hop_sig in s3_connotations:
                        # case E: S3 connotation bridge -> recurse (no side-candidate).
                        s3_hop = s3_connotations[hop_sig] + hops
                        accounted = decay(s3_hop)
                        yield from self.expand(
                            q_kline, c_kline, _visited=_visited, _top=False
                        )
                        break
            slot_values.append(accounted)

        # Matched nodes: grounded -> 1.0; matched-ungrounded -> decay(1).
        for n in s2_fit:
            kl = state.find(n)
            if kl is not None and state.is_grounded(kl):
                slot_values.append(1.0)
            else:
                # Ungrounded match OR not in state: one hop of doubt.
                slot_values.append(decay(1))

        if not slot_values:
            # Both klines node-less: vacuously fully accounted.
            slot_values = [1.0]

        significance = aggregator.compose_terminal(slot_values)
        if _top and self._sayable(candidate):
            yield KValue(candidate, significance)

    def _candidates(self, entry: KLine) -> list[KLine]:
        """Held klines selectable for the entry — Def 16: occurrence.

        A candidate is selectable when its signature occurs in one of the
        entry's nodes (containment in bit space — a compound node references
        what the candidate is). Content overlap is not selection: it routes
        the band inside expand.
        """
        signifier = self._state.signifier
        conns: list[KLine] = []

        for sig in self._state.where(
            lambda k: entry.signature != k.signature
            and not is_identity(k)
            and any(
                signifier.node_in(k.signature, n) for n in entry.nodes
            )
        ):
            conns.append(sig)
        return conns

    def _connotations(self, entry: KLine) -> dict[KNode, int]:
        """``sig -> min hops`` over edge-hop chains from the entry's nodes and
        from its underfit gap's covering bridges."""
        signifier = self._state.signifier
        conns: dict[KNode, int] = {}
        for node in entry.nodes:
            for hops, sig in self._edge_hops(node):
                if sig != entry.signature:
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

    def _nodes_of(self, signature: KNode) -> list[KNode]:
        """Held nodes whose bit pattern sits inside ``signature``."""
        out: list[KNode] = []
        signifier = self.signifier
        for kline in self._state.where(
            lambda k: signifier.sig_in(k.signature, signature) and is_identity(k)
        ):
            out.append(kline.signature)
        return out

    def _edge_hops(
        self, sig: KNode
    ) -> Iterator[tuple[int, KNode]]:
        """Yield ``(hops, sig)`` breadth-first over *every* non-terminal,
        non-identity resolution edge — not one deterministic path.

        BFS order is min-hops-first, so consumers halt at their k nearest
        results and never explore past them.
        """
        state = self._state
        signifier = self._state.signifier
        frontier: list[KNode] = [sig]
        visited: set[KNode] = {sig}
        hop_count = 0
        while frontier and hop_count < MAX_HOP:
            hop_count += 1
            next_frontier: list[KNode] = []
            for cur in frontier:
                for kline in state.find_sig(cur):
                    if (
                        kline is None
                        or is_terminal(kline)
                        or is_identity(kline)
                    ):
                        continue
                    reached = signifier.signature_of(kline.nodes)
                    if reached in visited:
                        continue
                    visited.add(reached)
                    yield hop_count, reached
                    next_frontier.append(reached)
            frontier = next_frontier

    def _sayable(self, kline: KLine) -> bool:
        """A proposal says something new: not an identity (an ask or a
        fact), not already grounded (nothing to ratify)."""
        return not is_identity(kline) and not self._state.is_grounded(kline)
