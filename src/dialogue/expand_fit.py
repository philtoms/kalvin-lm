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

from kalvin.kline import KLine, KNode, is_canon, is_terminal, is_identity, classify_misfit
from kalvin.kvalue import KValue
from kalvin.significance import (
    DEFAULT_AGGREGATOR,
    PROPOSAL_AGGREGATOR,
    SIG_S4,
    Aggregator,
)

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model

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


    def _nodes_of(self, signature: KNode) -> list[KNode]:
        """Held nodes whose bit pattern sits inside ``signature``."""
        out: list[KNode] = []
        signifier = self.signifier
        for kline in self._state.where(lambda k: signifier.bit_in(k.signature, signature) and is_identity(k)):
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
                for kline in state.find_bucket(cur):
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

    def propose(self, entry: KLine) -> Iterator[KValue]:
        for candidate in self._connotations(entry):
            c_kline = self._state.find(candidate)
            if c_kline is not None:
                yield from self.expand(entry, c_kline)
        return
    
    def expand(
        self,
        query: KLine,
        candidate: KLine,
        *,
        _visited: set[tuple[int, KNode]] | None = None,
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
        if _visited is None:
            _visited = set()

        key = (query.signature, candidate.signature)
        if key in _visited:
            return  # cycle detected
        _visited.add(key)

        state = self._state
        signifier = state.signifier

        q_set = set(query.nodes)
        c_set = set(candidate.nodes)
        mismatched_q = q_set - c_set
        mismatched_c = c_set - q_set
        matched = q_set & c_set

        s3_connotations: dict[KNode, int] = {}  # sig -> min hops from any query node

        # Per-node accountedness, in slot order. One float per slot.
        slot_values: list[float] = []

        aggregator = PROPOSAL_AGGREGATOR
        decay = aggregator.decay

        for n in mismatched_q:
            accounted = 0.0  # unresolvable default (case F)
            q_kline = state.find(n)
            if q_kline is not None:
                for hops, match_sig in self._edge_hops(n):
                    c_kline = state.find(match_sig)
                    if c_kline is None:
                        continue
                    if match_sig in mismatched_c:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        yield from self.expand(q_kline, c_kline, _visited=_visited)
                        break
                    elif signifier.signifies(n, match_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield KValue(c_kline, sig_byte)
                        break
                    elif match_sig not in s3_connotations or hops < s3_connotations[match_sig]:
                        s3_connotations[match_sig] = hops
            slot_values.append(accounted)

        for n in mismatched_c:
            accounted = 0.0
            q_kline = state.find(n)
            if q_kline is not None:
                for hops, match_sig in self._edge_hops(n):
                    c_kline = state.find(match_sig)
                    if c_kline is None:
                        continue
                    if match_sig in mismatched_q:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        yield from self.expand(q_kline, c_kline, _visited=_visited)
                        break
                    elif signifier.signifies(n, match_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield KValue(c_kline, sig_byte)
                        break
                    elif match_sig in s3_connotations:
                        # case E: S3 connotation bridge -> recurse (no side-candidate).
                        s3_hop = s3_connotations[match_sig] + hops
                        accounted = decay(s3_hop)
                        yield from self.expand(q_kline, c_kline, _visited=_visited)
                        break
            slot_values.append(accounted)

        # Matched nodes: grounded -> 1.0; matched-ungrounded -> decay(1).
        for n in matched:
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
        yield KValue(candidate, significance)

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

