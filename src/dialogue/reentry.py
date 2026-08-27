"""The S2 strategy: propose rational fills for a pending misfit.

Nothing is invented — every node in a proposal is either an original node,
the node of a grounded signature required by the entry's signature, or the
node of a grounded kline a required node connotates. Substitution is
structural only; nothing is grafted by adjacency.

An entry ``sig:[nodes]`` splits by structure:

- **fit** — nodes whose bit pattern sits inside the signature.
- **overfit** — nodes whose bit pattern is not contained in the signature.
- **underfit** — grounded signatures sitting inside the entry's signature
  that the fit nodes do not cover.

Three arms:

- **Fit** — no underfit and no overfit: the fit nodes are the proposal
  (an asked canon aligned to its grounded phrasing is the answer).
- **Underfit** — reenter under the same signature with the fit nodes plus
  the underfit signature's nodes; the reentry tends to the canon.
- **Overfit** — an overfit node is admissible only as a structural
  stand-in: a required node ``n`` whose connotation ``c`` (a grounded,
  non-terminal kline) accounts for the overfit nodes. The substitution
  ``o_sig = sig - n + c.signature`` yields a new head, reentered with the
  fit nodes plus ``c``'s nodes.

Reentry is breadth-first and halts at a per-call proposal budget; a
grounded entry short-circuits at S1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    KNode,
    is_canon,
    is_identity,
    is_terminal,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    PROPOSAL_AGGREGATOR,
    SIG8_MAX,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from kalvin.abstract import KSignifier

__all__ = ["Reentry"]

# Upper bound on edge hop chain depth (_edge_hops's traversal bound).
_MAX_HOP = 100

class Reentry:
    """The S2 strategy: rational fills for a pending misfit."""

    def __init__(
        self,
        state: EngineState,
    ) -> None:
        self._state = state

    @property
    def signifier(self) -> KSignifier:
        return self._state.signifier

    @property
    def state(self) -> EngineState:
        return self._state

    #: Per-call proposal budget: at most this many proposals per misfit;
    #: reentry proposals share the budget.
    BUDGET = 3

    def propose(
        self,
        entry: KLine,
        _depth: int = 2,
        _budget: int | None = None,
        _origin: KLine | None = None,
    ) -> Iterator[KValue]:
        """Yield rational proposals for ``entry``, breadth-first, budgeted.

        Emission happens only from a clean entry — one with no unresolved
        overfit or underfit. Each arm targets one outstanding misfit,
        structures it as its own proposal entry, and substitutes that entry's
        completions back into the parent; the parent emits when the
        substitution leaves it clean.
        """
        budget = self.BUDGET if _budget is None else _budget
        if budget <= 0:
            return
        state = self._state
        origin = entry if _origin is None else _origin

        if _origin is not None and state.is_grounded(entry):
            # A grounded reentry target has nothing to complete — and a
            # grounded kline is not a proposal (nothing to ratify).
            return

        nodes = list(entry.nodes)
        sig_nodes = self._nodes_of(entry.signature)
        fit = [n for n in nodes if self.signifier.bit_in(n, entry.signature)]
        overfit = [n for n in nodes if not self.signifier.bit_in(n, entry.signature)]
        underfit = [n for n in sig_nodes if n not in nodes]
        # underfit = self._underfit(entry, fit)

        if not overfit and not underfit:
            if fit:
                yield from self._emit(entry, fit, origin)
            return

        if _depth <= 0:
            return
        emitted = 0
        for u in underfit:
            if emitted >= budget:
                return
            for delivered in self._target(entry, u, fit, budget - emitted, _depth):
                if emitted >= budget:
                    break
                expanded = fit + [n for n in delivered if n not in fit]
                if sorted(expanded) == sorted(nodes):
                    continue
                for kv in self.propose(
                    KLine(entry.signature, expanded, entry.dbg),
                    _depth=_depth - 1,
                    _budget=budget - emitted,
                    _origin=origin,
                ):
                    emitted += 1
                    if not state.is_refused(kv.kline):
                        yield kv

        if not overfit:
            return
        overfit_set = set(overfit)
        for n in sig_nodes:
            if emitted >= budget:
                return
            for c in self._connotate(n):
                if emitted >= budget:
                    break
                if not set(c.nodes) <= overfit_set:
                    continue
                o_sig = (entry.signature & ~n) | c.signature
                expanded = fit + [m for m in c.nodes if m not in fit]
                target = KLine(o_sig, expanded, entry.dbg)
                if state.is_grounded(target):
                    continue
                for kv in self.propose(
                    target, _depth=_depth - 1, _budget=budget - emitted, _origin=origin
                ):
                    emitted += 1
                    if not state.is_refused(kv.kline):
                        yield kv

    def _target(
        self,
        entry: KLine,
        u: KNode,
        fit: list[KNode],
        budget: int,
        _depth: int,
    ) -> Iterator[list[KNode]]:
        """Node lists the underfit signature ``u`` delivers when completed.

        ``u`` is structured as its own proposal entry over the entry's nodes
        that sit inside it; that entry's completions are the deliveries. A
        grounded identity delivers itself.
        """
        state = self._state
        target = KLine(u, [n for n in fit if self.signifier.bit_in(n, u)], entry.dbg)
        delivered = [
            list(kv.kline.nodes)
            for kv in self.propose(
                target, _depth=_depth - 1, _budget=budget, _origin=entry
            )
        ]
        if not delivered and any(
            is_identity(k) for k in state.find_bucket(u)
        ):
            delivered = [[u]]
        yield from delivered

    def _emit(self, entry: KLine, nodes: list[KNode], origin: KLine) -> Iterator[KValue]:
        """The clean proposal: the entry's signature over its fit nodes."""
        canon = self._state.canon_nodes(entry.signature)
        if canon and any(node not in self._state.ltm for node in canon):
            # An unresolved canon member (an ask still standing) blocks
            # emission: the signature is not yet understood.
            return
        if sorted(nodes) == sorted(origin.nodes):
            # A self-fill reconstructs the entry: nothing new is said.
            return
        if any(k.nodes == nodes for k in self._state.where(lambda k: True)):
            # The node set is already grounded (under another head): saying
            # it back proposes nothing.
            return
        kline = KLine(entry.signature, nodes, entry.dbg)
        if is_terminal(kline) or is_identity(kline):
            # Only misfits are proposed: identities are asks or facts.
            return
        kline = self._align_to_grounded(kline)
        yield KValue(kline, self._grade(entry, kline))

    def _nodes_of(self, signature: KNode) -> list[KNode]:
        """Held nodes whose bit pattern sits inside ``signature``."""
        out: list[KNode] = []
        for kline in self._state.where(
            lambda k: self.signifier.bit_in(k.signature, signature) and is_identity(k)
        ):
            out.append(kline.signature)
        return out

    def _connotate(self, node: KNode) -> list[KLine]:
        """Grounded, non-terminal klines reached from ``node`` by edge hops."""
        out: list[KLine] = []
        for _, sig in self._edge_hops(node):
            for kline in self._state.find_bucket(sig):
                if not is_terminal(kline) and not is_identity(kline):
                    out.append(kline)
        return out

    def _align_to_grounded(self, kline: KLine) -> KLine:
        """Adopt a grounded kline's node order for the proposed nodes.

        The proposed multiset is the answer, but its order is an artifact of
        slot accounting. If the nodes' generated signature is already
        grounded, the grounded kline's order is the phrasing K knows — use
        it. This bypasses linguistic post-processing.
        """
        gen = self._state.signifier.signature_of(kline.nodes)
        for grounded in self._state.ltm.get(gen, []):
            if is_identity(grounded):
                continue
            if (
                grounded.nodes != kline.nodes
                and set(grounded.nodes) == set(kline.nodes)
            ):
                # A permutation of the proposed multiset: adopt its order.
                return KLine(kline.signature, list(grounded.nodes), kline.dbg)
            break
        return kline

    def _grade(self, entry: KLine, kline: KLine) -> int:
        """Post-hoc significance of a proposal for ``entry``: K's understanding
        of the proposal, per proposal node.

        1. The proposal itself is grounded (its exact shape is ratified) — S1.
        2. A node of the entry's canon — S2 (hop 1).
        3. A node crossover-reachable from a canon node in either direction —
           S3 at the crossover hops.
        4. Otherwise — S4: work assigned but unaccounted (no credit).
        """
        if self._state.is_grounded(kline):
            return SIG8_MAX
        canon = self._state.canon_nodes(entry.signature) or []
        aggregator = PROPOSAL_AGGREGATOR
        canon_set = set(canon)
        entities: list[dict[KNode, int]] = []
        for c in canon:
            entities.append(self._chain(c))
        for sub in self._state.where(
            lambda k: is_canon(k, self._state.signifier)
        ):
            sub_set = set(sub.nodes)
            if sub_set and sub_set < canon_set:
                chain = self._chain(sub.signature)
                for n in sub.nodes:
                    chain.setdefault(n, 0)
                entities.append(chain)
        slots: list[float] = []
        for n in kline.nodes:
            if n in canon:
                slots.append(1.0)
                continue
            nchain = self._chain(n, containment=True)
            hops = min(
                (
                    h + e_h
                    for e in entities
                    for sig, e_h in e.items()
                    if (h := nchain.get(sig)) is not None
                ),
                default=None,
            )
            slots.append(aggregator.decay(hops) if hops is not None else 0.0)
        return aggregator.compose_terminal(slots)

    def _chain(
        self, sig: KNode, containment: bool = False
    ) -> dict[KNode, int]:
        """``sig -> hops`` over the edge-hop chain from ``sig`` (self at 0).

        Beyond the bucket edges, a canon's word has a containment edge into
        it (hop 1): a word connotes the group the script defines it in.
        Containment is one-directional credit: a proposal node may reach
        through a canon it belongs to, but the canon-side entities must
        connotate by their own edges only — otherwise any two words sharing
        a canon credit each other (co-occurrence, not connotation).
        """
        state = self._state
        signifier = self._state.signifier
        chain: dict[KNode, int] = {sig: 0}
        frontier: list[KNode] = []
        if containment:
            for kline in state.where(
                lambda k: sig in k.nodes and is_canon(k, signifier)
            ):
                reached = signifier.signature_of(kline.nodes)
                if reached != sig and reached not in chain:
                    chain[reached] = 1
                    frontier.append(reached)
        for hops, reached in self._edge_hops(sig):
            if reached not in chain or hops < chain[reached]:
                chain[reached] = hops
        hop = 1
        seen = set(frontier) | {sig}
        while frontier:
            hop += 1
            nxt = []
            for cur in frontier:
                for h2, reached in self._edge_hops(cur):
                    total = hop + h2 - 1
                    if reached not in chain or total < chain[reached]:
                        chain[reached] = total
                    if reached not in seen:
                        seen.add(reached)
                        nxt.append(reached)
            frontier = nxt
        return chain

    def _edge_hops(
        self, sig: KNode
    ) -> Iterator[tuple[int, KNode]]:
        """Yield ``(hops, sig)`` breadth-first over every non-terminal,
        non-identity resolution edge.

        BFS order is min-hops-first, so consumers halt at their k nearest
        results and never explore past them.
        """
        state = self._state
        signifier = self._state.signifier
        frontier: list[KNode] = [sig]
        visited: set[KNode] = {sig}
        hop_count = 0
        while frontier and hop_count < _MAX_HOP:
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
