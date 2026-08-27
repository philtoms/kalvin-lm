"""The S2 strategy: propose true fills for a pending misfit.

Nothing is invented — every node in a proposal comes from the entry or
from grounded klines. Three proposal arms, in emission order:

- **Pivots**: a grounded canon sharing a node with the entry's canon is a
  pivot; the entry's canon is aligned onto it and the pivot's word form is
  grafted on.
- **Fills** (underfit only): connotations are gathered from the entry's
  nodes and its underfit gap's covering bridges; grounded canons whose
  chains cross a connotation contribute the fills, graded by crossover
  distance. An overfit's excess nodes are swapped out.
- **Reentry**: a proposal that is itself an ungrounded misfit is proposed
  from again, one hop further out.

All arms are breadth-first, so discovery order is significance order, and
halt at a per-call proposal budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    KNode,
    classify_misfit,
    is_canon,
    is_identity,
    is_misfit,
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

__all__ = ["PivotFill"]

# Upper bound on edge hop chain depth (_edge_hops's traversal bound).
_MAX_HOP = 100

# The BPE token ID occupies the lower 32 bits of a node value; distance
# calculations match word identity on this half (see NLPSignifier's packing).
_BPE_MASK = 0xFFFF_FFFF


class PivotFill:
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

    #: Per-call proposal budget: at most this many proposals per misfit;
    #: reentry proposals share the budget.
    BUDGET = 3

    def propose(self, entry: KLine) -> Iterator[KValue]:
        for candidate in self._candidates(entry):
            yield from self._expand(entry, candidate, _visited=set())
        return

    def _expand(
        self,
        entry: KLine,
        candidate: KLine,
        _visited: set[tuple[int, KNode]],
        _budget: int | None = None,
    ) -> Iterator[KValue]:
        """Yield proposals for ``entry`` breadth-first, halting at the budget.

        Nearer proposals (fewer hops) are discovered first, so discovery
        order is significance order — no global sort. Reentry proposals share
        the budget and are explored lazily as it remains.
        """
        key = (entry.signature, candidate.signature)
        if key in _visited:
            return  # cycle detected
        _visited.add(key)

        budget = self.BUDGET if _budget is None else _budget
        if budget <= 0:
            return
        
        signifier = self._state.signifier

        e_set = set(entry.nodes)
        c_set = set(candidate.nodes)
        underfit = e_set - c_set
        overfit = c_set - e_set
        fit = e_set & c_set

        s2_target: list[KNode] = list(fit)

        underfit, overfit = classify_misfit(entry, signifier)
        if not underfit and not overfit:
            # A gapless canon is not a misfit. The one exception: an asked
            # signature whose canon overlaps grounded knowledge — pivot
            # alignment aligns the overlap and turns the uncovered slots
            # into asks. Only that arm can handle it.
            if entry.signature not in self._state.asked or not is_canon(
                entry, signifier
            ):
                return
        nodes_sig = signifier.signature_of(entry.nodes)
        gap = signifier.residual(entry.signature, nodes_sig)
        excess = signifier.residual(nodes_sig, entry.signature)

        emitted: int = 0
        for kv in self._pivots(entry):
            if emitted >= budget:
                return
            if self._state.is_refused(kv.kline):
                continue
            emitted += 1
            yield kv
        for kv in self._fills(entry, underfit, overfit, gap, excess):
            if emitted >= budget:
                return
            if self._state.is_refused(kv.kline):
                continue
            emitted += 1
            yield kv
        if _depth > 0:
            # Reentry: each proposal's own nodes widen the connotation set;
            # propose from them to reach fills one hop further out.
            for target in self._reentry_targets(entry):
                if emitted >= budget:
                    return
                for sub in self._expand(
                    target, _depth=_depth - 1, _budget=budget - emitted
                ):
                    emitted += 1
                    yield sub

    def _reentry_targets(self, entry: KLine) -> list[KLine]:
        """Misfit proposals from this entry worth proposing from again."""
        signifier = self._state.signifier
        out: list[KLine] = []
        for kv in [*self._fills_through(entry), *self._pivot_proposals(entry)]:
            best = kv.kline
            if (
                is_misfit(best, signifier)
                and not self._state.is_grounded(best)
                and best not in out
            ):
                out.append(best)
        return out

    def _fills_through(self, entry: KLine) -> list[KValue]:
        """Every constructible fill proposal, ungated — the reentry targets."""
        signifier = self._state.signifier
        underfit, overfit = classify_misfit(entry, signifier)
        out: list[KValue] = []
        if underfit:
            conns = self._crossover_connotations(entry)
            base_nodes = list(entry.nodes)
            for sig in self._crossing_fills(entry, conns):
                expanded = base_nodes + [sig]
                out.append(KValue(KLine(entry.signature, expanded, entry.dbg), 0))
        out.extend(self._pivot_proposals(entry))
        return out

    def _fills(
        self,
        entry: KLine,
        underfit: bool,
        overfit: bool,
        gap: KNode,
        excess: KNode,
    ) -> Iterator[KValue]:
        """Yield gated, graded fill proposals, nearest first."""
        signifier = self._state.signifier
        if not underfit:
            return
        conns = self._crossover_connotations(entry)
        if not conns:
            return
        base_nodes = [
            n for n in entry.nodes if not (overfit and signifier.signifies(n, excess))
        ]
        fills = self._crossing_fills(entry, conns)
        # Nearest fills first: discovery order is significance order.
        for sig, _ in sorted(fills.items(), key=lambda item: item[1]):
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
            if is_terminal(kline) or is_identity(kline) or is_canon(kline, signifier):
                # Only misfits are proposed: identities are asks or facts,
                # canons are the script/compiler's own ground truth.
                continue
            kline = self._align_to_grounded(kline)
            if signifier.residual(
                entry.signature, signifier.signature_of(expanded)
            ) != 0:
                # Unaccounted work: the gap could not be filled.
                continue
            yield KValue(kline, self._grade(entry, kline))

    def _pivots(self, entry: KLine) -> Iterator[KValue]:
        for kv in self._pivot_proposals(entry):
            if self._state.is_refused(kv.kline):
                continue
            yield kv

    def _align_to_grounded(self, kline: KLine) -> KLine:
        """Adopt a grounded kline's node order for the proposed nodes.

        The proposed multiset is the misfit's answer, but its order is an
        artifact of slot accounting. If the nodes' generated signature is
        already grounded, the grounded kline's order is the phrasing K
        knows — use it. This bypasses linguistic post-processing.
        """
        gen = self._state.signifier.signature_of(kline.nodes)
        for grounded in self._state.ltm.get(gen, []):
            if is_identity(grounded):
                continue
            if grounded.nodes != kline.nodes:
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
        # Canon-side entities: each canon node, plus grounded sub-canon
        # groups (canon nodes resolving as a unit through their signature).
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
        it (hop 1): a word connotes the group the script defines it in. This
        is what links a gap fill to the canon node it answers — e.g. WDMH's
        ``what`` connotes Object = Query = ALL, whose canon carries
        ``a, little, lamb``: the fills reach ``what``'s chain through it.

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
        # Propagate through containment-seeded entries (the BFS above only
        # runs from ``sig`` itself).
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
            resolved: list[KNode | None] = []
            gaps: list[KNode] = []
            # Grouped resolution first: canon nodes forming a grounded
            # sub-canon resolve as a unit through its signature's path.
            grouped: set[KNode] = set()
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
                    resolved.append(None)
            # Fill the gap slots with the pivot's unassigned nodes (S4 fill:
            # work is assigned, if only by adjacency). A lone gap takes the
            # whole leftover residual as a grouped fill (appended); multiple
            # gaps take positional fills — each gap the leftover at its
            # relative position in canon/pivot order, which is alignment,
            # not guessing.
            leftovers = [n for n in pnodes if n not in resolved]
            if len(gaps) > 1 and len(leftovers) != len(gaps):
                # Positional fills need a peer: gap count must match
                # leftover count, or the pivot is not shaped like this ask.
                continue
            if gaps:
                if len(gaps) == 1:
                    # The lone gap takes the whole leftover residual as a
                    # grouped fill (appended).
                    resolved = [r for r in resolved if r is not None]
                    resolved.extend(leftovers)
                else:
                    # Gaps are canon-ordered, leftovers pivot-ordered: pair
                    # them positionally.
                    it = iter(leftovers)
                    resolved = [
                        r if r is not None else next(it) for r in resolved
                    ]
            nodes: list[KNode] = []
            for rn in resolved:
                if rn is not None and rn not in nodes:
                    nodes.append(rn)
            if sorted(nodes) == sorted(entry.nodes):
                continue
            kline = KLine(entry.signature, nodes, entry.dbg)
            if is_terminal(kline) or is_identity(kline) or is_canon(kline, signifier):
                # Only misfits are proposed: identities are asks or facts,
                # canons are the script/compiler's own ground truth.
                continue
            kline = self._align_to_grounded(kline)
            if not signifier.signifies(
                signifier.signature_of(nodes), entry.signature
            ):
                continue
            out.append(KValue(kline, self._grade(entry, kline)))
        return out

    def _crossover_connotations(self, entry: KLine) -> dict[KNode, int]:
        """``sig -> min hops`` over edge-hop chains from the entry's nodes and
        from its underfit gap's covering bridges."""
        signifier = self._state.signifier
        conns: dict[KNode, int] = {}
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
        self, entry: KLine, conns: dict[KNode, int]
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
        self, entry: KLine, conns: dict[KNode, int]
    ) -> dict[KNode, int]:
        """``fill sig -> total hops`` for candidate values crossing a connotation.

        A candidate's signature or node is a fill when it reaches a connotation
        through its own edge-hop chain; the distance is the connotation's hops
        plus the crossing hops.
        """
        fills: dict[KNode, int] = {}
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
