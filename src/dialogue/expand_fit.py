"""The expand S2 strategy — emit the single most significant proposal.

For a pending misfit ``entry``, every grounded candidate sharing a node value
with it is graded by :meth:`ExpandFit._expand` (a self-contained port of
``kalvin.expand.expand`` over the engine's :class:`EngineState`). Each yield
carries a real significance byte (a graded distance, not a band). The strategy
keeps the one yield with the highest byte — the most significant — reshapes
it, and emits that single proposal, stamped with the yield's actual byte.

Significance — not banding — is the selection criterion. S1 (0xFF) and S4
(0x00) are not gated: they are positions in the cascade. An S1-graded yield is
simply the highest byte and wins; an S4-graded yield is the lowest and loses.
No proposal is invented: every node in the reshape comes from a grounded
contributor.

The reshape recognises three shapes, all preserving the candidate's own
signature:

- **underfit** — the signature promises more than the nodes deliver → add a
  contributor's nodes.
- **overfit** — the nodes deliver more than the signature captures → trim the
  excess nodes.
- **dual** — both → swap the excess nodes for a contributor's nodes.

No invention: every node added comes from a grounded contributor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    classify_misfit,
    is_canon,
    is_identity,
    is_terminal,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    DEFAULT_AGGREGATOR,
    SIG_MASK,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

    from kalvin.abstract import KSignifier

__all__ = ["ExpandFit"]

# Upper bound on edge hop chain depth (_edge_hops's traversal bound).
_MAX_HOP = 100


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

    def propose_gap(
        self,
        candidate: KLine,
    ) -> list[KValue]:
        """Expand a misfit candidate by underfit, overfit or otherwise bad-fit expansion strategies"""
        if len(candidate.nodes) < 2:
            return []
        signifier = self._state.signifier
        underfit, overfit = classify_misfit(candidate, signifier)
        if not underfit and not overfit:
            return []

        candidate_sig = candidate.signature
        nodes_sig = signifier.signature_of(candidate.nodes)
        underfit_gap = signifier.residual(candidate_sig, nodes_sig)
        overfit_mask = signifier.residual(nodes_sig, candidate_sig)

        if underfit_gap and overfit_mask:
            proposals = self._expand_badfit(candidate, underfit_gap, overfit_mask)
        elif underfit_gap:
            proposals = self._expand_underfit(candidate, underfit_gap)
        else:
            proposals = self._expand_overfit(candidate, overfit_mask)
        graded = [self._grade(candidate, p) for p in proposals]
        graded.sort(key=lambda kv: kv.significance & SIG_MASK, reverse=True)
        return graded

    def _grade(self, entry: KLine, proposal: KLine) -> KValue:
        """Grade ``proposal`` against ``entry`` via the aggregator's terminal byte."""
        byte = list(self._expand(entry, proposal))[-1].significance
        return KValue(proposal, DEFAULT_AGGREGATOR.compose_terminal([byte & SIG_MASK]))


    def propose(
        self,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        state = self._state
        graded: list[KValue] = []
        for candidate in state.similar_fit_candidates(entry):
            graded.extend(self._expand(entry, candidate))
        if not graded:
            return []
        best = max(graded, key=lambda kv: kv.significance & SIG_MASK)
        proposals = self._reshape(best.kline)
        if not proposals:
            return []
        chosen = self._most_significant(entry, proposals)
        return [KValue(chosen.kline, chosen.significance & SIG_MASK)]

    def _reshape(
        self, candidate: KLine
    ) -> list[KLine]:
        """Reshape a misfit ``candidate`` into self-consistent proposal klines.

        Returns one kline per reshape (no companions). Yields nothing for a
        candidate whose signature faithfully covers its nodes (terminals and
        canons included).
        """
        signifier = self._state.signifier
        underfit, overfit = classify_misfit(candidate, signifier)
        if not underfit and not overfit:
            return []

        candidate_sig = candidate.signature
        nodes_sig = signifier.signature_of(candidate.nodes)
        underfit_gap = signifier.residual(candidate_sig, nodes_sig)
        overfit_mask = signifier.residual(nodes_sig, candidate_sig)

        if underfit_gap and overfit_mask:
            proposals = self._badfit(candidate, underfit_gap, overfit_mask)
        elif underfit_gap:
            proposals = self._underfit(candidate, underfit_gap)
        else:
            proposals = self._overfit(candidate, overfit_mask)

        return [p for p in proposals if not is_terminal(p)]

    def _most_significant(
        self, entry: KLine, proposals: list[KLine]
    ) -> KValue:
        """The reshape that grades highest when re-expanded against ``entry``.

        ``_expand``'s final yield is the grade for the pair itself; that is the
        byte compared.
        """
        graded = [
            list(self._expand(entry, p))[-1] for p in proposals
        ]
        return max(graded, key=lambda kv: kv.significance & SIG_MASK)

    # ── graph expansion (mirrors kalvin.expand over an EngineState) ──

    def _edge_hops(
        self, sig: int
    ) -> Iterator[tuple[int, int]]:
        """Yield ``(hop_count, next_sig)`` for each non-canonical resolution step.

        Follows: resolve sig → kline → signifier.signature_of(kline.nodes) → repeat.
        Stops at a dead end, an identity kline, a canonical kline, or a cycle.
        """
        state = self._state
        signifier = self._state.signifier
        hop_count = 0
        visited: set[int] = set()
        while hop_count < _MAX_HOP:
            if sig in visited:
                break  # cycle detected
            visited.add(sig)
            kline = state.find(sig)
            if kline is None or is_terminal(kline) or is_canon(kline, signifier):
                break
            hop_count += 1
            sig = signifier.signature_of(kline.nodes)
            yield hop_count, sig

    def _expand(
        self,
        query: KLine,
        candidate: KLine,
        *,
        aggregator: Aggregator | None = None,
        _visited: set[tuple[int, int]] | None = None,
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
        The final yield is always the terminal KValue for the original pair.
        """
        if aggregator is None:
            aggregator = DEFAULT_AGGREGATOR
        if _visited is None:
            _visited = set()

        key = (query.signature, candidate.signature)
        if key in _visited:
            return  # cycle detected
        _visited.add(key)

        state = self._state
        signifier = self._state.signifier
        q_set = set(query.nodes)
        c_set = set(candidate.nodes)
        mismatched_q = q_set - c_set
        mismatched_c = c_set - q_set
        matched = q_set & c_set

        s3_connotations: dict[int, int] = {}  # sig -> min hops from any query node

        slot_values: list[float] = []
        decay = aggregator.decay

        for n in mismatched_q:
            accounted = 0.0  # unresolvable default (case F)
            q_kline = state.find(n)
            if q_kline is not None:
                for hops, match_sig in self._edge_hops(n):
                    if match_sig in mismatched_c:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        c_kline = state.find(match_sig)
                        if c_kline is not None:
                            yield from self._expand(
                                q_kline, c_kline,
                                aggregator=aggregator, _visited=_visited,
                            )
                        break
                    elif signifier.signifies(n, match_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        c_kline = state.find(match_sig)
                        if c_kline is not None:
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
                    if match_sig in mismatched_q:
                        # case C: exact opposing match (S2 direct) -> recurse.
                        accounted = decay(hops)
                        c_kline = state.find(match_sig)
                        if c_kline is not None:
                            yield from self._expand(
                                q_kline, c_kline,
                                aggregator=aggregator, _visited=_visited,
                            )
                        break
                    elif signifier.signifies(n, match_sig):
                        # case D: signifies (S2 loose) -> side-candidate, no recurse.
                        accounted = decay(hops)
                        c_kline = state.find(match_sig)
                        if c_kline is not None:
                            sig_byte = aggregator.compose_terminal([decay(hops)])
                            yield KValue(c_kline, sig_byte)
                        break
                    elif match_sig in s3_connotations:
                        # case E: S3 connotation bridge -> recurse (no side-candidate).
                        s3_hop = s3_connotations[match_sig] + hops
                        accounted = decay(s3_hop)
                        c_kline = state.find(match_sig)
                        if c_kline is not None:
                            yield from self._expand(
                                q_kline, c_kline,
                                aggregator=aggregator, _visited=_visited,
                            )
                        break
            slot_values.append(accounted)

        # Matched nodes: grounded -> 1.0; matched-ungrounded -> decay(1).
        for n in matched:
            kl = state.find(n)
            if kl is not None and state.is_grounded(kl):
                slot_values.append(1.0)
            else:
                # Ungrounded match OR not in the store: one hop of doubt.
                slot_values.append(decay(1))

        if not slot_values:
            # Both klines node-less: vacuously fully accounted.
            slot_values = [1.0]

        significance = aggregator.compose_terminal(slot_values)
        yield KValue(candidate, significance)

    def _expand_underfit(
        self, entry: KLine, gap: int
    ) -> list[KLine]:
        """Fill the gap with the co-denotations of a gap-covering kline's nodes.

        A grounded kline whose signature covers the gap is a connotation bridge
        (``what:[Object]`` bridging a W gap). The grounded klines sharing its
        nodes (``lamb:[Object]``, ``ALL:[Object]``) denote what the gap stands
        for; their signatures fill the gap.
        """
        signifier = self._state.signifier
        proposals: list[KLine] = []
        for bridge in self._state.where(
            lambda k: signifier.signifies(k.signature, gap)
        ):
            bridge_nodes = set(bridge.nodes)
            for denotation in self._state.where(
                lambda k: k is not bridge
                and not is_identity(k)
                and set(k.nodes) & bridge_nodes
            ):
                expanded = list(entry.nodes) + [denotation.signature]
                if signifier.signifies(
                    signifier.signature_of(expanded), entry.signature
                ):
                    proposals.append(KLine(entry.signature, expanded, entry.dbg))
        return proposals

    def _expand_overfit(
        self, entry: KLine, excess: int
    ) -> list[KLine]:
        """Drop the nodes whose bits contribute to the excess."""
        signifier = self._state.signifier
        remaining = [n for n in entry.nodes if not signifier.signifies(n, excess)]
        if remaining == list(entry.nodes):
            return []
        kline = KLine(entry.signature, remaining, entry.dbg)
        if is_terminal(kline):
            return []
        return [kline]

    def _expand_badfit(
        self, entry: KLine, gap: int, excess: int
    ) -> list[KLine]:
        """Swap the excess nodes for gap-covering contributors' nodes."""
        signifier = self._state.signifier
        remaining = [n for n in entry.nodes if not signifier.signifies(n, excess)]
        proposals: list[KLine] = []
        for contributor in self._state.where(
            lambda k: signifier.signifies(k.signature, gap)
        ):
            kline = KLine(
                entry.signature, remaining + list(contributor.nodes), entry.dbg
            )
            if not is_terminal(kline):
                proposals.append(kline)
        return proposals

    def _underfit(
        self, kline: KLine, gap: int
    ) -> list[KLine]:
        """Add a contributor's nodes when they cover the gap."""
        signifier = self._state.signifier
        out: list[KLine] = []
        for contributor in self._state.where(lambda k: signifier.signifies(k.signature, gap)):
            expanded_nodes = list(kline.nodes) + list(contributor.nodes)
            if signifier.signifies(signifier.signature_of(expanded_nodes), kline.signature):
                out.append(KLine(kline.signature, expanded_nodes, kline.dbg))
        return out

    def _overfit(self, kline: KLine, excess: int) -> list[KLine]:
        """Drop the nodes whose bits contribute to the excess."""
        signifier = self._state.signifier
        remaining = [n for n in kline.nodes if not signifier.signifies(n, excess)]
        if remaining == list(kline.nodes):
            return []
        return [KLine(kline.signature, remaining, kline.dbg)]

    def _badfit(
        self, kline: KLine, gap: int, excess: int
    ) -> list[KLine]:
        """Swap the excess nodes for a gap-filling contributor's nodes."""
        signifier = self._state.signifier
        remaining = [n for n in kline.nodes if not signifier.signifies(n, excess)]
        out: list[KLine] = []
        for contributor in self._state.where(lambda k: signifier.signifies(k.signature, gap)):
            out.append(KLine(kline.signature, remaining + list(contributor.nodes), kline.dbg))
        return out
