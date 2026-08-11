"""The expand S2 strategy — emit the single most significant proposal.

For a pending misfit ``entry``, every grounded candidate sharing a node value
with it is graded via ``kalvin.expand.expand``. Each yield carries a real
significance byte (a graded distance, not a band). The strategy keeps the one
yield with the highest byte — the most significant — reshapes it through
:func:`propose_expansions`, and emits that single proposal, stamped with the
yield's actual byte.

Significance — not banding — is the selection criterion. S1 (0xFF) and S4
(0x00) are not gated: they are positions in the cascade. An S1-graded yield is
simply the highest byte and wins; an S4-graded yield is the lowest and loses.
No proposal is invented: every node in the reshape comes from a grounded
contributor.

``propose_expansions`` (below) reshapes a misfit candidate into one kline
per reshape (no companions; excess nodes are dropped, not re-wrapped). A
candidate is a misfit when its signature (what it promises) and the
signature of its nodes (what it delivers) diverge. Three shapes are
recognised, all preserving the candidate's own signature:

- **underfit** — the signature promises more than the nodes deliver → add a
  contributor's nodes.
- **overfit** — the nodes deliver more than the signature captures → trim the
  excess nodes.
- **dual** — both → swap the excess nodes for a contributor's nodes.

No invention: every node added comes from a grounded contributor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dialogue.misfit import GroundedModel, similar_fit_candidates
from kalvin.expand import expand
from kalvin.kline import KLine, classify_misfit, is_terminal
from kalvin.kvalue import KValue
from kalvin.significance import SIG_MASK

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from dialogue.engine import EngineState
    from kalvin.abstract import KSignifier

__all__ = ["ExpandFit", "propose_expansions", "SupportsWhere"]


@runtime_checkable
class SupportsWhere(Protocol):
    """The model surface :func:`propose_expansions` reads — just ``where``."""

    def where(self, predicate) -> list[KLine]: ...  # type: ignore[no-untyped-def]


def propose_expansions(
    model: SupportsWhere,
    candidate: KLine,
    signifier: KSignifier,
) -> list[KLine]:
    """Reshape a misfit ``candidate`` into self-consistent proposal klines.

    Returns one kline per reshape (no companions). Yields nothing for a
    candidate whose signature faithfully covers its nodes (terminals and
    canons included).
    """
    underfit, overfit = classify_misfit(candidate, signifier)
    if not underfit and not overfit:
        return []

    candidate_sig = candidate.signature
    nodes_sig = signifier.signature_of(candidate.nodes)
    underfit_gap = signifier.residual(candidate_sig, nodes_sig)
    overfit_mask = signifier.residual(nodes_sig, candidate_sig)

    if underfit_gap and overfit_mask:
        proposals = _dual(model, candidate, underfit_gap, overfit_mask, signifier)
    elif underfit_gap:
        proposals = _underfit(model, candidate, underfit_gap, signifier)
    else:
        proposals = _overfit(candidate, overfit_mask, signifier)

    return [p for p in proposals if not is_terminal(p)]


def _underfit(
    model: SupportsWhere, kline: KLine, gap: int, signifier: KSignifier
) -> list[KLine]:
    """Add a contributor's nodes when they cover the gap."""
    out: list[KLine] = []
    for contributor in model.where(lambda k: signifier.signifies(k.signature, gap)):
        expanded_nodes = list(kline.nodes) + list(contributor.nodes)
        if signifier.signifies(signifier.signature_of(expanded_nodes), kline.signature):
            out.append(KLine(kline.signature, expanded_nodes, kline.dbg))
    return out


def _overfit(kline: KLine, excess: int, signifier: KSignifier) -> list[KLine]:
    """Drop the nodes whose bits contribute to the excess."""
    remaining = [n for n in kline.nodes if not signifier.signifies(n, excess)]
    if remaining == list(kline.nodes):
        return []
    return [KLine(kline.signature, remaining, kline.dbg)]


def _dual(
    model: SupportsWhere, kline: KLine, gap: int, excess: int, signifier: KSignifier
) -> list[KLine]:
    """Swap the excess nodes for a gap-filling contributor's nodes."""
    remaining = [n for n in kline.nodes if not signifier.signifies(n, excess)]
    out: list[KLine] = []
    for contributor in model.where(lambda k: signifier.signifies(k.signature, gap)):
        out.append(KLine(kline.signature, remaining + list(contributor.nodes), kline.dbg))
    return out


class ExpandFit:
    """The S2 strategy: grade every candidate, emit the most significant proposal."""

    def propose(
        self,
        state: EngineState,
        signifier: KSignifier,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        model = GroundedModel(state)
        graded: list[KValue] = []
        for candidate in similar_fit_candidates(state, signifier, entry):
            graded.extend(expand(model, entry, candidate, signifier))
        if not graded:
            return []
        best = max(graded, key=lambda kv: kv.significance & SIG_MASK)
        proposals = propose_expansions(model, best.kline, signifier)
        if not proposals:
            return []
        chosen = self._most_significant(model, entry, proposals, signifier)
        return [KValue(chosen.kline, chosen.significance & SIG_MASK)]

    @staticmethod
    def _most_significant(
        model: GroundedModel, entry: KLine, proposals: list[KLine], signifier: KSignifier
    ) -> KValue:
        """The reshape that grades highest when re-expanded against ``entry``.

        ``expand``'s final yield is the grade for the pair itself; that is the
        byte compared.
        """
        graded = [list(expand(model, entry, p, signifier))[-1] for p in proposals]
        return max(graded, key=lambda kv: kv.significance & SIG_MASK)
