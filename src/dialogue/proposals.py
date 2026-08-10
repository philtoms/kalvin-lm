"""Expansion proposals for misfit candidates — the dialogue-local version.

A fork of ``kalvin.proposals``, simplified for the lean engine: no generator
semantics (returns a plain list) and no companion klines (each proposal is a
single reshaped kline; excess nodes are dropped, not re-wrapped).

A candidate is a misfit when its signature (what it promises) and the
signature of its nodes (what it delivers) diverge. Three shapes are
recognised, all preserving the candidate's own signature:

- **underfit** — the signature promises more than the nodes deliver → add a
  contributor's nodes.
- **overfit** — the nodes deliver more than the signature captures → trim the
  excess nodes.
- **dual** — both → swap the excess nodes for a contributor's nodes.

No invention: every node added comes from a grounded contributor. A terminal
or canonical candidate yields nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.kline import KLine, classify_misfit, is_canon, is_terminal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["propose_expansions", "SupportsWhere"]


@runtime_checkable
class SupportsWhere(Protocol):
    """The model surface ``propose_expansions`` reads — just ``where``."""

    def where(self, predicate) -> list[KLine]: ...  # type: ignore[no-untyped-def]


def propose_expansions(
    model: SupportsWhere,
    candidate: KLine,
    signifier: KSignifier,
) -> list[KLine]:
    """Reshape a misfit ``candidate`` into self-consistent proposal klines.

    Returns one kline per reshape (no companions). Yields nothing for a
    terminal or canonical candidate, or one whose signature faithfully covers
    its nodes.
    """
    if is_terminal(candidate) or is_canon(candidate, signifier):
        return []

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
