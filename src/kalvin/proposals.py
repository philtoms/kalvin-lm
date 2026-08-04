"""Proposals — expansion-proposal generation for misfit candidates.

This module owns the *misfit-comprehension* layer: classifying a candidate
kline whose signature and nodes' signature disagree, and proposing reshapes
that bring them back into agreement. A kline is a misfit when its
signature (what it promises) and its nodes' signature (what it delivers)
diverge — ``propose_expansions`` is the entry point the cogitator calls.

The proposals never invent and never leave orphans:
  - No invention: every signature used exists in the model.
  - No orphan nodes: removed nodes form a companion kline.

Three misfit shapes are recognised — underfitting (deliver less than
promised → add nodes), overfitting (deliver more than promised → trim
nodes), and dual (both → atomic replacement). The helpers
``generate_expansions``, ``_underfit_expansions``,
``_overfit_expansions``, and ``_dual_expansions`` reshape what is already
there; ``_split_excess`` partitions a kline's nodes by an excess residual.

This module builds on the graph layer (``kalvin.expand``) only insofar as
both share the significance algebra (``kalvin.significance``) and the model
(``kalvin.model``); the dependency on expand itself is one-way and
conceptual: expand walks the graph and yields connotations, proposals
reshapes misfits into self-consistent klines. The significance-model
grounding predicates (``is_terminal``, ``is_canon``, ``classify_misfit``) and the signifier
interface (``residual``, ``signifies``,
``signature_of``) are consumed here as stable seams.

Module-level types:
  propose_expansions, generate_expansions
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, classify_misfit, is_canon, is_terminal

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model


# Expansion Proposal Pipeline


def propose_expansions(
    model: Model,
    candidate: KLine,
    significance: int,
    signifier: KSignifier,
) -> Iterator[tuple[KLine, int]]:
    """Generate expansion proposals for a misfit candidate.

    Classifies the candidate's misfit type (underfitting, overfitting, or both)
    and generates expansion proposals. Each yield is ``(proposal_kline, significance)``.
    Yields nothing if the candidate is canonical or not a misfit.

    The caller is responsible for pairing proposals with the correct query kline
    for handler dispatch — for connotation yields from ``expand()``, the query
    is the WorkItem's inbound query, not the yielded KValue's kline.

    Expansion proposals must carry decomposition information, so terminal
    klines (the empty Unknown, self-referential ``{S: [S]}``, or a
    compound-word) are never emitted
    — neither as the proposal nor as a companion. A single removed node
    produces the companion ``{n: [n]}``, which is a terminal and is dropped.
    """
    if is_terminal(candidate) or is_canon(candidate, signifier):
        return  # terminal or canonical — nothing to expand

    underfit, overfit = classify_misfit(candidate, signifier)

    if not underfit and not overfit:
        return

    candidate_sig = candidate.signature
    nodes_sig = signifier.signature_of(candidate.nodes)
    underfit_gap = signifier.residual(candidate_sig, nodes_sig)
    overfit_mask = signifier.residual(nodes_sig, candidate_sig)

    for proposal, companions in generate_expansions(model, candidate, underfit_gap, overfit_mask, signifier):
        if is_terminal(proposal):
            continue
        yield (proposal, significance)
        for companion in companions:
            if is_terminal(companion):
                continue
            yield (companion, significance)


# Expansion Proposal Helpers (misfit comprehension)
#
# These back propose_expansions(). A kline is a misfit when its signature
# and its nodes' signature disagree; the helpers below classify the misfit
# type (underfitting, overfitting, or both) and generate expansion proposals
# that satisfy:
#   - No invention: every signature used exists in the model.
#   - No orphan nodes: removed nodes form a companion kline.


def generate_expansions(
    model: Model,
    kline: KLine,
    underfit_gap: int,
    overfit_mask: int,
    signifier: KSignifier,
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Generate expansion proposals for a misfit kline.

    Each yield is (proposal_kline, companion_klines) where:
    - proposal_kline is the expanded version of the input
    - companion_klines are klines formed from removed nodes (may be empty)

    Expansion proposals satisfy:
    - No invention: every signature used exists in the model
    - No orphan nodes: removed nodes form a companion kline
    """
    if underfit_gap and overfit_mask:
        yield from _dual_expansions(model, kline, underfit_gap, overfit_mask, signifier)
    else:
        if underfit_gap:
            yield from _underfit_expansions(model, kline, underfit_gap, signifier)

        if overfit_mask:
            yield from _overfit_expansions(kline, overfit_mask, signifier)


def _underfit_expansions(
    model: Model, kline: KLine, gap: int, signifier: KSignifier
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Add nodes whose signatures overlap and thus reduce the gap."""
    contributors = model.where(lambda k: signifier.signifies(k.signature, gap))

    for contributor in contributors:
        expanded_nodes = list(kline.nodes) + list(contributor.nodes)
        expanded_sig = kline.signature
        proposal = KLine(expanded_sig, expanded_nodes, kline.dbg)

        new_nodes_sig = signifier.signature_of(expanded_nodes)
        if signifier.signifies(new_nodes_sig, expanded_sig):
            yield (proposal, [])


def _split_excess(kline: KLine, excess: int, signifier: KSignifier) -> tuple[list[int], list[int]]:
    """Split kline nodes into (excess_nodes, remaining) by excess residual."""
    excess_nodes = [n for n in kline.nodes if signifier.signifies(n, excess)]
    remaining = [n for n in kline.nodes if n not in excess_nodes]
    return excess_nodes, remaining


def _overfit_expansions(kline: KLine, excess: int, signifier: KSignifier) -> Iterator[tuple[KLine, list[KLine]]]:
    """Remove nodes whose bits contribute to the excess."""
    excess_nodes, remaining = _split_excess(kline, excess, signifier)

    if not excess_nodes:
        return
    trimmed = KLine(kline.signature, remaining, kline.dbg)

    companion_sig = signifier.signature_of(excess_nodes)
    companion = KLine(companion_sig, excess_nodes)

    yield (trimmed, [companion])


def _dual_expansions(
    model: Model, kline: KLine, gap: int, excess: int, signifier: KSignifier
) -> Iterator[tuple[KLine, list[KLine]]]:
    """Atomic replacement: swap excess nodes for gap-filling nodes."""
    excess_nodes, remaining = _split_excess(kline, excess, signifier)

    contributors = model.where(lambda k: signifier.signifies(k.signature, gap))

    for contributor in contributors:
        replacement_nodes = remaining + list(contributor.nodes)
        replacement = KLine(kline.signature, replacement_nodes, kline.dbg)

        companion_sig = signifier.signature_of(excess_nodes)
        companion = KLine(companion_sig, excess_nodes)

        yield (replacement, [companion])
