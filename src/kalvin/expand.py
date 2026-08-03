"""Expand — graph expansion and expansion-proposal pipeline.

This module owns the *graph* layer: traversing the model to expand a
query|candidate pair into connotations and a terminal significance byte
(``expand``), promoting structurally-participating klines after ratification
(``promote_participating``), and generating expansion proposals for misfit
candidates (``propose_expansions`` plus the misfit-comprehension helpers).

It builds on the significance topology layer (``kalvin.significance``), which
owns the 8-bit distance algebra, the band layout, the decay/compose seams and
the ``Aggregator``, and the structural-grounding predicates. The dependency
is strictly one-way: expand → significance, never the reverse.

Misfit comprehension (underfit / overfit / dual expansion generation) lives
in the ``generate_expansions`` helpers at the bottom of this module — a kline
is a misfit when its signature and its nodes' signature disagree, and the
helpers reshape what is already there (no invention, no orphan nodes).

The module reads from the Model (storage) but is a separate responsibility:
Model indexes and retrieves; Expand walks the graph and proposes reshapes.

Module-level constants and types:
  MAX_HOP, edge_hops, expand, promote_participating, propose_expansions,
  generate_expansions
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, is_canon, is_terminal, is_unknown
from kalvin.kvalue import KValue
from kalvin.significance import (
    DEFAULT_AGGREGATOR,
    SIG_S4,
    Aggregator,
    is_s1,
)

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model

_log = logging.getLogger(__name__)

# Upper bound on edge hop chain depth (edge_hops's traversal bound).
MAX_HOP = 100


# Helper Functions


def edge_hops(model: Model, sig: int, signifier: KSignifier) -> Iterator[tuple[int, int]]:
    """Yield (hop_count, next_sig) for each non-canonical resolution step.

    Follows: resolve sig → kline → signifier.signature_of(kline.nodes) → repeat.
    Stops at a dead end, an identity kline, a canonical kline, or a cycle.
    """
    hop_count = 0
    visited: set[int] = set()
    while hop_count < MAX_HOP:
        if sig in visited:
            break  # cycle detected
        visited.add(sig)
        kline = model.find(sig)
        if kline is None or is_terminal(kline) or is_canon(kline, signifier):
            break
        hop_count += 1
        sig = signifier.signature_of(kline.nodes)
        yield hop_count, sig

# Structural Grounding re-export
#
# is_s1 / is_countersigned / structural_significance live in kalvin.significance.
# expand() calls is_s1 directly; the others are imported by callers from there.


# Graph Expansion


def expand(
    model: Model,
    query: KLine,
    candidate: KLine,
    signifier: KSignifier,
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
    The final yield is always the terminal KValue for the original
    pair.

    ``aggregator`` bundles the layout (S2_S3_BOUNDARY) and the two pluggable
    seams (DecayFunction, ComposeFunction); defaults to
    :data:`DEFAULT_AGGREGATOR`.
    """
    if aggregator is None:
        aggregator = DEFAULT_AGGREGATOR
    if _visited is None:
        _visited = set()

    key = (query.signature, candidate.signature)
    if key in _visited:
        return  # cycle detected
    _visited.add(key)

    q_set = set(query.nodes)
    c_set = set(candidate.nodes)
    mismatched_q = q_set - c_set
    mismatched_c = c_set - q_set
    matched = q_set & c_set

    s3_connotations: dict[int, int] = {}  # sig -> min hops from any query node

    # Per-node accountedness, in slot order. One float per slot.
    slot_values: list[float] = []

    decay = aggregator.decay

    for n in mismatched_q:
        accounted = 0.0  # unresolvable default (case F)
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_c:
                    # case C: exact opposing match (S2 direct) -> recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    # case D: signifies (S2 loose) -> side-candidate, no recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield KValue(c_kline, sig_byte)
                    break
                elif match_sig not in s3_connotations or hops < s3_connotations[match_sig]:
                    s3_connotations[match_sig] = hops
        slot_values.append(accounted)

    for n in mismatched_c:
        accounted = 0.0
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_q:
                    # case C: exact opposing match (S2 direct) -> recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    # case D: signifies (S2 loose) -> side-candidate, no recurse.
                    accounted = decay(hops)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_byte = aggregator.compose_terminal([decay(hops)])
                        yield KValue(c_kline, sig_byte)
                    break
                elif match_sig in s3_connotations:
                    # case E: S3 connotation bridge -> recurse (no side-candidate).
                    s3_hop = s3_connotations[match_sig] + hops
                    accounted = decay(s3_hop)
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model, q_kline, c_kline, signifier,
                            aggregator=aggregator, _visited=_visited,
                        )
                    break
        slot_values.append(accounted)

    # Matched nodes: grounded -> 1.0; matched-ungrounded -> decay(1).
    for n in matched:
        kl = model.find(n)
        if kl is not None and is_s1(model, kl, signifier):
            slot_values.append(1.0)
        else:
            # Ungrounded match OR not in model: one hop of doubt.
            slot_values.append(decay(1))

    if not slot_values:
        # Both klines node-less: vacuously fully accounted.
        slot_values = [1.0]

    significance = aggregator.compose_terminal(slot_values)
    yield KValue(candidate, significance)


# Promotion Helpers


def promote_participating(model: Model, query: KLine, candidate: KLine, signifier: KSignifier) -> None:
    """Promote klines that structurally participated in a ratification event.

    After S1 ratification between query and candidate, promote:
    1. The query and candidate themselves (always)
    2. Any STM kline whose signature is a node value in the query or
       candidate AND whose nodes are empty (Unknown frame), a single
       non-literal node (countersign/denote pair), or a canonical
       composition (canonization entry).

    Does NOT promote cogitator expansion proposals (multi-node non-
    canonical klines) that merely share signature bits.
    """
    # Signatures of node values participating in query/candidate.
    node_sigs: set[int] = set()
    for n in query.nodes:
        node_sigs.add(n)
    for n in candidate.nodes:
        node_sigs.add(n)
    node_sigs.add(query.signature)
    node_sigs.add(candidate.signature)

    to_promote: list[KLine] = []
    for kl in model.iter_stm():
        if kl.signature not in node_sigs:
            continue
        # Promote structural klines: Unknown frames, single-node entries,
        # or canonical compositions.
        if not kl.nodes:
            to_promote.append(kl)
        elif isinstance(kl.nodes, int):
            to_promote.append(kl)
        elif isinstance(kl.nodes, list) and len(kl.nodes) == 1:
            to_promote.append(kl)
        elif is_canon(kl, signifier):
            to_promote.append(kl)

    _log.info(
        "promote_participating: query=%#x candidate=%#x promoting %d structural + 2",
        query.signature,
        candidate.signature,
        len(to_promote),
    )

    to_promote.extend([query, candidate])

    for kl in to_promote:
        model.add_to_ltm(kl)


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

    underfit, overfit = signifier.classify_misfit(candidate.signature, candidate.nodes)

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
