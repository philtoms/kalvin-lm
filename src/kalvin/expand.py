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
after ratification is the agent's responsibility (see
:attr:`Rationaliser._promote_participating`).

Module-level constants and types:
  MAX_HOP, edge_hops, expand
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, is_canon, is_terminal
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
