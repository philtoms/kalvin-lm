"""Expand — graph expansion, significance computation, classification, and
expansion proposal pipeline.

This module owns the full significance → classification → expansion proposal
pipeline:

  1. **Significance computation** — expand() computes packed distances and
     yields QueryCandidate objects with connotation and terminal significance.
  2. **Boundary constants** — S2_S3_DISTANCE, boundaries(), classify() map
     significance values to S1/S2/S3/S4 bands.
  3. **Expansion proposals** — propose_expansions() classifies misfits and
     generates (proposal, significance) tuples for the caller to dispatch.
  4. **Structural grounding** — is_s1(), is_countersigned() verify S1 status.

The module reads from the Model (storage) but is a separate responsibility:
Model indexes and retrieves; Expand computes how far apart two KLines are.

Re-exported constants and types from the old model module:
  D_MAX, MASK64, MAX_HOP, _S3_BIAS, QueryCandidate

Band-anchored normalization (defined here, not re-exported):
  normalise_significance, S2_TOP, S2_FLOOR, S3_K
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, is_canon, is_identity
from kalvin.misfit import generate_expansions

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier
    from kalvin.model import Model

_log = logging.getLogger(__name__)

# _S3_BIAS — S3 connotation tier bias. Connotation hop counts are biased by
# this before linear distance addition, so S3 distances always exceed S2
# distances. With _S3_BIAS=1, minimum S3 distance = S2_S3_DISTANCE + 1 = 101.
_S3_BIAS = 1


# Public significance constants
D_MAX = 0xFFFF_FFFF_FFFF_FFFF  # maximum distance, also the significance of zero distance
MASK64 = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit mask for bitwise inversion

# Upper bound on edge hop chain depth; also the per-node distance penalty for
# a mismatched node that does not resolve to an exact opposing match
# (fully-unresolved OR S2 "signifies" loose case).
MAX_HOP = 100

# S2|S3 boundary — S2 direct hops stay below this threshold; S3 connotation
# hops start at S2_S3_DISTANCE + 1 = 101.
S2_S3_DISTANCE = 100

# Band-anchored normalization constants. Each band owns a fixed
# sub-range of [0.0, 1.0]; S3 is asymptotic, mapping its unbounded distance
# range injectively into an open interval without clamping.
S2_TOP = 0.99  # S2 anchor at distance 2 (closest S2 is now distance 1, ≈ 0.9950)
S2_FLOOR = 0.50  # S2|S3 boundary (distance 100); S3 asymptote
S3_K = 50  # decay rate (smaller compresses deep S3 faster)


# Significance Boundaries


def boundaries() -> tuple[int, int, int]:
    """Return the three significance boundaries.

    S1|S2 = D_MAX       (only exact S1 — distance 0 — qualifies)
    S2|S3 = ~S2_S3_DISTANCE
    S3|S4 = 0           (only a complete unresolvable is S4)
    """
    s12 = D_MAX
    s23 = (~S2_S3_DISTANCE) & MASK64
    s34 = 0
    return s12, s23, s34


def classify(sig: int, s12: int, s23: int, s34: int) -> str:
    """Classify a significance value against three boundaries.

    Returns "S1", "S2", "S3", or "S4".
    """
    if sig >= s12:
        return "S1"
    elif sig >= s23:
        return "S2"
    elif sig >= s34:
        return "S3"
    else:
        return "S4"


def normalise_significance(raw_sig: int) -> float:
    """Normalise a raw significance value to a band-anchored float in [0.0, 1.0].

    The single source of truth for significance normalization.
    Each band owns a fixed sub-range so S1/S2/S3/S4 are always ordered and
    visible; S3 uses an asymptotic curve so its unbounded distance range
    maps injectively into an open interval without ever being clamped.

    - S1 (distance 0)    -> 1.0
    - S2 (1..100)        -> linear in [0.50, 0.99] (closest S2 is distance 1)
    - S3 (>100)          -> asymptotic 0.50 * S3_K / (S3_K + (distance-100)),
                           never 0.0
    - raw 0 (S4)         -> 0.0
    """
    if raw_sig == 0:
        return 0.0
    distance = (~raw_sig) & MASK64
    if distance == 0:
        return 1.0
    if distance <= S2_S3_DISTANCE:
        return S2_FLOOR + (S2_TOP - S2_FLOOR) * (S2_S3_DISTANCE - distance) / (S2_S3_DISTANCE - 2)
    delta = distance - S2_S3_DISTANCE
    return S2_FLOOR * S3_K / (S3_K + delta)


class QueryCandidate:
    """A single query|candidate pair yielded by graph expansion.

    Replaces the NamedTuple from model.py with a class for forward
    compatibility. Still usable as a tuple: (query, candidate, significance).
    """

    __slots__ = ("query", "candidate", "significance")

    def __init__(self, query: KLine, candidate: KLine, significance: int):
        self.query = query
        self.candidate = candidate
        self.significance = significance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryCandidate):
            return NotImplemented
        return (
            self.query is other.query
            and self.candidate is other.candidate
            and self.significance == other.significance
        )

    def __repr__(self) -> str:
        return f"QueryCandidate(q={self.query!r}, c={self.candidate!r}, sig={self.significance:#x})"

    def __iter__(self):
        return iter((self.query, self.candidate, self.significance))


# Helper Functions


# ``is_canon`` and ``is_identity`` are defined in :mod:`kalvin.kline` (they
# are structural properties of a KLine). Re-exported here for backward
# compatibility; new code should import from ``kalvin.kline``.


def edge_hops(model: Model, sig: int, signifier: KSignifier) -> Iterator[tuple[int, int]]:
    """Yield (hop_count, next_sig) for each non-canonical resolution step.

    Follows: resolve sig → kline → signifier.make_signature(kline.nodes) → repeat.
    Stops at a dead end, an identity kline, a canonical kline, or a cycle.
    """
    hop_count = 0
    visited: set[int] = set()
    while hop_count < MAX_HOP:
        if sig in visited:
            break  # cycle detected
        visited.add(sig)
        kline = model.find(sig)
        if kline is None or is_identity(kline) or is_canon(kline, signifier):
            break
        hop_count += 1
        sig = signifier.make_signature(kline.nodes)
        yield hop_count, sig


def _as_kline(model: Model, node: int) -> KLine:
    """Resolve a node value to a KLine. Raises ValueError if not resolvable."""
    kline = model.find(node)
    if kline is None:
        raise ValueError(f"Node {node:#x} does not resolve to any KLine")
    return kline


# Structural Grounding


def is_s1(model: Model, kline: KLine, signifier: KSignifier) -> bool:
    """Determine if a kline is structurally grounded (S1).

    A kline is S1 if:
    1. Its signature fully describes its nodes (canonical), OR
    2. It is countersigned by another kline in the model.
    """
    if is_canon(kline, signifier):
        return True
    return is_countersigned(model, kline, signifier)


def is_countersigned(model: Model, kline: KLine, signifier: KSignifier) -> bool:
    """Check if kline is countersigned by any kline in the model.

    A kline is countersigned if its nodes_signature exists as a
    countersigning kline with one node — the countersigned kline's signature.

    Query = {Q: [A, B]}
    Countersigner = {AB: [Q]}

    A self-referential kline ``{S: [S]}`` is excluded: its nodes_signature
    is ``S`` and it is itself a one-node kline whose node is ``S``, so it
    would otherwise count as its own countersigner.
    """
    if is_identity(kline):
        return False
    nodes_signature = signifier.make_signature(kline.nodes)
    for countersigner in model.find_all(nodes_signature):
        if len(countersigner.nodes) == 1 and countersigner.nodes[0] == kline.signature:
            return True
    return False


# Graph Expansion


def expand(
    model: Model,
    query: KLine,
    candidate: KLine,
    signifier: KSignifier,
    distance: int = 0,
    *,
    _visited: set[tuple[int, int]] | None = None,
) -> Iterator[QueryCandidate]:
    """Expand a query-candidate pair, yielding connotations and terminal distance.

    For each discovered connotation (S2/S3 indirect relationship), recursively
    yields QueryCandidate items for the connotation pair.  The final yield is
    always the terminal QueryCandidate with the computed distance for the
    original pair.

    Distance is a single integer. S3 connotation hops use linear distance
    ``S2_S3_DISTANCE + hop_count`` to ensure S3 distances moderately
    exceed S2 distances — close enough for temperature to bridge,
    without the quadratic explosion of the previous packing function.
    """
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

    s3_connotations: dict[int, int] = {}  # sig → min hops from any query node

    total_distance = 0

    for n in mismatched_q:
        hop_distance = MAX_HOP
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_c:
                    hop_distance = hops
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model,
                            q_kline,
                            c_kline,
                            signifier,
                            hops,
                            _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_distance = distance + hops
                        significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                        yield QueryCandidate(q_kline, c_kline, significance)
                    break
                elif match_sig not in s3_connotations or hops < s3_connotations[match_sig]:
                    s3_connotations[match_sig] = hops
        total_distance += hop_distance

    for n in mismatched_c:
        hop_distance = MAX_HOP
        q_kline = model.find(n)
        if q_kline is not None:
            for hops, match_sig in edge_hops(model, n, signifier):
                if match_sig in mismatched_q:
                    hop_distance = hops
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model,
                            q_kline,
                            c_kline,
                            signifier,
                            hops,
                            _visited=_visited,
                        )
                    break
                elif signifier.signifies(n, match_sig):
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        sig_distance = distance + hops
                        significance = (~min(sig_distance, D_MAX - 1)) & MASK64
                        yield QueryCandidate(q_kline, c_kline, significance)
                    break
                elif match_sig in s3_connotations:
                    s3_hop = s3_connotations[match_sig] + hops
                    c_kline = model.find(match_sig)
                    if c_kline is not None:
                        yield from expand(
                            model,
                            q_kline,
                            c_kline,
                            signifier,
                            S2_S3_DISTANCE + s3_hop + _S3_BIAS - 1,
                            _visited=_visited,
                        )
                    hop_distance = 0
                    break
        total_distance += hop_distance

    # Matched but ungrounded nodes incur a small S2 penalty.
    for n in matched:
        kl = model.find(n)
        if kl is None or not is_s1(model, kl, signifier):
            total_distance += 1

    total_distance += distance

    significance = (~min(total_distance, D_MAX - 1)) & MASK64
    yield QueryCandidate(query, candidate, significance)


# Promotion Helpers


def promote_participating(model: Model, query: KLine, candidate: KLine, signifier: KSignifier) -> None:
    """Promote klines that structurally participated in a ratification event.

    After S1 ratification between query and candidate, promote:
    1. The query and candidate themselves (always)
    2. Any STM kline whose signature is a node value in the query or
       candidate AND whose nodes are empty (identity frame), a single
       non-literal node (countersign/undersign pair), or a canonical
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
        # Promote structural klines: identity frames, single-node entries,
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
    for handler dispatch — for connotation yields from ``expand()``, this is
    ``qc.query``, not the original WorkItem's query.

    Expansion proposals must carry decomposition information, so identity
    klines (empty nodes or self-referential ``{S: [S]}``) are never emitted
    — neither as the proposal nor as a companion. A single removed node
    produces the companion ``{n: [n]}``, which is identity and is dropped.
    """
    if is_identity(candidate) or is_canon(candidate, signifier):
        return  # identity or canonical — nothing to expand

    underfit, overfit = signifier.classify_misfit(candidate.signature, candidate.nodes)

    if not underfit and not overfit:
        return

    candidate_sig = candidate.signature
    nodes_sig = signifier.make_signature(candidate.nodes)
    underfit_gap = signifier.residual(candidate_sig, nodes_sig)
    overfit_mask = signifier.residual(nodes_sig, candidate_sig)

    for proposal, companions in generate_expansions(model, candidate, underfit_gap, overfit_mask, signifier):
        if is_identity(proposal):
            continue
        yield (proposal, significance)
        for companion in companions:
            if is_identity(companion):
                continue
            yield (companion, significance)
