"""Significance Pipeline — computes significance between query and candidate KLines.

Significance is the inverse of distance: significance = ~distance.

Three-step pipeline: Route → Distance → Invert.

Route (self-contained, no model call):
  - All nodes match (exist in candidate) → distance = 0 → S1
  - Some nodes match → s2_distance → S2
  - No nodes match → s3_distance → S3
  - No candidates → distance = MAX → S4

See specs/significance.md for the full specification.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from kalvin.abstract import KModel
from kalvin.kline import KLine, KSig

MASK64 = 0xFFFF_FFFF_FFFF_FFFF

# D_boundary separates S2 and S3 distance ranges
D_BOUNDARY = 0x8000_0000_0000_0000
D_MAX = 0xFFFF_FFFF_FFFF_FFFF


class SignificanceResult(NamedTuple):
    """Result of significance computation between a query and candidate."""
    significance: int     # ~distance (uint64)
    distance: int         # raw distance
    level: str            # "S1", "S2", "S3", or "S4"
    match_count: int      # number of nodes matched during routing
    total_nodes: int      # total nodes in query


def significance_pipeline(
    query: KLine,
    candidates: list[KLine],
    model: KModel,
) -> list[tuple[Optional[KLine], SignificanceResult]]:
    """Run significance computation for a query against all candidates.

    Handles all significance levels including S4.

    Args:
        query: The query KLine Q.
        candidates: Candidate KLines from the model (possibly empty).
        model: Model providing s2_distance, s3_distance.

    Returns:
        List of (candidate, SignificanceResult) tuples.
        When candidates is empty, returns [(None, S4_result)].
    """
    if not candidates:
        return [(None, no_candidates_result())]

    results: list[tuple[Optional[KLine], SignificanceResult]] = []

    for candidate in candidates:
        result = compute_significance(query, candidate, model)
        results.append((candidate, result))

    return results


def compute_significance(
    query: KLine,
    candidate: KLine,
    model: KModel,
) -> SignificanceResult:
    """Compute significance of a single query-candidate pair.

    Performs per-node S1 testing, routes to appropriate distance
    function, and inverts distance to significance.
    """
    nodes = query.nodes
    total = len(nodes)

    if total == 0:
        # Empty query → S4
        return SignificanceResult(
            significance=0,
            distance=D_MAX,
            level="S4",
            match_count=0,
            total_nodes=0,
        )

    # Step 1: Route — node-membership test (no model call)
    candidate_nodes = set(candidate.nodes)
    match_count = sum(1 for n in nodes if n in candidate_nodes)

    # Step 2: Distance — route to appropriate function
    if match_count == total:
        # All nodes match → S1
        distance = 0
        level = "S1"
    elif match_count > 0:
        # Some match → S2
        raw_distance = model.s2_distance(query, candidate)
        distance = max(1, min(raw_distance, D_BOUNDARY - 1))
        level = "S2"
    else:
        # No match → S3
        raw_distance = model.s3_distance(query, candidate)
        distance = max(D_BOUNDARY, min(raw_distance, D_MAX - 1))
        level = "S3"

    # Step 3: Invert
    significance = (~distance) & MASK64

    return SignificanceResult(
        significance=significance,
        distance=distance,
        level=level,
        match_count=match_count,
        total_nodes=total,
    )


def no_candidates_result() -> SignificanceResult:
    """Return the S4 result for no candidates."""
    return SignificanceResult(
        significance=0,
        distance=D_MAX,
        level="S4",
        match_count=0,
        total_nodes=0,
    )
