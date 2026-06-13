"""Signature — plain OR-reduction of raw node values.

See specs/signature.md for the full specification.

Key invariants:
  - Every node contributes its full value via OR-reduce — no masking,
    no branching, no special cases.
  - An empty kline produces signature 0.
  - Any uint64 may serve as a signature; there is no is_signature test.
"""

from __future__ import annotations

from collections.abc import Sequence


def make_signature(nodes: Sequence[int]) -> int:
    """Produce a signature from a sequence of nodes.

    Plain OR-reduce of raw node values. No masking, no branching, no
    special cases.

    Args:
        nodes: Sequence of uint64 node values.

    Returns:
        uint64 signature value.
    """
    sig = 0
    for node in nodes:
        sig |= node
    return sig


def signifies(a: int, b: int) -> bool:
    """Test whether two signatures overlap (bitwise AND ≠ 0).

    This is the basis for candidate retrieval in the rationalisation
    pipeline: two signatures that share at least one set bit are
    considered potentially significant.
    """
    return (a & b) != 0
