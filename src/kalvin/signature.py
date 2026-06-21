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

# Kalvin nodes pack a type word into the upper 32 bits and a BPE token ID
# into the lower 32 bits. signifies() compares only the upper (type) half —
# the BPE component is masked off so that two klines signify each other
# based on type-word overlap, not token identity. Kalvin does not interpret
# the type word; see the NLP specialisation for one meaning of these bits.
_TYPE_MASK = 0xFFFF_FFFF_0000_0000


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
    """Test whether two signatures overlap in their type-word bits.

    The lower 32 bits (BPE token IDs) are masked off so that only the
    upper 32 bits (the type word) participate. This is the basis for
    candidate retrieval in the rationalisation pipeline: two signatures
    that share at least one set type bit are considered potentially
    significant.
    """
    return (a & b & _TYPE_MASK) != 0
