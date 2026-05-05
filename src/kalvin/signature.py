"""Signature — OR-reduction of nodes with bit-0 literal-content flag.

See specs/signature.md for the full specification.

Key invariants:
  - Non-literal nodes contribute their full value via OR.
  - Literal nodes contribute bit 0 only (literal-content flag = 1).
  - An all-literal kline produces signature 1.
  - An empty kline produces signature 0.
  - Any uint64 may serve as a signature; there is no is_signature test.

Literal testing is a Kalvin-level concern, not a tokenizer concern.
The literal mask (lower 32 bits all set) is a standardized bit pattern
that all tokenizers must respect. Kalvin tests for it directly.
"""

from __future__ import annotations

from typing import Sequence

# ── Literal mask ──────────────────────────────────────────────────────────
# Lower 32 bits all set — unambiguous discriminator for literal nodes.
# This is a Kalvin convention: any uint64 whose lower 32 bits are all 1s
# is a literal node, regardless of which tokenizer produced it.
LITERAL_MASK = 0xFFFF_FFFF


def is_literal_node(node: int) -> bool:
    """Test whether a node carries the literal mask.

    A literal node has its lower 32 bits all set (0xFFFFFFFF), with the
    character code point stored in the upper 32 bits. This is a standardized
    Kalvin bit pattern — the test does not depend on the tokenizer.

    BPE tokens never produce this pattern (their vocab IDs are small integers
    combined with type prefixes). Mod tokens produce it only when encoding
    individual characters as literal nodes.
    """
    return (node & LITERAL_MASK) == LITERAL_MASK


def make_signature(nodes: Sequence[int]) -> int:
    """Produce a signature from a sequence of nodes.

    Non-literal nodes contribute their full value via OR-reduction.
    Literal nodes contribute bit 0 only (the literal-content flag).

    Args:
        nodes: Sequence of uint64 node values.

    Returns:
        uint64 signature value.
    """
    sig = 0
    for node in nodes:
        if is_literal_node(node):
            sig |= 1          # bit 0: literal-content flag
        else:
            sig |= node       # non-literal contributes full value
    return sig


def signifies(a: int, b: int) -> bool:
    """Test whether two signatures overlap (bitwise AND ≠ 0).

    This is the basis for candidate retrieval in the rationalisation
    pipeline: two signatures that share at least one set bit are
    considered potentially significant.
    """
    return (a & b) != 0
