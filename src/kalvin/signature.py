"""Signature — OR-reduction of nodes with bit-0 literal-content flag.

See specs/signature.md for the full specification.

Key invariants:
  - Non-literal nodes contribute their full value via OR.
  - Literal nodes contribute bit 0 only (literal-content flag = 1).
  - An all-literal kline produces signature 1.
  - An empty kline produces signature 0.
  - Any uint64 may serve as a signature; there is no is_signature test.
"""

from __future__ import annotations

from typing import Callable, Sequence


def make_signature(
    nodes: Sequence[int],
    is_literal_fn: Callable[[int], bool],
) -> int:
    """Produce a signature from a sequence of nodes.

    Non-literal nodes contribute their full value via OR-reduction.
    Literal nodes contribute bit 0 only (the literal-content flag).

    Args:
        nodes: Sequence of uint64 node values.
        is_literal_fn: Tokenizer function to test whether a node is literal.

    Returns:
        uint64 signature value.
    """
    sig = 0
    for node in nodes:
        if is_literal_fn(node):
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
