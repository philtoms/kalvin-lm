"""Signature — OR-reduction of nodes with bit-0 literal-content flag.

See specs/signature.md for the full specification.

Key invariants:
  - Literal nodes contribute bit 0 only (literal-content flag = 1).
  - NLP-BPE nodes contribute only their NLP type bits (high 32), masking
    out BPE token IDs to keep signatures vocabulary-independent.
  - Mod32 packed nodes contribute their full value via OR.
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

# ── NLP type mask ─────────────────────────────────────────────────────────
# High 32 bits — masks the NLP type portion of an NLP-BPE node.
# NLP-BPE nodes use the layout: (nlp_type32 << 32) | bpe_token_id.
# When computing signatures, only the NLP type bits contribute; BPE IDs
# (low 32 bits) are excluded to keep signatures vocabulary-independent.
NLP_TYPE_MASK = 0xFFFF_FFFF_0000_0000

# ── BPE token mask ────────────────────────────────────────────────────────
# Low 32 bits — masks the BPE token ID portion of an NLP-BPE node.
BPE_TOKEN_MASK = 0xFFFF_FFFF


def node_to_sig(node: int) -> int:
    """Convert a node value to its signature-equivalent form.

    Used when a raw node value is used as a model lookup key (e.g.
    ``model.find(node)``). The model indexes klines by their signature,
    so the lookup key must be in signature form.

    Three cases:
      - Literal nodes → 1 (bit 0: literal-content flag).
      - NLP-BPE nodes → NLP type bits only (high 32), BPE ID masked out.
      - Mod32 packed nodes → full value (node == signature).

    Args:
        node: A uint64 node value.

    Returns:
        The signature-equivalent value for model lookup.
    """
    if is_literal_node(node):
        return 1  # bit 0: literal-content flag
    if is_nlp_node(node):
        return node & NLP_TYPE_MASK  # NLP type bits only (high 32)
    return node  # Mod32 packed: full value


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


def is_nlp_node(node: int) -> bool:
    """Test whether a node is an NLP-BPE node.

    An NLP-BPE node has non-zero high 32 bits and is NOT a literal node.
    The high 32 bits carry the NLP type encoding (POS + DEP + MORPH);
    the low 32 bits carry the BPE token ID.

    **Mod32-only assumption**: This test correctly distinguishes NLP-BPE
    nodes from Mod32 packed nodes because Mod32 packed nodes use only
    bits 1–31 (high 32 bits are always zero). However, Mod64 packed nodes
    use bits 1–63 and could set bits in the high 32 range, causing a false
    positive. This is acceptable because:

    1. The current codebase uses Mod32 packed nodes exclusively at runtime.
    2. NLP-BPE nodes will coexist with Mod32 nodes, not Mod64.
    3. If Mod64 is ever used alongside NLP-BPE, this function will need a
       more discriminating test — but that is a future concern outside this
       task's scope.

    **Edge case**: An NLP-BPE node with ``nlp_type32 == 0`` would have
    zero high bits and be indistinguishable from a Mod32 packed node.
    In practice this cannot happen — unknown BPE tokens receive at least
    ``POS_X = 65536`` as their ``nlp_type32``, ensuring the high bits
    are always non-zero for valid NLP nodes.
    """
    if is_literal_node(node):
        return False
    return (node >> 32) != 0


def make_signature(nodes: Sequence[int]) -> int:
    """Produce a signature from a sequence of nodes.

    Uses ``node_to_sig()`` to extract the signature contribution of each node,
    then OR-reduces them into a single uint64.

    Args:
        nodes: Sequence of uint64 node values.

    Returns:
        uint64 signature value.
    """
    sig = 0
    for node in nodes:
        sig |= node_to_sig(node)
    return sig


def signifies(a: int, b: int) -> bool:
    """Test whether two signatures overlap (bitwise AND ≠ 0).

    This is the basis for candidate retrieval in the rationalisation
    pipeline: two signatures that share at least one set bit are
    considered potentially significant.
    """
    return (a & b) != 0
