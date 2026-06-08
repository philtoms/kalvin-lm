"""Tests for Signature — openspec/signature.md conformance."""

import pytest
from kalvin.signature import (
    make_signature,
    signifies,
    is_literal_node,
    is_nlp_node,
    LITERAL_MASK,
    NLP_TYPE_MASK,
)

MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def lit(codepoint: int) -> int:
    """Create a literal node value."""
    return (codepoint << 32) | 0xFFFF_FFFF


def packed(value: int) -> int:
    """Create a packed (non-literal) node value."""
    return value


class TestIsLiteralNode:
    """is_literal_node: lower 32 bits all set."""

    def test_literal_node(self):
        assert is_literal_node(lit(65)) is True

    def test_literal_node_zero_codepoint(self):
        assert is_literal_node((0 << 32) | 0xFFFFFFFF) is True

    def test_packed_node(self):
        assert is_literal_node(2) is False

    def test_packed_node_bit0_set(self):
        """Packed node with bit 0 set but not full mask."""
        assert is_literal_node(0b11) is False

    def test_zero(self):
        assert is_literal_node(0) is False

    def test_signature_with_bit0(self):
        """Signature with bit 0 set (literal-content flag) is not literal."""
        assert is_literal_node(0b110 | 1) is False


class TestMakeSignature:
    """make_signature: OR-reduction with bit-0 literal-content flag."""

    def test_empty_nodes(self):
        assert make_signature([]) == 0

    def test_single_non_literal(self):
        assert make_signature([42]) == 42

    def test_multiple_non_literal(self):
        assert make_signature([0b10, 0b100]) == 0b110

    def test_single_literal(self):
        """Literal contributes bit 0 only."""
        assert make_signature([lit(65)]) == 1

    def test_multiple_literals(self):
        """All literals contribute bit 0 — idempotent."""
        assert make_signature([lit(65), lit(66)]) == 1

    def test_literal_and_non_literal(self):
        """Mixed: bit 0 set from literal + full value from non-literal."""
        result = make_signature([lit(65), 0b110])
        assert result == 1 | 0b110  # 0b111 = 7

    def test_commutative(self):
        a = make_signature([lit(65), 0b10])
        b = make_signature([0b10, lit(65)])
        assert a == b

    def test_non_literal_identity(self):
        """make_signature([x]) == x for non-literal x."""
        assert make_signature([42]) == 42

    def test_well_known_zero(self):
        assert make_signature([]) == 0

    def test_well_known_one(self):
        """All-literal kline produces signature 1."""
        assert make_signature([lit(65)]) == 1

    def test_all_non_literal(self):
        """BPE-style tokens (no literals): full OR."""
        assert make_signature([42, 100]) == 42 | 100


class TestSignifies:
    """signifies(a, b) → (a & b) != 0."""

    def test_overlapping_bits(self):
        assert signifies(0b110, 0b010) is True

    def test_non_overlapping_bits(self):
        assert signifies(0b100, 0b010) is False

    def test_self(self):
        assert signifies(0b110, 0b110) is True

    def test_zero_signifies_nothing(self):
        assert signifies(0, 0) is False
        assert signifies(0, 42) is False
        assert signifies(42, 0) is False

    def test_all_literal_matches_all_literal(self):
        assert signifies(1, 1) is True

    def test_all_literal_vs_packed(self):
        """Signature 1 (all-literal) vs packed node 0b110."""
        assert signifies(1, 0b110) is False

    def test_mixed_matches_packed(self):
        """Signature 1|packed contains packed bits → match."""
        assert signifies(1 | 0b110, 0b110) is True

    def test_commutative(self):
        assert signifies(0b110, 0b010) == signifies(0b010, 0b110)


# ── NLP-BPE helpers for tests ────────────────────────────────────────────


def nlp_node(nlp_type: int, bpe_id: int) -> int:
    """Create an NLP-BPE node: (nlp_type32 << 32) | bpe_token_id."""
    return (nlp_type << 32) | bpe_id


class TestIsNlpNode:
    """is_nlp_node: non-literal nodes with non-zero high 32 bits."""

    def test_nlp_bpe_node(self):
        """NLP-BPE node with high bits set → True."""
        node = nlp_node(0x10000, 42)
        assert is_nlp_node(node) is True

    def test_mod32_packed_node(self):
        """Mod32 packed node (bits 1–31 only) → False."""
        assert is_nlp_node(6) is False

    def test_literal_node(self):
        """Literal node has high bits set but is literal → False."""
        node = (65 << 32) | 0xFFFFFFFF
        assert is_nlp_node(node) is False

    def test_zero(self):
        """Zero has no high bits → False."""
        assert is_nlp_node(0) is False


class TestMakeSignatureNLP:
    """make_signature: NLP-BPE nodes contribute only NLP type bits."""

    def test_nlp_same_type_different_bpe(self):
        """Two NLP nodes with same NLP type but different BPE IDs
        produce identical signatures — BPE IDs are masked out."""
        nlp_type = 0x10000
        node_a = nlp_node(nlp_type, 42)
        node_b = nlp_node(nlp_type, 99)
        assert make_signature([node_a]) == make_signature([node_b])

    def test_nlp_different_types_or_reduction(self):
        """NLP nodes with different NLP types — OR-reduction of high bits."""
        node_a = nlp_node(0x10000, 10)
        node_b = nlp_node(0x20000, 20)
        result = make_signature([node_a, node_b])
        expected = (node_a & NLP_TYPE_MASK) | (node_b & NLP_TYPE_MASK)
        assert result == expected

    def test_nlp_single_node_masks_bpe_id(self):
        """Single NLP node: signature has only the NLP type bits, not BPE ID."""
        nlp_type = 0x10000
        bpe_id = 42
        node = nlp_node(nlp_type, bpe_id)
        sig = make_signature([node])
        # Low 32 bits of signature should be 0 (BPE ID masked out)
        assert (sig & 0xFFFFFFFF) == 0
        # High 32 bits should carry the NLP type
        assert (sig >> 32) == nlp_type

    def test_mixed_nlp_and_literal(self):
        """NLP + literal nodes: bit 0 set (from literal) + NLP type bits."""
        nlp = nlp_node(0x10000, 42)
        literal = lit(65)
        result = make_signature([nlp, literal])
        expected = (nlp & NLP_TYPE_MASK) | 1
        assert result == expected

    def test_backward_compat_mod32(self):
        """Mod32 packed nodes still produce full OR-reduction."""
        assert make_signature([0b10, 0b100]) == 0b110

    def test_mixed_nlp_and_mod32(self):
        """NLP nodes contribute high bits, Mod32 contributes low bits."""
        nlp = nlp_node(0x10000, 42)
        mod32 = 0b110
        result = make_signature([nlp, mod32])
        expected = (nlp & NLP_TYPE_MASK) | mod32
        assert result == expected

    def test_nlp_commutative(self):
        """NLP signature computation is order-independent."""
        nlp_a = nlp_node(0x10000, 10)
        nlp_b = nlp_node(0x20000, 20)
        a = make_signature([nlp_a, nlp_b])
        b = make_signature([nlp_b, nlp_a])
        assert a == b

    def test_nlp_identity(self):
        """make_signature([nlp_node]) == nlp_type bits only (BPE ID excluded)."""
        node = nlp_node(0x10000, 42)
        assert make_signature([node]) == (node & NLP_TYPE_MASK)


class TestSignifiesNLP:
    """signifies works correctly with NLP-derived signatures."""

    def test_nlp_same_type_signifies(self):
        """Two signatures from NLP nodes of same type share bits → True."""
        node_a = nlp_node(0x10000, 42)
        node_b = nlp_node(0x10000, 99)
        assert signifies(make_signature([node_a]), make_signature([node_b])) is True

    def test_nlp_different_non_overlapping_types(self):
        """Two NLP types with no shared bits → False."""
        node_a = nlp_node(0x10000, 42)  # bit 16
        node_b = nlp_node(0x20000, 99)  # bit 17
        # OR-reduce each individually — no overlap between 0x10000 and 0x20000
        # But these are high 32-bit values, so:
        sig_a = make_signature([node_a])
        sig_b = make_signature([node_b])
        # 0x10000 << 32 vs 0x20000 << 32 — no overlap
        assert signifies(sig_a, sig_b) is False

    def test_nlp_overlapping_types(self):
        """Two NLP nodes with overlapping type bits → True."""
        node_a = nlp_node(0x30000, 42)  # bits 16+17
        node_b = nlp_node(0x10000, 99)  # bit 16
        assert signifies(make_signature([node_a]), make_signature([node_b])) is True

    def test_nlp_signature_vs_zero(self):
        """NLP-derived signature vs 0 → False."""
        node = nlp_node(0x10000, 42)
        assert signifies(make_signature([node]), 0) is False


# ── Backward compatibility integration tests ──────────────────────────


class TestSignatureNLPBackwardCompat:
    """Integration tests verifying NLP masking doesn't break Mod32 semantics.

    These are cross-module regression guards: Mod32 signatures must remain
    unchanged after the NLP-aware make_signature() was introduced.
    NLP and Mod32 node types must coexist correctly in mixed sequences.

    These tests complement the unit-level tests in TestMakeSignatureNLP
    by using real tokenizer instances and verifying cross-module behavior.
    """

    def test_mod32_signature_unchanged(self) -> None:
        """Mod32 packed nodes produce full-value OR-reduction — no masking.

        Regression guard: make_signature() on Mod32 nodes must still
        OR-reduce the full node value, not apply NLP masking.
        """
        from kalvin.mod_tokenizer import Mod32Tokenizer

        tok = Mod32Tokenizer()
        nodes = tok.encode("ABC")
        sig = make_signature(nodes)

        # Mod32 packed: single node with all bits set for A, B, C
        # Signature must equal the full node value (no masking)
        assert sig == nodes[0], (
            f"Mod32 signature should be {nodes[0]}, got {sig} — "
            f"NLP masking incorrectly applied to Mod32 nodes"
        )

    def test_nlp_signature_correct_masking(self) -> None:
        """NLP-BPE nodes: signature has only NLP type bits, BPE IDs excluded.

        Two NLP nodes for 'Tea' (nlp_type32=131200) and 'brewed'
        (nlp_type32=8421376).  The signature should be the OR of their
        high 32 NLP type bits only: (131200 | 8421376) << 32.
        """
        from kalvin.nlp_tokenizer import NLPTokenizer

        nlp = NLPTokenizer.from_files()
        tea = nlp.encode("Tea")
        brewed = nlp.encode("brewed")

        sig = make_signature(tea + brewed)

        # Expected: OR of high 32 NLP type bits only
        tea_type = tea[0] >> 32
        brewed_type = brewed[0] >> 32
        expected = (tea_type | brewed_type) << 32

        assert sig == expected, (
            f"Expected {expected:#x}, got {sig:#x} — "
            f"BPE IDs should be excluded from NLP signature"
        )

        # Low 32 bits must be zero (BPE IDs masked out)
        assert (sig & 0xFFFFFFFF) == 0, "BPE IDs leaked into signature"

    def test_signifies_cross_type(self) -> None:
        """Mod32 and NLP signatures don't falsely overlap.

        Mod32 bits are in positions 1–31, NLP type bits are in
        positions 32+.  Unless there's a coincidental bit overlap
        (which doesn't happen with our test data), signifies()
        should return False.
        """
        from kalvin.mod_tokenizer import Mod32Tokenizer
        from kalvin.nlp_tokenizer import NLPTokenizer

        mod32 = Mod32Tokenizer()
        nlp = NLPTokenizer.from_files()

        mod32_sig = make_signature(mod32.encode("ABC"))
        nlp_sig = make_signature(nlp.encode("Tea"))

        # Mod32 sig has bits only in positions 1–31
        # NLP sig has bits only in positions 32+
        # No overlap → signifies should be False
        assert signifies(mod32_sig, nlp_sig) is False, (
            f"Mod32 ({mod32_sig:#x}) and NLP ({nlp_sig:#x}) signatures "
            f"should not overlap"
        )

    def test_mixed_mod32_nlp_signature(self) -> None:
        """Mixed Mod32 + NLP node list: both types contribute correctly.

        A node list containing both Mod32 packed nodes and NLP-BPE nodes
        should produce a signature with Mod32 bits in the low positions
        and NLP type bits in the high positions.
        """
        from kalvin.mod_tokenizer import Mod32Tokenizer
        from kalvin.nlp_tokenizer import NLPTokenizer

        mod32 = Mod32Tokenizer()
        nlp = NLPTokenizer.from_files()

        mod32_nodes = mod32.encode("A")  # Single packed node
        nlp_nodes = nlp.encode("Tea")    # Single NLP-BPE node

        mixed = mod32_nodes + nlp_nodes
        sig = make_signature(mixed)

        # Mod32 contribution: full value (low bits)
        mod32_sig = make_signature(mod32_nodes)
        assert sig & mod32_sig == mod32_sig, (
            "Mod32 bits should be present in mixed signature"
        )

        # NLP contribution: type bits only (high bits)
        nlp_sig = make_signature(nlp_nodes)
        assert sig & nlp_sig == nlp_sig, (
            "NLP type bits should be present in mixed signature"
        )

        # Combined: OR of both
        assert sig == (mod32_sig | nlp_sig)
