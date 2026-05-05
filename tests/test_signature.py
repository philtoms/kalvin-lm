"""Tests for Signature — openspec/signature.md conformance."""

import pytest
from kalvin.signature import make_signature, signifies, is_literal_node, LITERAL_MASK

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
