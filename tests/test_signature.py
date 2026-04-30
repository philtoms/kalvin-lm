"""Tests for Signature — openspec/signature.md conformance."""

import pytest
from kalvin.signature import make_signature, signifies, significance_value, MASK64


# Mock is_literal function for testing
def is_literal_mask(node: int) -> bool:
    """Matches Mod tokenizer: lower 32 bits all set."""
    return (node & 0xFFFF_FFFF) == 0xFFFF_FFFF


def is_literal_always_false(node: int) -> bool:
    return False


def is_literal_always_true(node: int) -> bool:
    return True


def lit(codepoint: int) -> int:
    """Create a literal node value."""
    return (codepoint << 32) | 0xFFFF_FFFF


def packed(value: int) -> int:
    """Create a packed (non-literal) node value."""
    return value


class TestMakeSignature:
    """make_signature: OR-reduction with bit-0 literal-content flag."""

    def test_empty_nodes(self):
        assert make_signature([], is_literal_mask) == 0

    def test_single_non_literal(self):
        assert make_signature([42], is_literal_mask) == 42

    def test_multiple_non_literal(self):
        assert make_signature([0b10, 0b100], is_literal_mask) == 0b110

    def test_single_literal(self):
        """Literal contributes bit 0 only."""
        assert make_signature([lit(65)], is_literal_mask) == 1

    def test_multiple_literals(self):
        """All literals contribute bit 0 — idempotent."""
        assert make_signature([lit(65), lit(66)], is_literal_mask) == 1

    def test_literal_and_non_literal(self):
        """Mixed: bit 0 set from literal + full value from non-literal."""
        result = make_signature([lit(65), 0b110], is_literal_mask)
        assert result == 1 | 0b110  # 0b111 = 7

    def test_commutative(self):
        a = make_signature([lit(65), 0b10], is_literal_mask)
        b = make_signature([0b10, lit(65)], is_literal_mask)
        assert a == b

    def test_non_literal_identity(self):
        """make_signature([x]) == x for non-literal x."""
        assert make_signature([42], is_literal_mask) == 42

    def test_well_known_zero(self):
        assert make_signature([], is_literal_mask) == 0

    def test_well_known_one(self):
        """All-literal kline produces signature 1."""
        assert make_signature([lit(65)], is_literal_mask) == 1

    def test_with_always_false_literal(self):
        """BPE tokenizer: nothing is literal."""
        assert make_signature([42, 100], is_literal_always_false) == 42 | 100

    def test_with_always_true_literal(self):
        """Everything is literal → all contribute bit 0."""
        assert make_signature([42, 100], is_literal_always_true) == 1


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


class TestSignificanceValue:
    """significance = ~distance & MASK64."""

    def test_zero_distance(self):
        assert significance_value(0) == MASK64  # S1

    def test_max_distance(self):
        assert significance_value(MASK64) == 0  # S4

    def test_inversion(self):
        d = 0x0000_0000_0000_0005
        s = significance_value(d)
        assert (~d) & MASK64 == s
