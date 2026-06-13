"""Tests for Signature — specs/signature.md conformance."""

from kalvin.signature import make_signature, signifies


class TestMakeSignature:
    """make_signature: plain OR-reduction of raw node values."""

    def test_empty_nodes(self):
        assert make_signature([]) == 0

    def test_single_node(self):
        assert make_signature([42]) == 42

    def test_multiple_nodes(self):
        assert make_signature([0b10, 0b100]) == 0b110

    def test_commutative(self):
        a = make_signature([0b10, 0b100])
        b = make_signature([0b100, 0b10])
        assert a == b

    def test_identity(self):
        """make_signature([x]) == x for any single node."""
        assert make_signature([42]) == 42

    def test_well_known_zero(self):
        assert make_signature([]) == 0

    def test_or_reduce(self):
        """BPE-style tokens: full OR."""
        assert make_signature([42, 100]) == 42 | 100

    def test_backward_compat_mod32(self):
        """Mod32 packed nodes still produce full OR-reduction."""
        assert make_signature([0b10, 0b100]) == 0b110


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

    def test_commutative(self):
        assert signifies(0b110, 0b010) == signifies(0b010, 0b110)
