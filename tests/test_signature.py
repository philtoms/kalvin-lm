"""Tests for Signature — specs/signature.md conformance."""

from kalvin.signature import make_signature, signifies


def T(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits of a uint64.

    Typed nodes pack the sig word into the upper 32
    bits and the BPE token ID into the lower 32. signifies() compares
    only the upper (sig-word) half, so test values that must participate in
    significance matching are shifted up with this helper.
    """
    return bits << 32


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

    def test_or_reduction_of_packed_nodes(self):
        """OR-reduction of two packed node values produces their union (SIG-14)."""
        assert make_signature([0b10, 0b100]) == 0b110


class TestSignifies:
    """signifies(a, b) → overlap in the upper (sig-word) 32 bits only."""

    def test_overlapping_type_bits(self):
        assert signifies(T(0b110), T(0b010)) is True

    def test_non_overlapping_type_bits(self):
        assert signifies(T(0b100), T(0b010)) is False

    def test_self(self):
        assert signifies(T(0b110), T(0b110)) is True

    def test_zero_signifies_nothing(self):
        assert signifies(0, 0) is False
        assert signifies(0, T(42)) is False
        assert signifies(T(42), 0) is False

    def test_commutative(self):
        assert signifies(T(0b110), T(0b010)) == signifies(T(0b010), T(0b110))

    def test_bpe_component_masked_off(self):
        """Overlap confined to the lower (BPE) 32 bits does not signify."""
        # Same BPE token ID, no type bits → not significant.
        assert signifies(0b110, 0b010) is False
        assert signifies(0b110, 0b110) is False

    def test_type_overlap_beats_bpe_difference(self):
        """Different BPE IDs but shared type bits still signify."""
        a = T(0b110) | 0b0001  # type 0b110, BPE id 1
        b = T(0b010) | 0b0010  # type 0b010, BPE id 2
        assert signifies(a, b) is True
