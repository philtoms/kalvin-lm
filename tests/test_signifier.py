"""Tests for Signifier — specs/signifier.md conformance (NLPSignifier)."""

from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def T(bits: int) -> int:
    """Place type-word bits in the upper 32 bits of a uint64.

    NLPSignifier nodes pack the NLP type word into the upper 32 bits and
    the BPE token ID into the lower 32. signifies() compares only the
    upper (type-word) half, so test values that must participate in
    significance matching are shifted up with this helper.
    """
    return bits << 32


class TestMakeSignature:
    """make_signature: bitwise OR-reduction of raw node values."""

    def test_empty_nodes(self):
        assert signifier.make_signature([]) == 0

    def test_single_node(self):
        assert signifier.make_signature([42]) == 42

    def test_multiple_nodes(self):
        assert signifier.make_signature([0b10, 0b100]) == 0b110

    def test_commutative(self):
        a = signifier.make_signature([0b10, 0b100])
        b = signifier.make_signature([0b100, 0b10])
        assert a == b

    def test_identity(self):
        """make_signature([x]) == x for any single node."""
        assert signifier.make_signature([42]) == 42

    def test_well_known_zero(self):
        assert signifier.make_signature([]) == 0

    def test_or_reduce(self):
        """BPE-style tokens: full OR."""
        assert signifier.make_signature([42, 100]) == 42 | 100

    def test_or_reduction_of_packed_nodes(self):
        """OR-reduction of two packed node values produces their union (SIG-14)."""
        assert signifier.make_signature([0b10, 0b100]) == 0b110


class TestSignifies:
    """signifies(a, b) → overlap in the upper (type-word) 32 bits only."""

    def test_overlapping_type_bits(self):
        assert signifier.signifies(T(0b110), T(0b010)) is True

    def test_non_overlapping_type_bits(self):
        assert signifier.signifies(T(0b100), T(0b010)) is False

    def test_self(self):
        assert signifier.signifies(T(0b110), T(0b110)) is True

    def test_zero_signifies_nothing(self):
        assert signifier.signifies(0, 0) is False
        assert signifier.signifies(0, T(42)) is False
        assert signifier.signifies(T(42), 0) is False

    def test_commutative(self):
        assert signifier.signifies(T(0b110), T(0b010)) == signifier.signifies(T(0b010), T(0b110))

    def test_bpe_component_masked_off(self):
        """Overlap confined to the lower (BPE) 32 bits does not signify."""
        # Same BPE token ID, no type bits → not significant.
        assert signifier.signifies(0b110, 0b010) is False
        assert signifier.signifies(0b110, 0b110) is False

    def test_type_overlap_beats_bpe_difference(self):
        """Different BPE-IDs but shared type bits still signify."""
        a = T(0b110) | 0b0001  # type 0b110, BPE id 1
        b = T(0b010) | 0b0010  # type 0b010, BPE id 2
        assert signifier.signifies(a, b) is True


class TestAbstractConformance:
    """NLPSignifier satisfies the KSignifier ABC."""

    def test_is_ksignifier(self):
        from kalvin.abstract import KSignifier

        assert isinstance(signifier, KSignifier)

    def test_abc_is_abstract(self):
        from kalvin.abstract import KSignifier

        # KSignifier cannot be instantiated directly.
        import pytest

        with pytest.raises(TypeError):
            KSignifier()
