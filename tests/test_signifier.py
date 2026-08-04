"""Tests for Signifier — specs/signifier.md conformance (NLPSignifier)."""

from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def t(bits: int) -> int:
    """Place type-word bits in the upper 32 bits of a uint64.

    NLPSignifier nodes pack the NLP type word into the upper 32 bits and
    the BPE token ID into the lower 32. signifies() compares only the
    upper (type-word) half, so test values that must participate in
    significance matching are shifted up with this helper.
    """
    return bits << 32


class TestMakeSignature:
    """signature_of: bitwise OR-reduction of raw node values."""

    def test_empty_nodes(self):
        assert signifier.signature_of([]) == 0

    def test_single_node(self):
        assert signifier.signature_of([42]) == 42

    def test_multiple_nodes(self):
        assert signifier.signature_of([0b10, 0b100]) == 0b110

    def test_commutative(self):
        a = signifier.signature_of([0b10, 0b100])
        b = signifier.signature_of([0b100, 0b10])
        assert a == b

    def test_identity(self):
        """signature_of([x]) == x for any single node."""
        assert signifier.signature_of([42]) == 42

    def test_well_known_zero(self):
        assert signifier.signature_of([]) == 0

    def test_or_reduce(self):
        """BPE-style tokens: full OR."""
        assert signifier.signature_of([42, 100]) == 42 | 100

    def test_or_reduction_of_packed_nodes(self):
        """OR-reduction of two packed node values produces their union (SIG-14)."""
        assert signifier.signature_of([0b10, 0b100]) == 0b110


class TestSignifies:
    """signifies(a, b) → overlap in the upper (type-word) 32 bits only."""

    def test_overlapping_type_bits(self):
        assert signifier.signifies(t(0b110), t(0b010)) is True

    def test_non_overlapping_type_bits(self):
        assert signifier.signifies(t(0b100), t(0b010)) is False

    def test_self(self):
        assert signifier.signifies(t(0b110), t(0b110)) is True

    def test_zero_signifies_nothing(self):
        assert signifier.signifies(0, 0) is False
        assert signifier.signifies(0, t(42)) is False
        assert signifier.signifies(t(42), 0) is False

    def test_commutative(self):
        assert signifier.signifies(t(0b110), t(0b010)) == signifier.signifies(t(0b010), t(0b110))

    def test_bpe_component_masked_off(self):
        """Overlap confined to the lower (BPE) 32 bits does not signify."""
        # Same BPE token ID, no type bits → not significant.
        assert signifier.signifies(0b110, 0b010) is False
        assert signifier.signifies(0b110, 0b110) is False

    def test_type_overlap_beats_bpe_difference(self):
        """Different BPE-IDs but shared type bits still signify."""
        a = t(0b110) | 0b0001  # type 0b110, BPE id 1
        b = t(0b010) | 0b0010  # type 0b010, BPE id 2
        assert signifier.signifies(a, b) is True


class TestResidual:
    """residual: masked type-word set-difference."""

    def test_type_word_bits_in_a_not_in_b(self):
        """SIG-17: residual(t(0b110), t(0b010)) == t(0b100)."""
        assert signifier.residual(t(0b110), t(0b010)) == t(0b100)

    def test_empty_residual_for_equal_inputs(self):
        """SIG-18: residual(a, a) == 0 for any a."""
        assert signifier.residual(t(0b110), t(0b110)) == 0
        assert signifier.residual(0, 0) == 0

    def test_bpe_id_bits_masked_off(self):
        """SIG-19: BPE-id residuals are masked off."""
        # type 0b110 vs type 0b010, BPE ids 5 and 7 differ
        assert signifier.residual(t(0b110) | 5, t(0b010) | 7) == t(0b100)

    def test_directional(self):
        """residual(a, b) != residual(b, a) in general."""
        assert signifier.residual(t(0b110), t(0b010)) != signifier.residual(t(0b010), t(0b110))


class TestClassifyMisfitRemoved:
    """classify_misfit moved to kalvin.kline (structural concern).

    The SIG-20..23 conformance cases now live in tests/test_misfit.py as
    TestClassifyMisfitConformance; NLPSignifier no longer defines the method.
    """

    def test_no_method_on_signifier(self):
        """classify_misfit is not a Signifier method (moved to kline)."""
        assert not hasattr(signifier, "classify_misfit")


class TestAbstractConformance:
    """NLPSignifier satisfies the KSignifier ABC."""

    def test_is_ksignifier(self):
        from kalvin.abstract import KSignifier

        assert isinstance(signifier, KSignifier)

    def test_abc_is_abstract(self):
        # KSignifier cannot be instantiated directly.
        import pytest

        from kalvin.abstract import KSignifier

        with pytest.raises(TypeError):
            KSignifier()
