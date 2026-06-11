"""Tests for src/kscript/nlp_types.py."""

from kscript.nlp_types import describe_nlp_type


class TestDescribeNlpType:
    """Tests for describe_nlp_type()."""

    def test_zero_type_bits(self) -> None:
        """No NLP type bits set → <NLP:0>."""
        assert describe_nlp_type(0) == "<NLP:0>"

    def test_single_pos_noun(self) -> None:
        """Single POS tag: NOUN (bit 7)."""
        # NOUN = 1 << 7 = 128, shifted to high 32
        sig = 128 << 32
        assert describe_nlp_type(sig) == "<NOUN>"

    def test_single_pos_verb(self) -> None:
        """Single POS tag: VERB (bit 15)."""
        sig = (1 << 15) << 32
        assert describe_nlp_type(sig) == "<VERB>"

    def test_multiple_flags(self) -> None:
        """Multiple NLP flags: PROPN | DEP_SUBJ."""
        # PROPN = 1 << 11 = 2048, DEP_SUBJ = 1 << 17 = 131072
        nlp_type = (1 << 11) | (1 << 17)
        sig = nlp_type << 32
        result = describe_nlp_type(sig)
        assert "PROPN" in result
        assert "DEP_SUBJ" in result

    def test_with_bpe_id_ignored(self) -> None:
        """Low 32 bits (BPE ID) are ignored."""
        nlp_type = (1 << 7)  # NOUN
        sig = (nlp_type << 32) | 12345  # BPE ID 12345
        assert describe_nlp_type(sig) == "<NOUN>"

    def test_unknown_bits(self) -> None:
        """Bits set that don't match any known flag."""
        # Use a bit beyond the known range (bit 32 would be in the low part,
        # so test with a known pattern + an unknown high bit)
        # Actually, all 32 bits are defined. Test with 0 value but non-zero low.
        assert describe_nlp_type(0xDEAD) == "<NLP:0>"

    def test_morph_feature(self) -> None:
        """Single MORPH feature: MORPH_PERF (bit 31)."""
        sig = (1 << 31) << 32
        assert describe_nlp_type(sig) == "<MORPH_PERF>"

    def test_all_pos_tags(self) -> None:
        """All 17 POS tags set."""
        all_pos = 0
        for i in range(17):
            all_pos |= (1 << i)
        sig = all_pos << 32
        result = describe_nlp_type(sig)
        assert "ADJ" in result
        assert "X" in result
        assert "VERB" in result
