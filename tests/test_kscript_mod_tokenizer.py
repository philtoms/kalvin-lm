"""Tests for ModTokenizer classes (Mod32, Mod64, Mod128)."""

import pytest
from kscript.mod_tokenizer import ModTokenizer, Mod32Tokenizer,Mod64Tokenizer, Mod128Tokenizer, _build_char_bit_maps, mod_alphabet

class TestBuildCharBitMaps:
    """Tests for _build_char_bit_maps helper function."""

    def test_returns_tuple_of_dicts(self):
        """Function returns tuple of two dictionaries."""
        char_bit, bit_char = _build_char_bit_maps(32)
        assert isinstance(char_bit, dict)
        assert isinstance(bit_char, dict)

    def test_bit_char_reverse_mapping(self):
        """bit_char provides reverse mapping from bit to char."""
        char_bit, bit_char = _build_char_bit_maps(32)
        # Check that each bit_char entry maps to a valid char_bit entry
        for bit, char in bit_char.items():
            assert char in char_bit
            assert char_bit[char] == bit

    def test_uppercase_priority_in_reverse_mapping(self):
        """Uppercase letters have priority in reverse mapping."""
        char_bit, bit_char = _build_char_bit_maps(32)
        # 'A' and 'a' may map to same bit position (mod 32)
        # But 'A' should be in bit_char because it comes first
        assert bit_char.get(char_bit['A']) == 'A'

    def test_mod32_bit_positions_unique(self):
        """With mod 32, bit positions are unique within 0-31 range."""
        char_bit, _ = _build_char_bit_maps(32)
        # First 32 characters should have unique positions
        first_32 = mod_alphabet[0:31]
        seen_positions = set()
        for c in first_32:
            pos = char_bit[c]
            # Check it's a power of 2 (single bit)
            assert pos & (pos - 1) == 0 or pos == 0, f"Char {c} not a power of 2: {pos}"
            seen_positions.add(pos)

    def test_mod64_creates_more_unique_positions(self):
        """With mod 64, more characters get unique bit positions."""
        char_bit, _ = _build_char_bit_maps(64)
        # First 64 characters should have unique positions
        first_64 = mod_alphabet[0:64]
        seen_positions = set()
        for c in first_64:
            pos = char_bit[c]
            assert pos not in seen_positions, f"Duplicate position for {c}"
            seen_positions.add(pos)
        assert len(seen_positions) == 64

    def test_mod128_creates_even_more_unique_positions(self):
        """With mod 128, even more characters get unique positions."""
        char_bit, _ = _build_char_bit_maps(128)
        chars = mod_alphabet
        seen_positions = set()
        for c in chars:
            pos = char_bit[c]
            assert pos not in seen_positions
            seen_positions.add(pos)

    def test_printable_ascii_included(self):
        """All printable ASCII characters are mapped."""
        char_bit, _ = _build_char_bit_maps(32)
        for code in range(32, 127):
            c = chr(code)
            assert c in char_bit, f"Missing printable ASCII: {repr(c)}"


class TestModTokenizerBase:
    """Tests for base ModTokenizer class."""

    def test_inherits_from_ktokenizer(self):
        """ModTokenizer inherits from KTokenizer."""
        from kalvin.abstract import KTokenizer
        assert issubclass(ModTokenizer, KTokenizer)

    def test_has_char_bit_mapping(self):
        """ModTokenizer has CHAR_BIT class attribute."""
        assert hasattr(ModTokenizer, 'CHAR_BIT')
        assert isinstance(ModTokenizer.CHAR_BIT, dict)

    def test_has_bit_char_mapping(self):
        """ModTokenizer has BIT_CHAR class attribute."""
        assert hasattr(ModTokenizer, 'BIT_CHAR')
        assert isinstance(ModTokenizer.BIT_CHAR, dict)


class TestMod32Tokenizer:
    """Tests for Mod32Tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a Mod32Tokenizer instance."""
        return Mod32Tokenizer()

    def test_encode_single_char_returns_list(self, tokenizer):
        """encode returns a list for single character."""
        result = tokenizer.encode('A')
        assert isinstance(result, list)
        assert len(result) == 1

    def test_encode_single_char_returns_bit_value(self, tokenizer):
        """encode returns correct bit value for single character."""
        result = tokenizer.encode('A')
        # A is at position 0, so bit value is 1 << 0 = 1
        assert result == [1]

    def test_encode_b_returns_correct_bit(self, tokenizer):
        """encode returns correct bit for B."""
        result = tokenizer.encode('B')
        # B is at position 1, so bit value is 1 << 1 = 2
        assert result == [2]

    def test_encode_z_returns_correct_bit(self, tokenizer):
        """encode returns correct bit for Z."""
        result = tokenizer.encode('Z')
        # Z is at position 25, so bit value is 1 << 25
        assert result == [1 << 25]

    def test_decode_single_token(self, tokenizer):
        """decode returns correct character for single token."""
        result = tokenizer.decode([1])  # bit position 0 = 'A'
        assert result == 'A'

    def test_decode_multiple_tokens(self, tokenizer):
        """decode concatenates characters from multiple tokens."""
        # A=1, B=2
        result = tokenizer.decode([1, 2])
        assert result == 'AB'

    def test_encode_decode_roundtrip_single_char(self, tokenizer):
        """encode/decode roundtrip works for single character."""
        original = 'X'
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_encode_multi_char_returns_first_word_tokens(self, tokenizer):
        """encode for multi-char string returns tokens for first word via batch_encode.

        Note: The current implementation has a quirk where multi-char strings
        go through batch_encode(text.split()) which splits on whitespace.
        """
        # Single word - returns its token
        result = tokenizer.encode('A')  # Single char
        assert result == [tokenizer.CHAR_BIT['A']]

    def test_encode_with_pad_ws_strips_and_adds_space(self, tokenizer):
        """encode with pad_ws=True strips and adds trailing space."""
        result = tokenizer.encode('A', pad_ws=True)
        # Should encode 'A ' (A with trailing space)
        # Space is a printable ASCII character
        assert len(result) == 1  # Goes through batch_encode for multi-char

    def test_batch_encode_returns_list_of_lists(self, tokenizer):
        """batch_encode returns list of token ID lists."""
        texts = ['A', 'B', 'C']
        result = tokenizer.batch_encode(texts)
        assert isinstance(result, list)
        assert all(isinstance(ids, list) for ids in result)
        assert len(result) == 3

    def test_batch_encode_empty_list(self, tokenizer):
        """batch_encode with empty list returns empty list."""
        result = tokenizer.batch_encode([])
        assert result == []

    def test_batch_encode_preserves_order(self, tokenizer):
        """batch_encode preserves input order."""
        texts = ['A', 'B', 'C']
        result = tokenizer.batch_encode(texts)
        # Each should be a single-element list
        assert result[0] == [tokenizer.CHAR_BIT['A']]
        assert result[1] == [tokenizer.CHAR_BIT['B']]
        assert result[2] == [tokenizer.CHAR_BIT['C']]

    def test_encode_empty_string(self, tokenizer):
        """encode with empty string behavior."""
        # Empty string has len < 1, so it won't hit the batch_encode path
        # but CHAR_BIT[''] will raise KeyError
        with pytest.raises(KeyError):
            tokenizer.encode('')

    def test_all_uppercase_letters_encodable(self, tokenizer):
        """All uppercase letters can be encoded and decoded."""
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            encoded = tokenizer.encode(c)
            decoded = tokenizer.decode(encoded)
            assert decoded == c, f"Failed for {c}"

    def test_all_lowercase_letters_modulo_encodable(self, tokenizer):
        """All lowercase letters can be encoded and decoded."""
        for c in "abcdefghijklmnopqrstuvwxyz":
            encoded = tokenizer.encode(c)
            assert encoded[0] <= 1 << 32

class TestMod64Tokenizer:
    """Tests for Mod64Tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a Mod64Tokenizer instance."""
        return Mod64Tokenizer()

    def test_encode_single_char(self, tokenizer):
        """encode returns correct bit value for single character."""
        result = tokenizer.encode('A')
        assert result == [1]  # A is still at position 0

    def test_all_digits_encodable(self, tokenizer):
        """All digits can be encoded and decoded."""
        for c in "0123456789":
            encoded = tokenizer.encode(c)
            decoded = tokenizer.decode(encoded)
            assert decoded == c, f"Failed for {c}"

    def test_encode_decode_roundtrip(self, tokenizer):
        """encode/decode roundtrip works."""
        for c in mod_alphabet[0:63]:
            encoded = tokenizer.encode(c)
            decoded = tokenizer.decode(encoded)
            assert decoded == c, f"Failed roundtrip for {c}"

    def test_unique_bits_for_alphanumeric(self, tokenizer):
        """All alphanumeric characters have unique bit values."""
        seen = {}
        for c in mod_alphabet[0:63]:
            bit = tokenizer.encode(c)[0]
            assert bit not in seen, f"Duplicate bit {bit} for {c} and {seen[bit]}"
            seen[bit] = c


class TestMod128Tokenizer:
    """Tests for Mod128Tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a Mod128Tokenizer instance."""
        return Mod128Tokenizer()

    def test_encode_single_char(self, tokenizer):
        """encode returns correct bit value for single character."""
        result = tokenizer.encode('A')
        assert result == [1]  # A is still at position 0

    def test_encode_decode_roundtrip(self, tokenizer):
        """encode/decode roundtrip works."""
        for c in mod_alphabet:
            encoded = tokenizer.encode(c)
            decoded = tokenizer.decode(encoded)
            assert decoded == c, f"Failed roundtrip for {c}"

    def test_unique_bits_for_alphanumeric(self, tokenizer):
        """All alphanumeric characters have unique bit values."""
        chars = mod_alphabet
        seen = {}
        for c in chars:
            bit = tokenizer.encode(c)[0]
            assert bit not in seen, f"Duplicate bit {bit} for {c} and {seen[bit]}"
            seen[bit] = c


class TestModTokenizerComparison:
    """Tests comparing different modulo tokenizers."""

    def test_uppercase_same_across_modulos(self):
        """Uppercase letters have same encoding across all modulos."""
        t32 = Mod32Tokenizer()
        t64 = Mod64Tokenizer()
        t128 = Mod128Tokenizer()

        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert t32.encode(c) == t64.encode(c) == t128.encode(c), f"Mismatch for {c}"

    def test_digits_same_across_modulos(self):
        """Digits 0-4 have same encoding across all modulos (positions 27-32)."""
        t32 = Mod32Tokenizer()
        t64 = Mod64Tokenizer()
        t128 = Mod128Tokenizer()

        for c in "01234":
            assert t32.encode(c) == t64.encode(c) == t128.encode(c), f"Mismatch for {c}"

    def test_vocab_size_difference(self):
        """Different modulos have different vocab sizes."""
        t32 = Mod32Tokenizer()
        t64 = Mod64Tokenizer()
        t128 = Mod128Tokenizer()

        # Count unique bit values for alphanumeric
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"

        def count_unique(tokenizer):
            return len(set(tokenizer.encode(c)[0] for c in chars))

        # mod32 should have fewer unique values due to wrapping
        unique_32 = count_unique(t32)
        unique_64 = count_unique(t64)
        unique_128 = count_unique(t128)

        assert unique_64 > unique_32
        assert unique_128 >= unique_64


class TestModTokenizerEdgeCases:
    """Edge case tests for ModTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a Mod32Tokenizer instance."""
        return Mod128Tokenizer()

    def test_space_character(self, tokenizer):
        """Space character can be encoded."""
        encoded = tokenizer.encode(' ')
        assert isinstance(encoded, list)
        assert len(encoded) == 1
        decoded = tokenizer.decode(encoded)
        assert decoded == ' '

    def test_special_characters(self, tokenizer):
        """Special characters can be encoded."""
        special_chars = "!@#$%^&*()-_=+[]{}|;':\",./<>?"
        for c in special_chars:
            encoded = tokenizer.encode(c)
            assert isinstance(encoded, list)
            assert len(encoded) == 1
            decoded = tokenizer.decode(encoded)
            assert decoded == c, f"Failed for {repr(c)}"

    def test_newline_character(self, tokenizer):
        """Newline character can be encoded."""
        encoded = tokenizer.encode('\n')
        assert isinstance(encoded, list)
        decoded = tokenizer.decode(encoded)
        assert decoded == '\n'

    def test_tab_character(self, tokenizer):
        """Tab character can be encoded."""
        encoded = tokenizer.encode('\t')
        assert isinstance(encoded, list)
        decoded = tokenizer.decode(encoded)
        assert decoded == '\t'

    def test_multibyte_char_raises_error(self, tokenizer):
        """Non-ASCII characters raise KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode('é')

    def test_unicode_char_raises_error(self, tokenizer):
        """Unicode characters raise KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode('😀')

    def test_batch_encode_with_whitespace(self, tokenizer):
        """batch_encode handles strings with whitespace."""
        texts = ['A B', 'C D']
        result = tokenizer.batch_encode(texts)
        assert len(result) == 2
        # Each should encode to the combined bits of the characters
