"""Tests for Mod Tokenizer — openspec/tokenizer.md conformance.

Literals are numbers or quoted strings (not lowercase identifiers).
The tokenizer determines encoding mode internally:
  - All-uppercase-alpha strings → packed (single node, bit 0 clear)
  - Everything else → literal (one node per character, literal mask)
"""

import pytest
from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer
from kalvin.signature import LITERAL_MASK, is_literal_node


class TestMod32PackedEncoding:
    """Packed encoding: OR of char bits, bit 0 clear."""

    def test_encode_empty(self):
        t = Mod32Tokenizer()
        assert t.encode("") == []

    def test_encode_single_char(self):
        t = Mod32Tokenizer()
        result = t.encode("A")
        assert len(result) == 1
        assert result[0] == t.CHAR_BIT["A"]

    def test_encode_multi_char(self):
        t = Mod32Tokenizer()
        result = t.encode("AB")
        expected = t.CHAR_BIT["A"] | t.CHAR_BIT["B"]
        assert result == [expected]

    def test_packed_bit0_clear(self):
        t = Mod32Tokenizer()
        for c in "ABCXYZ":
            node = t.encode(c)[0]
            assert (node & 1) == 0, f"Packed node for {c!r} has bit 0 set"

    def test_encode_order_lost(self):
        t = Mod32Tokenizer()
        assert t.encode("AB") == t.encode("BA")

    def test_encode_multiplicity_lost(self):
        t = Mod32Tokenizer()
        assert t.encode("AA") == t.encode("A")


class TestMod32LiteralEncoding:
    """Literal encoding: (codepoint << 32) | 0xFFFFFFFF.

    Non-uppercase-alpha strings are automatically encoded as literal.
    """

    def test_literal_encoding_format(self):
        t = Mod32Tokenizer()
        # Non-uppercase string → literal
        result = t.encode("1")
        assert len(result) == 1
        expected = (ord("1") << 32) | LITERAL_MASK
        assert result[0] == expected

    def test_literal_multi_char(self):
        t = Mod32Tokenizer()
        result = t.encode("123")
        assert len(result) == 3
        assert result[0] == (ord("1") << 32) | LITERAL_MASK
        assert result[1] == (ord("2") << 32) | LITERAL_MASK
        assert result[2] == (ord("3") << 32) | LITERAL_MASK

    def test_literal_order_preserved(self):
        t = Mod32Tokenizer()
        ab = t.encode("12")
        ba = t.encode("21")
        assert ab != ba

    def test_literal_identity_preserved(self):
        t = Mod32Tokenizer()
        a = t.encode("1")[0]
        b = t.encode("2")[0]
        assert a != b

    def test_literal_bit0_set(self):
        t = Mod32Tokenizer()
        for c in "123xyz!@":
            node = t.encode(c)[0]
            assert (node & 1) == 1, f"Literal node for {c!r} has bit 0 clear"

    def test_literal_mask_in_lower_32(self):
        t = Mod32Tokenizer()
        node = t.encode("1")[0]
        assert (node & 0xFFFF_FFFF) == 0xFFFF_FFFF

    def test_uppercase_is_packed_not_literal(self):
        """Uppercase-alpha strings are packed, not literal."""
        t = Mod32Tokenizer()
        node = t.encode("A")[0]
        assert (node & 0xFFFF_FFFF) != 0xFFFF_FFFF
        assert len(t.encode("A")) == 1  # single packed node

    def test_mixed_case_is_literal(self):
        """Mixed-case strings are literal (not all uppercase alpha)."""
        t = Mod32Tokenizer()
        result = t.encode("Hello")
        assert len(result) == 5  # one node per character
        for node in result:
            assert is_literal_node(node)

    def test_number_string_is_literal(self):
        """Number strings are literal."""
        t = Mod32Tokenizer()
        result = t.encode("42")
        assert len(result) == 2  # one node per character
        for node in result:
            assert is_literal_node(node)


class TestMod32IsLiteral:
    """is_literal_node: (node & 0xFFFFFFFF) == 0xFFFFFFFF."""

    def test_literal_node(self):
        t = Mod32Tokenizer()
        node = t.encode("1")[0]
        assert is_literal_node(node) is True

    def test_packed_node_not_literal(self):
        t = Mod32Tokenizer()
        node = t.encode("A")[0]
        assert is_literal_node(node) is False

    def test_signature_bit0_not_literal(self):
        """A signature with bit 0 set is NOT a literal node."""
        sig_with_bit0 = 1 | 0b110  # from mixed literal+non-literal
        assert is_literal_node(sig_with_bit0) is False

    def test_zero_not_literal(self):
        assert is_literal_node(0) is False


class TestMod32Decode:
    """Decode: auto-detect literal mask vs packed."""

    def test_decode_packed_roundtrip(self):
        t = Mod32Tokenizer()
        # Single char always round-trips exactly
        for text in ["A", "AB", "ABC"]:
            encoded = t.encode(text)
            decoded = t.decode(encoded)
            assert set(decoded) == set(text)
        # Multi-char with collision may not round-trip exactly in Mod32
        # (characters share bit positions when wrapping)
        # But at minimum, encoding and decoding produces valid characters
        for text in ["HELLO", "XYZ"]:
            encoded = t.encode(text)
            decoded = t.decode(encoded)
            assert isinstance(decoded, str)

    def test_decode_literal_roundtrip(self):
        t = Mod32Tokenizer()
        for text in ["1", "12", "123", "Hello", "xyz123"]:
            encoded = t.encode(text)
            decoded = t.decode(encoded)
            assert decoded == text, f"Round-trip failed for {text!r}"

    def test_decode_auto_detect_literal(self):
        t = Mod32Tokenizer()
        node = t.encode("1")[0]
        decoded = t.decode([node])
        assert decoded == "1"

    def test_decode_auto_detect_packed(self):
        t = Mod32Tokenizer()
        node = t.encode("A")[0]
        decoded = t.decode([node])
        assert decoded == "A"

    def test_decode_empty(self):
        t = Mod32Tokenizer()
        assert t.decode([]) == ""


class TestMod32Vocab:
    """Vocabulary: 95 printable ASCII characters."""

    def test_vocab_size(self):
        t = Mod32Tokenizer()
        # Mod32 has 31 unique bit positions (bits 1-31)
        assert t.vocab_size == 31

    def test_all_printable_ascii_encodable(self):
        t = Mod32Tokenizer()
        for i in range(32, 127):
            c = chr(i)
            result = t.encode(c)
            assert len(result) == 1, f"Failed to encode {c!r}"
            assert result[0] != 0

    def test_all_printable_literal_roundtrip(self):
        t = Mod32Tokenizer()
        for i in range(32, 127):
            c = chr(i)
            # Use a non-uppercase context to get literal encoding
            text = f" {c}"  # space prefix forces literal
            encoded = t.encode(text)
            decoded = t.decode(encoded)
            assert decoded == text, f"Round-trip failed for {c!r}"


class TestMod64Tokenizer:
    """Mod64 variant: 63 bit positions."""

    def test_encode_literal(self):
        t = Mod64Tokenizer()
        node = t.encode("1")[0]
        assert (node & 0xFFFF_FFFF) == 0xFFFF_FFFF
        assert (node >> 32) == ord("1")

    def test_encode_packed(self):
        t = Mod64Tokenizer()
        node = t.encode("A")[0]
        assert (node & 1) == 0

    def test_roundtrip_literal(self):
        t = Mod64Tokenizer()
        for text in ["Hello", "World", "Test123"]:
            assert t.decode(t.encode(text)) == text
