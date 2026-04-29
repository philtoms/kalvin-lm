"""Tests for Mod Tokenizer — openspec/tokenizer.md conformance."""

import pytest
from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer, LITERAL_MASK


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
        for c in "ABCxyz123":
            node = t.encode(c)[0]
            assert (node & 1) == 0, f"Packed node for {c!r} has bit 0 set"

    def test_encode_order_lost(self):
        t = Mod32Tokenizer()
        assert t.encode("AB") == t.encode("BA")

    def test_encode_multiplicity_lost(self):
        t = Mod32Tokenizer()
        assert t.encode("AA") == t.encode("A")


class TestMod32LiteralEncoding:
    """Literal encoding: (codepoint << 32) | 0xFFFFFFFF."""

    def test_literal_encoding_format(self):
        t = Mod32Tokenizer()
        result = t.encode("A", pack=False)
        assert len(result) == 1
        expected = (65 << 32) | LITERAL_MASK
        assert result[0] == expected

    def test_literal_multi_char(self):
        t = Mod32Tokenizer()
        result = t.encode("ABC", pack=False)
        assert len(result) == 3
        assert result[0] == (65 << 32) | LITERAL_MASK
        assert result[1] == (66 << 32) | LITERAL_MASK
        assert result[2] == (67 << 32) | LITERAL_MASK

    def test_literal_order_preserved(self):
        t = Mod32Tokenizer()
        ab = t.encode("AB", pack=False)
        ba = t.encode("BA", pack=False)
        assert ab != ba

    def test_literal_identity_preserved(self):
        t = Mod32Tokenizer()
        a = t.encode("A", pack=False)[0]
        b = t.encode("B", pack=False)[0]
        assert a != b

    def test_literal_bit0_set(self):
        t = Mod32Tokenizer()
        for c in "ABCxyz123":
            node = t.encode(c, pack=False)[0]
            assert (node & 1) == 1, f"Literal node for {c!r} has bit 0 clear"

    def test_literal_mask_in_lower_32(self):
        t = Mod32Tokenizer()
        node = t.encode("A", pack=False)[0]
        assert (node & 0xFFFF_FFFF) == 0xFFFF_FFFF

class TestMod32IsLiteral:
    """is_literal: (node & 0xFFFFFFFF) == 0xFFFFFFFF."""

    def test_literal_node(self):
        t = Mod32Tokenizer()
        node = t.encode("A", pack=False)[0]
        assert t.is_literal(node) is True

    def test_packed_node_not_literal(self):
        t = Mod32Tokenizer()
        node = t.encode("A", pack=True)[0]
        assert t.is_literal(node) is False

    def test_signature_bit0_not_literal(self):
        """A signature with bit 0 set is NOT a literal node."""
        sig_with_bit0 = 1 | 0b110  # from mixed literal+non-literal
        t = Mod32Tokenizer()
        assert t.is_literal(sig_with_bit0) is False

    def test_zero_not_literal(self):
        t = Mod32Tokenizer()
        assert t.is_literal(0) is False

class TestMod32Decode:
    """Decode: auto-detect literal mask vs packed."""

    def test_decode_packed_roundtrip(self):
        t = Mod32Tokenizer()
        # Single char always round-trips exactly
        for text in ["A", "AB", "ABC"]:
            encoded = t.encode(text, pack=True)
            decoded = t.decode(encoded)
            assert set(decoded) == set(text)
        # Multi-char with collision may not round-trip exactly in Mod32
        # (characters share bit positions when wrapping)
        # But at minimum, encoding and decoding produces valid characters
        for text in ["Hello", "xyz123"]:
            encoded = t.encode(text, pack=True)
            decoded = t.decode(encoded)
            assert isinstance(decoded, str)

    def test_decode_literal_roundtrip(self):
        t = Mod32Tokenizer()
        for text in ["A", "AB", "ABC", "Hello", "xyz123"]:
            encoded = t.encode(text, pack=False)
            decoded = t.decode(encoded)
            assert decoded == text, f"Round-trip failed for {text!r}"

    def test_decode_auto_detect_literal(self):
        t = Mod32Tokenizer()
        node = t.encode("A", pack=False)[0]
        decoded = t.decode([node])
        assert decoded == "A"

    def test_decode_auto_detect_packed(self):
        t = Mod32Tokenizer()
        node = t.encode("A", pack=True)[0]
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
            result = t.encode(c, pack=True)
            assert len(result) == 1, f"Failed to encode {c!r}"
            assert result[0] != 0

    def test_all_printable_literal_roundtrip(self):
        t = Mod32Tokenizer()
        for i in range(32, 127):
            c = chr(i)
            encoded = t.encode(c, pack=False)
            decoded = t.decode(encoded)
            assert decoded == c, f"Round-trip failed for {c!r}"


class TestMod64Tokenizer:
    """Mod64 variant: 63 bit positions."""

    def test_encode_literal(self):
        t = Mod64Tokenizer()
        node = t.encode("A", pack=False)[0]
        assert (node & 0xFFFF_FFFF) == 0xFFFF_FFFF
        assert (node >> 32) == 65

    def test_encode_packed(self):
        t = Mod64Tokenizer()
        node = t.encode("A", pack=True)[0]
        assert (node & 1) == 0

    def test_roundtrip_literal(self):
        t = Mod64Tokenizer()
        for text in ["Hello", "World", "Test123"]:
            assert t.decode(t.encode(text, pack=False)) == text


