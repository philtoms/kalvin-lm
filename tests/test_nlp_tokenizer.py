"""Tests for NLPTokenizer — NLP-enriched BPE tokenizer.

Uses real BPE tokenizer and grammar dictionary from data/tokenizer/.
"""

from __future__ import annotations

import pytest

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import LITERAL_MASK, is_literal_node


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nlp() -> NLPTokenizer:
    """Load NLPTokenizer from standard file paths (once per session)."""
    return NLPTokenizer.from_files()


# ── Encode tests ──────────────────────────────────────────────────────────

class TestEncode:
    """Tests for the encode() method."""

    def test_encode_tea_single_node(self, nlp: NLPTokenizer) -> None:
        """'Tea' produces exactly one NLP-BPE node with correct type and ID."""
        nodes = nlp.encode("Tea")
        assert len(nodes) == 1
        node = nodes[0]

        # High 32 bits = nlp_type32 for "Tea" from grammar dict
        nlp_type32 = node >> 32
        bpe_id = node & 0xFFFFFFFF

        assert nlp_type32 == 131200, f"Expected nlp_type32=131200, got {nlp_type32}"
        assert bpe_id == 12465, f"Expected bpe_id=12465, got {bpe_id}"

    def test_encode_empty(self, nlp: NLPTokenizer) -> None:
        """Encoding empty string returns empty list."""
        assert nlp.encode("") == []

    def test_encode_pad_ws(self, nlp: NLPTokenizer) -> None:
        """pad_ws=True adds trailing space consistent with BPE tokenizer."""
        # "Tea " with pad_ws=False already includes a space token
        # "Tea" with pad_ws=True strips + adds trailing space
        nodes_no_pad = nlp.encode("Tea ", pad_ws=False)
        nodes_pad = nlp.encode("Tea", pad_ws=True)

        # Both should produce the same result: Tea + space
        assert nodes_no_pad == nodes_pad

        # Verify the space token is present (BPE token for space is 32)
        space_node = nodes_pad[-1]
        assert (space_node & 0xFFFFFFFF) == 32

    def test_nodes_not_literal(self, nlp: NLPTokenizer) -> None:
        """NLP-BPE nodes are not detected as literal nodes."""
        nodes = nlp.encode("Tea")
        for node in nodes:
            assert not is_literal_node(node), (
                f"Node {node} should not be a literal node — "
                f"BPE IDs are < vocab_size, never 0xFFFFFFFF"
            )


# ── Decode / Roundtrip tests ─────────────────────────────────────────────

class TestDecode:
    """Tests for the decode() method and round-trip behaviour."""

    def test_roundtrip(self, nlp: NLPTokenizer) -> None:
        """decode(encode(text)) == text for a known single word."""
        text = "Tea"
        assert nlp.decode(nlp.encode(text)) == text

    def test_roundtrip_phrase(self, nlp: NLPTokenizer) -> None:
        """decode(encode(text)) == text for a multi-word phrase."""
        text = "Tea brewed softly"
        assert nlp.decode(nlp.encode(text)) == text

    def test_decode_empty(self, nlp: NLPTokenizer) -> None:
        """Decoding empty list returns empty string."""
        assert nlp.decode([]) == ""


# ── Unknown token fallback ───────────────────────────────────────────────

class TestUnknownFallback:
    """Tests for unknown BPE tokens falling back to POS_X."""

    def test_unknown_token_fallback(self) -> None:
        """BPE tokens not in grammar dict get UNKNOWN_NLP_TYPE (65536)."""
        from kalvin.tokenizer import Tokenizer
        from kalvin.nlp_tokenizer import load_grammar_dict

        # Use an empty grammar dict — all tokens should be unknown
        bpe = Tokenizer.from_directory()
        nlp = NLPTokenizer(bpe, {})

        nodes = nlp.encode("Tea")
        assert len(nodes) >= 1

        for node in nodes:
            nlp_type32 = node >> 32
            assert nlp_type32 == 65536, (
                f"Expected UNKNOWN_NLP_TYPE=65536 for empty grammar, got {nlp_type32}"
            )


# ── Literal encoding tests ──────────────────────────────────────────────

class TestLiteralEncoding:
    """Tests for encode_literal()."""

    def test_literal_encoding(self, nlp: NLPTokenizer) -> None:
        """encode_literal('abc') produces 3 literal nodes."""
        text = "abc"
        nodes = nlp.encode_literal(text)

        assert len(nodes) == 3

        for i, (char, node) in enumerate(zip(text, nodes)):
            expected = (ord(char) << 32) | LITERAL_MASK
            assert node == expected, (
                f"Node {i}: expected {expected}, got {node}"
            )
            assert is_literal_node(node), (
                f"Node {i} ({node}) should be a literal node"
            )


# ── Properties tests ────────────────────────────────────────────────────

class TestProperties:
    """Tests for vocab_size and grammar_size properties."""

    def test_vocab_size(self, nlp: NLPTokenizer) -> None:
        """vocab_size matches BPE tokenizer's vocab size."""
        assert nlp.vocab_size == 17392

    def test_grammar_size(self, nlp: NLPTokenizer) -> None:
        """grammar_size matches grammar dict entry count."""
        assert nlp.grammar_size == 12871


# ── Factory tests ───────────────────────────────────────────────────────

class TestFactory:
    """Tests for from_files() class method."""

    def test_from_files_default_args(self) -> None:
        """from_files() with no args loads a working tokenizer."""
        nlp = NLPTokenizer.from_files()
        nodes = nlp.encode("Tea")
        assert len(nodes) >= 1, "from_files() should produce a working tokenizer"
