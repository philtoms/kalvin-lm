"""Tests for NLPTokenizer — NLP-enriched BPE tokenizer.

Uses real BPE tokenizer and grammar dictionary from data/tokenizer/.
"""

from __future__ import annotations

import pytest

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import (
    LITERAL_MASK,
    NLP_TYPE_MASK,
    is_literal_node,
    is_nlp_node,
    make_signature,
)


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
        assert nlp.grammar_size == 17392


# ── Factory tests ───────────────────────────────────────────────────────

class TestFactory:
    """Tests for from_files() class method."""

    def test_from_files_default_args(self) -> None:
        """from_files() with no args loads a working tokenizer."""
        nlp = NLPTokenizer.from_files()
        nodes = nlp.encode("Tea")
        assert len(nodes) >= 1, "from_files() should produce a working tokenizer"


# ── Integration: end-to-end encode → decode → signature pipeline ──────

class TestNLPEncodingPipeline:
    """Cross-module integration tests exercising encode → decode → signature.

    These verify the NLP tokenizer produces correct NLP-BPE node values,
    that round-trip through encode/decode preserves text, and that
    make_signature correctly handles NLP-BPE nodes vs. space tokens
    vs. literal nodes.

    Key insight: multi-word phrases produce space BPE tokens (ID 32) with
    nlp_type32=0, making them Mod32-style packed nodes rather than NLP-BPE
    nodes.  Single words are pure NLP-BPE nodes.
    """

    def test_pipeline_encode_decode_roundtrip(self, nlp: NLPTokenizer) -> None:
        """Full encode → decode round-trip with node-level inspection.

        'Tea brewed softly' produces 5 BPE tokens: 3 words + 2 spaces.
        The round-trip must recover the original text exactly.
        """
        text = "Tea brewed softly"
        nodes = nlp.encode(text)

        # 5 nodes: Tea, <space>, brewed, <space>, softly
        assert len(nodes) == 5, (
            f"Expected 5 nodes (3 words + 2 spaces), got {len(nodes)}"
        )

        # Verify round-trip
        assert nlp.decode(nodes) == text

    def test_pipeline_node_format(self, nlp: NLPTokenizer) -> None:
        """Individual word encodings match the expected NLP-BPE node format.

        node = (nlp_type32 << 32) | bpe_token_id

        Uses known BPE tokens from the grammar dictionary:
          Tea     → bpe_id=12465, nlp_type32=131200
          brewed  → bpe_id=4964,  nlp_type32=8421376
          softly  → bpe_id=977,   nlp_type32=2097156
        """
        # Each word is a single BPE token → single NLP-BPE node
        for word, expected_type32, expected_bpe_id in [
            ("Tea", 131200, 12465),
            ("brewed", 8421376, 4964),
            ("softly", 2097156, 977),
        ]:
            nodes = nlp.encode(word)
            assert len(nodes) == 1, f"'{word}' should encode to 1 node, got {len(nodes)}"
            expected = (expected_type32 << 32) | expected_bpe_id
            assert nodes[0] == expected, (
                f"'{word}': expected node {expected}, got {nodes[0]}"
            )

    def test_pipeline_signature_nlp_only(self, nlp: NLPTokenizer) -> None:
        """Signature of pure NLP-BPE nodes contains only high-32 NLP type bits.

        Using a single word ('Tea') avoids space tokens (which have
        nlp_type32=0 and contribute to the low bits).  The signature
        should have no bits set in the low 32 — BPE IDs are masked out.
        """
        nodes = nlp.encode("Tea")
        assert len(nodes) == 1
        assert is_nlp_node(nodes[0]), "Single-word node should be NLP-BPE"

        sig = make_signature(nodes)

        # Non-zero
        assert sig != 0

        # Low 32 bits must be 0 (BPE IDs masked out by NLP-aware make_signature)
        assert (sig & 0xFFFFFFFF) == 0, (
            f"Signature {sig:#x} has bits in low 32 — BPE IDs leaked"
        )

        # High bits match the nlp_type32 of the node
        assert sig == (nodes[0] & NLP_TYPE_MASK)

    def test_pipeline_signature_with_literal(self, nlp: NLPTokenizer) -> None:
        """Mixed NLP-BPE + literal nodes: bit 0 set + NLP type bits.

        Create a mixed sequence: NLP-BPE node from encode("Tea") and
        literal nodes from encode_literal("X").  The signature should
        have bit 0 set (literal-content flag) and NLP type bits in
        the high positions.
        """
        nlp_nodes = nlp.encode("Tea")
        literal_nodes = nlp.encode_literal("X")

        all_nodes = nlp_nodes + literal_nodes
        sig = make_signature(all_nodes)

        # Bit 0 must be set (literal present)
        assert sig & 1 == 1, "Bit 0 should be set from literal node"

        # NLP type bits must be present in high positions
        nlp_type_bits = nlp_nodes[0] & NLP_TYPE_MASK
        assert sig & nlp_type_bits != 0, "NLP type bits should be present"

        # Low bits (excluding bit 0) should only be from NLP masking
        # (NLP nodes contribute only high bits, literals contribute only bit 0)
        expected = nlp_type_bits | 1
        assert sig == expected

    def test_pipeline_space_nodes_not_nlp(self, nlp: NLPTokenizer) -> None:
        """Space tokens (nlp_type32=0) are NOT NLP nodes — they act as Mod32 packed.

        In multi-word phrases, space BPE tokens (ID 32) have nlp_type32=0
        in the grammar dictionary.  Their resulting node value is just 32,
        which is_nlp_node() returns False for.  These contribute to the
        low bits of the signature via full-value OR-reduction.
        """
        nodes = nlp.encode("Tea brewed")
        # nodes = [Tea_node, space_node, brewed_node]
        assert len(nodes) == 3

        space_node = nodes[1]
        # Space BPE token ID is 32; nlp_type32=0 → node = (0 << 32) | 32 = 32
        assert space_node == 32, f"Space node should be 32, got {space_node}"
        assert not is_nlp_node(space_node), "Space node should NOT be NLP"
        assert not is_literal_node(space_node), "Space node should NOT be literal"

        # Signature of the full phrase includes bit 5 from the space node
        sig = make_signature(nodes)
        # Space contributes its full value (32 = bit 5) to the signature
        assert sig & 32 != 0, "Space node should contribute bit 5 to signature"
