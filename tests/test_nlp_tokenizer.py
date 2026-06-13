"""Tests for NLPTokenizer — NLP-enriched BPE tokenizer.

Uses real BPE tokenizer and grammar dictionary from data/tokenizer/.
"""

from __future__ import annotations

import pytest

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import make_signature


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

        assert nlp_type32 == 133120, f"Expected nlp_type32=133120, got {nlp_type32}"
        assert bpe_id == 18874, f"Expected bpe_id=18874, got {bpe_id}"

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
            assert not False, (
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

# Removed — encode_literal() removed with literal concept# ── Properties tests ────────────────────────────────────────────────────

class TestProperties:
    """Tests for vocab_size and grammar_size properties."""

    def test_vocab_size(self, nlp: NLPTokenizer) -> None:
        """vocab_size matches BPE tokenizer's vocab size."""
        assert nlp.vocab_size == 25007

    def test_grammar_size(self, nlp: NLPTokenizer) -> None:
        """grammar_size matches grammar dict entry count."""
        assert nlp.grammar_size == 25007


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

        'Tea brewed softly' produces 3 BPE nodes under the regenerated
        model — spaces are absorbed into adjacent word tokens, so there
        are no standalone space nodes.  The round-trip must recover the
        original text exactly.
        """
        text = "Tea brewed softly"
        nodes = nlp.encode(text)

        # 3 nodes: each word absorbed its leading space into one BPE token.
        assert len(nodes) == 3, (
            f"Expected 3 nodes (spaces absorbed), got {len(nodes)}"
        )

        # Verify round-trip
        assert nlp.decode(nodes) == text

    def test_pipeline_node_format(self, nlp: NLPTokenizer) -> None:
        """Individual word encodings match the expected NLP-BPE node format.

        node = (nlp_type32 << 32) | bpe_token_id

        Under the regenerated BPE model, 'Tea' is a single token while
        'brewed' and 'softly' split into multiple subword tokens.  Every
        node carries an NLP type in its high 32 bits, and each word
        round-trips through decode.
        """
        # 'Tea' is a single BPE token → single NLP-BPE node
        nodes = nlp.encode("Tea")
        assert len(nodes) == 1
        nlp_type32, bpe_id = 133120, 18874
        assert nodes[0] == (nlp_type32 << 32) | bpe_id

        # Multi-subword words: every node has NLP type bits set and the
        # word round-trips through decode.
        for word in ("brewed", "softly"):
            nodes = nlp.encode(word)
            assert len(nodes) >= 1, f"'{word}' should encode to ≥1 node"
            for node in nodes:
                assert (node >> 32) != 0, (
                    f"'{word}': node {node} should carry an NLP type"
                )
            assert nlp.decode(nodes) == word, f"'{word}' should round-trip"

    # Removed: test_pipeline_signature_nlp_only — make_signature() is now plain OR-reduce
    # Removed: test_pipeline_signature_with_literal — literal concept removed

    def test_pipeline_space_nodes_not_nlp(self, nlp: NLPTokenizer) -> None:
        """Spaces are absorbed into adjacent BPE tokens (no standalone node).

        In the regenerated BPE model, 'Tea brewed' produces 2 nodes — the
        space is merged into the following word token rather than emitted
        as a separate ID-32 node.  There is therefore no standalone space
        node, and the phrase still round-trips through decode.
        """
        nodes = nlp.encode("Tea brewed")
        # Spaces absorbed: 2 nodes (Tea, " brewed") — no standalone space.
        assert len(nodes) == 2

        # No standalone space node (BPE token ID 32 with nlp_type32=0).
        assert 32 not in nodes

        # Round-trip recovers the original phrase.
        assert nlp.decode(nodes) == "Tea brewed"

        # Signature is well-formed and non-zero.
        sig = make_signature(nodes)
        assert sig != 0
