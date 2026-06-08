"""Tests for TokenEncoder and CompiledEntry — symbolic→token conversion and encode/decode.

Tests TokenEncoder in isolation (no AST traversal), plus CompiledEntry
encode/decode round-trip tests ported from test_kscript.py.
"""

import pytest

from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer, ModTokenizer
from kalvin.signature import is_nlp_node
from kscript.ast_emitter import SymbolicEntry
from kscript.token_encoder import CompiledEntry, TokenEncoder

# Conditional NLP tokenizer import
try:
    from kalvin.nlp_tokenizer import NLPTokenizer
    _has_nlp = True
except ImportError:
    _has_nlp = False

_nlp_skip = pytest.mark.skipif(not _has_nlp, reason="NLPTokenizer not available")


# ── Shared fixtures ──────────────────────────────────────────────────────────

_tok64 = Mod64Tokenizer()
_tok32 = Mod32Tokenizer()


# =============================================================================
# 1. TokenEncoder — basic encoding
# =============================================================================


class TestTokenEncoderBasic:
    def test_encode_unsigned(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([SymbolicEntry("A", None, "UNSIGNED")])
        assert len(entries) == 1
        e = entries[0]
        # Signature should be a valid token ID
        assert isinstance(e.signature, int)
        assert e.signature == _tok64.encode("A")[0]
        # None → KLine normalizes to []
        assert e.nodes == []

    def test_encode_countersign(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([SymbolicEntry("A", "B", "COUNTERSIGN")])
        assert len(entries) == 1
        e = entries[0]
        assert e.signature == _tok64.encode("A")[0]
        # B is a signature, so encoded as single int
        assert e.nodes == [_tok64.encode("B")[0]]


# =============================================================================
# 2. TokenEncoder — literal nodes
# =============================================================================


class TestTokenEncoderLiteral:
    def test_encode_literal_nodes(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([SymbolicEntry("A", "hello", "CONNOTATE")])
        assert len(entries) == 1
        e = entries[0]
        assert e.signature == _tok64.encode("A")[0]
        # "hello" is a literal, encoded as multiple token IDs
        assert isinstance(e.nodes, list)
        assert len(e.nodes) > 1


# =============================================================================
# 3. TokenEncoder — list nodes
# =============================================================================


class TestTokenEncoderListNode:
    def test_encode_list_nodes(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([SymbolicEntry("AB", ["A", "B"], "CANONIZE")])
        assert len(entries) == 1
        e = entries[0]
        assert e.signature == _tok64.encode("AB")[0]
        assert e.nodes == [_tok64.encode("A")[0], _tok64.encode("B")[0]]

    def test_decode_list_nodes_round_trip(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([SymbolicEntry("AB", ["A", "B"], "CANONIZE")])
        sig, nodes = entries[0].decode(_tok64)
        assert sig == "AB"
        assert nodes == ["A", "B"]


# =============================================================================
# 4. TokenEncoder — dev mode
# =============================================================================


class TestTokenEncoderDevMode:
    def test_dev_mode_dbg_text_unsigned(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64, dev=True)
        entries = encoder.encode_entries([SymbolicEntry("A", None, "UNSIGNED")])
        assert entries[0].dbg_text == "[S4] A: None"

    def test_dev_mode_dbg_text_countersign(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64, dev=True)
        entries = encoder.encode_entries([SymbolicEntry("A", "B", "COUNTERSIGN")])
        assert entries[0].dbg_text == "[S1] A: B"

    def test_dev_mode_dbg_text_canonize_list(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64, dev=True)
        entries = encoder.encode_entries([SymbolicEntry("AB", ["A", "B"], "CANONIZE")])
        assert entries[0].dbg_text == "[S2] AB: ['A', 'B']"

    def test_no_dev_mode_no_dbg_text(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64, dev=False)
        entries = encoder.encode_entries([SymbolicEntry("A", None, "UNSIGNED")])
        assert entries[0].dbg_text == ""


# =============================================================================
# 5. TokenEncoder — multiple entries
# =============================================================================


class TestTokenEncoderMultiple:
    def test_encode_multiple_entries(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        symbolic = [
            SymbolicEntry("A", None, "UNSIGNED"),
            SymbolicEntry("B", "A", "COUNTERSIGN"),
            SymbolicEntry("AB", ["A", "B"], "CANONIZE"),
        ]
        entries = encoder.encode_entries(symbolic)
        assert len(entries) == 3

    def test_empty_input(self) -> None:
        encoder = TokenEncoder(tokenizer=_tok64)
        entries = encoder.encode_entries([])
        assert entries == []


# =============================================================================
# 6. CompiledEntry encode — ported from test_kscript.py
# =============================================================================


class TestCompiledEntryEncode:
    def test_encode_sig_only(self) -> None:
        entry = CompiledEntry.encode("A", None, _tok64)
        assert entry.nodes == []  # None normalizes to []
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == ""  # empty nodes decode to ''

    def test_encode_sig_to_sig(self) -> None:
        entry = CompiledEntry.encode("A", "B", _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == ["B"]  # single-node signature stored as list

    def test_encode_sig_to_literal(self) -> None:
        entry = CompiledEntry.encode("A", "hello", _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == "hello"

    def test_encode_sig_to_list(self) -> None:
        entry = CompiledEntry.encode("AB", ["A", "B"], _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "AB"
        assert nodes == ["A", "B"]


# =============================================================================
# 7. CompiledEntry decode — round-trip verification
# =============================================================================


class TestCompiledEntryDecode:
    def test_decode_unsigned(self) -> None:
        entry = CompiledEntry.encode("A", None, _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == ""

    def test_decode_sig_to_sig(self) -> None:
        entry = CompiledEntry.encode("A", "B", _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == ["B"]

    def test_decode_sig_to_literal(self) -> None:
        entry = CompiledEntry.encode("A", "hello", _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "A"
        assert nodes == "hello"

    def test_decode_sig_to_list(self) -> None:
        entry = CompiledEntry.encode("AB", ["A", "B"], _tok64)
        sig, nodes = entry.decode(_tok64)
        assert sig == "AB"
        assert nodes == ["A", "B"]

    def test_decode_with_mod32(self) -> None:
        entry = CompiledEntry.encode("A", "B", _tok32)
        sig, nodes = entry.decode(_tok32)
        assert sig == "A"
        assert nodes == ["B"]


# =============================================================================
# 8. CompiledEntry encode/decode with NLPTokenizer
# =============================================================================

@_nlp_skip
class TestTokenEncoderNLP:
    """Tests exercising the token encoder pipeline with NLPTokenizer.

    These verify that the widened KTokenizer type hints work with an
    actual alternative tokenizer. All tests skip if NLPTokenizer or its
    data files are unavailable.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        try:
            self._nlp_tok = NLPTokenizer.from_files()
        except Exception:
            pytest.skip("NLPTokenizer data files not available")

    def test_encode_unsigned_with_nlp(self) -> None:
        """Encode an unsigned entry with NLPTokenizer — sig should be an NLP-BPE node."""
        entry = CompiledEntry.encode("MHALLO", None, self._nlp_tok)
        # Signature token ID should be a valid NLP node (high 32 bits non-zero)
        assert is_nlp_node(entry.signature)

    def test_encode_countersign_with_nlp(self) -> None:
        """Encode a countersign entry — both sig and node should be NLP-BPE encoded."""
        entry = CompiledEntry.encode("A", "B", self._nlp_tok)
        assert is_nlp_node(entry.signature)
        # Node is a list (KLine normalization); all should be NLP-BPE nodes
        assert isinstance(entry.nodes, list)
        assert len(entry.nodes) == 1
        assert is_nlp_node(entry.nodes[0])

    def test_encode_literal_with_nlp(self) -> None:
        """Encode a literal connotate — NLP tokenizer may BPE-encode 'hello'."""
        entry = CompiledEntry.encode("A", "hello", self._nlp_tok)
        assert is_nlp_node(entry.signature)
        # "hello" is encoded by NLP tokenizer as BPE tokens
        assert isinstance(entry.nodes, list)
        assert len(entry.nodes) >= 1

    def test_encode_and_decode_roundtrip_nlp(self) -> None:
        """Encode then decode an entry — string values should be recovered."""
        entry = CompiledEntry.encode("A", "B", self._nlp_tok)
        sig, nodes = entry.decode(self._nlp_tok)
        assert sig == "A"
        # Single BPE-encoded sig node
        assert nodes == ["B"]

    def test_encode_literal_roundtrip_nlp(self) -> None:
        """Encode/decode a literal connotate — text should round-trip."""
        entry = CompiledEntry.encode("A", "hello", self._nlp_tok)
        sig, nodes = entry.decode(self._nlp_tok)
        assert sig == "A"
        # NLP tokenizer should round-trip the literal text
        assert nodes == "hello"
