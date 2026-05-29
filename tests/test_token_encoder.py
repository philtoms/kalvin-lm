"""Tests for TokenEncoder and CompiledEntry — symbolic→token conversion and encode/decode.

Tests TokenEncoder in isolation (no AST traversal), plus CompiledEntry
encode/decode round-trip tests ported from test_kscript.py.
"""

import pytest

from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer, ModTokenizer
from kscript.ast_emitter import SymbolicEntry
from kscript.token_encoder import CompiledEntry, TokenEncoder


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
