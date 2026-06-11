"""Tests for NLP signature description in the decompiler.

Covers acceptance criteria:
- NB-24: NLP node decodes to word text
- NB-25: NLP signature shows type description

All tests construct klines directly with known NLP type values — no
dependency on the compiler, BindingScope, or spaCy.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from kalvin.kline import KLine
from kscript.decompiler import Decompiler
from kscript.nlp_types import describe_nlp_type


# ── NLP type bit values (mirrors create_nlp_type32 from dev/nlp/nlp_analyzer.py) ──

# POS tags (bits 0–16)
POS_ADJ   = 1 << 0
POS_ADP   = 1 << 1
POS_ADV   = 1 << 2
POS_AUX   = 1 << 3
POS_CCONJ = 1 << 4
POS_DET   = 1 << 5
POS_INTJ  = 1 << 6
POS_NOUN  = 1 << 7
POS_NUM   = 1 << 8
POS_PART  = 1 << 9
POS_PRON  = 1 << 10
POS_PROPN = 1 << 11
POS_PUNCT = 1 << 12
POS_SCONJ = 1 << 13
POS_SYM   = 1 << 14
POS_VERB  = 1 << 15
POS_X     = 1 << 16

# DEP groups (bits 17–24)
DEP_SUBJ   = 1 << 17
DEP_OBJ    = 1 << 18
DEP_OBL    = 1 << 19
DEP_COMP   = 1 << 20
DEP_MOD    = 1 << 21
DEP_FUNC   = 1 << 22
DEP_STRUCT = 1 << 23
DEP_PUNCT  = 1 << 24

# MORPH features (bits 25–31)
MORPH_PLUR = 1 << 25
MORPH_PRES = 1 << 26
MORPH_IMP  = 1 << 27
MORPH_P1   = 1 << 28
MORPH_P2   = 1 << 29
MORPH_P3   = 1 << 30
MORPH_PERF = 1 << 31


def _make_nlp_sig(nlp_type: int) -> int:
    """Create a pure NLP-type signature (high 32 bits = nlp_type, low 32 bits = 0)."""
    return nlp_type << 32


def _make_nlp_node(nlp_type: int, bpe_id: int) -> int:
    """Create an NLP-BPE node: (nlp_type << 32) | bpe_id."""
    return (nlp_type << 32) | bpe_id


def _mock_nlp_tokenizer() -> MagicMock:
    """Create a mock NLP tokenizer (supports_mcs=False) with default empty decode."""
    tok = MagicMock()
    tok.supports_mcs = False
    tok.decode = MagicMock(return_value="")
    return tok


# ═══════════════════════════════════════════════════════════════════════════
# NB-24: NLP node decodes to word
# ═══════════════════════════════════════════════════════════════════════════

class TestNB24:
    """NB-24: NLP node decodes to word text."""

    def test_decompile_nlp_node_readable(self) -> None:
        """A KLine with an NLP-BPE node decodes the node to its word text.

        The signature is the NLP type value; the node is a valid NLP-BPE node
        that the tokenizer can decode to a word.
        """
        tok = MagicMock()
        tok.supports_mcs = False
        # Node decode returns the word; sig decode returns empty (not a BPE ID)
        tok.decode = MagicMock(side_effect=lambda ids: "cat" if ids == [_make_nlp_node(POS_NOUN, 42)] else "")

        decomp = Decompiler(tokenizer=tok)

        # Signature is pure NLP type; node is NLP-BPE token
        sig = _make_nlp_sig(POS_NOUN)
        node = _make_nlp_node(POS_NOUN, 42)
        kline = KLine(signature=sig, nodes=node)

        entries = decomp.decompile([kline])
        assert len(entries) == 1
        # The node should decode to "cat"
        assert entries[0].nodes == "cat"


# ═══════════════════════════════════════════════════════════════════════════
# NB-25: NLP signature shows type description
# ═══════════════════════════════════════════════════════════════════════════

class TestNB25:
    """NB-25: NLP signature shows type description for OR'd NLP type bits."""

    def test_decompile_nlp_signature_type_description(self) -> None:
        """Multiple OR'd POS bits produce pipe-joined description.

        (POS_PROPN | POS_VERB | POS_DET | POS_ADJ | POS_NOUN) << 32
        should produce "<PROPN|VERB|DET|ADJ|NOUN>".
        """
        tok = _mock_nlp_tokenizer()
        decomp = Decompiler(tokenizer=tok)

        nlp_type = POS_PROPN | POS_VERB | POS_DET | POS_ADJ | POS_NOUN
        sig = _make_nlp_sig(nlp_type)

        result = decomp._decode_sig(sig)
        assert result == "<ADJ|DET|NOUN|PROPN|VERB>"

    def test_decompile_nlp_signature_single_pos(self) -> None:
        """A single POS bit produces a single-name description."""
        tok = _mock_nlp_tokenizer()
        decomp = Decompiler(tokenizer=tok)

        sig = _make_nlp_sig(POS_NOUN)
        result = decomp._decode_sig(sig)
        assert result == "<NOUN>"

    def test_decompile_nlp_signature_pos_with_dep(self) -> None:
        """POS and DEP bits both appear in the description.

        DEP_STRUCT corresponds to the "root" dependency in the coarse grouping.
        """
        tok = _mock_nlp_tokenizer()
        decomp = Decompiler(tokenizer=tok)

        nlp_type = POS_VERB | DEP_STRUCT
        sig = _make_nlp_sig(nlp_type)
        result = decomp._decode_sig(sig)
        assert result == "<VERB|DEP_STRUCT>"

    def test_decompile_nlp_signature_full_type(self) -> None:
        """POS, DEP, and MORPH bits all appear in the description."""
        tok = _mock_nlp_tokenizer()
        decomp = Decompiler(tokenizer=tok)

        nlp_type = POS_NOUN | DEP_SUBJ | MORPH_PLUR
        sig = _make_nlp_sig(nlp_type)
        result = decomp._decode_sig(sig)
        assert result == "<NOUN|DEP_SUBJ|MORPH_PLUR>"


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and precedence tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNLPSigEdgeCases:
    """Edge cases: precedence, Mod32 unaffected, zero bits."""

    def test_decompile_mod32_sig_unchanged(self) -> None:
        """A Mod32 signature (high 32 bits zero) does NOT trigger NLP description.

        Mod32 sigs go through tokenizer.decode as before.
        """
        tok = MagicMock()
        tok.supports_mcs = False
        tok.decode = MagicMock(return_value="HELLO")

        decomp = Decompiler(tokenizer=tok)

        # Mod32: high 32 bits are zero, (node >> 32) == 0 for Mod32
        sig = 0x00000000_00000005
        result = decomp._decode_sig(sig)
        assert result == "HELLO"
        # High 32 bits are zero for Mod32 sig
        assert not (sig >> 32) != 0

    def test_decompile_nlp_sig_in_mcs_names(self) -> None:
        """An NLP-type sig that IS in _mcs_names returns the recovered name.

        _mcs_names takes precedence over type description.
        """
        tok = _mock_nlp_tokenizer()
        decomp = Decompiler(tokenizer=tok)

        sig = _make_nlp_sig(POS_NOUN)
        decomp._mcs_names[sig] = "recovered_name"

        result = decomp._decode_sig(sig)
        assert result == "recovered_name"

    def test_decompile_nlp_sig_zero_type_bits(self) -> None:
        """Edge case: NLP node with zero type bits returns <NLP:0>.

        In practice this shouldn't happen (NLP nodes always have non-zero
        high bits), but we handle it gracefully.
        """
        result = describe_nlp_type(0)
        assert result == "<NLP:0>"

    def test_describe_nlp_type_preserves_order(self) -> None:
        """Flag names appear in bit-position order regardless of OR combination."""
        # OR bits out of order — description should still be in bit order
        nlp_type = POS_VERB | POS_NOUN | POS_ADJ  # ADJ=0, NOUN=7, VERB=15
        sig = _make_nlp_sig(nlp_type)
        result = describe_nlp_type(sig)
        assert result == "<ADJ|NOUN|VERB>"

    def test_describe_nlp_type_all_pos_flags(self) -> None:
        """All POS flags at once produce all POS names."""
        all_pos = (POS_ADJ | POS_ADP | POS_ADV | POS_AUX | POS_CCONJ |
                   POS_DET | POS_INTJ | POS_NOUN | POS_NUM | POS_PART |
                   POS_PRON | POS_PROPN | POS_PUNCT | POS_SCONJ | POS_SYM |
                   POS_VERB | POS_X)
        sig = _make_nlp_sig(all_pos)
        result = describe_nlp_type(sig)
        assert result == "<ADJ|ADP|ADV|AUX|CCONJ|DET|INTJ|NOUN|NUM|PART|PRON|PROPN|PUNCT|SCONJ|SYM|VERB|X>"
