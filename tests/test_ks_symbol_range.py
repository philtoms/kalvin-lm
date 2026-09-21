"""Symbol-range regressions: digits and word-internal punctuation.

Identifiers admit alphanumerics plus `- . _ '` (an allowlist — operator
and structural marks `= > < ( ) #` never enter it). A caseless first
character (digit, punctuation) starts a literal word exactly as a
lowercase letter does; ALL-UPPER compounds are alnum-only, so a punctured
identifier never char-decomposes.
"""

from __future__ import annotations

import pytest

from ks.compiler import compile_source
from ks.lexer import Lexer, LexerError


def _decoded(source: str) -> list[str]:
    return [
        f"{kv.kline.dbg.op} | {kv.kline.dbg.decoded}"
        for kv in compile_source(source, dev=True)
    ]


class TestCaselessWords:
    def test_digit_word_node(self):
        assert _decoded("A = 42\n") == ["DENOTES | A:[42]"]

    def test_punctuated_digit_word(self):
        assert _decoded("Pi = 3.14\n") == ["DENOTES | Pi:[3.14]"]

    def test_hyphenated_word(self):
        assert _decoded("X = a-b\n") == ["DENOTES | X:[a-b]"]

    def test_apostrophe_word(self):
        assert _decoded("Word = don't\n") == ["DENOTES | Word:[don't]"]

    def test_caseless_first_sig(self):
        # 7up: caseless-first, no MTS, no expansion — a literal word sig.
        assert _decoded("7up = X\n") == ["DENOTES | 7up:[X]"]

    def test_bare_caseless_word_is_ask(self):
        assert _decoded("42\n") == ["ASK | 42:[42]"]

    def test_caseless_never_attracts(self):
        # The bare digit is a literal word even under a word list
        # containing it — ambient attraction is uppercase-only.
        assert _decoded("(2 fast)\n2 = X\n") == ["DENOTES | 2:[X]"]

    def test_punctuated_word_matches_prose(self):
        # The word-list tier carries punctuated words; the kline-position
        # identifier can now equal them (W attracts don't, the literal
        # node matches, self-denote collapses to identity).
        assert _decoded("(don't stop)\nD = don't\n") == [
            "IDENTITY | don't:[don't]"
        ]


class TestCompoundGuard:
    def test_punctured_identifier_is_not_a_compound(self):
        # A-1 isupper() is True (uncased chars ignored) but not alnum:
        # no MTS canon, just the word.
        assert _decoded("A-1 = X\n") == ["DENOTES | A-1:[X]"]

    def test_alnum_compound_still_fires(self):
        assert "CANONICALISES | M2:[M, 2]" in _decoded("M2 = X\n")


class TestReservedSymbols:
    def test_operators_still_parse_after_punctuated_idents(self):
        toks = Lexer("a-b = X\n").tokenize()
        assert [(t.type.name, t.value) for t in toks][:4] == [
            ("SIGNATURE", "a-b"),
            ("DENOTES", "="),
            ("SIGNATURE", "X"),
            ("NEWLINE", "\n"),
        ]

    def test_countersigns_unaffected(self):
        toks = Lexer("x == y\n").tokenize()
        assert (toks[1].type.name, toks[1].value) == ("COUNTERSIGNS", "==")

    def test_comment_mark_still_trailing_comment(self):
        toks = Lexer("a-b#note\n").tokenize()
        assert [(t.type.name, t.value) for t in toks] == [
            ("SIGNATURE", "a-b"),
            ("NEWLINE", "\n"),
            ("EOF", ""),
        ]

    def test_parens_still_annotation(self):
        toks = Lexer("D(ob) = 42\n").tokenize()
        assert [(t.type.name, t.value) for t in toks][:4] == [
            ("SIGNATURE", "D"),
            ("ANNOTATION", "(ob)"),
            ("DENOTES", "="),
            ("SIGNATURE", "42"),
        ]

    def test_held_back_symbols_still_rejected(self):
        for ch in "?!,:;":
            with pytest.raises(LexerError):
                Lexer(f"A = x{ch}\n").tokenize()

    def test_reserved_punctuated_word_via_bracketed_witness(self):
        # `? ! ,` stay reserved: punctuated words are written bracketed
        # (the tail is free-form text), and attraction already matches them.
        assert _decoded("w(hat?) = Q\n") == ["DENOTES | what?:[Q]"]
        assert _decoded("(what? yes!)\nW = X\n") == ["DENOTES | what?:[X]"]
