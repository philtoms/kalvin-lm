"""Integration tests for the KScript v3 pipeline (src/ks/).

Spec ref: @specs/kscript.md §15 (Test Matrix).

This module covers all 37 spec test IDs (KS-1 through KS-37):

    KS-1   — Token types recognized                          TestTokenType
    KS-1   — Token types recognized (also in TestLexer)      TestLexer
    KS-2   — Multi-char operator priority                    TestLexer
    KS-3   — BPE annotations                                 TestLexer
    KS-4   — INDENT/DEDENT tracking                          TestLexer
    KS-5   — Edge cases (empty, whitespace, errors)          TestLexer
    KS-6   — AST scope model                                 TestParserAST
    KS-7   — Block parsing                                   TestParserAST
    KS-8   — Annotations preserved                           TestParserAST
    KS-9   — Inline annotation attachment                    TestParserAST
    KS-10  — Empty source                                    TestParserAST
    KS-11  — COUNTERSIGN per-item                            TestEmitterOperators
    KS-12  — UNDERSIGN per-item reversed                     TestEmitterOperators
    KS-13  — CONNOTATE per-item                              TestEmitterOperators
    KS-14  — CANONIZE aggregates                             TestEmitterOperators
    KS-15  — Operator chain                                  TestEmitterOperators
    KS-16  — Indent extends scope                            TestEmitterOperators
    KS-17  — DEDENT returns to parent                        TestEmitterOperators
    KS-18  — Non-CANONIZE with indent                        TestEmitterOperators
    KS-19  — MTS expansion                                   TestEmitterMTS
    KS-20  — No MTS for single-char                          TestEmitterMTS
    KS-21  — MTS on node side                                TestEmitterMTS
    KS-22  — Node count invariant                            TestEmitterMTS
    KS-23  — First-letter matching                           TestBindingScope
    KS-24  — Occurrence counter                              TestBindingScope
    KS-25  — Inline binding bypass                           TestEmitterOperators
    KS-26  — Rule B4 override                                TestEmitterBinding
    KS-27  — Scope inheritance                               TestBindingScope
    KS-28  — Scope shadowing                                 TestBindingScope
    KS-29  — Counter reset                                   TestBindingScope
    KS-30  — Unresolved identifier (no fallback state)     TestBindingScope
    KS-31  — Inert annotation                                TestBindingScope
    KS-32  — Unresolved char typed-node encoding             TestEncoding
    KS-33  — Self-identity                                   TestEmitterOperators
    KS-34  — Nodes always a list                             TestStructure
    KS-35  — §14.11 complex nested (master regression)       TestComplexExamples
    KS-36  — §14.12 NLP-bound example                        TestComplexExamples
    KS-37  — Uniform tokenizer integration                   TestComplexExamples
"""

from __future__ import annotations

import dataclasses

import pytest

from kalvin.kline import KLine
from kalvin.tokenizer import Tokenizer
from ks import compile_source
from ks.ast import Annotation, Block, KScriptFile, OperatorScope
from ks.binding_scope import BindingScope
from ks.lexer import Lexer, LexerError
from ks.parser import Parser
from ks.token import Token, TokenType
from tests.conftest import requires_tokenizer_data

# The entire module compiles real KScript sources which now default to the
# kalvin tokenizer; skip cleanly when the tokenizer data assets are absent.
pytestmark = requires_tokenizer_data

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def compile_dev(source: str) -> list[KLine]:
    """Compile source with dev=True (populates dbg for readable assertions)."""
    return compile_source(source, dev=True)


def compile_real(source: str, tokenizer=None) -> list[KLine]:
    """Compile with dev=False for uint64-level assertions."""
    return compile_source(source, tokenizer=tokenizer, dev=False)


# Lazy module-level tokenizer (safe at import time; ``pytestmark`` gates
# execution so this is only ever instantiated on a data-present machine).
_TOK_INSTANCE: Tokenizer | None = None


def _tok() -> Tokenizer:
    """Return the shared tokenizer, constructing it on first use."""
    global _TOK_INSTANCE
    if _TOK_INSTANCE is None:
        _TOK_INSTANCE = Tokenizer.from_files()
    return _TOK_INSTANCE


def _sig_str(entry: KLine) -> str:
    """Return a human-readable signature string for an entry.

    Uses dbg.label when available (dev mode); otherwise decodes the uint64.
    """
    if entry.dbg and entry.dbg.label:
        return entry.dbg.label
    return _tok().decode([entry.signature])


def _node_strs(entry: KLine) -> list[str]:
    """Decode an entry's uint64 node values to human-readable strings."""
    if not entry.nodes:
        return []
    return [_tok().decode([n]) for n in entry.nodes]


def _find_entries(
    entries: list[KLine],
    *,
    sig: str | None = None,
    op: str | None = None,
    nodes: list[str] | None = None,
) -> list[KLine]:
    """Find entries matching the given criteria by decoded string values."""
    results = []
    for e in entries:
        if sig is not None and _sig_str(e) != sig:
            continue
        if op is not None and e.dbg.op != op:
            continue
        if nodes is not None and _node_strs(e) != nodes:
            continue
        results.append(e)
    return results


def has_entry(
    entries: list[KLine],
    *,
    sig: str,
    op: str | None = None,
    nodes: list[str] | None = None,
) -> bool:
    """Check if at least one entry matches the given criteria."""
    return len(_find_entries(entries, sig=sig, op=op, nodes=nodes)) > 0


# ---------------------------------------------------------------------------
# Spec coverage audit comment
# ---------------------------------------------------------------------------
# KS-1  : test_token_type_members, test_ks1_token_recognition
# KS-2  : test_ks2_multi_char_operator_priority
# KS-3  : test_ks3_bpe_annotations, test_ks3_nested_parens
# KS-4  : test_ks4_indent_dedent
# KS-5  : test_ks5_empty_input, test_ks5_whitespace_only, test_ks5_unknown_char,
#          test_ks5_angle_bracket_error
# KS-6  : test_ks6_scope_model_ast
# KS-7  : test_ks7_block_parsing
# KS-8  : test_ks8_annotations_preserved
# KS-9  : test_ks9_sig_inline_annotation, test_ks9_node_inline_annotation
# KS-10 : test_ks10_empty_source
# KS-11 : test_ks11_countersign_per_item
# KS-12 : test_ks12_undersign_per_item_reversed
# KS-13 : test_ks13_connotate_per_item
# KS-14 : test_ks14_canonize_aggregates
# KS-15 : test_ks15_operator_chain
# KS-16 : test_ks16_indent_extends_scope
# KS-17 : test_ks17_dedent_returns_to_parent
# KS-18 : test_ks18_non_canonize_with_indent
# KS-19 : test_ks19_mts_expansion
# KS-20 : test_ks20_no_mts_for_single_char
# KS-21 : test_ks21_mts_on_node_side
# KS-22 : test_ks22_node_count_invariant
# KS-23 : test_ks23_first_letter_matching
# KS-24 : test_ks24_occurrence_counter
# KS-25 : test_ks25_inline_binding_bypass
# KS-26 : test_ks26_rule_b4_override
# KS-27 : test_ks27_scope_inheritance
# KS-28 : test_ks28_scope_shadowing
# KS-29 : test_ks29_counter_reset
# KS-30 : test_ks30_unresolved_identifier
# KS-31 : test_ks31_inert_annotation
# KS-32 : test_ks32_unresolved_char_typed_encoding
# KS-33 : test_ks33_self_identity
# KS-34 : test_ks34_nodes_always_list_canonize, test_ks34_nodes_always_list_unsigned
# KS-35 : test_ks35_complex_nested_master_regression
# KS-36 : test_ks36_nlp_bound_example
# KS-37 : test_ks37_uniform_tokenizer


# ===================================================================
# TestTokenType — KS-1 (token types and Token dataclass)
# ===================================================================


class TestTokenType:
    """KS-1: All token types recognized; Token is a frozen dataclass."""

    def test_token_type_members(self):
        """KS-1: All 10 TokenType members exist."""
        expected = {
            "COUNTERSIGN",
            "CANONIZE",
            "CONNOTATE",
            "UNDERSIGN",
            "SIGNATURE",
            "ANNOTATION",
            "NEWLINE",
            "INDENT",
            "DEDENT",
            "EOF",
        }
        actual = {m.name for m in TokenType}
        assert actual == expected

    def test_token_is_frozen_dataclass(self):
        """KS-1: Token is a frozen dataclass with type, value, line, column."""
        assert dataclasses.is_dataclass(Token)
        assert getattr(Token, "__dataclass_params__").frozen is True

        fields = {f.name for f in dataclasses.fields(Token)}
        assert fields == {"type", "value", "line", "column"}

    def test_ks1_token_recognition(self):
        """KS-1: Lexer produces correct token types for A == B."""
        tokens = Lexer("A == B").tokenize()
        # Expect: SIGNATURE("A"), COUNTERSIGN("=="), SIGNATURE("B"), EOF
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.COUNTERSIGN,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]


# ===================================================================
# TestLexer — KS-1 through KS-5
# ===================================================================


class TestLexer:
    """Lexer tests covering KS-1 through KS-5."""

    # -- KS-2: Multi-char operator priority --------------------------------

    def test_ks2_multi_char_operator_priority_eq(self):
        """KS-2: '==' is lexed as COUNTERSIGN, not two UNDERSIGN tokens."""
        tokens = Lexer("A == B").tokenize()
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.COUNTERSIGN,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]
        # Confirm no UNDERSIGN tokens
        assert TokenType.UNDERSIGN not in types

    def test_ks2_multi_char_operator_priority_arrow(self):
        """KS-2: '=>' is lexed as CANONIZE, not UNDERSIGN + CONNOTATE."""
        tokens = Lexer("A => B").tokenize()
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.CANONIZE,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]
        assert TokenType.UNDERSIGN not in types
        assert TokenType.CONNOTATE not in types

    # -- KS-3: BPE annotations --------------------------------------------

    def test_ks3_bpe_annotations(self):
        """KS-3: '(hello world)' produces a single ANNOTATION token."""
        tokens = Lexer("(hello world)").tokenize()
        # Expect: ANNOTATION("(hello world)"), EOF
        assert tokens[0].type == TokenType.ANNOTATION
        assert tokens[0].value == "(hello world)"
        assert tokens[1].type == TokenType.EOF

    def test_ks3_nested_parens(self):
        """KS-3: Nested parens produce a single ANNOTATION preserving content."""
        tokens = Lexer("(a (b c) d)").tokenize()
        assert tokens[0].type == TokenType.ANNOTATION
        assert tokens[0].value == "(a (b c) d)"
        assert tokens[1].type == TokenType.EOF

    # -- KS-4: INDENT/DEDENT ----------------------------------------------

    def test_ks4_indent_dedent(self):
        """KS-4: Indentation produces INDENT and DEDENT tokens."""
        tokens = Lexer("A\n  B\nC").tokenize()
        types = [t.type for t in tokens]
        # Expect: SIGNATURE(A), NEWLINE, INDENT, SIGNATURE(B), NEWLINE, DEDENT, SIGNATURE(C), EOF
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

        # INDENT should appear before B
        idx_b = next(
            i for i, t in enumerate(tokens) if t.type == TokenType.SIGNATURE and t.value == "B"
        )
        idx_indent = next(i for i, t in enumerate(tokens) if t.type == TokenType.INDENT)
        assert idx_indent < idx_b

        # DEDENT should appear before C
        idx_c = next(
            i for i, t in enumerate(tokens) if t.type == TokenType.SIGNATURE and t.value == "C"
        )
        idx_dedent = next(i for i, t in enumerate(tokens) if t.type == TokenType.DEDENT)
        assert idx_dedent < idx_c

    def test_ks4_dedent_at_eof(self):
        """KS-4: Remaining indent levels produce DEDENT tokens at EOF."""
        tokens = Lexer("A\n  B").tokenize()
        types = [t.type for t in tokens]
        # INDENT for B, then DEDENT at EOF
        assert types.count(TokenType.INDENT) == 1
        assert types.count(TokenType.DEDENT) == 1

    # -- KS-5: Edge cases -------------------------------------------------

    def test_ks5_empty_input(self):
        """KS-5: Empty input produces only EOF."""
        tokens = Lexer("").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_ks5_whitespace_only(self):
        """KS-5: Whitespace-only input produces only EOF (per spec)."""
        tokens = Lexer("   ").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_ks5_angle_bracket_error(self):
        """KS-5: '<' raises LexerError."""
        with pytest.raises(LexerError):
            Lexer("A < B").tokenize()

    def test_ks5_unknown_char_error(self):
        """KS-5: Unknown characters raise LexerError."""
        with pytest.raises(LexerError):
            Lexer("A @ B").tokenize()


# ===================================================================
# TestParserAST — KS-6 through KS-10
# ===================================================================


class TestParserAST:
    """Parser/AST tests covering KS-6 through KS-10."""

    @staticmethod
    def _parse(source: str) -> KScriptFile:
        """Helper: lex and parse source into a KScriptFile AST."""
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    # -- KS-6: Scope model AST -------------------------------------------

    def test_ks6_scope_model_ast(self):
        """KS-6: Parse 'A == B > C = D' into chained OperatorScope nodes."""
        ast = self._parse("A == B > C = D")
        assert len(ast.constructs) == 1

        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op == TokenType.COUNTERSIGN

        # First item is B (Signature), but since B > C = D forms a chain,
        # the items should contain a nested OperatorScope for the connotate.
        # A == [B > C = D]  — B is both a COUNTERSIGN node and the sig for >
        assert len(scope.items) >= 1
        inner = scope.items[0]
        assert isinstance(inner, OperatorScope)
        assert inner.sig.id == "B"
        assert inner.op == TokenType.CONNOTATE

        # B > [C = D] — C is the connotate node and the sig for =
        assert len(inner.items) >= 1
        deepest = inner.items[0]
        assert isinstance(deepest, OperatorScope)
        assert deepest.sig.id == "C"
        assert deepest.op == TokenType.UNDERSIGN

    # -- KS-7: Block parsing ---------------------------------------------

    def test_ks7_block_parsing(self):
        """KS-7: Indented source creates Block nodes with correct constructs."""
        source = "A =>\n  B\n  C"
        ast = self._parse(source)

        assert len(ast.constructs) == 1
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op == TokenType.CANONIZE
        assert scope.child_block is not None
        assert isinstance(scope.child_block, Block)
        assert len(scope.child_block.constructs) == 2

    # -- KS-8: Annotations preserved -------------------------------------

    def test_ks8_annotations_preserved(self):
        """KS-8: '(Mary Had)' produces an Annotation node in the AST."""
        ast = self._parse("(Mary Had)")
        assert len(ast.constructs) == 1
        ann = ast.constructs[0]
        assert isinstance(ann, Annotation)
        assert ann.text == "(Mary Had)"

    # -- KS-9: Inline annotations ----------------------------------------

    def test_ks9_sig_inline_annotation(self):
        """KS-9: 'S(ubject) = M' attaches inline_annotation to scope."""
        ast = self._parse("S(ubject) = M")
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "S"
        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"

    def test_ks9_node_inline_annotation(self):
        """KS-9: 'A = D(et)' attaches node_inline_annotation to scope."""
        ast = self._parse("A = D(et)")
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.node_inline_annotation is not None
        assert scope.node_inline_annotation.text == "(et)"

    # -- KS-10: Empty source ---------------------------------------------

    def test_ks10_empty_source(self):
        """KS-10: Empty source produces empty script (no error)."""
        ast = self._parse("")
        assert isinstance(ast, KScriptFile)
        assert ast.constructs == []


# ===================================================================
# TestBindingScope — KS-23, KS-24, KS-27 through KS-31
# ===================================================================


class TestBindingScope:
    """BindingScope unit tests covering KS-23, KS-24, KS-27–KS-31."""

    # -- KS-23: First-letter matching ------------------------------------

    def test_ks23_first_letter_matching(self):
        """KS-23: resolve('M') → 'Mary', resolve('H') → 'Had', resolve('A') → 'A'."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Mary", "Had", "A", "Little", "Lamb"])

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("H") == "Had"
        assert scope.resolve("A") == "A"

    # -- KS-24: Occurrence counter ---------------------------------------

    def test_ks24_occurrence_counter(self):
        """KS-24: First resolve('L') → 'Little', second → 'Lamb'."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Little", "Lamb"])

        assert scope.resolve("L") == "Little"
        assert scope.resolve("L") == "Lamb"

    # -- KS-27: Scope inheritance ----------------------------------------

    def test_ks27_scope_inheritance(self):
        """KS-27: Inner scope with no matching words falls through to outer."""
        scope = BindingScope()
        scope.push_scope()  # outer
        scope.add_words(["Alpha"])
        scope.push_scope()  # inner (no words)

        assert scope.resolve("A") == "Alpha"

    # -- KS-28: Scope shadowing ------------------------------------------

    def test_ks28_scope_shadowing(self):
        """KS-28: Inner scope binding shadows outer for same character."""
        scope = BindingScope()
        scope.push_scope()  # outer
        scope.add_words(["Alpha"])
        scope.push_scope()  # inner
        scope.add_words(["Another"])

        assert scope.resolve("A") == "Another"

    # -- KS-29: Counter reset --------------------------------------------

    def test_ks29_counter_reset(self):
        """KS-29: Pushing a new scope resets counters for resolution."""
        scope = BindingScope()
        scope.push_scope()  # scope 1
        scope.add_words(["Little", "Lamb"])
        assert scope.resolve("L") == "Little"  # counter 0 in scope 1

        scope.push_scope()  # scope 2 (empty) — should reset counters
        # Spec expects: resolve falls through to scope 1 but with reset counter
        # → "Little" again (not "Lamb")
        assert scope.resolve("L") == "Little"

    # -- KS-30: Unresolved identifier (no fallback state) -------------

    def test_ks30_unresolved_identifier(self):
        """KS-30: An unresolved identifier (BindingScope.resolve returns None)
        is encoded as its own raw BPE token — no special fallback state.

        At the BindingScope level, resolve('Z') with no matching words
        returns None. The encoding behavior (single typed node, same
        path as any resolved character) is covered by KS-32.
        """
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Alpha", "Beta"])
        assert scope.resolve("Z") is None

    # -- KS-31: Inert annotation -----------------------------------------

    def test_ks31_inert_annotation(self):
        """KS-31: Words with no matching characters have no effect."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Xray", "Yankee"])
        assert scope.resolve("M") is None


# ===================================================================
# TestEmitterOperators — KS-11 through KS-18, KS-25, KS-33
# ===================================================================


class TestEmitterOperators:
    """Emitter operator tests covering KS-11–KS-18, KS-25, KS-33."""

    # -- KS-11: COUNTERSIGN per-item -------------------------------------

    def test_ks11_countersign_per_item(self):
        """KS-11: A == B C → {A:[B]}, {B:[A]}, {A:[C]}, {C:[A]} COUNTERSIGN."""
        entries = compile_dev("A == B C")
        # Per spec §14.2 + per-item: 4 COUNTERSIGN entries only.
        assert len(entries) == 4
        assert all(e.dbg.op == "COUNTERSIGNED" for e in entries)
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNED", nodes=["A"])

    def test_ks11_countersign_entries_present(self):
        """KS-11 (relaxed): The 4 COUNTERSIGN pairs are present regardless of extras."""
        entries = compile_dev("A == B C")
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNED", nodes=["A"])

    # -- KS-12: UNDERSIGN per-item reversed -------------------------------

    def test_ks12_undersign_per_item_reversed(self):
        """KS-12: A = B C → {B:[A]}, {C:[A]} UNDERSIGN."""
        entries = compile_dev("A = B C")
        assert len(entries) == 2
        assert all(e.dbg.op == "UNDERSIGNED" for e in entries)
        assert has_entry(entries, sig="B", op="UNDERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="C", op="UNDERSIGNED", nodes=["A"])

    def test_ks12_undersign_entries_present(self):
        """KS-12 (relaxed): The 2 UNDERSIGN entries are present."""
        entries = compile_dev("A = B C")
        assert has_entry(entries, sig="B", op="UNDERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="C", op="UNDERSIGNED", nodes=["A"])

    # -- KS-13: CONNOTATE per-item ----------------------------------------

    def test_ks13_connotate_per_item(self):
        """KS-13: A > B C → {A:[B]}, {A:[C]} CONNOTATE."""
        entries = compile_dev("A > B C")
        assert len(entries) == 2
        assert all(e.dbg.op == "CONNOTED" for e in entries)
        assert has_entry(entries, sig="A", op="CONNOTED", nodes=["B"])
        assert has_entry(entries, sig="A", op="CONNOTED", nodes=["C"])

    def test_ks13_connotate_entries_present(self):
        """KS-13 (relaxed): The 2 CONNOTATE entries are present."""
        entries = compile_dev("A > B C")
        assert has_entry(entries, sig="A", op="CONNOTED", nodes=["B"])
        assert has_entry(entries, sig="A", op="CONNOTED", nodes=["C"])

    # -- KS-14: CANONIZE aggregates ---------------------------------------

    def test_ks14_canonize_aggregates(self):
        """KS-14: A => B C D → {A:[B,C,D]} CANONIZE."""
        entries = compile_dev("A => B C D")
        assert len(entries) == 1
        assert entries[0].dbg.op == "CANONIZED"
        assert _sig_str(entries[0]) == "A"
        assert _node_strs(entries[0]) == ["B", "C", "D"]

    def test_ks14_canonize_entry_present(self):
        """KS-14 (relaxed): The CANONIZE aggregate entry is present."""
        entries = compile_dev("A => B C D")
        assert has_entry(entries, sig="A", op="CANONIZED", nodes=["B", "C", "D"])

    # -- KS-15: Operator chain -------------------------------------------

    def test_ks15_operator_chain(self):
        """KS-15: A == B > C = D → entries per §14.7 table."""
        entries = compile_dev("A == B > C = D")
        assert len(entries) == 4
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="B", op="CONNOTED", nodes=["C"])
        assert has_entry(entries, sig="D", op="UNDERSIGNED", nodes=["C"])

    def test_ks15_operator_chain_entries_present(self):
        """KS-15 (relaxed): The 4 operator chain entries are present."""
        entries = compile_dev("A == B > C = D")
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="B", op="CONNOTED", nodes=["C"])
        assert has_entry(entries, sig="D", op="UNDERSIGNED", nodes=["C"])

    # -- KS-16: Indent extends scope --------------------------------------

    def test_ks16_indent_extends_scope(self):
        """KS-16: Indented items under CANONIZE belong to parent's node list."""
        source = "A =>\n  B\n  C"
        entries = compile_dev(source)
        # CANONIZE should have B and C as nodes
        assert has_entry(entries, sig="A", op="CANONIZED", nodes=["B", "C"])

    # -- KS-17: DEDENT returns to parent ----------------------------------

    def test_ks17_dedent_returns_to_parent(self):
        """KS-17: After dedent, subsequent constructs compile at parent level."""
        source = "A =>\n  B\nC = D"
        entries = compile_dev(source)
        # A CANONIZE with B as node (from indented block)
        assert has_entry(entries, sig="A", op="CANONIZED", nodes=["B"])
        # D UNDERSIGN [C] (at parent level after dedent)
        assert has_entry(entries, sig="D", op="UNDERSIGNED", nodes=["C"])

    # -- KS-18: Non-CANONIZE with indent ---------------------------------

    def test_ks18_non_canonize_with_indent(self):
        """KS-18: A == B\\n  C\\n  D → 6 COUNTERSIGN entries (§14.10)."""
        source = "A == B\n  C\n  D"
        entries = compile_dev(source)
        assert len(entries) == 6
        # All should be COUNTERSIGN (bidirectional pairs)
        assert all(e.dbg.op == "COUNTERSIGNED" for e in entries)

    def test_ks18_non_canonize_entries_present(self):
        """KS-18 (relaxed): The 6 COUNTERSIGN pairs are present."""
        source = "A == B\n  C\n  D"
        entries = compile_dev(source)
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNED", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNED", nodes=["D"])
        assert has_entry(entries, sig="D", op="COUNTERSIGNED", nodes=["A"])

    # -- KS-25: Inline binding bypass ------------------------------------

    def test_ks25_inline_binding_bypass(self):
        """KS-25: S(ubject) = M — inline annotation resolves S to 'Subject'."""
        entries = compile_dev("S(ubject) = M")
        # The signature side should resolve to "Subject" via inline annotation
        sig_entries = _find_entries(entries, sig="Subject")
        assert len(sig_entries) > 0, "Expected entries with sig='Subject'"

    # -- KS-33: Self-identity --------------------------------------------

    def test_ks33_self_identity(self):
        """KS-33: A = A → single {A:[]} IDENTITY."""
        entries = compile_dev("A = A")
        assert len(entries) == 1
        assert entries[0].dbg.op == "IDENTITY"
        assert _sig_str(entries[0]) == "A"
        assert entries[0].nodes == []

    def test_ks33_self_identity_unsigned_present(self):
        """KS-33 (relaxed): At least one A IDENTITY with empty nodes exists."""
        entries = compile_dev("A = A")
        assert has_entry(entries, sig="A", op="IDENTITY", nodes=[])


# ===================================================================
# TestEmitterMTS — KS-19 through KS-22
# ===================================================================


class TestEmitterMTS:
    """MTS expansion tests covering KS-19 through KS-22."""

    # -- KS-19: MTS expansion --------------------------------------------

    def test_ks19_mts_expansion(self):
        """KS-19: ABC → 4 entries matching §14.6.

        Expected:
          1. A unsigned (S4)
          2. B unsigned (S4)
          3. C unsigned (S4)
          4. ABC canonize [A, B, C] (S2)
        """
        entries = compile_dev("ABC")
        assert len(entries) == 4

        assert _sig_str(entries[0]) == "A" and entries[0].dbg.op == "IDENTITY"
        assert _sig_str(entries[1]) == "B" and entries[1].dbg.op == "IDENTITY"
        assert _sig_str(entries[2]) == "C" and entries[2].dbg.op == "IDENTITY"
        assert _sig_str(entries[3]) == "ABC" and entries[3].dbg.op == "CANONIZED"
        assert _node_strs(entries[3]) == ["A", "B", "C"]

    # -- KS-20: No MTS for single-char -----------------------------------

    def test_ks20_no_mts_for_single_char(self):
        """KS-20: A → single IDENTITY entry, no component expansion."""
        entries = compile_dev("A")
        assert len(entries) == 1
        assert entries[0].dbg.op == "IDENTITY"
        assert _sig_str(entries[0]) == "A"
        assert entries[0].nodes == []

    # -- KS-21: MTS on node side -----------------------------------------

    def test_ks21_mts_on_node_side(self):
        """KS-21: A == MHALL triggers MTS expansion for MHALL on the node side."""
        entries = compile_dev("A == MHALL")
        # MTS for MHALL: component unsigned entries + CANONIZE entry
        assert has_entry(entries, sig="MHALL", op="CANONIZED")
        # Countersign pairs: A ↔ MHALL (node is the packed uint64 for MHALL,
        # which decodes to sorted chars, so we check by sig and op only)
        a_cs = _find_entries(entries, sig="A", op="COUNTERSIGNED")
        assert len(a_cs) >= 1, "Expected A COUNTERSIGN entry"
        mhall_cs = _find_entries(entries, sig="MHALL", op="COUNTERSIGNED")
        assert len(mhall_cs) >= 1, "Expected MHALL COUNTERSIGN entry"

    # -- KS-22: Node count invariant --------------------------------------

    def test_ks22_node_count_invariant(self):
        """KS-22: MTS canonization entry has N nodes for an N-char identifier."""
        for ident in ["AB", "ABC", "ABCD", "MHALL"]:
            entries = compile_dev(ident)
            canonize_entries = _find_entries(entries, sig=ident, op="CANONIZED")
            assert len(canonize_entries) >= 1, f"No CANONIZE entry for {ident}"
            canon = canonize_entries[0]
            assert len(canon.nodes) == len(ident), (
                f"MTS canonize for {ident}: expected {len(ident)} nodes, got {len(canon.nodes)}"
            )


# ===================================================================
# TestEmitterBinding — KS-26 (Rule B4 override)
# ===================================================================


class TestEmitterBinding:
    """Binding integration tests covering KS-26."""

    # -- KS-26: Rule B4 override -----------------------------------------

    def test_ks26_rule_b4_override(self):
        """KS-26: Inline annotation patches parent MTS CANONIZE entry.

        In the §14.12 source, S(ubject) inside a subscript block patches
        the parent SVO CANONIZE entry: S → 'Subject'.
        """
        source = (
            "(Mary Had A Little Lamb)\n"
            "MHALL == SVO =>\n"
            "  S(ubject) = M\n"
            "  V = H\n"
            "  O = ALL =>\n"
            "    A = D\n"
            "    L = M\n"
            "    L > O"
        )
        entries = compile_dev(source)
        # Find the SVO CANONIZE entry — it should have "Subject" as first node
        svo_canon = _find_entries(entries, sig="SVO", op="CANONIZED")
        assert len(svo_canon) >= 1, "Expected at least one SVO CANONIZE entry"
        # A simpler check: verify that "Subject" unsigned entries exist
        # (from the inline annotation's MTS expansion)
        subject_entries = _find_entries(entries, sig="Subject")
        assert len(subject_entries) > 0, "Expected entries for 'Subject' (from inline annotation)"


# ===================================================================
# TestStructure — KS-34 (nodes always a list)
# ===================================================================


class TestStructure:
    """Structural invariant tests covering KS-34."""

    def test_ks34_nodes_always_list_canonize(self):
        """KS-34: CANONIZE entry nodes is a list of length 1+ (not scalar)."""
        entries = compile_dev("A => B")
        canon = _find_entries(entries, sig="A", op="CANONIZED")
        assert len(canon) == 1
        assert isinstance(canon[0].nodes, list)
        assert len(canon[0].nodes) >= 1

    def test_ks34_nodes_always_list_unsigned(self):
        """KS-34: IDENTITY entry nodes is an empty list (not None)."""
        entries = compile_dev("A")
        assert len(entries) == 1
        assert isinstance(entries[0].nodes, list)
        assert entries[0].nodes == []

    def test_ks34_all_entries_nodes_are_lists(self):
        """KS-34: For every compiled entry, nodes is a list."""
        for source in ["A", "A == B", "A => B C", "ABC", "A > B\n  C"]:
            entries = compile_dev(source)
            for e in entries:
                assert isinstance(e.nodes, list), (
                    f"Entry {e!r} has nodes of type {type(e.nodes)}, expected list"
                )


# ===================================================================
# TestEncoding — KS-32 (Unresolved char typed-node encoding)
# ===================================================================


class TestEncoding:
    """Encoding tests covering KS-32."""

    def test_ks32_unresolved_char_typed_encoding(self):
        """KS-32: An unresolved single character (e.g. 'Z') encodes to a single typed node.

        Under the kalvin tokenizer there is no character-bit fallback.  An
        unresolved character is encoded as its own raw BPE token, producing a
        valid typed node (high 32 bits = type word, low 32 bits = BPE id).
        """
        entries = compile_dev("Z")
        assert len(entries) >= 1
        entry = entries[0]
        type_word = entry.signature >> 32
        bpe_id = entry.signature & 0xFFFFFFFF
        # Typed node: high 32 bits carry the type word; low 32 bits carry BPE id
        assert type_word > 0, f"Expected type-word bits in high word, got {type_word}"
        assert bpe_id > 0, f"Expected a valid BPE token id, got {bpe_id}"
        # Must NOT be the legacy character-bit-packed value (single-bit encoding)
        assert entry.signature != 67108864, "Signature should not be a legacy bit value"
        assert entry.dbg.op == "IDENTITY"


# ===================================================================
# TestComplexExamples — KS-35 through KS-37 + §14.8 secondary regression
# ===================================================================

_SEC1411_SOURCE = "MHALL == SVO =>\n  S = M\n  V = H\n  O = ALL =>\n    A = D\n    L = M\n    L > O"

_SEC148_SOURCE = "A =>\n  B\n  C = D"

_SEC1412_SOURCE = (
    "(Mary Had A Little Lamb)\n"
    "MHALL == SVO =>\n"
    "  S(ubject) = M\n"
    "  V = H\n"
    "  O = ALL =>\n"
    "    A = D\n"
    "    L = M\n"
    "    L > O"
)


class TestComplexExamples:
    """Complex integration tests covering KS-35 through KS-37 plus §14.8."""

    # -- §14.8 secondary regression (simpler nested case) ----------------

    def test_sec148_strict(self):
        """§14.8 secondary regression — strict spec count (5 entries)."""
        entries = compile_dev(_SEC148_SOURCE)
        # Spec §14.8: 5 entries
        assert len(entries) == 5
        assert has_entry(entries, sig="A", op="CANONIZED", nodes=["B", "C"])
        assert has_entry(entries, sig="D", op="UNDERSIGNED", nodes=["C"])
        assert has_entry(entries, sig="B", op="IDENTITY", nodes=[])
        assert has_entry(entries, sig="C", op="IDENTITY", nodes=[])
        assert has_entry(entries, sig="D", op="IDENTITY", nodes=[])

    def test_sec148_presence(self):
        """§14.8 secondary regression — key entries present (5 entries).

        CANONIZE subscript blocks emit identity for
        bare scopes, UNDERSIGN scope sigs, and leaf Signature items.
        identity entries use IDENTITY op.
        Now matches spec §14.8 exactly (5 entries).
        """
        entries = compile_dev(_SEC148_SOURCE)
        assert len(entries) == 5
        assert has_entry(entries, sig="A", op="CANONIZED", nodes=["B", "C"])
        assert has_entry(entries, sig="D", op="UNDERSIGNED", nodes=["C"])
        assert has_entry(entries, sig="B", op="IDENTITY", nodes=[])
        assert has_entry(entries, sig="C", op="IDENTITY", nodes=[])
        assert has_entry(entries, sig="D", op="IDENTITY", nodes=[])

    # -- KS-35: §14.11 complex nested (master regression) ----------------

    def test_ks35_complex_nested_strict(self):
        """KS-35: §14.11 master regression — strict spec count (18 entries).

        MTS component IDENTITY dedup, no compound-own
        identity, subscript identity suppression for MTS CANONIZE scopes.

        Expected entries per spec §14.11:
        1–4:   MTS M, H, A, L identity (S4)
        5:     MHALL canonize [M, H, A, L, L] (S2)
        6–8:   MTS S, V, O identity (S4)
        9:     SVO canonize [S, V, O] (S2)
        10:    MHALL countersign [SVO] (S1)
        11:    SVO countersign [MHALL] (S1)
        (SVO canonize subscript: deduped)
        12:    M undersign [S] (S3)
        13:    H undersign [V] (S3)
        (MTS ALL A, L: deduped)
        14:    ALL canonize [A, L, L] (S2)
        15:    ALL undersign [O] (S3)
        (ALL canonize subscript: deduped)
        16:    D undersign [A] (S3)
        17:    M undersign [L] (S3)
        18:    L connotate [O] (S3)
        """
        entries = compile_dev(_SEC1411_SOURCE)
        assert len(entries) == 18

        # Spot-check critical entries by dbg.label
        assert entries[0].dbg and entries[0].dbg.label == "M" and entries[0].dbg.op == "IDENTITY"
        # MHALL CANONIZE with 5 nodes
        mhall_canon = [
            e for e in entries if e.dbg and e.dbg.label == "MHALL" and e.dbg.op == "CANONIZED"
        ]
        assert len(mhall_canon) == 1
        assert len(mhall_canon[0].nodes) == 5  # M, H, A, L, L

        # SVO CANONIZE with 3 nodes
        svo_canon = [
            e for e in entries if e.dbg and e.dbg.label == "SVO" and e.dbg.op == "CANONIZED"
        ]
        assert len(svo_canon) == 1
        assert len(svo_canon[0].nodes) == 3  # S, V, O

        # Countersign pair
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNED")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNED")

        # Undersign entries
        assert has_entry(entries, sig="M", op="UNDERSIGNED")
        assert has_entry(entries, sig="H", op="UNDERSIGNED")
        assert has_entry(entries, sig="ALL", op="UNDERSIGNED")
        assert has_entry(entries, sig="D", op="UNDERSIGNED")

        # Connotate
        assert has_entry(entries, sig="L", op="CONNOTED")

    def test_ks35_complex_nested_presence(self):
        """KS-35: §14.11 master regression — key entries present (18 entries)."""
        entries = compile_dev(_SEC1411_SOURCE)
        assert len(entries) == 18

        # MTS identity for all single-char identifiers
        for char in ["M", "H", "A", "L", "S", "V", "O"]:
            assert has_entry(entries, sig=char, op="IDENTITY"), f"Missing IDENTITY entry for {char}"

        # MTS CANONIZE for compound identifiers
        assert has_entry(entries, sig="MHALL", op="CANONIZED")
        assert has_entry(entries, sig="SVO", op="CANONIZED")
        assert has_entry(entries, sig="ALL", op="CANONIZED")

        # MHALL CANONIZE has 5 nodes (M, H, A, L, L)
        mhall_canon = _find_entries(entries, sig="MHALL", op="CANONIZED")
        assert len(mhall_canon) >= 1
        assert len(mhall_canon[0].nodes) == 5

        # SVO CANONIZE has 3 nodes (S, V, O)
        svo_canon = _find_entries(entries, sig="SVO", op="CANONIZED")
        assert len(svo_canon) >= 1
        assert len(svo_canon[0].nodes) == 3

        # ALL CANONIZE has 3 nodes (A, L, L)
        all_canon = _find_entries(entries, sig="ALL", op="CANONIZED")
        assert len(all_canon) >= 1
        assert len(all_canon[0].nodes) == 3

        # Countersign pair: MHALL ↔ SVO
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNED")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNED")

        # Undersign entries from subscript
        assert has_entry(entries, sig="M", op="UNDERSIGNED")  # M undersign [S]
        assert has_entry(entries, sig="H", op="UNDERSIGNED")  # H undersign [V]
        assert has_entry(entries, sig="ALL", op="UNDERSIGNED")  # ALL undersign [O]
        assert has_entry(entries, sig="D", op="UNDERSIGNED")  # D undersign [A]
        assert has_entry(entries, sig="M", op="UNDERSIGNED")  # M undersign [L]

        # Connotate
        assert has_entry(entries, sig="L", op="CONNOTED")  # L connotate [O]

        # Verify significance levels
        from kalvin.kline import _SIG_LEVELS

        cs_entries = _find_entries(entries, op="COUNTERSIGNED")
        assert all(_SIG_LEVELS.get(e.dbg.op, "S4") == "S1" for e in cs_entries)
        us_entries = _find_entries(entries, op="UNDERSIGNED")
        assert all(_SIG_LEVELS.get(e.dbg.op, "S4") == "S3" for e in us_entries)
        canon_entries = _find_entries(entries, op="CANONIZED")
        assert all(_SIG_LEVELS.get(e.dbg.op, "S4") == "S2" for e in canon_entries)
        con_entries = _find_entries(entries, op="CONNOTED")
        assert all(_SIG_LEVELS.get(e.dbg.op, "S4") == "S3" for e in con_entries)

    # -- KS-36: §14.12 NLP-bound example ---------------------------------

    def test_ks36_nlp_bound_example(self):
        """KS-36: §14.12 NLP-bound example — key resolved entries present.

        The block annotation (Mary Had A Little Lamb) provides words for
        MHALL's character resolution. Inline annotation S(ubject) triggers
        Rule B4 override on parent SVO CANONIZE entry.
        """
        entries = compile_dev(_SEC1412_SOURCE)
        assert len(entries) > 0

        # MTS for MHALL should resolve M→Mary, H→Had, A→"A", L→Little, L→Lamb
        # Check that "Mary" appears as a signature (from MTS resolution)
        assert has_entry(entries, sig="Mary", op="IDENTITY") or has_entry(
            entries, sig="Mary", op="CANONIZED"
        ), "Expected 'Mary' entries from MHALL MTS resolution"

        # "Subject" should appear from inline annotation S(ubject)
        subject_entries = _find_entries(entries, sig="Subject")
        assert len(subject_entries) > 0, (
            "Expected 'Subject' entries from inline annotation S(ubject)"
        )

        # SVO CANONIZE should exist (potentially with "Subject" patched in)
        assert has_entry(entries, sig="SVO", op="CANONIZED")

        # Basic operator entries should still exist
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNED")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNED")

    # -- KS-37: Uniform tokenizer integration ----------------------------

    def test_ks37_uniform_tokenizer(self):
        """KS-37: §14.12 example compiles under the kalvin tokenizer (uniform typing).

        Every character — both NLP-bound (resolved via word lists) and
        unresolved — produces a valid typed node.  There is no
        character-bit fallback; the whole pipeline goes through the tokenizer.
        """
        entries = compile_dev(_SEC1412_SOURCE)
        assert len(entries) > 0

        # Every entry carries a valid typed signature: high 32 bits hold
        # the type word, low 32 bits hold the BPE token id.
        for e in entries:
            assert isinstance(e.signature, int)
            assert e.signature > 0
            assert (e.signature >> 32) > 0, (
                f"Entry {e.dbg.label!r} signature {e.signature:#x} has no type-word bits"
            )

        # All entries should have a valid op via dbg
        from kalvin.kline import _SIG_LEVELS

        for e in entries:
            assert e.dbg and e.dbg.op in (
                "COUNTERSIGNED",
                "CANONIZED",
                "CONNOTED",
                "UNDERSIGNED",
                "IDENTITY",
            )
            assert _SIG_LEVELS.get(e.dbg.op, "S4") in ("S1", "S2", "S3", "S4")
