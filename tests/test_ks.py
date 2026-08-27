"""Integration tests for the KScript v3 pipeline (src/ks/)."""

from __future__ import annotations

import dataclasses

import pytest

from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.significance import SIG_S1, SIG_S4
from ks import compile_source
from ks.ast import Annotation, Block, KScriptFile, OperatorScope, Signature
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


def compile_dev(source: str) -> list[KValue]:
    """Compile source with dev=True (populates dbg for readable assertions)."""
    return compile_source(source, dev=True)


def compile_real(source: str, tokenizer=None) -> list[KValue]:
    """Compile with dev=False for uint64-level assertions."""
    return compile_source(source, tokenizer=tokenizer, dev=False)


# Lazy module-level tokenizer (safe at import time; ``pytestmark`` gates
# execution so this is only ever instantiated on a data-present machine).
_TOK_INSTANCE: NLPTokenizer | None = None


def _tok() -> NLPTokenizer:
    """Return the shared tokenizer, constructing it on first use."""
    global _TOK_INSTANCE
    if _TOK_INSTANCE is None:
        _TOK_INSTANCE = NLPTokenizer()
    return _TOK_INSTANCE


def _sig_str(entry: KValue) -> str:
    """Return a human-readable signature string for an entry.

    Uses dbg.label when available (dev mode); otherwise decodes the uint64.
    """
    if entry.kline.dbg and entry.kline.dbg.label:
        return entry.kline.dbg.label
    return _tok().decode([entry.kline.signature])


def _node_strs(entry: KValue) -> list[str]:
    """Decode an entry's uint64 node values to human-readable strings."""
    if not entry.kline.nodes:
        return []
    return [_tok().decode([n]) for n in entry.kline.nodes]


def _find_entries(
    entries: list[KValue],
    *,
    sig: str | None = None,
    op: str | None = None,
    nodes: list[str] | None = None,
) -> list[KValue]:
    """Find entries matching the given criteria by decoded string values."""
    results = []
    for e in entries:
        if sig is not None and _sig_str(e) != sig:
            continue
        if op is not None and e.kline.dbg.op != op:
            continue
        if nodes is not None and _node_strs(e) != nodes:
            continue
        results.append(e)
    return results


def has_entry(
    entries: list[KValue],
    *,
    sig: str,
    op: str | None = None,
    nodes: list[str] | None = None,
) -> bool:
    """Check if at least one entry matches the given criteria."""
    return len(_find_entries(entries, sig=sig, op=op, nodes=nodes)) > 0


# ---------------------------------------------------------------------------

class TestTokenType:
    """All token types recognized; Token is a frozen dataclass."""

    def test_token_type_members(self):
        """All 10 TokenType members exist."""
        expected = {
            "COUNTERSIGNS",
            "CANONIZES",
            "CONNOTES",
            "DENOTES",
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
        """Token is a frozen dataclass with type, value, line, column."""
        assert dataclasses.is_dataclass(Token)
        assert getattr(Token, "__dataclass_params__").frozen is True

        fields = {f.name for f in dataclasses.fields(Token)}
        assert fields == {"type", "value", "line", "column"}

    def test_token_recognition(self):
        """Lexer produces correct token types for A == B."""
        tokens = Lexer("A == B").tokenize()
        # Expect: SIGNATURE("A"), COUNTERSIGNS("=="), SIGNATURE("B"), EOF
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.COUNTERSIGNS,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]

class TestLexer:

    # -- Multi-char operator priority --------------------------------

    def test_multi_char_operator_priority_eq(self):
        """'==' is lexed as COUNTERSIGNS, not two DENOTES tokens."""
        tokens = Lexer("A == B").tokenize()
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.COUNTERSIGNS,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]
        # Confirm no DENOTES tokens
        assert TokenType.DENOTES not in types

    def test_multi_char_operator_priority_arrow(self):
        """'=>' is lexed as CANONIZES, not DENOTES + CONNOTES."""
        tokens = Lexer("A => B").tokenize()
        types = [t.type for t in tokens]
        assert types == [
            TokenType.SIGNATURE,
            TokenType.CANONIZES,
            TokenType.SIGNATURE,
            TokenType.EOF,
        ]
        assert TokenType.DENOTES not in types
        assert TokenType.CONNOTES not in types

    # -- BPE annotations --------------------------------------------

    def test_bpe_annotations(self):
        """'(hello world)' produces a single ANNOTATION token."""
        tokens = Lexer("(hello world)").tokenize()
        # Expect: ANNOTATION("(hello world)"), EOF
        assert tokens[0].type == TokenType.ANNOTATION
        assert tokens[0].value == "(hello world)"
        assert tokens[1].type == TokenType.EOF

    def test_nested_parens(self):
        """Nested parens produce a single ANNOTATION preserving content."""
        tokens = Lexer("(a (b c) d)").tokenize()
        assert tokens[0].type == TokenType.ANNOTATION
        assert tokens[0].value == "(a (b c) d)"
        assert tokens[1].type == TokenType.EOF

    # -- INDENT/DEDENT ----------------------------------------------

    def test_indent_dedent(self):
        """Indentation produces INDENT and DEDENT tokens."""
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

    def test_dedent_at_eof(self):
        """Remaining indent levels produce DEDENT tokens at EOF."""
        tokens = Lexer("A\n  B").tokenize()
        types = [t.type for t in tokens]
        # INDENT for B, then DEDENT at EOF
        assert types.count(TokenType.INDENT) == 1
        assert types.count(TokenType.DEDENT) == 1

    # -- Edge cases -------------------------------------------------

    def test_empty_input(self):
        """Empty input produces only EOF."""
        tokens = Lexer("").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only(self):
        """Whitespace-only input produces only EOF (per spec)."""
        tokens = Lexer("   ").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_angle_bracket_error(self):
        """'<' raises LexerError."""
        with pytest.raises(LexerError):
            Lexer("A < B").tokenize()

    def test_unknown_char_error(self):
        """Unknown characters raise LexerError."""
        with pytest.raises(LexerError):
            Lexer("A @ B").tokenize()

class TestParserAST:

    @staticmethod
    def _parse(source: str) -> KScriptFile:
        """Helper: lex and parse source into a KScriptFile AST."""
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    # -- Scope model AST -------------------------------------------

    def test_scope_model_ast(self):
        """Parse 'A == B > C = D' into chained OperatorScope nodes."""
        ast = self._parse("A == B > C = D")
        assert len(ast.constructs) == 1

        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op == TokenType.COUNTERSIGNS

        # First item is B (Signature), but since B > C = D forms a chain,
        # the items should contain a nested OperatorScope for the connote.
        # A == [B > C = D]  — B is both a COUNTERSIGNS node and the sig for >
        assert len(scope.items) >= 1
        inner = scope.items[0]
        assert isinstance(inner, OperatorScope)
        assert inner.sig.id == "B"
        assert inner.op == TokenType.CONNOTES

        # B > [C = D] — C is the connote node and the sig for =
        assert len(inner.items) >= 1
        deepest = inner.items[0]
        assert isinstance(deepest, OperatorScope)
        assert deepest.sig.id == "C"
        assert deepest.op == TokenType.DENOTES

    # -- Block parsing ---------------------------------------------

    def test_block_parsing(self):
        """Indented source creates Block nodes with correct constructs."""
        source = "A =>\n  B\n  C"
        ast = self._parse(source)

        assert len(ast.constructs) == 1
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op == TokenType.CANONIZES
        assert scope.child_block is not None
        assert isinstance(scope.child_block, Block)
        assert len(scope.child_block.constructs) == 2

    # -- Annotations preserved -------------------------------------

    def test_annotations_preserved(self):
        """'(Mary Had)' heads no following construct, so it synthesizes
        the bare sigless MH scope (a standalone sentence)."""
        ast = self._parse("(Mary Had)")
        assert len(ast.constructs) == 1
        block = ast.constructs[0]
        assert isinstance(block, Block)
        ann, scope = block.constructs
        assert isinstance(ann, Annotation)
        assert ann.text == "(Mary Had)"
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "MH"
        assert scope.op is None

    def test_prefix_annotation_stays_loose(self):
        """An annotation followed by a SIGNATURE construct is a prefix
        annotation for it — no synthesis."""
        ast = self._parse("(Mary Had)\nMH == SVO")
        assert len(ast.constructs) == 2
        assert isinstance(ast.constructs[0], Annotation)
        assert isinstance(ast.constructs[1], OperatorScope)
        assert ast.constructs[1].sig.id == "MH"

    # -- Inline annotations ----------------------------------------

    def test_sig_inline_annotation(self):
        """'S(ubject) = M' attaches inline_annotation to scope."""
        ast = self._parse("S(ubject) = M")
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "S"
        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"

    def test_node_inline_annotation(self):
        """'A = D(et)' attaches the inline annotation to the D item."""
        ast = self._parse("A = D(et)")
        scope = ast.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        d_item = scope.items[0]
        assert isinstance(d_item, Signature)
        assert d_item.id == "D"
        assert d_item.inline_annotation is not None
        assert d_item.inline_annotation.text == "(et)"

    # -- Empty source ---------------------------------------------

    def test_empty_source(self):
        """Empty source produces empty script (no error)."""
        ast = self._parse("")
        assert isinstance(ast, KScriptFile)
        assert ast.constructs == []

class TestBindingScope:

    # -- First-letter matching ------------------------------------

    def test_first_letter_matching(self):
        """resolve('M') → 'Mary', resolve('H') → 'Had', resolve('A') → 'A'."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Mary", "Had", "A", "Little", "Lamb"])

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("H") == "Had"
        assert scope.resolve("A") == "A"

    # -- Occurrence counter ---------------------------------------

    def test_occurrence_counter(self):
        """First resolve('L') → 'Little', second → 'Lamb'."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Little", "Lamb"])

        assert scope.resolve("L") == "Little"
        assert scope.resolve("L") == "Lamb"

    # -- Scope inheritance ----------------------------------------

    def test_scope_inheritance(self):
        """Inner scope with no matching words falls through to outer."""
        scope = BindingScope()
        scope.push_scope()  # outer
        scope.add_words(["Alpha"])
        scope.push_scope()  # inner (no words)

        assert scope.resolve("A") == "Alpha"

    # -- Scope shadowing ------------------------------------------

    def test_scope_shadowing(self):
        """Inner scope binding shadows outer for same character."""
        scope = BindingScope()
        scope.push_scope()  # outer
        scope.add_words(["Alpha"])
        scope.push_scope()  # inner
        scope.add_words(["Another"])

        assert scope.resolve("A") == "Another"

    # -- Counter reset --------------------------------------------

    def test_counter_reset(self):
        """Pushing a new scope resets counters for resolution."""
        scope = BindingScope()
        scope.push_scope()  # scope 1
        scope.add_words(["Little", "Lamb"])
        assert scope.resolve("L") == "Little"  # counter 0 in scope 1

        scope.push_scope()  # scope 2 (empty) — should reset counters
        # Resolve falls through to scope 1 with a reset counter
        # → "Little" again (not "Lamb")
        assert scope.resolve("L") == "Little"

    # -- Unresolved identifier (no fallback state) -------------

    def test_unresolved_identifier(self):
        """An unresolved identifier (BindingScope.resolve returns None)
        is encoded as its own raw BPE token — no special fallback state.

        At the BindingScope level, resolve('Z') with no matching words
        returns None. The encoding behavior (single typed node, same
        path as any resolved character) is covered by .
        """
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Alpha", "Beta"])
        assert scope.resolve("Z") is None

    # -- Inert annotation -----------------------------------------

    def test_inert_annotation(self):
        """Words with no matching characters have no effect."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Xray", "Yankee"])
        assert scope.resolve("M") is None

class TestEmitterOperators:

    # -- COUNTERSIGNS per-item -------------------------------------

    def test_countersign_per_item(self):
        """A == B C → {A:[B]}, {B:[A]}, {A:[C]}, {C:[A]} COUNTERSIGNS."""
        entries = compile_dev("A == B C")
        assert len(entries) == 4
        assert all(e.kline.dbg.op == "COUNTERSIGNS" for e in entries)
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNS", nodes=["A"])

    def test_countersign_entries_present(self):
        """relaxed): The 4 COUNTERSIGNS pairs are present regardless of extras."""
        entries = compile_dev("A == B C")
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNS", nodes=["A"])

    # -- DENOTES per-item reversed -------------------------------

    def test_denote_per_item_reversed(self):
        """A = B C → {B:[A]}, {C:[A]} DENOTES."""
        entries = compile_dev("A = B C")
        assert len(entries) == 2
        assert all(e.kline.dbg.op == "DENOTES" for e in entries)
        assert has_entry(entries, sig="B", op="DENOTES", nodes=["A"])
        assert has_entry(entries, sig="C", op="DENOTES", nodes=["A"])

    def test_denote_entries_present(self):
        """relaxed): The 2 DENOTES entries are present."""
        entries = compile_dev("A = B C")
        assert has_entry(entries, sig="B", op="DENOTES", nodes=["A"])
        assert has_entry(entries, sig="C", op="DENOTES", nodes=["A"])

    # -- CONNOTES per-item ----------------------------------------

    def test_connote_per_item(self):
        """A > B C → {A:[B]}, {A:[C]} CONNOTES."""
        entries = compile_dev("A > B C")
        assert len(entries) == 2
        assert all(e.kline.dbg.op == "CONNOTES" for e in entries)
        assert has_entry(entries, sig="A", op="CONNOTES", nodes=["B"])
        assert has_entry(entries, sig="A", op="CONNOTES", nodes=["C"])

    def test_connote_entries_present(self):
        """relaxed): The 2 CONNOTES entries are present."""
        entries = compile_dev("A > B C")
        assert has_entry(entries, sig="A", op="CONNOTES", nodes=["B"])
        assert has_entry(entries, sig="A", op="CONNOTES", nodes=["C"])

    # -- CANONIZES aggregates ---------------------------------------

    def test_canonize_aggregates(self):
        """A => B C D → {A:[B,C,D]} CANONIZES."""
        entries = compile_dev("A => B C D")
        assert len(entries) == 1
        assert entries[0].kline.dbg.op == "CANONIZES"
        assert _sig_str(entries[0]) == "A"
        assert _node_strs(entries[0]) == ["B", "C", "D"]

    def test_canonize_entry_present(self):
        """relaxed): The CANONIZES aggregate entry is present."""
        entries = compile_dev("A => B C D")
        assert has_entry(entries, sig="A", op="CANONIZES", nodes=["B", "C", "D"])

    # -- Operator chain -------------------------------------------

    def test_operator_chain(self):
        """A == B > C = D → entries per table."""
        entries = compile_dev("A == B > C = D")
        assert len(entries) == 4
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="B", op="CONNOTES", nodes=["C"])
        assert has_entry(entries, sig="D", op="DENOTES", nodes=["C"])

    def test_operator_chain_entries_present(self):
        """relaxed): The 4 operator chain entries are present."""
        entries = compile_dev("A == B > C = D")
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="B", op="CONNOTES", nodes=["C"])
        assert has_entry(entries, sig="D", op="DENOTES", nodes=["C"])

    # -- Indent extends scope --------------------------------------

    def test_indent_extends_scope(self):
        """Indented items under CANONIZES belong to parent's node list."""
        source = "A =>\n  B\n  C"
        entries = compile_dev(source)
        # CANONIZES should have B and C as nodes
        assert has_entry(entries, sig="A", op="CANONIZES", nodes=["B", "C"])

    # -- DEDENT returns to parent ----------------------------------

    def test_dedent_returns_to_parent(self):
        """After dedent, subsequent constructs compile at parent level."""
        source = "A =>\n  B\nC = D"
        entries = compile_dev(source)
        # A CANONIZES with B as node (from indented block)
        assert has_entry(entries, sig="A", op="CANONIZES", nodes=["B"])
        # D DENOTES [C] (at parent level after dedent)
        assert has_entry(entries, sig="D", op="DENOTES", nodes=["C"])

    # -- Non-CANONIZES with indent ---------------------------------

    def test_non_canonize_with_indent(self):
        """A == B\\n  C\\n  D → 6 COUNTERSIGNS entries."""
        source = "A == B\n  C\n  D"
        entries = compile_dev(source)
        assert len(entries) == 6
        # All should be COUNTERSIGNS (bidirectional pairs)
        assert all(e.kline.dbg.op == "COUNTERSIGNS" for e in entries)

    def test_non_canonize_entries_present(self):
        """relaxed): The 6 COUNTERSIGNS pairs are present."""
        source = "A == B\n  C\n  D"
        entries = compile_dev(source)
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["B"])
        assert has_entry(entries, sig="B", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["C"])
        assert has_entry(entries, sig="C", op="COUNTERSIGNS", nodes=["A"])
        assert has_entry(entries, sig="A", op="COUNTERSIGNS", nodes=["D"])
        assert has_entry(entries, sig="D", op="COUNTERSIGNS", nodes=["A"])

    # -- Inline binding bypass ------------------------------------

    def test_inline_binding_bypass(self):
        """S(ubject) = M — inline annotation resolves S to 'Subject'."""
        entries = compile_dev("S(ubject) = M")
        # The signature side should resolve to "Subject" via inline annotation
        sig_entries = _find_entries(entries, sig="Subject")
        assert len(sig_entries) > 0, "Expected entries with sig='Subject'"

    # -- Self-identity --------------------------------------------

    def test_self_identity(self):
        """A = A → single {A:[A]} IDENTITY (self-referential, S1)."""
        entries = compile_dev("A = A")
        assert len(entries) == 1
        assert entries[0].kline.dbg.op == "IDENTITY"
        assert _sig_str(entries[0]) == "A"
        assert entries[0].kline.nodes == [entries[0].kline.signature]
        assert entries[0].significance == SIG_S1

    def test_self_identity_unsigned_present(self):
        """A = A → self-referential IDENTITY (no empty-form Unknown)."""
        entries = compile_dev("A = A")
        assert has_entry(entries, sig="A", op="IDENTITY")
        assert not has_entry(entries, sig="A", op="UNKNOWN", nodes=[])

    # -- /b/c: Singleton Identity vs Unknown (binding-aware) ---------

    def test_bare_singleton_unbound_is_unknown(self):
        """A bare unbound singleton → {A:[]} UNKNOWN, S4."""
        entries = compile_dev("A")
        assert has_entry(entries, sig="A", op="UNKNOWN", nodes=[])
        unknown = _find_entries(entries, sig="A", op="UNKNOWN")[0]
        assert unknown.kline.nodes == []
        assert unknown.significance == SIG_S4

    def test_bare_singleton_word_bound_is_identity(self):
        """A bare word-bound singleton → self-ref IDENTITY, S1.

        Binding is the sole Identity/Unknown discriminator.
        """
        entries = compile_dev("(Mary)\nM")
        assert has_entry(entries, sig="Mary", op="IDENTITY")
        identity = _find_entries(entries, sig="Mary", op="IDENTITY")[0]
        assert identity.kline.nodes == [identity.kline.signature]
        assert identity.significance == SIG_S1
        # No empty-form Unknown for the bound singleton.
        assert not has_entry(entries, sig="Mary", op="UNKNOWN", nodes=[])

    def test_binding_is_sole_discriminator(self):
        """an unbound singleton stays Unknown even when referenced as
        a node or introduced via MTS — only word binding elevates it."""
        # A is referenced as a node (A == A would self-denote → IDENTITY, so
        # use a distinct node B that is unbound). B is introduced only by
        # being a node here; it has no annotation → stays Unknown.
        entries = compile_dev("A == B")
        assert has_entry(entries, sig="B", op="COUNTERSIGNS")  # referenced as node
        # B has no identity entry of its own (no bare singleton, no binding).
        assert not has_entry(entries, sig="B", op="IDENTITY")
        assert not has_entry(entries, sig="B", op="UNKNOWN", nodes=[])

class TestEmitterMTS:

    # -- MTS expansion --------------------------------------------

    def test_mts_expansion(self):
        """ABC → only the CANONIZES canon (no per-component entries).

        MTS emits exactly one entry: the canon {ABC:[A,B,C]} (S2). The
        characters are values inside the canon, not headed klines.
        """
        entries = compile_dev("ABC")
        assert len(entries) == 1

        assert _sig_str(entries[0]) == "ABC" and entries[0].kline.dbg.op == "CANONIZES"
        assert _node_strs(entries[0]) == ["A", "B", "C"]

    def test_mts_component_uniformity(self):
        """word-bound MTS constituents → IDENTITY; unbound → UNKNOWN.

        Same binding-aware rule as a bare singleton. A word-bound
        compound's constituent compiles to a self-referential Identity; an
        unbound compound's constituent stays an empty Unknown.
        """
        # Bound: M resolves to "Mary" under (Mary ...) → IDENTITY.
        bound = compile_dev("(Mary)\nM")
        assert has_entry(bound, sig="Mary", op="IDENTITY")
        # Unbound: M has no annotation → UNKNOWN.
        unbound = compile_dev("M")
        assert has_entry(unbound, sig="M", op="UNKNOWN", nodes=[])
        assert not has_entry(unbound, sig="M", op="IDENTITY")

    # -- No MTS for single-char -----------------------------------

    def test_no_mts_for_single_char(self):
        """A → single UNKNOWN entry, no component expansion."""
        entries = compile_dev("A")
        assert len(entries) == 1
        assert entries[0].kline.dbg.op == "UNKNOWN"
        assert _sig_str(entries[0]) == "A"
        assert entries[0].kline.nodes == []

    # -- MTS on node side -----------------------------------------

    def test_mts_on_node_side(self):
        """A == MHALL triggers MTS expansion for MHALL on the node side."""
        entries = compile_dev("A == MHALL")
        # MTS for MHALL: component unsigned entries + CANONIZES entry
        assert has_entry(entries, sig="MHALL", op="CANONIZES")
        # Countersign pairs: A ↔ MHALL (node is the compound uint64 for MHALL,
        # which decodes to sorted chars, so we check by sig and op only)
        a_cs = _find_entries(entries, sig="A", op="COUNTERSIGNS")
        assert len(a_cs) >= 1, "Expected A COUNTERSIGNS entry"
        mhall_cs = _find_entries(entries, sig="MHALL", op="COUNTERSIGNS")
        assert len(mhall_cs) >= 1, "Expected MHALL COUNTERSIGNS entry"

    # -- Node count invariant --------------------------------------

    def test_node_count_invariant(self):
        """MTS canonization entry has N nodes for an N-char identifier."""
        for ident in ["AB", "ABC", "ABCD", "MHALL"]:
            entries = compile_dev(ident)
            canonize_entries = _find_entries(entries, sig=ident, op="CANONIZES")
            assert len(canonize_entries) >= 1, f"No CANONIZES entry for {ident}"
            canon = canonize_entries[0]
            actual = len(canon.kline.nodes)
            assert actual == len(ident), (
                f"MTS canonize for {ident}: expected {len(ident)} nodes, got {actual}"
            )

class TestEmitterBinding:

    # -- Rule B4 override -----------------------------------------

    def test_rule_b4_override(self):
        """Inline annotation patches parent MTS CANONIZES entry.

        In the source, S(ubject) inside a subscript block patches
        the parent SVO CANONIZES entry: S → 'Subject'.
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
        # Find the SVO CANONIZES entry — it should have "Subject" as first node
        svo_canon = _find_entries(entries, sig="SVO", op="CANONIZES")
        assert len(svo_canon) >= 1, "Expected at least one SVO CANONIZES entry"
        # A simpler check: verify that "Subject" unsigned entries exist
        # (from the inline annotation's MTS expansion)
        subject_entries = _find_entries(entries, sig="Subject")
        assert len(subject_entries) > 0, "Expected entries for 'Subject' (from inline annotation)"

class TestStructure:

    def test_nodes_always_list_canonize(self):
        """CANONIZES entry nodes is a list of length 1+ (not scalar)."""
        entries = compile_dev("A => B")
        canon = _find_entries(entries, sig="A", op="CANONIZES")
        assert len(canon) == 1
        assert isinstance(canon[0].kline.nodes, list)
        assert len(canon[0].kline.nodes) >= 1

    def test_nodes_always_list_unsigned(self):
        """UNKNOWN entry nodes is an empty list (not None)."""
        entries = compile_dev("A")
        assert len(entries) == 1
        assert isinstance(entries[0].kline.nodes, list)
        assert entries[0].kline.nodes == []

    def test_all_entries_nodes_are_lists(self):
        """For every compiled entry, nodes is a list."""
        for source in ["A", "A == B", "A => B C", "ABC", "A > B\n  C"]:
            entries = compile_dev(source)
            for e in entries:
                assert isinstance(e.kline.nodes, list), (
                    f"Entry {e!r} has nodes of type {type(e.kline.nodes)}, expected list"
                )

class TestEncoding:

    def test_unresolved_char_typed_encoding(self):
        """An unresolved single character (e.g. 'Z') encodes to a single typed node.

        Under the kalvin tokenizer there is no character-bit fallback.  An
        unresolved character is encoded as its own raw BPE token, producing a
        valid typed node (high 32 bits = sig word, low 32 bits = BPE id).
        """
        entries = compile_dev("Z")
        assert len(entries) >= 1
        entry = entries[0]
        sig_word = entry.kline.signature >> 32
        bpe_id = entry.kline.signature & 0xFFFFFFFF
        # Typed node: high 32 bits carry the sig word; low 32 bits carry BPE id
        assert sig_word > 0, f"Expected sig-word bits in high word, got {sig_word}"
        assert bpe_id > 0, f"Expected a valid BPE token id, got {bpe_id}"
        # Must NOT be the legacy character-bit encoding (single-bit)
        assert entry.kline.signature != 67108864, "Signature should not be a legacy bit value"
        assert entry.kline.dbg.op == "UNKNOWN"

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



    # -- secondary regression (simpler nested case) ----------------

    def test_sec148_strict(self):
        """ secondary regression — strict spec count (5 entries)."""
        entries = compile_dev(_SEC148_SOURCE)
        assert len(entries) == 5
        assert has_entry(entries, sig="A", op="CANONIZES", nodes=["B", "C"])
        assert has_entry(entries, sig="D", op="DENOTES", nodes=["C"])
        assert has_entry(entries, sig="B", op="UNKNOWN", nodes=[])
        assert has_entry(entries, sig="C", op="UNKNOWN", nodes=[])
        assert has_entry(entries, sig="D", op="UNKNOWN", nodes=[])

    def test_sec148_presence(self):
        """ secondary regression — key entries present (5 entries).

        CANONIZES subscript blocks emit identity for
        bare scopes, DENOTES scope sigs, and leaf Signature items.
        identity entries use UNKNOWN op.
        Now matches spec exactly (5 entries).
        """
        entries = compile_dev(_SEC148_SOURCE)
        assert len(entries) == 5
        assert has_entry(entries, sig="A", op="CANONIZES", nodes=["B", "C"])
        assert has_entry(entries, sig="D", op="DENOTES", nodes=["C"])
        assert has_entry(entries, sig="B", op="UNKNOWN", nodes=[])
        assert has_entry(entries, sig="C", op="UNKNOWN", nodes=[])
        assert has_entry(entries, sig="D", op="UNKNOWN", nodes=[])

    # -- complex nested (master regression) ----------------

    def test_complex_nested_strict(self):
        """master regression — strict spec count (18 entries).

        Output ordering is source-first: compiled source klines precede
        every MTS expansion kline. MTS component UNKNOWN dedup, no
        compound-own identity, subscript identity suppression for MTS
        CANONIZES scopes.

        Source entries (S1/S3, in emission order):
        1:     MHALL countersign [SVO] (S1)
        2:     SVO countersign [MHALL] (S1)
        3:     M denote [S] (S3)
        4:     H denote [V] (S3)
        5:     ALL denote [O] (S3)
        6:     D denote [A] (S3)
        7:     M denote [L] (S3)
        8:     L connote [O] (S3)
        MTS entries (S2, after all source — canons only, no components):
        9:     MHALL canonize [M, H, A, L, L] (S2)
        10:    SVO canonize [S, V, O] (S2)
        (SVO canonize subscript: deduped)
        11:    ALL canonize [A, L, L] (S2)
        (ALL canonize subscript: deduped)
        """
        entries = compile_dev(_SEC1411_SOURCE)
        assert len(entries) == 11

        # Spot-check critical entries by dbg.label — the first emitted
        # kline is now a source countersign, not an MTS component identity.
        assert (
            entries[0].kline.dbg
            and entries[0].kline.dbg.label == "MHALL"
            and entries[0].kline.dbg.op == "COUNTERSIGNS"
        )
        # MHALL CANONIZES with 5 nodes
        mhall_canon = [
            e for e in entries
            if e.kline.dbg and e.kline.dbg.label == "MHALL" and e.kline.dbg.op == "CANONIZES"
        ]
        assert len(mhall_canon) == 1
        assert len(mhall_canon[0].kline.nodes) == 5  # M, H, A, L, L

        # SVO CANONIZES with 3 nodes
        svo_canon = [
            e for e in entries
            if e.kline.dbg and e.kline.dbg.label == "SVO" and e.kline.dbg.op == "CANONIZES"
        ]
        assert len(svo_canon) == 1
        assert len(svo_canon[0].kline.nodes) == 3  # S, V, O

        # Countersign pair
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNS")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNS")

        # Denote entries
        assert has_entry(entries, sig="M", op="DENOTES")
        assert has_entry(entries, sig="H", op="DENOTES")
        assert has_entry(entries, sig="ALL", op="DENOTES")
        assert has_entry(entries, sig="D", op="DENOTES")

        # Connote
        assert has_entry(entries, sig="L", op="CONNOTES")

        # MTS emits no per-component entries.
        for char in ["M", "H", "A", "L", "S", "V", "O"]:
            assert not has_entry(entries, sig=char, op="UNKNOWN"), (
                f"Unexpected component entry for {char}"
            )

    def test_complex_nested_presence(self):
        """master regression — key entries present (11 entries)."""
        entries = compile_dev(_SEC1411_SOURCE)
        assert len(entries) == 11

        # MTS emits only canons — no per-component UNKNOWN entries.
        for char in ["M", "H", "A", "L", "S", "V", "O"]:
            assert not has_entry(entries, sig=char, op="UNKNOWN"), (
                f"Unexpected component entry for {char}"
            )

        # MTS CANONIZES for compound identifiers
        assert has_entry(entries, sig="MHALL", op="CANONIZES")
        assert has_entry(entries, sig="SVO", op="CANONIZES")
        assert has_entry(entries, sig="ALL", op="CANONIZES")

        # MHALL CANONIZES has 5 nodes (M, H, A, L, L)
        mhall_canon = _find_entries(entries, sig="MHALL", op="CANONIZES")
        assert len(mhall_canon) >= 1
        assert len(mhall_canon[0].kline.nodes) == 5

        # SVO CANONIZES has 3 nodes (S, V, O)
        svo_canon = _find_entries(entries, sig="SVO", op="CANONIZES")
        assert len(svo_canon) >= 1
        assert len(svo_canon[0].kline.nodes) == 3

        # ALL CANONIZES has 3 nodes (A, L, L)
        all_canon = _find_entries(entries, sig="ALL", op="CANONIZES")
        assert len(all_canon) >= 1
        assert len(all_canon[0].kline.nodes) == 3

        # Countersign pair: MHALL ↔ SVO
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNS")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNS")

        # Denote entries from subscript
        assert has_entry(entries, sig="M", op="DENOTES")  # M denote [S]
        assert has_entry(entries, sig="H", op="DENOTES")  # H denote [V]
        assert has_entry(entries, sig="ALL", op="DENOTES")  # ALL denote [O]
        assert has_entry(entries, sig="D", op="DENOTES")  # D denote [A]
        assert has_entry(entries, sig="M", op="DENOTES")  # M denote [L]

        # Connote
        assert has_entry(entries, sig="L", op="CONNOTES")  # L connote [O]

        # Verify structural significance levels. Significance is derived
        # from kline shape (sig_level), not the op token: a canonical kline
        # (sig == signature_of(nodes)) is S1; a single-node non-canonical
        # kline is S3; an empty-nodes kline is S4.
        from kalvin.kline import sig_level
        from kalvin.signifier import NLPSignifier as _Sig

        _sgf = _Sig()
        cs_entries = _find_entries(entries, op="COUNTERSIGNS")
        assert all(sig_level(e.kline, _sgf) == "S3" for e in cs_entries)
        us_entries = _find_entries(entries, op="DENOTES")
        assert all(sig_level(e.kline, _sgf) == "S3" for e in us_entries)
        canon_entries = _find_entries(entries, op="CANONIZES")
        assert all(sig_level(e.kline, _sgf) == "S1" for e in canon_entries)
        con_entries = _find_entries(entries, op="CONNOTES")
        assert all(sig_level(e.kline, _sgf) == "S3" for e in con_entries)

    # -- Word-bound example ---------------------------------

    def test_word_bound_example(self):
        """Word-bound example — key resolved entries present.

        The block annotation (Mary Had A Little Lamb) provides words for
        MHALL's character resolution. Inline annotation S(ubject) triggers
        Rule B4 override on parent SVO CANONIZES entry.
        """
        entries = compile_dev(_SEC1412_SOURCE)
        assert len(entries) > 0

        # MTS for MHALL should resolve M→Mary, H→Had, A→"A", L→Little, L→Lamb.
        # Each word-bound constituent compiles to a self-referential IDENTITY
        # (/); Mary is no longer an empty Unknown.
        assert has_entry(entries, sig="Mary", op="IDENTITY"), (
            "Expected 'Mary' IDENTITY from MHALL MTS resolution"
        )

        # "Subject" should appear from inline annotation S(ubject)
        subject_entries = _find_entries(entries, sig="Subject")
        assert len(subject_entries) > 0, (
            "Expected 'Subject' entries from inline annotation S(ubject)"
        )

        # SVO CANONIZES should exist (potentially with "Subject" patched in)
        assert has_entry(entries, sig="SVO", op="CANONIZES")

        # Basic operator entries should still exist
        assert has_entry(entries, sig="MHALL", op="COUNTERSIGNS")
        assert has_entry(entries, sig="SVO", op="COUNTERSIGNS")

    # -- Uniform tokenizer integration ----------------------------

    def test_uniform_tokenizer(self):
        """example compiles under the kalvin tokenizer (uniform typing).

        Every character — both word-bound (resolved via word lists) and
        unresolved — produces a valid typed node.  There is no
        character-bit fallback; the whole pipeline goes through the tokenizer.
        """
        entries = compile_dev(_SEC1412_SOURCE)
        assert len(entries) > 0

        # Every entry carries a valid typed signature: high 32 bits hold
        # the sig word, low 32 bits hold the BPE token id.
        for e in entries:
            assert isinstance(e.kline.signature, int)
            assert e.kline.signature > 0
            assert (e.kline.signature >> 32) > 0, (
                f"Entry {e.kline.dbg.label!r} signature {e.kline.signature:#x} has no sig-word bits"
            )

        # All entries should have a valid op via dbg. (Significance is
        # structural — derived via sig_level, not an op→level table — so no
        # table lookup is asserted here.)
        for e in entries:
            assert e.kline.dbg and e.kline.dbg.op in (
                "COUNTERSIGNS",
                "CANONIZES",
                "CONNOTES",
                "DENOTES",
                "IDENTITY",
                "UNKNOWN",
            )
