"""Tests for KScript compiler pipeline (lexer → parser → compiler → decompiler)."""

import json
import tempfile
from pathlib import Path

import pytest

from kalvin.kline import KLine
from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer
from kalvin.signature import is_nlp_node
from kscript.ast import (
    Block,
    Comment,
    Construct,
    KScriptFile,
    Literal,
    PrimaryConstruct,
    Script,
    Signature,
)
from kscript.compiler import CompiledEntry, Compiler, compile_source
from kscript.decompiler import Decompiler, DecompiledEntry
from kscript.lexer import Lexer, LexerError
from kscript.output import read_bin, read_json, write_bin, write_json, write_jsonl
from kscript.parser import ParseError, Parser
from kscript.token import Token, TokenType


# ── Shared fixtures ──────────────────────────────────────────────────────────

_tok32 = Mod32Tokenizer()
_tok64 = Mod64Tokenizer()


def compile32(source: str) -> list[CompiledEntry]:
    return compile_source(source, tokenizer=_tok32, dev=True)


def compile64(source: str) -> list[CompiledEntry]:
    return compile_source(source, tokenizer=_tok64, dev=True)


def decode_entries(entries: list[CompiledEntry], tok=None) -> list[tuple[str, str | list[str] | None]]:
    tok = tok or _tok64
    return [e.decode(tok) for e in entries]


def entries_to_multidict(entries: list[CompiledEntry], tok=None) -> dict[str, list]:
    """Map sig → list of all decoded node values (preserving order and duplicates).

    Node values are whatever decode() returns: str, list[str], or '' (for empty).
    """
    tok = tok or _tok64
    result: dict[str, list] = {}
    for e in entries:
        sig, nodes = e.decode(tok)
        result.setdefault(sig, []).append(nodes)
    return result


def _md(entries: list[CompiledEntry], tok=None) -> dict[str, list]:
    """Shorthand for entries_to_multidict."""
    return entries_to_multidict(entries, tok)


# Conditional NLP tokenizer import
try:
    from kalvin.nlp_tokenizer import NLPTokenizer
    _has_nlp = True
except ImportError:
    _has_nlp = False

_nlp_skip = pytest.mark.skipif(not _has_nlp, reason="NLPTokenizer not available")


def _has_node(md: dict[str, list], sig: str, node_value) -> bool:
    """Check if a sig has a specific decoded node value in its list."""
    return node_value in md.get(sig, [])


# =============================================================================
# 1. Token type tests
# =============================================================================

class TestTokenType:
    def test_all_token_types_exist(self) -> None:
        expected = [
            "COUNTERSIGN", "CANONIZE", "CONNOTATE",
            "UNDERSIGN",
            "SIGNATURE", "LITERAL", "COMMENT",
            "NEWLINE", "INDENT", "DEDENT", "EOF",
        ]
        for name in expected:
            assert hasattr(TokenType, name), f"Missing TokenType.{name}"

    def test_token_frozen(self) -> None:
        t = Token(TokenType.SIGNATURE, "A", 1, 1)
        assert t.type == TokenType.SIGNATURE
        assert t.value == "A"
        assert t.line == 1
        assert t.column == 1


# =============================================================================
# 2. Lexer tests
# =============================================================================

class TestLexer:
    def test_simple_signature(self) -> None:
        tokens = Lexer("A").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.SIGNATURE]

    def test_multi_char_signature(self) -> None:
        tokens = Lexer("ABC").tokenize()
        sigs = [t for t in tokens if t.type == TokenType.SIGNATURE]
        assert len(sigs) == 1
        assert sigs[0].value == "ABC"

    def test_literal_number(self) -> None:
        tokens = Lexer("42").tokenize()
        lits = [t for t in tokens if t.type == TokenType.LITERAL]
        assert len(lits) == 1
        assert lits[0].value == "42"

    def test_literal_string(self) -> None:
        tokens = Lexer('"hello"').tokenize()
        lits = [t for t in tokens if t.type == TokenType.LITERAL]
        assert len(lits) == 1
        assert lits[0].value == '"hello"'

    def test_literal_lowercase(self) -> None:
        """Lowercase identifiers are now a lexer error.
        Use quoted strings for non-signature text.
        """
        with pytest.raises(LexerError):
            Lexer("hello").tokenize()

    def test_quoted_string(self) -> None:
        tokens = Lexer('"hello world"').tokenize()
        lits = [t for t in tokens if t.type == TokenType.LITERAL]
        assert len(lits) == 1
        assert lits[0].value == '"hello world"'

    def test_operators(self) -> None:
        tokens = Lexer("A == B").tokenize()
        types = [t.type for t in tokens if t.type not in (TokenType.EOF, TokenType.SIGNATURE)]
        assert TokenType.COUNTERSIGN in types

    def test_canonize(self) -> None:
        tokens = Lexer("A => B").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.CANONIZE in types

    def test_connotate(self) -> None:
        tokens = Lexer("A > B").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.CONNOTATE in types

    def test_underscore(self) -> None:
        tokens = Lexer("A = B").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.UNDERSIGN in types

    def test_less_than_is_error(self) -> None:
        with pytest.raises(LexerError):
            Lexer("A < B").tokenize()

    def test_le_is_error(self) -> None:
        """<= is no longer a valid operator, so <= should error on <."""
        with pytest.raises(LexerError):
            Lexer("A <= B").tokenize()

    def test_comment(self) -> None:
        tokens = Lexer("A (inline comment) => B").tokenize()
        assert any(t.type == TokenType.COMMENT for t in tokens)

    def test_multi_line_comment(self) -> None:
        source = "A => B\n(this is\na\ncomment)\nC => D"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comments) == 1
        assert "this is" in comments[0].value

    def test_nested_parens_in_comment(self) -> None:
        source = "A => B\n(outer (inner)\nstill outer)\nC => D"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        assert len(comments) == 1
        assert "inner" in comments[0].value

    def test_indent_dedent(self) -> None:
        source = "A =>\n  B\n  C"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

    def test_eof_always_present(self) -> None:
        tokens = Lexer("").tokenize()
        assert tokens[-1].type == TokenType.EOF

    def test_empty_source(self) -> None:
        tokens = Lexer("").tokenize()
        non_eof = [t for t in tokens if t.type != TokenType.EOF]
        assert len(non_eof) == 0


# =============================================================================
# 3. Parser AST tests
# =============================================================================

class TestParserAST:
    def test_simple_signature(self) -> None:
        tokens = Lexer("A").tokenize()
        kfile = Parser(tokens).parse()
        script = kfile.scripts[0]
        assert len(script.constructs) >= 1

    def test_chain_construct_canonize(self) -> None:
        tokens = Lexer("A => B").tokenize()
        kfile = Parser(tokens).parse()
        construct = kfile.scripts[0].constructs[0]
        assert isinstance(construct.inner, list)
        assert construct.chain_op == TokenType.CANONIZE
        assert construct.chain_right is not None

    def test_block_construct(self) -> None:
        tokens = Lexer("A =>\n  B\n  C").tokenize()
        kfile = Parser(tokens).parse()
        construct = kfile.scripts[0].constructs[0]
        assert construct.chain_op == TokenType.CANONIZE
        right = construct.chain_right
        assert isinstance(right.inner, Block)
        assert len(right.inner.constructs) >= 1

    def test_implicit_group(self) -> None:
        tokens = Lexer("A B => CD").tokenize()
        kfile = Parser(tokens).parse()
        assert len(kfile.scripts[0].constructs) >= 1

    def test_parse_error_on_invalid(self) -> None:
        tokens = Lexer("1 => A").tokenize()
        with pytest.raises(ParseError):
            Parser(tokens).parse()

    def test_empty_source_produces_empty_script(self) -> None:
        tokens = Lexer("").tokenize()
        kfile = Parser(tokens).parse()
        assert len(kfile.scripts) == 1
        assert len(kfile.scripts[0].constructs) == 0


# =============================================================================
# 3b. Comment parsing tests
# =============================================================================

class TestCommentParsing:
    """Tests for COMMENT token preservation and inline comment attachment."""

    def test_standalone_comment_preserved_as_comment_node(self) -> None:
        """Standalone comment on its own line is preserved as a Comment AST node."""
        tokens = Lexer("(a comment)\nA => B").tokenize()
        kfile = Parser(tokens).parse()
        constructs = kfile.scripts[0].constructs
        # First construct should be a Comment
        assert isinstance(constructs[0].inner, Comment)
        assert constructs[0].inner.text == "(a comment)"

    def test_inline_comment_attached_to_primary_construct(self) -> None:
        """Inline comment S(ubject) is attached to PrimaryConstruct.inline_comment."""
        tokens = Lexer("S(ubject) = M").tokenize()
        kfile = Parser(tokens).parse()
        construct = kfile.scripts[0].constructs[0]
        assert isinstance(construct.inner, list)
        primary = construct.inner[0]
        assert primary.sig.id == "S"
        assert primary.inline_comment is not None
        assert primary.inline_comment.text == "(ubject)"

    def test_inline_comment_on_unsigned_primary(self) -> None:
        """Inline comment on unsigned primary: S has inline_comment, B does not."""
        tokens = Lexer("S(note) B").tokenize()
        kfile = Parser(tokens).parse()
        construct = kfile.scripts[0].constructs[0]
        assert isinstance(construct.inner, list)
        assert len(construct.inner) == 2
        assert construct.inner[0].sig.id == "S"
        assert construct.inner[0].inline_comment is not None
        assert construct.inner[0].inline_comment.text == "(note)"
        assert construct.inner[1].sig.id == "B"
        assert construct.inner[1].inline_comment is None

    def test_inline_comment_in_block(self) -> None:
        """Inline comment inside a block construct is attached correctly."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V = H"
        tokens = Lexer(source).tokenize()
        kfile = Parser(tokens).parse()
        top = kfile.scripts[0].constructs[0]
        block = top.chain_right.inner
        assert isinstance(block, Block)
        # First construct in block: S(ubject) = M
        s_construct = block.constructs[0]
        assert isinstance(s_construct.inner, list)
        s_primary = s_construct.inner[0]
        assert s_primary.sig.id == "S"
        assert s_primary.inline_comment is not None
        assert s_primary.inline_comment.text == "(ubject)"

    def test_comment_between_constructs(self) -> None:
        """Standalone comment between constructs is parsed as its own construct."""
        tokens = Lexer("A => B\n(a note)\nC => D").tokenize()
        kfile = Parser(tokens).parse()
        constructs = kfile.scripts[0].constructs
        # Should have 3 constructs: A=>B, (a note), C=>D
        assert len(constructs) == 3
        # First: A => B
        assert constructs[0].chain_op == TokenType.CANONIZE
        # Second: (a note) as Comment
        assert isinstance(constructs[1].inner, Comment)
        assert constructs[1].inner.text == "(a note)"
        # Third: C => D
        assert constructs[2].chain_op == TokenType.CANONIZE

    def test_construct_without_comment_unchanged(self) -> None:
        """Backward compatibility: no Comment nodes, no inline_comment set."""
        tokens = Lexer("A => B").tokenize()
        kfile = Parser(tokens).parse()
        construct = kfile.scripts[0].constructs[0]
        assert isinstance(construct.inner, list)
        for primary in construct.inner:
            assert primary.inline_comment is None


# =============================================================================
# 4. Compiler basic tests
# =============================================================================

class TestCompilerBasic:
    def test_unsigned_identity(self) -> None:
        """Bare signature 'A' emits unsigned entry with empty nodes."""
        entries = compile64("A")
        md = _md(entries)
        # Unsigned entry has empty nodes (decoded as '')
        assert _has_node(md, "A", "")

    def test_countersign(self) -> None:
        entries = compile64("A == B")
        md = _md(entries)
        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])

    def test_undersign(self) -> None:
        entries = compile64("A = B")
        md = _md(entries)
        assert _has_node(md, "B", ["A"])

    def test_connotate(self) -> None:
        entries = compile64("A > B")
        md = _md(entries)
        assert _has_node(md, "A", ["B"])

    def test_canonize(self) -> None:
        entries = compile64("A => B")
        md = _md(entries)
        assert _has_node(md, "A", ["B"])

    def test_literal_node_connotate_string(self) -> None:
        entries = compile64('A > "hello"')
        md = _md(entries)
        assert _has_node(md, "A", '"hello"')

    def test_literal_node_connotate(self) -> None:
        entries = compile64("A > 1")
        md = _md(entries)
        assert _has_node(md, "A", "1")

    def test_quoted_string_literal(self) -> None:
        entries = compile64('A => "hello"')
        md = _md(entries)
        assert _has_node(md, "A", '"hello"')


# =============================================================================
# 5. MCS expansion tests
# =============================================================================

class TestMCSExpansion:
    def test_mcs_simple(self) -> None:
        """ABC emits component identities + MCS canonization."""
        entries = compile64("ABC")
        md = _md(entries)
        # Component identities (empty nodes)
        assert _has_node(md, "A", "")
        assert _has_node(md, "B", "")
        assert _has_node(md, "C", "")
        # MCS canonization: {ABC: [A, B, C]}
        assert _has_node(md, "ABC", ["A", "B", "C"])
        # ABC unsigned
        assert _has_node(md, "ABC", "")

    def test_mcs_in_construct(self) -> None:
        entries = compile64("ABC => X")
        md = _md(entries)
        assert _has_node(md, "ABC", ["X"])

    def test_no_mcs_for_single_char(self) -> None:
        entries = compile64("A => X")
        md = _md(entries)
        assert _has_node(md, "A", ["X"])

    def test_mcs_countersign(self) -> None:
        entries = compile64("ABC == X")
        md = _md(entries)
        assert _has_node(md, "ABC", ["X"])
        assert _has_node(md, "X", ["ABC"])


# =============================================================================
# 6. Chain tests
# =============================================================================

class TestChains:
    def test_canonize_chain(self) -> None:
        entries = compile64("A => B => C")
        md = _md(entries)
        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["C"])

    def test_per_item_canonize(self) -> None:
        entries = compile64("A => B C")
        md = _md(entries)
        assert _has_node(md, "A", ["B", "C"])

    def test_subscript_block(self) -> None:
        source = "A =>\n  B\n  C"
        entries = compile64(source)
        md = _md(entries)
        assert _has_node(md, "A", ["B", "C"])


# =============================================================================
# 7. Nested subscript tests
# =============================================================================

class TestNestedSubscripts:
    def test_nested_block(self) -> None:
        source = "A =>\n  B =>\n    C\n    D"
        entries = compile64(source)
        md = _md(entries)
        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["C", "D"])

    def test_subscript_with_inline_ops(self) -> None:
        source = "A =>\n  B\n  C = D"
        entries = compile64(source)
        md = _md(entries)
        assert _has_node(md, "A", ["B", "C"])
        assert _has_node(md, "D", ["C"])


# =============================================================================
# 8. Complex examples
# =============================================================================

class TestComplexExamples:
    def test_ab_arrow_a_b(self) -> None:
        entries = compile64("AB => A B")
        md = _md(entries)
        assert _has_node(md, "AB", ["A", "B"])

    def test_ab_double_equal_cd(self) -> None:
        entries = compile64("AB == CD")
        md = _md(entries)
        assert _has_node(md, "AB", ["A", "B"])  # MCS
        assert _has_node(md, "CD", ["C", "D"])  # MCS
        assert _has_node(md, "AB", ["CD"])      # countersign
        assert _has_node(md, "CD", ["AB"])      # reverse countersign

    def test_ab_gt_c(self) -> None:
        entries = compile64("AB > C")
        md = _md(entries)
        assert _has_node(md, "AB", ["C"])


# =============================================================================
# 9. Literal edge cases
# =============================================================================

class TestLiteralEdgeCases:
    def test_bare_literal_identity(self) -> None:
        entries = compile64("1")
        assert len(entries) == 1
        sig, nodes = entries[0].decode(_tok64)
        assert sig == "1"
        assert nodes == ""  # empty nodes → ''

    def test_bare_multiple_literals(self) -> None:
        entries = compile64("1 2 3")
        assert len(entries) == 3
        sigs = [e.decode(_tok64)[0] for e in entries]
        assert sigs == ["1", "2", "3"]

    def test_canonize_single_literal(self) -> None:
        entries = compile64("A => 1")
        md = _md(entries)
        assert _has_node(md, "A", "1")

    def test_block_mixed_literal_and_sig(self) -> None:
        source = "A =>\n  1\n  B"
        entries = compile64(source)
        md = _md(entries)
        assert _has_node(md, "A", ["1", "B"])

    def test_block_all_literals(self) -> None:
        source = "A =>\n  1\n  2\n  3"
        entries = compile64(source)
        md = _md(entries)
        # Consecutive literal tokens decode as a single concatenated string
        assert _has_node(md, "A", "123")

    def test_literal_cannot_own_chain(self) -> None:
        with pytest.raises(ParseError):
            tokens = Lexer("1 => A").tokenize()
            Parser(tokens).parse()

    def test_cannot_chain_through_literal(self) -> None:
        with pytest.raises(ParseError):
            tokens = Lexer("A => 1 => B").tokenize()
            Parser(tokens).parse()


# =============================================================================
# 10. Decompiler tests
# =============================================================================

class TestDecompiler:
    def _roundtrip(self, source: str) -> list[DecompiledEntry]:
        entries = compile64(source)
        return Decompiler(_tok64).decompile(entries)

    def _find(self, entries: list[DecompiledEntry], sig: str, level: str | None = None) -> dict | None:
        for e in entries:
            if e.sig == sig and (level is None or e.level == level):
                return e.to_dict()
        return None

    def _has_sig(self, entries: list[DecompiledEntry], sig: str) -> bool:
        return any(e.sig == sig for e in entries)

    def test_decompile_unsigned(self) -> None:
        result = self._roundtrip("A")
        entry = self._find(result, "A")
        assert entry is not None
        assert entry["nodes"] is None

    def test_decompile_undersign(self) -> None:
        result = self._roundtrip("A = B")
        entry = self._find(result, "B")
        assert entry is not None
        assert entry["nodes"] == "A"

    def test_decompile_countersign(self) -> None:
        result = self._roundtrip("A == B")
        entry = self._find(result, "A")
        assert entry is not None
        assert entry["nodes"] == "B"

    def test_decompile_connotate(self) -> None:
        result = self._roundtrip("A > B")
        entry = self._find(result, "A")
        assert entry is not None
        assert entry["nodes"] == "B"

    def test_decompile_literal_connotate(self) -> None:
        result = self._roundtrip('A > "hello"')
        entry = self._find(result, "A")
        assert entry is not None
        assert entry["nodes"] == '"hello"'

    def test_decompile_empty_input(self) -> None:
        result = Decompiler(_tok64).decompile([])
        assert result == []

    def test_decompile_subscript_block(self) -> None:
        source = "A =>\n  B\n  C"
        result = self._roundtrip(source)
        assert self._has_sig(result, "A")
        assert self._has_sig(result, "B")
        assert self._has_sig(result, "C")

    def test_decompile_mcs_preserves_name(self) -> None:
        result = self._roundtrip("ABC")
        entry = self._find(result, "ABC")
        assert entry is not None

    def test_decompile_mcs_with_countersign(self) -> None:
        result = self._roundtrip("AB == CD")
        assert self._has_sig(result, "AB")
        assert self._has_sig(result, "CD")

    def test_decompile_complex_nested(self) -> None:
        source = """MHALL == SVO =>
  S(ubject) = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O"""
        result = self._roundtrip(source)
        assert self._has_sig(result, "MHALL")
        assert self._has_sig(result, "SVO")

    def test_decompiled_entry_to_kscript(self) -> None:
        entry = DecompiledEntry(level="S4", sig="A", nodes=None)
        ks = entry.to_kscript()
        assert "A" in ks

    def test_decompiled_entry_to_dict(self) -> None:
        entry = DecompiledEntry(level="S1", sig="A", nodes="B")
        d = entry.to_dict()
        assert d["level"] == "S1"
        assert d["sig"] == "A"
        assert d["nodes"] == "B"


# =============================================================================
# 11. Decompiler MCS with Mod32
# =============================================================================

class TestDecompilerMCSMod32:
    def test_mcs_name_recovery(self) -> None:
        entries = compile32("AB")
        result = Decompiler(_tok32).decompile(entries)
        assert any(e.sig == "AB" for e in result)

    def test_mcs_with_mod32_tokenizer(self) -> None:
        entries = compile32("ABC => X")
        result = Decompiler(_tok32).decompile(entries)
        assert any(e.sig == "ABC" for e in result)

    def test_mcs_countersign_roundtrip(self) -> None:
        entries = compile32("AB == CD")
        result = Decompiler(_tok32).decompile(entries)
        assert any(e.sig == "AB" for e in result)
        assert any(e.sig == "CD" for e in result)


# =============================================================================
# 12. Output I/O tests
# =============================================================================

class TestOutputIO:
    def test_write_and_read_bin(self) -> None:
        entries = compile64("A = B")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            write_bin(entries, path)
            loaded = read_bin(path)
            assert len(loaded) == len(entries)
            for orig, loaded_entry in zip(entries, loaded):
                assert orig.signature == loaded_entry.signature
                assert orig.nodes == loaded_entry.nodes
        finally:
            path.unlink(missing_ok=True)

    def test_write_and_read_json(self) -> None:
        entries = compile64("A = B")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            write_json(entries, path, _tok64)
            loaded = read_json(path, _tok64)
            assert len(loaded) == len(entries)
        finally:
            path.unlink(missing_ok=True)

    def test_write_jsonl(self) -> None:
        entries = compile64("A = B")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            write_jsonl(entries, path, _tok64)
            content = path.read_text().strip()
            lines = content.split("\n")
            assert len(lines) == len(entries)
            for line in lines:
                obj = json.loads(line)
                assert isinstance(obj, dict)
        finally:
            path.unlink(missing_ok=True)

    def test_read_bin_invalid_magic(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"XXXX")
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="Invalid magic"):
                read_bin(path)
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# 13. CompiledEntry encode/decode tests
# =============================================================================

class TestCompiledEntryEncodeDecode:
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
# 14. KScript API tests
# =============================================================================

class TestKScriptAPI:
    def test_inline_source(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        assert len(ks.entries) >= 1

    def test_output_json(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            ks.output(path)
            data = json.loads(path.read_text())
            assert isinstance(data, list)
            assert len(data) >= 1
        finally:
            path.unlink(missing_ok=True)

    def test_output_bin(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            ks.output(path)
            content = path.read_bytes()
            assert content[:4] == b"KSC1"
        finally:
            path.unlink(missing_ok=True)

    def test_to_jsonl(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        lines = ks.to_jsonl()
        assert len(lines) >= 1
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_extend_base(self) -> None:
        from kscript import KScript
        base = KScript("A = B")
        extended = KScript("C = D", base=base)
        assert len(extended.entries) > len(base.entries)

    def test_load_from_bin_file(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            ks.output(path)
            loaded = KScript(path)
            assert len(loaded.entries) == len(ks.entries)
        finally:
            path.unlink(missing_ok=True)

    def test_load_from_json_file(self) -> None:
        from kscript import KScript
        ks = KScript("A = B")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            ks.output(path)
            loaded = KScript(path)
            assert len(loaded.entries) == len(ks.entries)
        finally:
            path.unlink(missing_ok=True)


# =============================================================================
# 15. KScript pipeline with NLPTokenizer
# =============================================================================

def _get_nlp_tokenizer():
    """Get NLPTokenizer instance, or None if unavailable."""
    if not _has_nlp:
        return None
    try:
        return NLPTokenizer.from_files()
    except Exception:
        return None


def compile_nlp(source: str) -> list[CompiledEntry]:
    """Compile with NLPTokenizer."""
    tok = _get_nlp_tokenizer()
    assert tok is not None, "NLPTokenizer not available"
    return compile_source(source, tokenizer=tok, dev=True)


@_nlp_skip
class TestKScriptNLP:
    """Tests exercising the full KScript pipeline with NLPTokenizer.

    These verify that the widened KTokenizer type hints work end-to-end.
    All tests skip if NLPTokenizer or its data files are unavailable.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_compile_simple_with_nlp(self) -> None:
        """Compile 'A == B' with NLP tokenizer — entries should decode back."""
        entries = compile_nlp("A == B")
        assert len(entries) >= 2
        # Verify all entries can be decoded without error
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_compile_unsigned_with_nlp(self) -> None:
        """Compile 'A' with NLP tokenizer — produces at least one entry."""
        entries = compile_nlp("A")
        assert len(entries) >= 1

    def test_compile_canonize_with_nlp(self) -> None:
        """Compile 'AB => X' with NLP tokenizer — produces entries."""
        entries = compile_nlp("AB => X")
        assert len(entries) >= 1

    def test_decompile_nlp(self) -> None:
        """Compile with NLP, decompile with same NLP tokenizer."""
        entries = compile_nlp("A == B")
        decompiled = Decompiler(self._nlp_tok).decompile(entries)
        assert len(decompiled) >= 1
        # Verify decompiled entries have correct sig strings
        sigs = [e.sig for e in decompiled]
        assert "A" in sigs
        assert "B" in sigs

    def test_kscript_api_with_nlp(self) -> None:
        """KScript API works with NLPTokenizer."""
        from kscript import KScript
        ks = KScript("A = B", tokenizer=self._nlp_tok)
        assert len(ks.entries) >= 1


# =============================================================================
# 16. KScript NLP integration tests (cross-module)
# =============================================================================

@_nlp_skip
class TestKScriptNLPIntegration:
    """Cross-module integration tests for KScript with NLP tokenizer.

    Verifies that compiled entries contain actual NLP-BPE nodes (high 32
    bits non-zero), that the decompiler can reconstruct source from them,
    and that encode/decode round-trips preserve symbolic names.

    These complement TestKScriptNLP by focusing on the node-level properties
    of the compilation output rather than just structural correctness.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_compiled_entries_have_nlp_bpe_nodes(self) -> None:
        """Compiled entry signatures are NLP-BPE nodes (high 32 bits non-zero)."""
        entries = compile_source("A => B", tokenizer=self._nlp_tok, dev=True)

        # All entries should have NLP-BPE signatures (high 32 bits set)
        for entry in entries:
            assert is_nlp_node(entry.signature), (
                f"Entry signature {entry.signature:#x} should be an NLP-BPE node"
            )

            # Non-empty node entries should have NLP-BPE node values
            for node in entry.nodes:
                assert is_nlp_node(node), (
                    f"Entry node {node:#x} should be an NLP-BPE node"
                )

    def test_decompiler_roundtrip_with_nlp(self) -> None:
        """Compile → decompile → verify symbolic names recovered."""
        entries = compile_source(
            "A == B\nC => D", tokenizer=self._nlp_tok, dev=True
        )
        decompiled = Decompiler(self._nlp_tok).decompile(entries)

        # All signature names should be recovered
        sigs = {e.sig for e in decompiled}
        assert "A" in sigs
        assert "B" in sigs
        assert "C" in sigs
        assert "D" in sigs

        # Verify node relationships survived
        countersign_entry = next(e for e in decompiled if e.sig == "A")
        assert countersign_entry.nodes == "B"

        canonize_entry = next(e for e in decompiled if e.sig == "C")
        assert canonize_entry.nodes == "D"

    def test_encode_decode_preserves_symbolic_names(self) -> None:
        """CompiledEntry.encode → decode recovers original symbolic names.

        Note: NLP tokenizer BPE-encodes multi-char strings to multiple nodes,
        but CompiledEntry.signature stores only a single int.  Single-char
        signatures (like 'A', 'B') encode to single NLP-BPE nodes and
        round-trip correctly.
        """
        entry = CompiledEntry.encode("A", ["B"], self._nlp_tok)
        sig, nodes = entry.decode(self._nlp_tok)

        assert sig == "A"
        assert nodes == ["B"]
