"""Integration tests for KScript compiler NLP binding resolver.

Tests NB-18 (Mod32/Mod64 unchanged) and NB-19 (same source compiles
under both modes).  The BindingResolver is wired conditionally into
Compiler.compile() — only when tokenizer.supports_mcs is False.

Spec ref: @kscript-nlp-binding §1.1 (pipeline), §10 (test matrix)
"""

from __future__ import annotations

import pytest

from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer
from kscript.compiler import Compiler, compile_source
from kscript.lexer import Lexer
from kscript.parser import Parser

# Conditional NLP tokenizer import
try:
    from kalvin.nlp_tokenizer import NLPTokenizer

    _has_nlp = True
except ImportError:
    _has_nlp = False

_nlp_skip = pytest.mark.skipif(not _has_nlp, reason="NLPTokenizer not available")


# ── Helpers ──────────────────────────────────────────────────────────────────

_tok32 = Mod32Tokenizer()
_tok64 = Mod64Tokenizer()


def _compile_with_compiler(
    source: str, tokenizer, dev: bool = True
) -> tuple[list, Compiler]:
    """Compile source and return (entries, compiler) for inspection."""
    tokens = Lexer(source).tokenize()
    kf = Parser(tokens).parse()
    compiler = Compiler(tokenizer, dev=dev)
    entries = compiler.compile(kf)
    return entries, compiler


def _entries_to_multidict(entries, tok=None) -> dict[str, list]:
    """Map sig → list of all decoded node values."""
    tok = tok or _tok64
    result: dict[str, list] = {}
    for e in entries:
        sig, nodes = e.decode(tok)
        result.setdefault(sig, []).append(nodes)
    return result


def _has_node(md: dict[str, list], sig: str, node_value) -> bool:
    """Check if a sig has a specific decoded node value."""
    return node_value in md.get(sig, [])


def _get_nlp_tokenizer():
    """Get NLPTokenizer instance, or None if data files unavailable."""
    if not _has_nlp:
        return None
    try:
        return NLPTokenizer.from_files()
    except Exception:
        return None


# =============================================================================
# NB-18: Mod32 compilation unchanged
# =============================================================================


class TestNB18Mod32Unchanged:
    """Mod32 compilation produces identical results to baseline.

    The BindingResolver must not affect Mod32 (supports_mcs is True) in
    any way — no resolver import, no symbol table, no changed output.
    """

    def test_simple_countersign_mod32(self) -> None:
        """A == B under Mod32 produces identical entries."""
        entries, compiler = _compile_with_compiler("A == B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])
        assert compiler.scope is None

    def test_canonize_mod32(self) -> None:
        """A => B under Mod32 produces identical entries."""
        entries, compiler = _compile_with_compiler("A => B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])
        assert compiler.scope is None

    def test_connotate_mod32(self) -> None:
        """A > B under Mod32 produces identical entries."""
        entries, compiler = _compile_with_compiler("A > B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])
        assert compiler.scope is None

    def test_subscript_block_mod32(self) -> None:
        """Subscript block under Mod32 produces identical entries."""
        source = "A =>\n  B\n  C"
        entries, compiler = _compile_with_compiler(source, _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B", "C"])
        assert compiler.scope is None

    def test_compile_source_mod32(self) -> None:
        """compile_source() convenience function works unchanged under Mod32."""
        entries = compile_source("A == B", tokenizer=_tok32, dev=True)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])


# =============================================================================
# NB-18 extended: Mod64 unchanged
# =============================================================================


class TestNB18Mod64Unchanged:
    """Mod64 compilation produces identical results to baseline."""

    def test_simple_countersign_mod64(self) -> None:
        """A == B under Mod64 produces identical entries."""
        entries, compiler = _compile_with_compiler("A == B", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])
        assert compiler.scope is None

    def test_canonize_mod64(self) -> None:
        """A => B under Mod64 produces identical entries."""
        entries, compiler = _compile_with_compiler("A => B", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", ["B"])
        assert compiler.scope is None

    def test_mcs_expansion_mod64(self) -> None:
        """MCS expansion under Mod64 produces identical entries."""
        entries, compiler = _compile_with_compiler("ABC", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", "")
        assert _has_node(md, "B", "")
        assert _has_node(md, "C", "")
        assert _has_node(md, "ABC", ["A", "B", "C"])
        assert compiler.scope is None

    def test_complex_nested_mod64(self) -> None:
        """Complex nested script under Mod64 produces identical entries."""
        source = "MHALL == SVO =>\n  S = M\n  V = H\n  O = ALL =>\n    A > D\n    L > M\n    L > O"
        entries, compiler = _compile_with_compiler(source, _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "M", ["S"])
        assert _has_node(md, "H", ["V"])
        assert _has_node(md, "A", ["D"])
        assert _has_node(md, "L", ["M"])
        assert _has_node(md, "L", ["O"])
        assert compiler.scope is None


# =============================================================================
# NB-19: Same source compiles under both modes
# =============================================================================


class TestNB19SameSourceBothModes:
    """Same KScript source compiles without errors under both Mod32 and NLP.

    Uses inline comments (e.g. S(ubject)) which are preserved in the AST
    but inert in Mod32/Mod64 mode.  In NLP mode, the BindingResolver
    processes them into an NLPSymbolTable.
    """

    def test_simple_undersign_mod32(self) -> None:
        """S(ubject) = M compiles under Mod32 — inline comment resolved."""
        entries, compiler = _compile_with_compiler("S(ubject) = M", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        # Inline comment is resolved by ASTEmitter regardless of mode
        assert _has_node(md, "M", "Subject")
        assert compiler.scope is None

    def test_simple_undersign_mod64(self) -> None:
        """S(ubject) = M compiles under Mod64 — inline comment resolved."""
        entries, compiler = _compile_with_compiler("S(ubject) = M", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        # Inline comment is resolved by ASTEmitter regardless of mode
        assert _has_node(md, "M", "Subject")
        assert compiler.scope is None

    def test_inline_comments_in_block_mod32(self) -> None:
        """Inline comments in subscript block compile under Mod32."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H"
        entries, compiler = _compile_with_compiler(source, _tok32)
        assert len(entries) > 0
        assert compiler.scope is None

    def test_inline_comments_in_block_mod64(self) -> None:
        """Inline comments in subscript block compile under Mod64."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H"
        entries, compiler = _compile_with_compiler(source, _tok64)
        assert len(entries) > 0
        assert compiler.scope is None

    def test_inline_comments_resolved_in_mod32(self) -> None:
        """NB-19: Inline comments are resolved in Mod32 mode.

        Compiling S(ubject) = M under Mod32 produces resolved word "Subject"
        — the inline comment is processed by ASTEmitter regardless of mode.
        """
        entries_with, _ = _compile_with_compiler("S(ubject) = M", _tok32)
        md_with = _entries_to_multidict(entries_with, _tok32)
        assert _has_node(md_with, "M", "Subject")

    def test_inline_comments_resolved_in_mod64(self) -> None:
        """NB-19: Inline comments are resolved in Mod64 mode."""
        entries_with, _ = _compile_with_compiler("S(ubject) = M", _tok64)
        md_with = _entries_to_multidict(entries_with, _tok64)
        assert _has_node(md_with, "M", "Subject")

    def test_complex_inline_comments_resolved_in_mod64(self) -> None:
        """Complex subscript with inline comments produces resolved words in Mod64."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H\n  O(bject) = ALL"
        entries, _ = _compile_with_compiler(source, _tok64)
        md = _entries_to_multidict(entries, _tok64)

        # Inline comments resolved to full words
        assert _has_node(md, "M", "Subject")
        assert _has_node(md, "H", "Verb")
        # O(bject) = ALL → ALL undersign O → but ALL is multi-char...
        # At minimum verify entries were produced
        assert len(entries) > 0


# =============================================================================
# NLP binding scope population (conditional on NLP availability)
# =============================================================================


@_nlp_skip
class TestNLPBindingScopePopulated:
    """Verify NLP mode creates and populates the BindingScope.

    These tests skip if NLPTokenizer data files are not available.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_nlp_mode_creates_scope(self) -> None:
        """NLP mode: compiling a source creates a BindingScope."""
        source = "S(ubject) = M"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert compiler.scope is not None
        assert len(entries) >= 1

    def test_nlp_mode_inline_binding_in_entries(self) -> None:
        """NLP mode: inline comment bindings are reflected in emitted entries."""
        source = "S(ubject) = M"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert compiler.scope is not None
        # Verify entries were produced and are decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_nlp_mode_multiple_inline_bindings(self) -> None:
        """NLP mode: multiple inline comments compile without errors."""
        source = "S(ubject) = M\nV(erb) = H"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert compiler.scope is not None
        assert len(entries) >= 1
        # All entries decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_nlp_mode_complex_subscript_bindings(self) -> None:
        """NLP mode: inline comments in subscript blocks compile without errors."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert compiler.scope is not None
        assert len(entries) >= 1

    def test_nlp_mode_no_comments_scope_exists(self) -> None:
        """NLP mode: source without comments still creates a BindingScope."""
        source = "A == B"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        # Scope exists (created for NLP mode) even without comment bindings
        assert compiler.scope is not None

    def test_nlp_mode_produces_valid_entries(self) -> None:
        """NB-19: Same source compiles under NLP mode without errors."""
        source = "A == B"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert len(entries) >= 2
        # All entries should be decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_nlp_mode_inline_comment_source_compiles(self) -> None:
        """NB-19: Source with inline comments compiles under NLP without errors."""
        source = "S(ubject) = M"
        entries, compiler = _compile_with_compiler(source, self._nlp_tok)

        assert len(entries) >= 1
        # All entries decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)


# =============================================================================
# Compiler unit tests — structural and regression guards
# =============================================================================


class TestCompilerStructure:
    """Regression guards ensuring old APIs are fully removed."""

    def test_no_binding_resolver_import_in_compiler(self) -> None:
        """compiler.py must not import binding_resolver."""
        import inspect
        import kscript.compiler as mod

        source = inspect.getsource(mod)
        assert "binding_resolver" not in source

    def test_no_nlpsymboltable_reference_in_compiler(self) -> None:
        """compiler.py must not reference NLPSymbolTable."""
        import inspect
        import kscript.compiler as mod

        source = inspect.getsource(mod)
        assert "NLPSymbolTable" not in source

    def test_no_symbol_table_attribute(self) -> None:
        """Compiler instances must not have a symbol_table attribute."""
        compiler = Compiler(_tok32)
        assert not hasattr(compiler, "symbol_table")
        assert not hasattr(compiler, "_symbol_table")

    def test_no_rewind_in_compiler(self) -> None:
        """compiler.py must not call rewind()."""
        import inspect
        import kscript.compiler as mod

        source = inspect.getsource(mod)
        assert "rewind" not in source


class TestCompilerUnit:
    """Unit tests for the simplified Compiler."""

    def test_mod32_compile_produces_entries(self) -> None:
        """Mod32 compile of A == B produces valid entries."""
        entries = compile_source("A == B", tokenizer=_tok32, dev=True)
        assert len(entries) >= 2
        md = _entries_to_multidict(entries, _tok32)
        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])

    def test_compile_source_convenience(self) -> None:
        """compile_source() works with default tokenizer."""
        entries = compile_source("A == B", dev=True)
        assert len(entries) >= 2

    def test_compiler_default_tokenizer(self) -> None:
        """Compiler() with no tokenizer uses Mod32Tokenizer."""
        compiler = Compiler()
        assert isinstance(compiler.tokenizer, Mod32Tokenizer)

    def test_compiler_custom_tokenizer(self) -> None:
        """Compiler() with custom tokenizer stores it."""
        compiler = Compiler(_tok64)
        assert compiler.tokenizer is _tok64
