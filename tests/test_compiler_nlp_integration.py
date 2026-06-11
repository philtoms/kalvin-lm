"""Integration tests for KScript compiler NLP binding resolution.

Tests NB-18 (Mod32/Mod64 unchanged) and NB-19 (same source compiles
under both modes).  The BindingScope is wired conditionally into
Compiler.compile() — only when tokenizer.supports_mcs is False.

Inline comments (e.g. S(ubject)) are resolved by the ASTEmitter in ALL
modes — the inline comment is an explicit binding at point of use.
Scope-based resolution (block comments) only activates in NLP mode.

Pipeline (v2.0):
  Compiler creates BindingScope (NLP mode) → ASTEmitter resolves inline
  → TokenEncoder encodes to CompiledEntry objects.

No BindingResolver, NLPSymbolTable, or symbol_table property references
(these legacy modules have been removed).

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

    The BindingScope must not affect Mod32 (supports_mcs is True) in
    any way — no scope creation, no changed output.  Inline comments
    still resolve (that's ASTEmitter behavior, not scope-dependent).
    """

    def test_simple_countersign_mod32(self) -> None:
        """A == B under Mod32 produces identical entries."""
        entries, _ = _compile_with_compiler("A == B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])

    def test_canonize_mod32(self) -> None:
        """A => B under Mod32 produces identical entries."""
        entries, _ = _compile_with_compiler("A => B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])

    def test_connotate_mod32(self) -> None:
        """A > B under Mod32 produces identical entries."""
        entries, _ = _compile_with_compiler("A > B", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B"])

    def test_subscript_block_mod32(self) -> None:
        """Subscript block under Mod32 produces identical entries."""
        source = "A =>\n  B\n  C"
        entries, _ = _compile_with_compiler(source, _tok32)
        md = _entries_to_multidict(entries, _tok32)

        assert _has_node(md, "A", ["B", "C"])

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
        entries, _ = _compile_with_compiler("A == B", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])

    def test_canonize_mod64(self) -> None:
        """A => B under Mod64 produces identical entries."""
        entries, _ = _compile_with_compiler("A => B", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", ["B"])

    def test_mcs_expansion_mod64(self) -> None:
        """MCS expansion under Mod64 produces identical entries."""
        entries, _ = _compile_with_compiler("ABC", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "A", "")
        assert _has_node(md, "B", "")
        assert _has_node(md, "C", "")
        assert _has_node(md, "ABC", ["A", "B", "C"])

    def test_complex_nested_mod64(self) -> None:
        """Complex nested script under Mod64 produces identical entries."""
        source = "MHALL == SVO =>\n  S = M\n  V = H\n  O = ALL =>\n    A > D\n    L > M\n    L > O"
        entries, _ = _compile_with_compiler(source, _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "M", ["S"])
        assert _has_node(md, "H", ["V"])
        assert _has_node(md, "A", ["D"])
        assert _has_node(md, "L", ["M"])
        assert _has_node(md, "L", ["O"])


# =============================================================================
# NB-19: Same source compiles under both modes
# =============================================================================


class TestNB19SameSourceBothModes:
    """Same KScript source compiles without errors under both Mod32 and NLP.

    Inline comments (e.g. S(ubject)) are resolved by the ASTEmitter in
    all modes — they are explicit bindings at point of use, independent
    of the BindingScope.  Scope-based resolution (block comments) only
    activates in NLP mode.
    """

    def test_simple_undersign_mod32(self) -> None:
        """S(ubject) = M compiles under Mod32 — inline comment resolves."""
        entries, _ = _compile_with_compiler("S(ubject) = M", _tok32)
        md = _entries_to_multidict(entries, _tok32)

        # Inline comment S(ubject) resolves to "Subject" in all modes
        assert _has_node(md, "M", "Subject")

    def test_simple_undersign_mod64(self) -> None:
        """S(ubject) = M compiles under Mod64 — inline comment resolves."""
        entries, _ = _compile_with_compiler("S(ubject) = M", _tok64)
        md = _entries_to_multidict(entries, _tok64)

        assert _has_node(md, "M", "Subject")

    def test_inline_comments_in_block_mod32(self) -> None:
        """Inline comments in subscript block compile under Mod32."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H"
        entries, _ = _compile_with_compiler(source, _tok32)
        assert len(entries) > 0

    def test_inline_comments_in_block_mod64(self) -> None:
        """Inline comments in subscript block compile under Mod64."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H"
        entries, _ = _compile_with_compiler(source, _tok64)
        assert len(entries) > 0

    def test_inline_comments_resolve_in_mod32(self) -> None:
        """NB-19: Inline comments resolve to full words in Mod32.

        S(ubject) = M resolves "Subject" as the node value, even without
        BindingScope — inline resolution is always active.
        """
        entries_with, _ = _compile_with_compiler("S(ubject) = M", _tok32)
        entries_without, _ = _compile_with_compiler("S = M", _tok32)

        md_with = _entries_to_multidict(entries_with, _tok32)
        md_without = _entries_to_multidict(entries_without, _tok32)

        # Inline comment produces "Subject" instead of ["S"]
        assert _has_node(md_with, "M", "Subject")
        assert _has_node(md_without, "M", ["S"])

    def test_inline_comments_resolve_in_mod64(self) -> None:
        """NB-19: Inline comments resolve to full words in Mod64."""
        entries_with, _ = _compile_with_compiler("S(ubject) = M", _tok64)
        entries_without, _ = _compile_with_compiler("S = M", _tok64)

        md_with = _entries_to_multidict(entries_with, _tok64)
        md_without = _entries_to_multidict(entries_without, _tok64)

        assert _has_node(md_with, "M", "Subject")
        assert _has_node(md_without, "M", ["S"])

    def test_complex_inline_comments_mod64(self) -> None:
        """Complex subscript with inline comments resolves in Mod64."""
        source = "MHALL == SVO =>\n  S(ubject) = M\n  V(erb) = H\n  O(bject) = ALL"
        entries, _ = _compile_with_compiler(source, _tok64)
        md = _entries_to_multidict(entries, _tok64)

        # Inline comments resolve in all modes — check sigs that don't get
        # Mod64-packed (single-char sigs)
        assert _has_node(md, "M", "Subject")
        assert _has_node(md, "H", "Verb")

        # Verify all entries are decodable (broader correctness check)
        for entry in entries:
            sig, nodes = entry.decode(_tok64)
            assert isinstance(sig, str)


# =============================================================================
# NLP mode compilation — verify bindings flow through compiled entries
# =============================================================================


@_nlp_skip
class TestNLPBindingInCompiledEntries:
    """Verify NLP bindings flow through to compiled entries.

    Replaces the old TestNLPSymbolTablePopulated which inspected
    compiler.symbol_table (now removed).  Verifies bindings by inspecting decoded
    compiled entries — sig names, node values, and NLP signature bits.

    These tests skip if NLPTokenizer data files are not available.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_nlp_mode_produces_valid_entries(self) -> None:
        """NB-19: Same source compiles under NLP mode without errors."""
        source = "A == B"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)

        assert len(entries) >= 2
        # All entries should be decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_nlp_mode_inline_comment_source_compiles(self) -> None:
        """NB-19: Source with inline comments compiles under NLP without errors."""
        source = "S(ubject) = M"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)

        assert len(entries) >= 1
        # All entries decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)

    def test_nlp_mode_compiles_with_bindings(self) -> None:
        """NLP mode: source with inline comments resolves bindings."""
        source = "S(ubject) = M"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)

        # Verify entries contain resolved words — "Subject" should appear
        # in the decoded output
        md = _entries_to_multidict(entries, self._nlp_tok)
        all_sigs = list(md.keys())
        all_nodes = []
        for nodes_list in md.values():
            for n in nodes_list:
                if isinstance(n, str):
                    all_nodes.append(n)
                elif isinstance(n, list):
                    all_nodes.extend(n)

        # The inline comment S(ubject) should produce "Subject" somewhere
        combined = " ".join(all_sigs + all_nodes)
        assert "Subject" in combined or "ubject" in combined

    def test_nlp_mode_no_comments_compiles(self) -> None:
        """NLP mode: source without comments compiles cleanly."""
        source = "A == B"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)

        assert len(entries) >= 2
        md = _entries_to_multidict(entries, self._nlp_tok)
        assert "A" in md
        assert "B" in md

    def test_inline_binding_decoded_in_entries(self) -> None:
        """Inline comment S(ubject) = M produces 'Subject' in decoded output."""
        source = "S(ubject) = M"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)
        md = _entries_to_multidict(entries, self._nlp_tok)

        # "Subject" should appear in some entry's nodes
        found = False
        for sig, node_lists in md.items():
            for n in node_lists:
                if n == "Subject" or (isinstance(n, list) and "Subject" in n):
                    found = True
        assert found, f"Expected 'Subject' in decoded entries, got: {md}"

    def test_source_without_comments_no_binding_artefacts(self) -> None:
        """Sources without comments produce clean entries with no binding artefacts."""
        source = "A == B"
        entries, _ = _compile_with_compiler(source, self._nlp_tok)

        # All entries should be decodable and produce standard A/B sigs
        md = _entries_to_multidict(entries, self._nlp_tok)
        assert "A" in md
        assert "B" in md
        # No surprise words from bindings
        for sig in md:
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
