"""Tests for ASTEmitter — AST traversal and symbolic entry emission.

Tests the pure AST logic of walking a parsed KScript AST and collecting
SymbolicEntry tuples, without any tokenizer involvement.
"""

from kscript.ast_emitter import ASTEmitter, SymbolicEntry
from kscript.lexer import Lexer
from kscript.parser import Parser

# ── Helpers ──────────────────────────────────────────────────────────────────

def emit_symbolic(source: str) -> list[SymbolicEntry]:
    """Parse source to AST, then emit symbolic entries."""
    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return ASTEmitter(dev=True).emit(kfile)


def find_entries(entries: list[SymbolicEntry], op: str) -> list[SymbolicEntry]:
    """Filter entries by operator."""
    return [e for e in entries if e.op == op]


# =============================================================================
# 1. Unsigned entries
# =============================================================================


class TestASTEmitterUnsigned:
    def test_bare_sig(self) -> None:
        entries = emit_symbolic("A")
        assert len(entries) == 1
        assert entries[0] == SymbolicEntry("A", None, "UNSIGNED")


# =============================================================================
# 2. Countersign
# =============================================================================


class TestASTEmitterCountersign:
    def test_countersign_bidirectional(self) -> None:
        entries = emit_symbolic("A == B")
        cs = find_entries(entries, "COUNTERSIGN")
        assert SymbolicEntry("A", "B", "COUNTERSIGN") in cs
        assert SymbolicEntry("B", "A", "COUNTERSIGN") in cs

    def test_countersign_count(self) -> None:
        entries = emit_symbolic("A == B")
        cs = find_entries(entries, "COUNTERSIGN")
        assert len(cs) == 2


# =============================================================================
# 3. Undersign
# =============================================================================


class TestASTEmitterUndersign:
    def test_undersign(self) -> None:
        entries = emit_symbolic("A = B")
        us = find_entries(entries, "UNDERSIGN")
        assert len(us) == 1
        assert us[0] == SymbolicEntry("B", "A", "UNDERSIGN")

    def test_undersign_identity(self) -> None:
        entries = emit_symbolic("A = A")
        ident = find_entries(entries, "IDENTITY")
        assert len(ident) == 1
        assert ident[0] == SymbolicEntry("A", None, "IDENTITY")


# =============================================================================
# 4. Connotate
# =============================================================================


class TestASTEmitterConnotate:
    def test_connotate(self) -> None:
        entries = emit_symbolic("A > B")
        con = find_entries(entries, "CONNOTATE")
        assert len(con) == 1
        assert con[0] == SymbolicEntry("A", "B", "CONNOTATE")



# =============================================================================
# 5. Canonize
# =============================================================================


class TestASTEmitterCanonize:
    def test_canonize_single(self) -> None:
        entries = emit_symbolic("A => B")
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 1
        assert can[0] == SymbolicEntry("A", "B", "CANONIZE")

    def test_canonize_multi(self) -> None:
        entries = emit_symbolic("A => B C")
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 1
        assert can[0] == SymbolicEntry("A", ["B", "C"], "CANONIZE")

    def test_canonize_chain(self) -> None:
        entries = emit_symbolic("A => B => C")
        can = find_entries(entries, "CANONIZE")
        # Two canonize steps: A=>B, B=>C
        assert len(can) == 2
        assert can[0] == SymbolicEntry("A", "B", "CANONIZE")
        assert can[1] == SymbolicEntry("B", "C", "CANONIZE")


# =============================================================================
# 6. Multi-Character Signatures (MCS)
# =============================================================================


class TestASTEmitterMCS:
    def test_mcs_three_char(self) -> None:
        entries = emit_symbolic("ABC")
        # Should contain unsigned for each char, canonize for the composite,
        # and unsigned for the composite itself
        unsigned = find_entries(entries, "UNSIGNED")
        can = find_entries(entries, "CANONIZE")
        assert SymbolicEntry("A", None, "UNSIGNED") in unsigned
        assert SymbolicEntry("B", None, "UNSIGNED") in unsigned
        assert SymbolicEntry("C", None, "UNSIGNED") in unsigned
        assert SymbolicEntry("ABC", None, "UNSIGNED") in unsigned
        assert SymbolicEntry("ABC", ["A", "B", "C"], "CANONIZE") in can

    def test_mcs_single_char_no_mcs(self) -> None:
        entries = emit_symbolic("A")
        # Single char should not produce MCS canonize entries
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 0


# =============================================================================
# 7. Subscripts
# =============================================================================


class TestASTEmitterSubscript:
    def test_basic_subscript(self) -> None:
        entries = emit_symbolic("A =>\n  B\n  C")
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 1
        assert can[0] == SymbolicEntry("A", ["B", "C"], "CANONIZE")

    def test_nested_subscript(self) -> None:
        entries = emit_symbolic("A =>\n  B =>\n    C\n    D")
        can = find_entries(entries, "CANONIZE")
        # A=>B (singleton unwrap), B=>[C,D]
        assert SymbolicEntry("A", "B", "CANONIZE") in can
        assert SymbolicEntry("B", ["C", "D"], "CANONIZE") in can


# =============================================================================
# 8. Dedup
# =============================================================================


class TestASTEmitterDedup:
    def test_no_duplicate_entries(self) -> None:
        entries = emit_symbolic("A == A")
        # Should produce exactly one COUNTERSIGN entry (A, A)
        # (no bidirectional since both sides are the same)
        cs = find_entries(entries, "COUNTERSIGN")
        assert len(cs) == 1
        assert cs[0] == SymbolicEntry("A", "A", "COUNTERSIGN")

    def test_dedup_across_constructs(self) -> None:
        # Two identical unsigned entries should be deduped
        entries = emit_symbolic("A\nA")
        unsigned = find_entries(entries, "UNSIGNED")
        assert len(unsigned) == 1


# =============================================================================
# 9. Singleton Unwrap
# =============================================================================


class TestASTEmitterSingletonUnwrap:
    def test_singleton_list_unwrapped(self) -> None:
        # A canonize with a single item should unwrap to a string
        entries = emit_symbolic("A => B")
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 1
        # nodes should be "B" (string), not ["B"] (list)
        assert can[0].nodes == "B"
        assert isinstance(can[0].nodes, str)

    def test_multi_item_list_preserved(self) -> None:
        entries = emit_symbolic("A => B C")
        can = find_entries(entries, "CANONIZE")
        assert len(can) == 1
        assert can[0].nodes == ["B", "C"]
        assert isinstance(can[0].nodes, list)


# =============================================================================
# 10. Comment constructs
# =============================================================================


class TestASTEmitterComment:
    def test_standalone_comment_produces_no_entries(self) -> None:
        """A standalone comment construct should be silently skipped."""
        entries = emit_symbolic("(Mary had a little lamb)")
        assert entries == []

    def test_comment_mixed_with_real_constructs(self) -> None:
        """Comments interleaved with real constructs should not produce
        entries or crash."""
        entries = emit_symbolic("A\n(Mary had a little lamb)\nB")
        unsigned = find_entries(entries, "UNSIGNED")
        assert len(unsigned) == 2
        assert SymbolicEntry("A", None, "UNSIGNED") in unsigned
        assert SymbolicEntry("B", None, "UNSIGNED") in unsigned
        # No comment-derived entries
        assert all("Mary" not in str(e) for e in entries)

    def test_inline_comment_on_primary_still_works(self) -> None:
        """Inline comments on primary constructs should not affect emission."""
        entries = emit_symbolic("(note) A == B")
        cs = find_entries(entries, "COUNTERSIGN")
        assert SymbolicEntry("A", "B", "COUNTERSIGN") in cs
        assert SymbolicEntry("B", "A", "COUNTERSIGN") in cs
