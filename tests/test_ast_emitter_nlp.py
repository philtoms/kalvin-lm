"""Tests for ASTEmitter NLP symbol table integration.

Covers KB-161 acceptance criteria:
  - NB-15 proxy: emitter produces resolved NLP word as sig
  - NB-16 proxy: emitter produces resolved NLP word as node
  - NB-17: mixed MCS with bound and unbound characters
"""

from __future__ import annotations

import pytest

from kscript.ast import (
    Block,
    Construct,
    KScriptFile,
    Literal,
    PrimaryConstruct,
    Script,
    Signature,
)
from kscript.ast_emitter import ASTEmitter, SymbolicEntry
from kscript.symbol_table import NLPSymbolTable
from kscript.token import TokenType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(**bindings: str) -> NLPSymbolTable:
    """Create an NLPSymbolTable with the given char→word bindings."""
    table = NLPSymbolTable()
    table.push_scope()
    for char, word in bindings.items():
        table.bind(char, word)
    return table


def _make_file(*constructs: Construct) -> KScriptFile:
    """Create a minimal KScriptFile wrapping the given constructs."""
    return KScriptFile(scripts=[Script(constructs=list(constructs))])


# ---------------------------------------------------------------------------
# Step 1: _resolve_sig_word unit tests
# ---------------------------------------------------------------------------

class TestResolveSigWord:
    """Tests for ASTEmitter._resolve_sig_word."""

    def test_bound_char_returns_word(self) -> None:
        """When M is bound to 'Mary', _resolve_sig_word returns 'Mary'."""
        table = _make_table(M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        assert emitter._resolve_sig_word("M") == "Mary"

    def test_unbound_char_returns_char(self) -> None:
        """When Z has no binding, _resolve_sig_word returns 'Z'."""
        table = _make_table(M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        assert emitter._resolve_sig_word("Z") == "Z"

    def test_no_symbol_table_returns_char(self) -> None:
        """With symbol_table=None, _resolve_sig_word returns the raw char."""
        emitter = ASTEmitter()
        assert emitter._resolve_sig_word("M") == "M"

    def test_empty_symbol_table_returns_char(self) -> None:
        """With an empty symbol table (no bindings), returns raw char."""
        table = NLPSymbolTable()
        table.push_scope()
        emitter = ASTEmitter(symbol_table=table)
        assert emitter._resolve_sig_word("M") == "M"

    def test_single_char_passthrough(self) -> None:
        """Non-alpha chars are passed through unchanged."""
        emitter = ASTEmitter()
        assert emitter._resolve_sig_word("1") == "1"


# ---------------------------------------------------------------------------
# Step 2: _emit_entry uses resolved words
# ---------------------------------------------------------------------------

class TestEmitEntryResolved:
    """Tests for _emit_entry with NLP word resolution."""

    def test_single_char_bound_sig(self) -> None:
        """Single-char sig with binding → sig field carries resolved word."""
        table = _make_table(M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("M", None, "UNSIGNED")
        assert len(emitter.entries) == 1
        assert emitter.entries[0].sig == "Mary"
        assert emitter.entries[0].op == "UNSIGNED"

    def test_single_char_unbound_sig(self) -> None:
        """Single-char sig without binding → sig field stays raw char."""
        emitter = ASTEmitter()
        emitter._emit_entry("Z", None, "UNSIGNED")
        assert len(emitter.entries) == 1
        assert emitter.entries[0].sig == "Z"

    def test_single_char_bound_countersign(self) -> None:
        """Countersign with bound sig → resolved sig in entry."""
        table = _make_table(S="Subject", M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("S", "M", "COUNTERSIGN")
        # sig resolved, node is raw str "M" — _emit_entry only resolves sig
        entry = emitter.entries[0]
        assert entry.sig == "Subject"
        # node stays as raw string "M" — node resolution happens in _emit_primary

    def test_multi_char_sig_not_resolved_in_emit_entry(self) -> None:
        """Multi-char sig is NOT resolved by _emit_entry — handled by _emit_mcs."""
        table = _make_table(M="Mary", H="had")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("MH", None, "UNSIGNED")
        assert len(emitter.entries) == 1
        # Multi-char sig should pass through unchanged in _emit_entry
        assert emitter.entries[0].sig == "MH"

    def test_dedup_with_resolved_words(self) -> None:
        """Two emissions of the same resolved word are deduped."""
        table = _make_table(M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("M", None, "UNSIGNED")
        emitter._emit_entry("M", None, "UNSIGNED")
        assert len(emitter.entries) == 1
        assert emitter.entries[0].sig == "Mary"

    def test_dedup_different_resolved_words_both_kept(self) -> None:
        """Two different chars that resolve to different words are both kept."""
        table = _make_table(M="Mary", H="had")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("M", None, "UNSIGNED")
        emitter._emit_entry("H", None, "UNSIGNED")
        assert len(emitter.entries) == 2
        assert emitter.entries[0].sig == "Mary"
        assert emitter.entries[1].sig == "had"

    def test_dedup_raw_char_no_collision_with_resolved(self) -> None:
        """A raw 'Mary' sig and a resolved M→'Mary' DO collide (same string)."""
        table = _make_table(M="Mary")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_entry("M", None, "UNSIGNED")  # resolves to "Mary"
        emitter._emit_entry("Mary", None, "UNSIGNED")  # raw "Mary"
        # Both have sig="Mary", None, so they dedup — this is correct behaviour
        assert len(emitter.entries) == 1


# ---------------------------------------------------------------------------
# Step 3: _emit_mcs per-character resolution
# ---------------------------------------------------------------------------

class TestEmitMcsResolved:
    """Tests for _emit_mcs with per-character NLP resolution."""

    def test_mixed_mcs_bound_and_unbound(self) -> None:
        """MCS 'MHALL' with M→Mary, H→had, A→a bound but L unbound.

        NB-17: unsigned entries carry resolved words for bound chars,
        raw chars for unbound. Canonization entry has original sig string
        with resolved-word nodes.
        """
        table = _make_table(M="Mary", H="had", A="a")
        emitter = ASTEmitter(symbol_table=table)
        result = emitter._emit_mcs("MHALL")

        assert result is True
        # 4 unique unsigned entries + 1 canonize = 5 total
        # (both L's dedup to the same "L" entry)
        assert len(emitter.entries) == 5

        # Unsigned entries: M→Mary, H→had, A→a, L→L (unbound)
        unsigned_entries = [e for e in emitter.entries if e.op == "UNSIGNED"]
        assert len(unsigned_entries) == 4
        assert unsigned_entries[0].sig == "Mary"
        assert unsigned_entries[1].sig == "had"
        assert unsigned_entries[2].sig == "a"
        # L is unbound → stays "L"; second L deduped
        assert unsigned_entries[3].sig == "L"

        # Canonize entry keeps original sig "MHALL"
        canonize_entries = [e for e in emitter.entries if e.op == "CANONIZE"]
        assert len(canonize_entries) == 1
        assert canonize_entries[0].sig == "MHALL"
        # Nodes contain resolved words for bound chars, raw chars for unbound
        assert canonize_entries[0].nodes == ["Mary", "had", "a", "L", "L"]

    def test_all_bound_mcs(self) -> None:
        """MCS with all characters bound → all unsigned entries carry words."""
        table = _make_table(M="Mary", H="had", A="a", L="little")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_mcs("MHA")

        unsigned_entries = [e for e in emitter.entries if e.op == "UNSIGNED"]
        assert len(unsigned_entries) == 3
        assert unsigned_entries[0].sig == "Mary"
        assert unsigned_entries[1].sig == "had"
        assert unsigned_entries[2].sig == "a"

        canonize = [e for e in emitter.entries if e.op == "CANONIZE"][0]
        assert canonize.sig == "MHA"
        assert canonize.nodes == ["Mary", "had", "a"]

    def test_all_unbound_mcs_no_symbol_table(self) -> None:
        """MCS with no symbol table → behaviour identical to current."""
        emitter = ASTEmitter()
        emitter._emit_mcs("ABC")

        unsigned_entries = [e for e in emitter.entries if e.op == "UNSIGNED"]
        assert len(unsigned_entries) == 3
        assert unsigned_entries[0].sig == "A"
        assert unsigned_entries[1].sig == "B"
        assert unsigned_entries[2].sig == "C"

        canonize = [e for e in emitter.entries if e.op == "CANONIZE"][0]
        assert canonize.sig == "ABC"
        assert canonize.nodes == ["A", "B", "C"]

    def test_mcs_dedup_with_resolved(self) -> None:
        """MCS dedup works — duplicate resolved words are deduped."""
        table = _make_table(A="alpha", B="alpha")
        emitter = ASTEmitter(symbol_table=table)
        emitter._emit_mcs("AB")

        unsigned_entries = [e for e in emitter.entries if e.op == "UNSIGNED"]
        # Both A and B resolve to "alpha" → first emitted, second deduped
        assert len(unsigned_entries) == 1
        assert unsigned_entries[0].sig == "alpha"


# ---------------------------------------------------------------------------
# Step 4: _emit_primary node resolution
# ---------------------------------------------------------------------------

class TestEmitPrimaryResolved:
    """Tests for _emit_primary with NLP word resolution for sig and node."""

    def _emit_primary_with_table(
        self, sig: str, op: TokenType | None, node_id: str | None,
        **bindings: str,
    ) -> list[SymbolicEntry]:
        """Helper: emit a PrimaryConstruct and return all entries."""
        table = _make_table(**bindings)
        emitter = ASTEmitter(symbol_table=table)

        sig_node = Signature(sig, line=1, column=1)
        node = Signature(node_id, line=1, column=2) if node_id is not None else None
        pc = PrimaryConstruct(sig=sig_node, op=op, node=node)

        emitter._emit_primary(pc)
        return emitter.entries

    def test_countersign_both_bound(self) -> None:
        """S = M with S→Subject, M→Mary → both fields carry resolved words."""
        entries = self._emit_primary_with_table(
            "S", TokenType.COUNTERSIGN, "M",
            S="Subject", M="Mary",
        )
        # First entry: sig=Subject, nodes=Mary, COUNTERSIGN
        cs_entries = [e for e in entries if e.op == "COUNTERSIGN"]
        assert len(cs_entries) >= 1
        assert cs_entries[0].sig == "Subject"
        assert cs_entries[0].nodes == "Mary"

    def test_countersign_sig_bound_node_unbound(self) -> None:
        """S = M with only S bound → sig resolved, node raw char."""
        entries = self._emit_primary_with_table(
            "S", TokenType.COUNTERSIGN, "M",
            S="Subject",
        )
        cs_entries = [e for e in entries if e.op == "COUNTERSIGN"]
        assert cs_entries[0].sig == "Subject"
        assert cs_entries[0].nodes == "M"

    def test_countersign_reverse_entry_resolved(self) -> None:
        """S = M (both Signature) → reverse countersign M → S also resolved."""
        entries = self._emit_primary_with_table(
            "S", TokenType.COUNTERSIGN, "M",
            S="Subject", M="Mary",
        )
        cs_entries = [e for e in entries if e.op == "COUNTERSIGN"]
        # Should have 2 countersign entries: S→M and M→S
        assert len(cs_entries) == 2
        assert cs_entries[0].sig == "Subject"
        assert cs_entries[0].nodes == "Mary"
        assert cs_entries[1].sig == "Mary"
        assert cs_entries[1].nodes == "Subject"

    def test_undersign_bound(self) -> None:
        """L > O with L→little, O→Object → undersign resolved."""
        entries = self._emit_primary_with_table(
            "L", TokenType.UNDERSIGN, "O",
            L="little", O="Object",
        )
        us_entries = [e for e in entries if e.op == "UNDERSIGN"]
        assert len(us_entries) == 1
        assert us_entries[0].sig == "Object"
        assert us_entries[0].nodes == "little"

    def test_connotate_bound(self) -> None:
        """S : M with both bound → connotate resolved."""
        entries = self._emit_primary_with_table(
            "S", TokenType.CONNOTATE, "M",
            S="Subject", M="Mary",
        )
        con_entries = [e for e in entries if e.op == "CONNOTATE"]
        assert len(con_entries) == 1
        assert con_entries[0].sig == "Subject"
        assert con_entries[0].nodes == "Mary"

    def test_unsigned_primary_bound(self) -> None:
        """Primary with no op but bound sig → unsigned with resolved word."""
        entries = self._emit_primary_with_table(
            "M", None, None,
            M="Mary",
        )
        assert len(entries) >= 1
        unsigned = [e for e in entries if e.op == "UNSIGNED"]
        assert any(e.sig == "Mary" for e in unsigned)

    def test_no_symbol_table_unchanged(self) -> None:
        """With no symbol table, _emit_primary behaves as before."""
        emitter = ASTEmitter()
        pc = PrimaryConstruct(
            sig=Signature("S", line=1, column=1),
            op=TokenType.COUNTERSIGN,
            node=Signature("M", line=1, column=2),
        )
        emitter._emit_primary(pc)
        cs_entries = [e for e in emitter.entries if e.op == "COUNTERSIGN"]
        assert len(cs_entries) == 2
        assert cs_entries[0].sig == "S"
        assert cs_entries[0].nodes == "M"
        assert cs_entries[1].sig == "M"
        assert cs_entries[1].nodes == "S"


# ---------------------------------------------------------------------------
# Step 5: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure default behaviour (no symbol_table) is unchanged."""

    def test_emit_mcs_no_table(self) -> None:
        """MCS emission without symbol table produces same results as before."""
        emitter = ASTEmitter()
        emitter._emit_mcs("ABC")

        assert len(emitter.entries) == 4  # 3 unsigned + 1 canonize
        unsigned = [e for e in emitter.entries if e.op == "UNSIGNED"]
        canonize = [e for e in emitter.entries if e.op == "CANONIZE"]
        assert [e.sig for e in unsigned] == ["A", "B", "C"]
        assert canonize[0].sig == "ABC"
        assert canonize[0].nodes == ["A", "B", "C"]

    def test_emit_entry_no_table(self) -> None:
        """_emit_entry without symbol table is a passthrough."""
        emitter = ASTEmitter()
        emitter._emit_entry("M", None, "UNSIGNED")
        assert emitter.entries[0].sig == "M"

    def test_resolve_returns_char_when_no_table(self) -> None:
        """_resolve_sig_word returns char when no table is set."""
        emitter = ASTEmitter()
        assert emitter._resolve_sig_word("X") == "X"

    def test_full_emit_no_table(self) -> None:
        """Full emit cycle without symbol table produces baseline results."""
        source = _make_file(
            Construct([
                PrimaryConstruct(
                    Signature("S", line=1, column=1),
                    TokenType.COUNTERSIGN,
                    Signature("M", line=1, column=2),
                ),
            ]),
        )
        emitter = ASTEmitter()
        entries = emitter.emit(source)

        # Should produce: MCS for S, MCS for M, unsigned S, unsigned M,
        # countersign S→M, countersign M→S
        sigs = [(e.sig, e.nodes, e.op) for e in entries]
        assert ("S", "M", "COUNTERSIGN") in sigs
        assert ("M", "S", "COUNTERSIGN") in sigs


# ---------------------------------------------------------------------------
# KB-166: Scope-aware emission tests
# ---------------------------------------------------------------------------

def _sig(id: str) -> Signature:
    """Shorthand for creating a Signature."""
    return Signature(id=id, line=1, column=1)


def _pc(sig_id: str, op=None, node_id: str | None = None) -> PrimaryConstruct:
    """Shorthand for creating a PrimaryConstruct."""
    node = Signature(node_id, line=1, column=2) if node_id is not None else None
    return PrimaryConstruct(sig=_sig(sig_id), op=op, node=node)


def _find_entry(entries: list[SymbolicEntry], sig: str, op: str) -> SymbolicEntry | None:
    """Find first entry matching sig and op."""
    for e in entries:
        if e.sig == sig and e.op == op:
            return e
    return None


# ---------------------------------------------------------------------------
# NB-8: Upward traversal
# ---------------------------------------------------------------------------


class TestNB8UpwardTraversal:
    """NB-8 — M in inner subscript resolves to 'Mary' from root scope.

    When processing S=M inside a subscript, M resolves upward through the
    scope parent chain to find M→'Mary' in root scope.
    """

    def test_m_resolves_upward_in_subscript(self) -> None:
        """S=M in subscript: M resolves to 'Mary' from root scope."""
        # Build scopes: root (M→Mary), child (S→Subject)
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # child (1)
        table.bind("S", "Subject")
        table.pop_scope()
        table.pop_scope()
        table.rewind()

        # AST: MHALL => Block([S = M])
        ast = _make_file(
            Construct(
                inner=[_pc("MHALL")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", TokenType.COUNTERSIGN, "M")]
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=True, symbol_table=table)
        entries = emitter.emit(ast)

        # Countersign entries should have resolved NLP words
        cs = [e for e in entries if e.op == "COUNTERSIGN"]
        # S=M → sig="Subject" (resolved at child), nodes="Mary" (upward from root)
        assert any(e.sig == "Subject" and e.nodes == "Mary" for e in cs), (
            f"Expected countersign (Subject, Mary); got {cs}"
        )
        # Reverse: M=S → sig="Mary" (upward from root), nodes="Subject"
        assert any(e.sig == "Mary" and e.nodes == "Subject" for e in cs), (
            f"Expected reverse countersign (Mary, Subject); got {cs}"
        )

    def test_upward_with_no_parent_binding_returns_raw(self) -> None:
        """Unbound char in subscript stays raw (no parent binding)."""
        table = NLPSymbolTable()
        table.push_scope()  # root (0) — no bindings
        table.push_scope()  # child (1) — S→Subject
        table.bind("S", "Subject")
        table.pop_scope()
        table.pop_scope()
        table.rewind()

        # AST: X => Block([S = M])
        ast = _make_file(
            Construct(
                inner=[_pc("X")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", TokenType.COUNTERSIGN, "M")]
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=True, symbol_table=table)
        entries = emitter.emit(ast)

        cs = [e for e in entries if e.op == "COUNTERSIGN"]
        # M has no binding anywhere → stays raw "M"
        assert any(e.sig == "Subject" and e.nodes == "M" for e in cs)


# ---------------------------------------------------------------------------
# NB-9: Downward traversal
# ---------------------------------------------------------------------------


class TestNB9DownwardTraversal:
    """NB-9 — S in SVO resolves to 'Subject' via child scope preview.

    When processing MHALL == SVO at root level, SVO's characters are
    decomposed via _emit_mcs. The emitter peeks at the child scope to
    discover S→Subject, V→Verb, O→Object before processing left primaries.
    """

    def test_svo_mcs_resolves_via_child_preview(self) -> None:
        """SVO MCS decomposition resolves S→Subject via peek."""
        # Scopes: root (M→Mary), child (S→Subject, V→Verb, O→Object)
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # child (1)
        table.bind("S", "Subject")
        table.bind("V", "Verb")
        table.bind("O", "Object")
        table.pop_scope()
        table.pop_scope()
        table.rewind()

        # AST: MHALL == SVO => Block([S = M])
        ast = _make_file(
            Construct(
                inner=[
                    _pc("MHALL", TokenType.COUNTERSIGN, "SVO"),
                ],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", TokenType.COUNTERSIGN, "M")]
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=False, symbol_table=table)
        entries = emitter.emit(ast)

        # Verify SVO MCS decomposition resolved S→Subject, V→Verb, O→Object
        unsigned = [e for e in entries if e.op == "UNSIGNED"]
        unsigned_sigs = {e.sig for e in unsigned}

        assert "Subject" in unsigned_sigs, (
            f"Expected 'Subject' in unsigned sigs; got {unsigned_sigs}"
        )
        assert "Verb" in unsigned_sigs, (
            f"Expected 'Verb' in unsigned sigs; got {unsigned_sigs}"
        )
        assert "Object" in unsigned_sigs, (
            f"Expected 'Object' in unsigned sigs; got {unsigned_sigs}"
        )

        # Verify canonize entry for SVO has resolved nodes
        svo_canon = _find_entry(entries, "SVO", "CANONIZE")
        assert svo_canon is not None
        assert svo_canon.nodes == ["Subject", "Verb", "Object"], (
            f"Expected SVO canonize nodes [Subject, Verb, Object]; got {svo_canon.nodes}"
        )

    def test_without_rewind_s_stays_raw(self) -> None:
        """Without rewind(), S in SVO stays raw 'S' (no scope navigation)."""
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # child (1)
        table.bind("S", "Subject")
        table.bind("V", "Verb")
        table.bind("O", "Object")
        table.pop_scope()   # child popped; root still on stack
        # No rewind — walk mode disabled, resolve uses active stack (root only)

        ast = _make_file(
            Construct(
                inner=[
                    _pc("MHALL", TokenType.COUNTERSIGN, "SVO"),
                ],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", TokenType.COUNTERSIGN, "M")]
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=False, symbol_table=table)
        entries = emitter.emit(ast)

        # SVO MCS: S stays raw "S" (no downward traversal, only root scope visible)
        svo_canon = _find_entry(entries, "SVO", "CANONIZE")
        assert svo_canon is not None
        assert "S" in svo_canon.nodes, (
            f"Without rewind, SVO nodes should contain raw 'S'; got {svo_canon.nodes}"
        )


# ---------------------------------------------------------------------------
# NB-12: Shadowing
# ---------------------------------------------------------------------------


class TestNB12Shadowing:
    """NB-12 — M→'Mod' in inner scope shadows M→'Mary' from root.

    Nested subscripts where root has M→Mary and inner subscript rebinds
    M→Mod. Inner scope entries see M→Mod; outer scope entries see M→Mary.
    """

    def test_inner_scope_shadows_outer(self) -> None:
        """L > M in inner subscript resolves M→'Mod' (shadows Mary)."""
        # Scopes: root (M→Mary), scope1 (empty), scope2 (M→Mod)
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # scope1 (1)
        table.push_scope()  # scope2 (2)
        table.bind("M", "Mod")
        table.pop_scope()   # scope2 popped
        table.pop_scope()   # scope1 popped
        table.pop_scope()   # root popped
        table.rewind()

        # AST: MHALL => Block([O = ALL => Block([L > M])])
        ast = _make_file(
            Construct(
                inner=[_pc("MHALL")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("O", TokenType.COUNTERSIGN, "ALL")],
                            chain_op=TokenType.CANONIZE,
                            chain_right=Construct(
                                inner=Block([
                                    Construct(
                                        inner=[_pc("L", TokenType.UNDERSIGN, "M")]
                                    )
                                ])
                            )
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=True, symbol_table=table)
        entries = emitter.emit(ast)

        # L > M undersign in scope2: M resolves to "Mod" (shadowed)
        us = [e for e in entries if e.op == "UNDERSIGN"]
        assert any(e.sig == "Mod" and e.nodes == "L" for e in us), (
            f"Expected undersign (Mod, L); got {us}"
        )

    def test_scope_restoration_after_shadowing(self) -> None:
        """After exiting shadowed scope, M resolves back to 'Mary'."""
        # Scopes: root (M→Mary), scope1 (empty), scope2 (M→Mod)
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # scope1 (1)
        table.push_scope()  # scope2 (2)
        table.bind("M", "Mod")
        table.pop_scope()   # scope2
        table.pop_scope()   # scope1
        table.pop_scope()   # root
        table.rewind()

        # AST: MHALL => Block([O = ALL => Block([L > M]), X = M])
        # After L > M in scope2 (M→Mod), exit to scope1, then X = M
        # where M resolves to "Mary" (restored)
        ast = _make_file(
            Construct(
                inner=[_pc("MHALL")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("O", TokenType.COUNTERSIGN, "ALL")],
                            chain_op=TokenType.CANONIZE,
                            chain_right=Construct(
                                inner=Block([
                                    Construct(
                                        inner=[_pc("L", TokenType.UNDERSIGN, "M")]
                                    )
                                ])
                            )
                        ),
                        Construct(
                            inner=[_pc("X", TokenType.COUNTERSIGN, "M")]
                        )
                    ])
                )
            )
        )

        emitter = ASTEmitter(skip_mcs=True, symbol_table=table)
        entries = emitter.emit(ast)

        # L > M in scope2: M→Mod (shadowed)
        us = [e for e in entries if e.op == "UNDERSIGN"]
        assert any(e.sig == "Mod" for e in us), (
            f"Expected Mod undersign; got {us}"
        )

        # X = M at scope1: M→Mary (restored)
        cs = [e for e in entries if e.op == "COUNTERSIGN"]
        # Forward: X→Mary countersign
        assert any(e.sig == "X" and e.nodes == "Mary" for e in cs), (
            f"Expected countersign (X, Mary); got {cs}"
        )
        # Reverse: Mary→X countersign
        assert any(e.sig == "Mary" and e.nodes == "X" for e in cs), (
            f"Expected reverse countersign (Mary, X); got {cs}"
        )

    def test_root_level_unaffected_by_shadow(self) -> None:
        """Root-level bindings are unaffected by inner shadowing."""
        # Same scope structure as above
        table = NLPSymbolTable()
        table.push_scope()  # root (0)
        table.bind("M", "Mary")
        table.push_scope()  # scope1 (1)
        table.push_scope()  # scope2 (2)
        table.bind("M", "Mod")
        table.pop_scope()
        table.pop_scope()
        table.pop_scope()
        table.rewind()

        # Simple AST: just M at root level (no subscript)
        ast = _make_file(
            Construct(inner=[_pc("M")])
        )

        emitter = ASTEmitter(skip_mcs=True, symbol_table=table)
        entries = emitter.emit(ast)

        # M at root resolves to "Mary" (shadow never applied at root)
        unsigned = [e for e in entries if e.op == "UNSIGNED"]
        assert any(e.sig == "Mary" for e in unsigned), (
            f"Expected unsigned 'Mary'; got {unsigned}"
        )
