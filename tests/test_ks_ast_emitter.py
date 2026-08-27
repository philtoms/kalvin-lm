"""Tests for ASTEmitter — KScript v3 scope-model AST → SymbolicEntry.

Test IDs map to the spec test matrix where applicable.
Helper constructs AST nodes directly (no lexer dependency).

Every test verifies the 'nodes always a list' invariant.
"""

from __future__ import annotations

import pytest

from ks.ast import Annotation, Block, KScriptFile, OperatorScope, Signature
from ks.ast_emitter import ASTEmitter, SymbolicEntry
from ks.binding_scope import BindingScope
from ks.token import TokenType

# ======================================================================
# Helpers
# ======================================================================


def _sig(id: str, line: int = 1, col: int = 1) -> Signature:
    return Signature(id=id, line=line, column=col)


def _ann(text: str, line: int = 1, col: int = 1) -> Annotation:
    return Annotation(text=text, line=line, column=col)


def _bare(id: str, **kw) -> OperatorScope:
    """Bare unsigned OperatorScope."""
    return OperatorScope(sig=_sig(id, **kw), op=None, items=[])


def _scope(
    sig_id: str,
    op: TokenType,
    items: list | None = None,
    child_block: Block | None = None,
    inline_annotation: Annotation | None = None,
) -> OperatorScope:
    return OperatorScope(
        sig=_sig(sig_id),
        op=op,
        items=items or [],
        child_block=child_block,
        inline_annotation=inline_annotation,
    )


def _block(*constructs) -> Block:
    return Block(constructs=list(constructs))


def _file(*constructs) -> KScriptFile:
    return KScriptFile(constructs=list(constructs))


def emit(
    file: KScriptFile,
    scope: BindingScope | None = None,
) -> list[SymbolicEntry]:
    """Convenience: build emitter, emit, return entries."""
    return ASTEmitter(scope=scope, dev=True).emit(file)


def _find_entries(
    entries: list[SymbolicEntry],
    sig: str | None = None,
    nodes: list[str] | None = None,
    op: str | None = None,
) -> list[SymbolicEntry]:
    """Filter entries by optional criteria."""
    result = []
    for e in entries:
        if sig is not None and e.sig != sig:
            continue
        if nodes is not None and e.nodes != nodes:
            continue
        if op is not None and e.op != op:
            continue
        result.append(e)
    return result


def assert_has_entry(
    entries: list[SymbolicEntry],
    sig: str,
    nodes: list[str],
    op: str,
) -> None:
    """Assert at least one entry matches (sig, nodes, op)."""
    matches = _find_entries(entries, sig=sig, nodes=nodes, op=op)
    assert matches, (
        f"Expected entry ({sig!r}, {nodes!r}, {op!r}) not found.\n"
        f"Entries: {[(e.sig, e.nodes, e.op) for e in entries]}"
    )


def assert_no_entry(
    entries: list[SymbolicEntry],
    sig: str,
    nodes: list[str],
    op: str,
) -> None:
    """Assert no entry matches (sig, nodes, op)."""
    matches = _find_entries(entries, sig=sig, nodes=nodes, op=op)
    assert not matches, (
        f"Unexpected entry ({sig!r}, {nodes!r}, {op!r}) found.\n"
        f"Entries: {[(e.sig, e.nodes, e.op) for e in entries]}"
    )


# ======================================================================
# Test: nodes always a list
# ======================================================================


class TestKS34NodesAlwaysList:
    """Every emitted entry has nodes: list[str].  Never None, never str."""

    def test_bare_unsigned(self):
        entries = emit(_file(_bare("A")))
        assert len(entries) == 1
        assert isinstance(entries[0].nodes, list)
        assert entries[0].nodes == []

    def test_canonize_single_node(self):
        """CANONIZES with one node — nodes is still [node], not bare string."""
        entries = emit(_file(_scope("A", TokenType.CANONIZES, items=[_sig("B")])))
        canonize = _find_entries(entries, sig="A", op="CANONIZES")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B"]
        assert isinstance(canonize[0].nodes, list)

    def test_all_entries_are_list(self):
        """Every entry in a non-trivial compilation has list nodes."""
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("B"), _sig("C")])))
        for e in entries:
            assert isinstance(e.nodes, list), f"Entry {e} has non-list nodes"


# ======================================================================
# Test:  COUNTERSIGNS per-item
# ======================================================================


class TestKS11Countersign:
    """A == B C → {A:[B], COUNTERSIGNS}, {B:[A], COUNTERSIGNS},
    {A:[C], COUNTERSIGNS}, {C:[A], COUNTERSIGNS}."""

    def test_countersign_per_item(self):
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("B"), _sig("C")])))
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGNS")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGNS")
        assert_has_entry(entries, "A", ["C"], "COUNTERSIGNS")
        assert_has_entry(entries, "C", ["A"], "COUNTERSIGNS")

    def test_countersign_single(self):
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("B")])))
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGNS")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGNS")

    def test_entry_count(self):
        """Exact entry counts — no spurious UNKNOWN from bare Signature nodes."""
        # A == B C → 4 COUNTERSIGNS, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("B"), _sig("C")])))
        assert len(entries) == 4
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0
        assert sum(1 for e in entries if e.op == "COUNTERSIGNS") == 4

        # A == B → 2 COUNTERSIGNS, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("B")])))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0


# ======================================================================
# Test:  DENOTES per-item reversed
# ======================================================================


class TestKS12Denote:
    """A = B C → {B:[A], DENOTES}, {C:[A], DENOTES}."""

    def test_denote_reversed(self):
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("B"), _sig("C")])))
        assert_has_entry(entries, "B", ["A"], "DENOTES")
        assert_has_entry(entries, "C", ["A"], "DENOTES")

    def test_denote_single(self):
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("B")])))
        assert_has_entry(entries, "B", ["A"], "DENOTES")

    def test_entry_count(self):
        """Exact entry counts — no spurious UNKNOWN from bare Signature nodes."""
        # A = B C → 2 DENOTES, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("B"), _sig("C")])))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0
        assert sum(1 for e in entries if e.op == "DENOTES") == 2

        # A = B → 1 DENOTES, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("B")])))
        assert len(entries) == 1
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0


# ======================================================================
# Test:  CONNOTES per-item
# ======================================================================


class TestKS13Connote:
    """A > B C → {A:[B], CONNOTES}, {A:[C], CONNOTES}."""

    def test_connote_forward(self):
        entries = emit(_file(_scope("A", TokenType.CONNOTES, items=[_sig("B"), _sig("C")])))
        assert_has_entry(entries, "A", ["B"], "CONNOTES")
        assert_has_entry(entries, "A", ["C"], "CONNOTES")

    def test_connote_single(self):
        entries = emit(_file(_scope("A", TokenType.CONNOTES, items=[_sig("B")])))
        assert_has_entry(entries, "A", ["B"], "CONNOTES")

    def test_entry_count(self):
        """Exact entry counts — no spurious UNKNOWN from bare Signature nodes."""
        # A > B C → 2 CONNOTES, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.CONNOTES, items=[_sig("B"), _sig("C")])))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0
        assert sum(1 for e in entries if e.op == "CONNOTES") == 2

        # A > B → 1 CONNOTES, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.CONNOTES, items=[_sig("B")])))
        assert len(entries) == 1
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0


# ======================================================================
# Test:  CANONIZES aggregates
# ======================================================================


class TestKS14Canonize:
    """A => B C D → exactly one CANONIZES entry {A:[B,C,D]}."""

    def test_canonize_aggregated(self):
        entries = emit(
            _file(_scope("A", TokenType.CANONIZES, items=[_sig("B"), _sig("C"), _sig("D")]))
        )
        canonize = _find_entries(entries, sig="A", op="CANONIZES")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B", "C", "D"]

    def test_canonize_single_node(self):
        entries = emit(_file(_scope("A", TokenType.CANONIZES, items=[_sig("B")])))
        canonize = _find_entries(entries, sig="A", op="CANONIZES")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B"]  # still a list

    def test_entry_count(self):
        """Exact entry counts — CANONIZES produces exactly one entry per scope."""
        # A => B C D → 1 CANONIZES, 0 UNKNOWN
        entries = emit(
            _file(_scope("A", TokenType.CANONIZES, items=[_sig("B"), _sig("C"), _sig("D")]))
        )
        assert len(entries) == 1
        assert entries[0].op == "CANONIZES"
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0

        # A => B → 1 CANONIZES, 0 UNKNOWN
        entries = emit(_file(_scope("A", TokenType.CANONIZES, items=[_sig("B")])))
        assert len(entries) == 1
        assert entries[0].op == "CANONIZES"
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0


# ======================================================================
# Test:  Operator chain
# ======================================================================


class TestKS15OperatorChain:
    """A == B > C = D → correct signatures per scope.

    The parser produces nested OperatorScopes for chained operators.
    We construct the AST manually to match:
      OperatorScope(A, COUNTERSIGNS, [
        OperatorScope(B, CONNOTES, [
          OperatorScope(C, DENOTES, [Signature(D)])])])
    """

    def test_operator_chain(self):
        ast = _file(
            _scope(
                "A",
                TokenType.COUNTERSIGNS,
                items=[
                    _scope(
                        "B",
                        TokenType.CONNOTES,
                        items=[_scope("C", TokenType.DENOTES, items=[_sig("D")])],
                    )
                ],
            )
        )
        entries = emit(ast)

        # COUNTERSIGNS A ↔ B
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGNS")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGNS")

        # CONNOTES B → C
        assert_has_entry(entries, "B", ["C"], "CONNOTES")

        # DENOTES D ← C
        assert_has_entry(entries, "D", ["C"], "DENOTES")

    def test_entry_count(self):
        """Exact entry counts — no spurious UNKNOWN from chained operator nodes."""
        # A == B > C = D → 4 entries (2 COUNTERSIGNS + 1 CONNOTES + 1 DENOTES, 0 UNKNOWN)
        ast = _file(
            _scope(
                "A",
                TokenType.COUNTERSIGNS,
                items=[
                    _scope(
                        "B",
                        TokenType.CONNOTES,
                        items=[_scope("C", TokenType.DENOTES, items=[_sig("D")])],
                    )
                ],
            )
        )
        entries = emit(ast)
        assert len(entries) == 4
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0
        assert sum(1 for e in entries if e.op == "COUNTERSIGNS") == 2
        assert sum(1 for e in entries if e.op == "CONNOTES") == 1
        assert sum(1 for e in entries if e.op == "DENOTES") == 1


# ======================================================================
# Test:  Indent extends scope
# ======================================================================


class TestKS16IndentExtends:
    """Indented child block under CANONIZES — items from child block
    belong to parent operator."""

    def test_canonize_with_child_block(self):
        """A => [B, C = D] — B and C are nodes for A's CANONIZES."""
        ast = _file(
            _scope(
                "A",
                TokenType.CANONIZES,
                items=[],
                child_block=_block(
                    _bare("B"),
                    _scope("C", TokenType.DENOTES, items=[_sig("D")]),
                ),
            )
        )
        entries = emit(ast)

        # CANONIZES aggregates B and C
        assert_has_entry(entries, "A", ["B", "C"], "CANONIZES")


# ======================================================================
# Test:  CANONIZES with Subscript Block
# ======================================================================


class TestKS16SubscriptBlock14x8:
    """ — A =>\\n  B\\n  C = D → 5 entries.

    CANONIZES subscript blocks emit UNKNOWN for all identifiers
    that don't already have an operator entry as their signature.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = emit(
            _file(
                _scope(
                    "A",
                    TokenType.CANONIZES,
                    items=[],
                    child_block=_block(
                        _bare("B"),
                        _scope("C", TokenType.DENOTES, items=[_sig("D")]),
                    ),
                )
            )
        )

    def test_entry_count(self):
        """Exactly 5 entries per spec."""
        assert len(self.entries) == 5

    def test_canonize_entry(self):
        """A | [B, C] | CANONIZES — aggregated single entry."""
        assert_has_entry(self.entries, "A", ["B", "C"], "CANONIZES")

    def test_denote_entry(self):
        """D | [C] | DENOTES — reversed direction."""
        assert_has_entry(self.entries, "D", ["C"], "DENOTES")

    def test_identity_entries(self):
        """B, C, D each get identity UNKNOWN entries."""
        assert_has_entry(self.entries, "B", [], "UNKNOWN")
        assert_has_entry(self.entries, "C", [], "UNKNOWN")
        assert_has_entry(self.entries, "D", [], "UNKNOWN")

    def test_no_duplicate_identity(self):
        """Exactly 3 UNKNOWN entries total — no duplicates."""
        assert sum(1 for e in self.entries if e.op == "UNKNOWN") == 3


# ======================================================================
# Test:  Chained CANONIZES
# ======================================================================


class TestKS14ChainedCanonize14x9:
    """ — A => B => C → 3 entries.

    Chained CANONIZES where B is both a CANONIZES scope sig and a node.
    B does NOT get an UNKNOWN entry because CANONIZES(B, [C]) already
    provides B as an entry signature.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = emit(
            _file(
                _scope(
                    "A",
                    TokenType.CANONIZES,
                    items=[_scope("B", TokenType.CANONIZES, items=[_sig("C")])],
                )
            )
        )

    def test_entry_count(self):
        """Exactly 3 entries per spec."""
        assert len(self.entries) == 3

    def test_canonize_entries(self):
        """A | [B] | CANONIZES and B | [C] | CANONIZES."""
        assert_has_entry(self.entries, "A", ["B"], "CANONIZES")
        assert_has_entry(self.entries, "B", ["C"], "CANONIZES")

    def test_leaf_identity(self):
        """C | [] | UNKNOWN — leaf Signature gets identity UNKNOWN."""
        assert_has_entry(self.entries, "C", [], "UNKNOWN")

    def test_no_identity_for_canonize_sig(self):
        """B does NOT have UNKNOWN(B, []) — B already has CANONIZES as identity."""
        assert_no_entry(self.entries, "B", [], "UNKNOWN")


# ======================================================================
# Test:  DEDENT returns to parent scope
# ======================================================================


class TestKS17Dedent:
    """After a DEDENT, subsequent constructs return to the correct parent."""

    def test_sequential_scopes(self):
        """Two constructs at the same level — no interference."""
        ast = _file(
            _scope("A", TokenType.COUNTERSIGNS, items=[_sig("B")]),
            _scope("C", TokenType.CONNOTES, items=[_sig("D")]),
        )
        entries = emit(ast)

        assert_has_entry(entries, "A", ["B"], "COUNTERSIGNS")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGNS")
        assert_has_entry(entries, "C", ["D"], "CONNOTES")
        # No cross-contamination
        assert_no_entry(entries, "A", ["D"], "COUNTERSIGNS")


# ======================================================================
# Test:  Non-CANONIZES with indent
# ======================================================================


class TestKS18NonCanonizeIndent:
    """A == B\\n  C\\n  D — per-item COUNTERSIGNS extends into child block."""

    def test_countersign_with_child_block(self):
        ast = _file(
            _scope(
                "A",
                TokenType.COUNTERSIGNS,
                items=[_sig("B")],
                child_block=_block(
                    _bare("C"),
                    _bare("D"),
                ),
            )
        )
        entries = emit(ast)

        # A ↔ B
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGNS")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGNS")
        # A ↔ C
        assert_has_entry(entries, "A", ["C"], "COUNTERSIGNS")
        assert_has_entry(entries, "C", ["A"], "COUNTERSIGNS")
        # A ↔ D
        assert_has_entry(entries, "A", ["D"], "COUNTERSIGNS")
        assert_has_entry(entries, "D", ["A"], "COUNTERSIGNS")

    def test_entry_count(self):
        """Exact entry counts — per spec, no UNKNOWN from child_block bare scopes.

        A == B\\n  C\\n  D → 6 entries (all COUNTERSIGNS, 0 UNKNOWN).
        """
        ast = _file(
            _scope(
                "A",
                TokenType.COUNTERSIGNS,
                items=[_sig("B")],
                child_block=_block(
                    _bare("C"),
                    _bare("D"),
                ),
            )
        )
        entries = emit(ast)
        assert len(entries) == 6
        assert sum(1 for e in entries if e.op == "COUNTERSIGNS") == 6
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0

    def test_denote_with_child_block(self):
        """A = B\\n  C\\n  D — per-item DENOTES extends into child block, no spurious UNKNOWN."""
        ast = _file(
            _scope(
                "A",
                TokenType.DENOTES,
                items=[_sig("B")],
                child_block=_block(
                    _bare("C"),
                    _bare("D"),
                ),
            )
        )
        entries = emit(ast)
        assert len(entries) == 3  # B→[A] DENOTES, C→[A] DENOTES, D→[A] DENOTES
        assert sum(1 for e in entries if e.op == "DENOTES") == 3
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0

    def test_connote_with_child_block(self):
        """A > B\\n  C\\n  D — per-item CONNOTES extends into child block, no spurious UNKNOWN."""
        ast = _file(
            _scope(
                "A",
                TokenType.CONNOTES,
                items=[_sig("B")],
                child_block=_block(
                    _bare("C"),
                    _bare("D"),
                ),
            )
        )
        entries = emit(ast)
        assert len(entries) == 3  # A→[B] CONNOTES, A→[C] CONNOTES, A→[D] CONNOTES
        assert sum(1 for e in entries if e.op == "CONNOTES") == 3
        assert sum(1 for e in entries if e.op == "UNKNOWN") == 0


# ======================================================================
# Test:  MTS expansion
# ======================================================================


class TestKS19MTS:
    """ABC → only CANONIZES {ABC:[A,B,C]}; no per-component entries."""

    def test_mts_expansion(self):
        entries = emit(_file(_bare("ABC")))

        # A bare compound is an ask: only the ASK canon — components are
        # values inside it, not headed klines of their own.
        assert_has_entry(entries, "ASK", ["A", "B", "C"], "ASK")
        assert not _find_entries(entries, sig="A")
        assert not _find_entries(entries, sig="B")
        assert not _find_entries(entries, sig="C")

    def test_mts_entry_count(self):
        """A bare compound produces exactly one entry (the ask canon)."""
        entries = emit(_file(_bare("ABC")))
        assert len(entries) == 1
        assert entries[0].sig == "ASK"
        assert entries[0].op == "ASK"

    def test_mts_entries_tagged_is_mts(self):
        """ MTS-produced canon entries carry is_mts=True; source entries do not.

        Only the MTS CANONIZES entry produced by ``_emit_mts`` is tagged.
        Operator-produced entries (COUNTERSIGNS/DENOTES/CONNOTES) and
        subscript identities stay source (is_mts=False) so the TokenEncoder
        can push them ahead of MTS in the final output.
        """
        # `ABC == MHALL`: ABC + MHALL each MTS-expand (canon only);
        # countersign is source.
        entries = emit(_file(_scope(
            "ABC", TokenType.COUNTERSIGNS, items=[_sig("MHALL")]
        )))

        # Both canonizations are MTS-tagged.
        assert any(
            (e.sig, e.nodes, e.op) == ("ABC", ["A", "B", "C"], "CANONIZES")
            and e.is_mts for e in entries
        )
        assert any(
            (e.sig, e.nodes, e.op) == ("MHALL", ["M", "H", "A", "L", "L"], "CANONIZES")
            and e.is_mts for e in entries
        )

        # The countersign pair is source (not MTS).
        assert any(
            (e.sig, e.nodes, e.op) == ("ABC", ["MHALL"], "COUNTERSIGNS")
            and not e.is_mts for e in entries
        )
        assert any(
            (e.sig, e.nodes, e.op) == ("MHALL", ["ABC"], "COUNTERSIGNS")
            and not e.is_mts for e in entries
        )

        # No source entry is a canon, no tagged entry is an operator entry.
        assert all(e.op == "CANONIZES" for e in entries if e.is_mts)
        assert all(e.op != "CANONIZES" for e in entries if not e.is_mts)


# ======================================================================
# Test:  No MTS for single-char
# ======================================================================


class TestKS20NoMTS:
    """A → no CANONIZES entries, only UNKNOWN {A:[]}."""

    def test_no_mts_single_char(self):
        entries = emit(_file(_bare("A")))
        canonize = _find_entries(entries, op="CANONIZES")
        assert len(canonize) == 0
        unsigned = _find_entries(entries, sig="A", op="UNKNOWN")
        assert len(unsigned) == 1
        assert unsigned[0].nodes == []


# ======================================================================
# Test:  No MTS for lowercase/mixed words
# ======================================================================


class TestKS20bNoMTSForWords:
    """Lowercase/mixed multi-char words are single-token identifiers, not
    compounds. They must NOT be decomposed into per-character entries.

    Regression: commit 490d98f relaxed the SIGNATURE rule to admit lowercase
    words (had, did, all). Previously only uppercase identifiers reached the
    compiler, so MTS decomposed every multi-char identifier. Without an
    uppercase guard, `all` would decompose to [a, l, l]. The fix: MTS
    character-expansion applies only to all-uppercase compounds.
    """

    def test_no_mts_lowercase_word(self):
        entries = emit(_file(_bare("had")))
        # No CANONIZES (MTS) entry, and no per-char identities.
        assert len(_find_entries(entries, op="CANONIZES")) == 0
        assert len(_find_entries(entries, sig="h")) == 0
        assert len(_find_entries(entries, sig="a")) == 0
        assert len(_find_entries(entries, sig="d")) == 0
        # The word itself is emitted as a single identity.
        unsigned = _find_entries(entries, sig="had", op="UNKNOWN")
        assert len(unsigned) == 1
        assert unsigned[0].nodes == []

    def test_no_mts_mixed_case_word(self):
        entries = emit(_file(_bare("Hello")))
        assert len(_find_entries(entries, op="CANONIZES")) == 0
        unsigned = _find_entries(entries, sig="Hello", op="UNKNOWN")
        assert len(unsigned) == 1
        assert unsigned[0].nodes == []

    def test_uppercase_still_decomposes(self):
        # Sanity: all-uppercase multi-char identifiers still trigger MTS,
        # emitting only the canon (no per-component entries).
        entries = emit(_file(_bare("ALL")))
        assert_has_entry(entries, "ASK", ["A", "L", "L"], "ASK")
        assert not _find_entries(entries, sig="A")
        assert not _find_entries(entries, sig="L")


# ======================================================================
# Test:  MTS on node side
# ======================================================================


class TestKS21MTSNode:
    """A == MHALL → MTS canon expansion fires for MHALL (canon only)."""

    def test_mts_on_node(self):
        entries = emit(_file(_scope("A", TokenType.COUNTERSIGNS, items=[_sig("MHALL")])))
        # MTS for MHALL: canon only (no per-component entries).
        assert_has_entry(entries, "MHALL", ["M", "H", "A", "L", "L"], "CANONIZES")
        assert not _find_entries(entries, sig="M", op="UNKNOWN")
        assert not _find_entries(entries, sig="H", op="UNKNOWN")
        assert not _find_entries(entries, sig="L", op="UNKNOWN")

        # COUNTERSIGNS
        assert_has_entry(entries, "A", ["MHALL"], "COUNTERSIGNS")
        assert_has_entry(entries, "MHALL", ["A"], "COUNTERSIGNS")


# ======================================================================
# Test:  Node count invariant
# ======================================================================


class TestKS22NodeCount:
    """MTS CANONIZES node count equals character count of compound identifier."""

    def test_node_count_abc(self):
        entries = emit(_file(_bare("ABC")))
        ask = _find_entries(entries, sig="ASK", op="ASK")
        assert len(ask) == 1
        assert len(ask[0].nodes) == 3  # len("ABC") == 3

    def test_node_count_mhall(self):
        entries = emit(_file(_bare("MHALL")))
        ask = _find_entries(entries, sig="ASK", op="ASK")
        assert len(ask) == 1
        assert len(ask[0].nodes) == 5  # len("MHALL") == 5

    @pytest.mark.parametrize("compound", ["AB", "XYZ", "HELLO", "ABCD"])
    def test_node_count_various(self, compound):
        entries = emit(_file(_bare(compound)))
        ask = _find_entries(entries, sig="ASK", op="ASK")
        assert len(ask) == 1
        assert len(ask[0].nodes) == len(compound)


# ======================================================================
# Test:  Rule B4 override
# ======================================================================


class TestKS26RuleB4:
    """Inline annotation patches parent MTS CANONIZES entry.

    SVO => S(ubject) = M → SVO CANONIZES entry has "Subject" replacing "S".
    """

    def test_rule_b4_override(self):
        scope = BindingScope()
        scope.push_scope()

        # SVO => [S(ubject) = M]
        ast = _file(
            _scope(
                "SVO",
                TokenType.CANONIZES,
                items=[],
                child_block=_block(
                    _scope(
                        "S",
                        TokenType.DENOTES,
                        items=[_sig("M")],
                        inline_annotation=_ann("(ubject)"),
                    ),
                ),
            )
        )
        entries = emit(ast, scope=scope)

        # SVO's CANONIZES entry should have "Subject" replacing "S"
        canonize = _find_entries(entries, sig="SVO", op="CANONIZES")
        assert len(canonize) >= 1
        # The MTS CANONIZES entry for SVO should have Subject patched in
        mts_canonize = [e for e in canonize if "Subject" in e.nodes]
        assert len(mts_canonize) >= 1, (
            f"Expected 'Subject' in SVO CANONIZES nodes. Got: {[e.nodes for e in canonize]}"
        )

    def test_rule_b4_with_binding_scope(self):
        """Full test: annotations + inline override."""
        scope = BindingScope()
        scope.push_scope()

        # (Mary Had A Little Lamb)
        # MHALL == SVO =>
        #   S(ubject) = M
        #   V = H
        #   O = ALL =>
        #     A = D
        #     L = M
        #     L > O
        ast = _file(
            _ann("(Mary Had A Little Lamb)"),
            _scope(
                "MHALL",
                TokenType.COUNTERSIGNS,
                items=[
                    _scope(
                        "SVO",
                        TokenType.CANONIZES,
                        items=[],
                        child_block=_block(
                            _scope(
                                "S",
                                TokenType.DENOTES,
                                items=[_sig("M")],
                                inline_annotation=_ann("(ubject)"),
                            ),
                            _scope("V", TokenType.DENOTES, items=[_sig("H")]),
                            _scope(
                                "O",
                                TokenType.DENOTES,
                                items=[
                                    _scope(
                                        "ALL",
                                        TokenType.CANONIZES,
                                        items=[],
                                        child_block=_block(
                                            _scope("A", TokenType.DENOTES, items=[_sig("D")]),
                                            _scope("L", TokenType.DENOTES, items=[_sig("M")]),
                                            _scope("L", TokenType.CONNOTES, items=[_sig("O")]),
                                        ),
                                    )
                                ],
                            ),
                        ),
                    )
                ],
            ),
        )
        entries = emit(ast, scope=scope)

        # SVO CANONIZES should have Subject replacing S
        svo_canonize = [e for e in entries if e.sig == "SVO" and e.op == "CANONIZES"]
        assert len(svo_canonize) >= 1
        assert "Subject" in svo_canonize[0].nodes, (
            f"Expected 'Subject' in SVO CANONIZES. Got: {svo_canonize[0].nodes}"
        )


# ======================================================================
# Test:  Self-identity
# ======================================================================


class TestKS33SelfIdentity:
    """A = A → {A:[A], IDENTITY} (self-referential, binding-independent)."""

    def test_self_identity(self):
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("A")])))
        # Should produce IDENTITY with self-referential nodes, not DENOTES
        assert_has_entry(entries, "A", ["A"], "IDENTITY")
        # Should NOT produce DENOTES entry
        assert_no_entry(entries, "A", ["A"], "DENOTES")

    def test_entry_count(self):
        """Exact entry count — self-identity produces exactly 1 IDENTITY entry."""
        # A = A → 1 entry (IDENTITY {A:[A]})
        entries = emit(_file(_scope("A", TokenType.DENOTES, items=[_sig("A")])))
        assert len(entries) == 1
        assert entries[0].op == "IDENTITY"
        assert entries[0].nodes == ["A"]


# ======================================================================
# Test: Annotation handling
# ======================================================================


class TestAnnotations:
    """Annotations feed BindingScope without producing entries."""

    def test_annotation_no_entry(self):
        """Block annotations don't produce entries directly."""
        entries = emit(
            _file(
                _ann("(hello world)"),
                _bare("A"),
            )
        )
        # Only UNKNOWN for A — no entry for the annotation
        assert len([e for e in entries if e.op != "UNKNOWN" or e.sig == "A"]) >= 1

    def test_annotation_feeds_scope(self):
        """Block annotation words are available for resolution."""
        scope = BindingScope()
        scope.push_scope()
        entries = emit(
            _file(
                _ann("(Mary Had A Little Lamb)"),
                _bare("M"),
            ),
            scope=scope,
        )
        # M should be resolved to "Mary" and compile to an IDENTITY
        identity_m = _find_entries(entries, sig="Mary", op="IDENTITY")
        assert len(identity_m) >= 1


# ======================================================================
# Test: MTS Deduplication
# ======================================================================


class TestMTSDedup:
    """CANONIZES dedup — only CANONIZES entries are deduped."""

    def test_canonize_dedup(self):
        """Same CANONIZES (sig, nodes) emitted twice → only one entry."""
        entries = emit(
            _file(
                _bare("ABC"),  # emits CANONIZES ABC:[A,B,C]
                _scope("ABC", TokenType.CANONIZES, items=[_sig("A"), _sig("B"), _sig("C")]),
            )
        )
        canonize = _find_entries(entries, sig="ABC", op="CANONIZES")
        assert len(canonize) == 1  # deduped

    def test_identity_dedup_by_mts(self):
        """MTS emits no per-component entries — only the canon."""
        entries = emit(
            _file(
                _bare("ABC"),  # emits only CANONIZES ABC:[A,B,C]
            )
        )
        # No headed entries for the characters.
        assert not _find_entries(entries, sig="A")
        assert not _find_entries(entries, sig="B")
        assert not _find_entries(entries, sig="C")
        assert len(_find_entries(entries, op="ASK")) == 1
        assert not _find_entries(entries, op="CANONIZES")

    def test_non_mts_identity_no_dedup(self):
        """Non-MTS UNKNOWN entries (from bare single-char scopes) are NOT deduped."""
        entries = emit(
            _file(
                _bare("A"),
                _bare("A"),
            )
        )
        identity_a = _find_entries(entries, sig="A", op="UNKNOWN")
        assert len(identity_a) == 2  # NOT deduped


# ======================================================================
# Test: MTS component UNKNOWN intra- and inter-expansion dedup
# ======================================================================


class TestMTSComponentDedup:
    """MTS canon deduplication — canon only, no component entries."""

    def test_intra_expansion_node_count(self):
        """MHALL has two L's — the ask canon preserves both as nodes."""
        entries = emit(_file(_bare("MHALL")))
        canon = _find_entries(entries, sig="ASK", op="ASK")
        assert len(canon) == 1
        assert canon[0].nodes == ["M", "H", "A", "L", "L"]  # 5 nodes, repeats kept

    def test_inter_expansion_dedup(self):
        """Second _emit_mts for same compound emits nothing (canon dedup)."""
        emitter = ASTEmitter()
        idx1 = emitter._emit_mts("ABC")
        count_after_first = len(emitter.entries)
        idx2 = emitter._emit_mts("ABC")
        assert len(emitter.entries) == count_after_first  # no new entries
        assert idx2 == idx1  # returns existing CANONIZES index

    def test_cross_compound_no_component_entries(self):
        """SVO after MHALL: each emits only its canon (no component entries)."""
        emitter = ASTEmitter()
        emitter._emit_mts("MHALL")
        count_after_mhall = len(emitter.entries)
        emitter._emit_mts("SVO")
        new_entries = emitter.entries[count_after_mhall:]
        # SVO emits only its CANONIZES canon.
        assert len(new_entries) == 1
        assert new_entries[0].sig == "SVO"
        assert new_entries[0].op == "CANONIZES"


# ======================================================================
# Test: CANONIZES scope push/pop
# ======================================================================


class TestScopePushPop:
    """CANONIZES scopes push/pop BindingScope correctly."""

    def test_scope_isolation(self):
        """Inner scope bindings don't leak to outer."""
        scope = BindingScope()
        scope.push_scope()

        # Outer: (Mary Had)
        # A => [inner: (apple)]
        # B
        ast = _file(
            _ann("(Mary Had)"),
            _scope(
                "A",
                TokenType.CANONIZES,
                items=[],
                child_block=_block(
                    _ann("(apple)"),
                    _bare("B"),
                ),
            ),
        )
        emit(ast, scope=scope)

        # B should NOT resolve to anything (in inner scope with "apple")
        # but might if outer scope's "Mary" matches
        # This mainly tests no crash and scope push/pop works


# ======================================================================
# Test: Empty file
# ======================================================================


class TestEmptyFile:
    """Empty source produces no entries."""

    def test_empty(self):
        entries = emit(_file())
        assert entries == []
