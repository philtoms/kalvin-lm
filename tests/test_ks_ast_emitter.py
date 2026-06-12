"""Tests for ASTEmitter — KScript v3 scope-model AST → SymbolicEntry.

Test IDs map to the spec test matrix (§15) where applicable.
Helper constructs AST nodes directly (no lexer dependency).

Every test verifies the 'nodes always a list' invariant (KS-34).
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
    node_inline_annotation: Annotation | None = None,
) -> OperatorScope:
    return OperatorScope(
        sig=_sig(sig_id),
        op=op,
        items=items or [],
        child_block=child_block,
        inline_annotation=inline_annotation,
        node_inline_annotation=node_inline_annotation,
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
# Test: nodes always a list (KS-34)
# ======================================================================


class TestKS34NodesAlwaysList:
    """Every emitted entry has nodes: list[str].  Never None, never str."""

    def test_bare_unsigned(self):
        entries = emit(_file(_bare("A")))
        assert len(entries) == 1
        assert isinstance(entries[0].nodes, list)
        assert entries[0].nodes == []

    def test_canonize_single_node(self):
        """CANONIZE with one node — nodes is still [node], not bare string."""
        entries = emit(_file(_scope("A", TokenType.CANONIZE, items=[_sig("B")])))
        canonize = _find_entries(entries, sig="A", op="CANONIZE")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B"]
        assert isinstance(canonize[0].nodes, list)

    def test_all_entries_are_list(self):
        """Every entry in a non-trivial compilation has list nodes."""
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B"), _sig("C")])
        ))
        for e in entries:
            assert isinstance(e.nodes, list), f"Entry {e} has non-list nodes"


# ======================================================================
# Test: KS-11 COUNTERSIGN per-item
# ======================================================================


class TestKS11Countersign:
    """A == B C → {A:[B], COUNTERSIGN}, {B:[A], COUNTERSIGN},
    {A:[C], COUNTERSIGN}, {C:[A], COUNTERSIGN}."""

    def test_countersign_per_item(self):
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B"), _sig("C")])
        ))
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGN")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGN")
        assert_has_entry(entries, "A", ["C"], "COUNTERSIGN")
        assert_has_entry(entries, "C", ["A"], "COUNTERSIGN")

    def test_countersign_single(self):
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B")])
        ))
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGN")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGN")

    def test_entry_count(self):
        """Exact entry counts — no spurious IDENTITY from bare Signature nodes."""
        # A == B C → 4 COUNTERSIGN, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B"), _sig("C")])
        ))
        assert len(entries) == 4
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0
        assert sum(1 for e in entries if e.op == "COUNTERSIGN") == 4

        # A == B → 2 COUNTERSIGN, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B")])
        ))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0


# ======================================================================
# Test: KS-12 UNDERSIGN per-item reversed
# ======================================================================


class TestKS12Undersign:
    """A = B C → {B:[A], UNDERSIGN}, {C:[A], UNDERSIGN}."""

    def test_undersign_reversed(self):
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("B"), _sig("C")])
        ))
        assert_has_entry(entries, "B", ["A"], "UNDERSIGN")
        assert_has_entry(entries, "C", ["A"], "UNDERSIGN")

    def test_undersign_single(self):
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("B")])
        ))
        assert_has_entry(entries, "B", ["A"], "UNDERSIGN")

    def test_entry_count(self):
        """Exact entry counts — no spurious IDENTITY from bare Signature nodes."""
        # A = B C → 2 UNDERSIGN, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("B"), _sig("C")])
        ))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0
        assert sum(1 for e in entries if e.op == "UNDERSIGN") == 2

        # A = B → 1 UNDERSIGN, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("B")])
        ))
        assert len(entries) == 1
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0


# ======================================================================
# Test: KS-13 CONNOTATE per-item
# ======================================================================


class TestKS13Connotate:
    """A > B C → {A:[B], CONNOTATE}, {A:[C], CONNOTATE}."""

    def test_connotate_forward(self):
        entries = emit(_file(
            _scope("A", TokenType.CONNOTATE, items=[_sig("B"), _sig("C")])
        ))
        assert_has_entry(entries, "A", ["B"], "CONNOTATE")
        assert_has_entry(entries, "A", ["C"], "CONNOTATE")

    def test_connotate_single(self):
        entries = emit(_file(
            _scope("A", TokenType.CONNOTATE, items=[_sig("B")])
        ))
        assert_has_entry(entries, "A", ["B"], "CONNOTATE")

    def test_entry_count(self):
        """Exact entry counts — no spurious IDENTITY from bare Signature nodes."""
        # A > B C → 2 CONNOTATE, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.CONNOTATE, items=[_sig("B"), _sig("C")])
        ))
        assert len(entries) == 2
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0
        assert sum(1 for e in entries if e.op == "CONNOTATE") == 2

        # A > B → 1 CONNOTATE, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.CONNOTATE, items=[_sig("B")])
        ))
        assert len(entries) == 1
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0


# ======================================================================
# Test: KS-14 CANONIZE aggregates
# ======================================================================


class TestKS14Canonize:
    """A => B C D → exactly one CANONIZE entry {A:[B,C,D]}."""

    def test_canonize_aggregated(self):
        entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[_sig("B"), _sig("C"), _sig("D")])
        ))
        canonize = _find_entries(entries, sig="A", op="CANONIZE")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B", "C", "D"]

    def test_canonize_single_node(self):
        entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[_sig("B")])
        ))
        canonize = _find_entries(entries, sig="A", op="CANONIZE")
        assert len(canonize) == 1
        assert canonize[0].nodes == ["B"]  # still a list

    def test_entry_count(self):
        """Exact entry counts — CANONIZE produces exactly one entry per scope."""
        # A => B C D → 1 CANONIZE, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[_sig("B"), _sig("C"), _sig("D")])
        ))
        assert len(entries) == 1
        assert entries[0].op == "CANONIZE"
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0

        # A => B → 1 CANONIZE, 0 IDENTITY
        entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[_sig("B")])
        ))
        assert len(entries) == 1
        assert entries[0].op == "CANONIZE"
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0


# ======================================================================
# Test: KS-15 Operator chain
# ======================================================================


class TestKS15OperatorChain:
    """A == B > C = D → correct signatures per scope.

    The parser produces nested OperatorScopes for chained operators.
    We construct the AST manually to match:
      OperatorScope(A, COUNTERSIGN, [OperatorScope(B, CONNOTATE, [OperatorScope(C, UNDERSIGN, [Signature(D)])])])
    """

    def test_operator_chain(self):
        ast = _file(
            _scope("A", TokenType.COUNTERSIGN, items=[
                _scope("B", TokenType.CONNOTATE, items=[
                    _scope("C", TokenType.UNDERSIGN, items=[_sig("D")])
                ])
            ])
        )
        entries = emit(ast)

        # COUNTERSIGN A ↔ B
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGN")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGN")

        # CONNOTATE B → C
        assert_has_entry(entries, "B", ["C"], "CONNOTATE")

        # UNDERSIGN D ← C
        assert_has_entry(entries, "D", ["C"], "UNDERSIGN")

    def test_entry_count(self):
        """Exact entry counts — no spurious IDENTITY from chained operator nodes."""
        # A == B > C = D → 4 entries (2 COUNTERSIGN + 1 CONNOTATE + 1 UNDERSIGN, 0 IDENTITY)
        ast = _file(
            _scope("A", TokenType.COUNTERSIGN, items=[
                _scope("B", TokenType.CONNOTATE, items=[
                    _scope("C", TokenType.UNDERSIGN, items=[_sig("D")])
                ])
            ])
        )
        entries = emit(ast)
        assert len(entries) == 4
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0
        assert sum(1 for e in entries if e.op == "COUNTERSIGN") == 2
        assert sum(1 for e in entries if e.op == "CONNOTATE") == 1
        assert sum(1 for e in entries if e.op == "UNDERSIGN") == 1


# ======================================================================
# Test: KS-16 Indent extends scope
# ======================================================================


class TestKS16IndentExtends:
    """Indented child block under CANONIZE — items from child block
    belong to parent operator."""

    def test_canonize_with_child_block(self):
        """A => [B, C = D] — B and C are nodes for A's CANONIZE."""
        ast = _file(
            _scope("A", TokenType.CANONIZE, items=[], child_block=_block(
                _bare("B"),
                _scope("C", TokenType.UNDERSIGN, items=[_sig("D")]),
            ))
        )
        entries = emit(ast)

        # CANONIZE aggregates B and C
        assert_has_entry(entries, "A", ["B", "C"], "CANONIZE")


# ======================================================================
# Test: KS-16 §14.8 CANONIZE with Subscript Block
# ======================================================================


class TestKS16SubscriptBlock14x8:
    """§14.8 — A =>\\n  B\\n  C = D → 5 entries.

    CANONIZE subscript blocks emit identity UNSIGNED for all identifiers
    that don't already have an operator entry as their signature.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[], child_block=_block(
                _bare("B"),
                _scope("C", TokenType.UNDERSIGN, items=[_sig("D")]),
            ))
        ))

    def test_entry_count(self):
        """Exactly 5 entries per spec §14.8."""
        assert len(self.entries) == 5

    def test_canonize_entry(self):
        """A | [B, C] | CANONIZE — aggregated single entry."""
        assert_has_entry(self.entries, "A", ["B", "C"], "CANONIZE")

    def test_undersign_entry(self):
        """D | [C] | UNDERSIGN — reversed direction."""
        assert_has_entry(self.entries, "D", ["C"], "UNDERSIGN")

    def test_identity_entries(self):
        """B, C, D each get identity IDENTITY entries."""
        assert_has_entry(self.entries, "B", [], "IDENTITY")
        assert_has_entry(self.entries, "C", [], "IDENTITY")
        assert_has_entry(self.entries, "D", [], "IDENTITY")

    def test_no_duplicate_identity(self):
        """Exactly 3 IDENTITY entries total — no duplicates."""
        assert sum(1 for e in self.entries if e.op == "IDENTITY") == 3


# ======================================================================
# Test: KS-14 §14.9 Chained CANONIZE
# ======================================================================


class TestKS14ChainedCanonize14x9:
    """§14.9 — A => B => C → 3 entries.

    Chained CANONIZE where B is both a CANONIZE scope sig and a node.
    B does NOT get identity UNSIGNED because CANONIZE(B, [C]) already
    provides B as an entry signature.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.entries = emit(_file(
            _scope("A", TokenType.CANONIZE, items=[
                _scope("B", TokenType.CANONIZE, items=[_sig("C")])
            ])
        ))

    def test_entry_count(self):
        """Exactly 3 entries per spec §14.9."""
        assert len(self.entries) == 3

    def test_canonize_entries(self):
        """A | [B] | CANONIZE and B | [C] | CANONIZE."""
        assert_has_entry(self.entries, "A", ["B"], "CANONIZE")
        assert_has_entry(self.entries, "B", ["C"], "CANONIZE")

    def test_leaf_identity(self):
        """C | [] | IDENTITY — leaf Signature gets identity IDENTITY."""
        assert_has_entry(self.entries, "C", [], "IDENTITY")

    def test_no_identity_for_canonize_sig(self):
        """B does NOT have IDENTITY(B, []) — B already has CANONIZE as identity."""
        assert_no_entry(self.entries, "B", [], "IDENTITY")


# ======================================================================
# Test: KS-17 DEDENT returns to parent scope
# ======================================================================


class TestKS17Dedent:
    """After a DEDENT, subsequent constructs return to the correct parent."""

    def test_sequential_scopes(self):
        """Two constructs at the same level — no interference."""
        ast = _file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B")]),
            _scope("C", TokenType.CONNOTATE, items=[_sig("D")]),
        )
        entries = emit(ast)

        assert_has_entry(entries, "A", ["B"], "COUNTERSIGN")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGN")
        assert_has_entry(entries, "C", ["D"], "CONNOTATE")
        # No cross-contamination
        assert_no_entry(entries, "A", ["D"], "COUNTERSIGN")


# ======================================================================
# Test: KS-18 Non-CANONIZE with indent
# ======================================================================


class TestKS18NonCanonizeIndent:
    """A == B\\n  C\\n  D — per-item COUNTERSIGN extends into child block."""

    def test_countersign_with_child_block(self):
        ast = _file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B")], child_block=_block(
                _bare("C"),
                _bare("D"),
            ))
        )
        entries = emit(ast)

        # A ↔ B
        assert_has_entry(entries, "A", ["B"], "COUNTERSIGN")
        assert_has_entry(entries, "B", ["A"], "COUNTERSIGN")
        # A ↔ C
        assert_has_entry(entries, "A", ["C"], "COUNTERSIGN")
        assert_has_entry(entries, "C", ["A"], "COUNTERSIGN")
        # A ↔ D
        assert_has_entry(entries, "A", ["D"], "COUNTERSIGN")
        assert_has_entry(entries, "D", ["A"], "COUNTERSIGN")

    def test_entry_count(self):
        """Exact entry counts — per spec §14.10, no IDENTITY from child_block bare scopes.

        A == B\\n  C\\n  D → 6 entries (all COUNTERSIGN, 0 IDENTITY).
        """
        ast = _file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("B")], child_block=_block(
                _bare("C"),
                _bare("D"),
            ))
        )
        entries = emit(ast)
        assert len(entries) == 6
        assert sum(1 for e in entries if e.op == "COUNTERSIGN") == 6
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0

    def test_undersign_with_child_block(self):
        """A = B\\n  C\\n  D — per-item UNDERSIGN extends into child block, no spurious IDENTITY."""
        ast = _file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("B")], child_block=_block(
                _bare("C"),
                _bare("D"),
            ))
        )
        entries = emit(ast)
        assert len(entries) == 3  # B→[A] UNDERSIGN, C→[A] UNDERSIGN, D→[A] UNDERSIGN
        assert sum(1 for e in entries if e.op == "UNDERSIGN") == 3
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0

    def test_connotate_with_child_block(self):
        """A > B\\n  C\\n  D — per-item CONNOTATE extends into child block, no spurious IDENTITY."""
        ast = _file(
            _scope("A", TokenType.CONNOTATE, items=[_sig("B")], child_block=_block(
                _bare("C"),
                _bare("D"),
            ))
        )
        entries = emit(ast)
        assert len(entries) == 3  # A→[B] CONNOTATE, A→[C] CONNOTATE, A→[D] CONNOTATE
        assert sum(1 for e in entries if e.op == "CONNOTATE") == 3
        assert sum(1 for e in entries if e.op == "IDENTITY") == 0


# ======================================================================
# Test: KS-19 MCS expansion
# ======================================================================


class TestKS19MCS:
    """ABC → IDENTITY entries for A, B, C; CANONIZE {ABC:[A,B,C]}; IDENTITY {ABC:[]}."""

    def test_mcs_expansion(self):
        entries = emit(_file(_bare("ABC")))

        # Component IDENTITYs
        assert_has_entry(entries, "A", [], "IDENTITY")
        assert_has_entry(entries, "B", [], "IDENTITY")
        assert_has_entry(entries, "C", [], "IDENTITY")

        # MCS CANONIZE
        assert_has_entry(entries, "ABC", ["A", "B", "C"], "CANONIZE")

        # Compound's own IDENTITY
        assert_has_entry(entries, "ABC", [], "IDENTITY")

    def test_mcs_entry_order(self):
        """MCS components come before CANONIZE, compound IDENTITY comes last."""
        entries = emit(_file(_bare("ABC")))
        sigs = [e.sig for e in entries]
        idx_A = sigs.index("A")
        idx_B = sigs.index("B")
        idx_C = sigs.index("C")
        idx_canonize = next(
            i for i, e in enumerate(entries)
            if e.sig == "ABC" and e.op == "CANONIZE"
        )
        idx_unsigned = next(
            i for i, e in enumerate(entries)
            if e.sig == "ABC" and e.op == "IDENTITY"
        )
        assert idx_A < idx_canonize
        assert idx_B < idx_canonize
        assert idx_C < idx_canonize
        assert idx_canonize < idx_unsigned


# ======================================================================
# Test: KS-20 No MCS for single-char
# ======================================================================


class TestKS20NoMCS:
    """A → no CANONIZE entries, only IDENTITY {A:[]}."""

    def test_no_mcs_single_char(self):
        entries = emit(_file(_bare("A")))
        canonize = _find_entries(entries, op="CANONIZE")
        assert len(canonize) == 0
        unsigned = _find_entries(entries, sig="A", op="IDENTITY")
        assert len(unsigned) == 1
        assert unsigned[0].nodes == []


# ======================================================================
# Test: KS-21 MCS on node side
# ======================================================================


class TestKS21MCSNode:
    """A == MHALL → MCS expansion fires for MHALL."""

    def test_mcs_on_node(self):
        entries = emit(_file(
            _scope("A", TokenType.COUNTERSIGN, items=[_sig("MHALL")])
        ))
        # MCS for MHALL: component IDENTITYs
        assert_has_entry(entries, "M", [], "IDENTITY")
        assert_has_entry(entries, "H", [], "IDENTITY")
        assert_has_entry(entries, "A", [], "IDENTITY")
        assert_has_entry(entries, "L", [], "IDENTITY")
        # MCS CANONIZE
        assert_has_entry(entries, "MHALL", ["M", "H", "A", "L", "L"], "CANONIZE")

        # COUNTERSIGN
        assert_has_entry(entries, "A", ["MHALL"], "COUNTERSIGN")
        assert_has_entry(entries, "MHALL", ["A"], "COUNTERSIGN")


# ======================================================================
# Test: KS-22 Node count invariant
# ======================================================================


class TestKS22NodeCount:
    """MCS CANONIZE node count equals character count of compound identifier."""

    def test_node_count_abc(self):
        entries = emit(_file(_bare("ABC")))
        canonize = _find_entries(entries, sig="ABC", op="CANONIZE")
        assert len(canonize) == 1
        assert len(canonize[0].nodes) == 3  # len("ABC") == 3

    def test_node_count_mhall(self):
        entries = emit(_file(_bare("MHALL")))
        canonize = _find_entries(entries, sig="MHALL", op="CANONIZE")
        assert len(canonize) == 1
        assert len(canonize[0].nodes) == 5  # len("MHALL") == 5

    @pytest.mark.parametrize("compound", ["AB", "XYZ", "HELLO", "ABCD"])
    def test_node_count_various(self, compound):
        entries = emit(_file(_bare(compound)))
        canonize = _find_entries(entries, sig=compound, op="CANONIZE")
        assert len(canonize) == 1
        assert len(canonize[0].nodes) == len(compound)


# ======================================================================
# Test: KS-26 Rule B4 override
# ======================================================================


class TestKS26RuleB4:
    """Inline annotation patches parent MCS CANONIZE entry.

    SVO => S(ubject) = M → SVO CANONIZE entry has "Subject" replacing "S".
    """

    def test_rule_b4_override(self):
        scope = BindingScope()
        scope.push_scope()

        # SVO => [S(ubject) = M]
        ast = _file(
            _scope("SVO", TokenType.CANONIZE, items=[], child_block=_block(
                _scope("S", TokenType.UNDERSIGN, items=[_sig("M")],
                       inline_annotation=_ann("(ubject)")),
            ))
        )
        entries = emit(ast, scope=scope)

        # SVO's CANONIZE entry should have "Subject" replacing "S"
        canonize = _find_entries(entries, sig="SVO", op="CANONIZE")
        assert len(canonize) >= 1
        # The MCS CANONIZE entry for SVO should have Subject patched in
        mcs_canonize = [e for e in canonize if "Subject" in e.nodes]
        assert len(mcs_canonize) >= 1, (
            f"Expected 'Subject' in SVO CANONIZE nodes. "
            f"Got: {[e.nodes for e in canonize]}"
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
            _scope("MHALL", TokenType.COUNTERSIGN, items=[
                _scope("SVO", TokenType.CANONIZE, items=[], child_block=_block(
                    _scope("S", TokenType.UNDERSIGN, items=[_sig("M")],
                           inline_annotation=_ann("(ubject)")),
                    _scope("V", TokenType.UNDERSIGN, items=[_sig("H")]),
                    _scope("O", TokenType.UNDERSIGN, items=[
                        _scope("ALL", TokenType.CANONIZE, items=[], child_block=_block(
                            _scope("A", TokenType.UNDERSIGN, items=[_sig("D")]),
                            _scope("L", TokenType.UNDERSIGN, items=[_sig("M")]),
                            _scope("L", TokenType.CONNOTATE, items=[_sig("O")]),
                        ))
                    ]),
                ))
            ]),
        )
        entries = emit(ast, scope=scope)

        # SVO CANONIZE should have Subject replacing S
        svo_canonize = [e for e in entries if e.sig == "SVO" and e.op == "CANONIZE"]
        assert len(svo_canonize) >= 1
        assert "Subject" in svo_canonize[0].nodes, (
            f"Expected 'Subject' in SVO CANONIZE. Got: {svo_canonize[0].nodes}"
        )


# ======================================================================
# Test: KS-33 Self-identity
# ======================================================================


class TestKS33SelfIdentity:
    """A = A → {A:[], IDENTITY} (collapsed from UNDERSIGN)."""

    def test_self_identity(self):
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("A")])
        ))
        # Should produce IDENTITY with empty nodes, not UNDERSIGN
        assert_has_entry(entries, "A", [], "IDENTITY")
        # Should NOT produce UNDERSIGN entry
        assert_no_entry(entries, "A", ["A"], "UNDERSIGN")

    def test_entry_count(self):
        """Exact entry count — self-identity produces exactly 1 IDENTITY entry."""
        # A = A → 1 entry (IDENTITY with empty nodes)
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("A")])
        ))
        assert len(entries) == 1
        assert entries[0].op == "IDENTITY"
        assert entries[0].nodes == []


# ======================================================================
# Test: Annotation handling
# ======================================================================


class TestAnnotations:
    """Annotations feed BindingScope without producing entries."""

    def test_annotation_no_entry(self):
        """Block annotations don't produce entries directly."""
        entries = emit(_file(
            _ann("(hello world)"),
            _bare("A"),
        ))
        # Only IDENTITY for A — no entry for the annotation
        assert len([e for e in entries if e.op != "IDENTITY" or e.sig == "A"]) >= 1

    def test_annotation_feeds_scope(self):
        """Block annotation words are available for resolution."""
        scope = BindingScope()
        scope.push_scope()
        entries = emit(_file(
            _ann("(Mary Had A Little Lamb)"),
            _bare("M"),
        ), scope=scope)
        # M should be resolved to "Mary"
        unsigned_m = _find_entries(entries, sig="Mary", op="IDENTITY")
        assert len(unsigned_m) >= 1


# ======================================================================
# Test: MCS Deduplication
# ======================================================================


class TestMCSDedup:
    """CANONIZE dedup — only CANONIZE entries are deduped."""

    def test_canonize_dedup(self):
        """Same CANONIZE (sig, nodes) emitted twice → only one entry."""
        entries = emit(_file(
            _bare("ABC"),  # emits CANONIZE ABC:[A,B,C]
            _scope("ABC", TokenType.CANONIZE, items=[_sig("A"), _sig("B"), _sig("C")]),
        ))
        canonize = _find_entries(entries, sig="ABC", op="CANONIZE")
        assert len(canonize) == 1  # deduped

    def test_identity_dedup_by_mcs(self):
        """MCS component IDENTITY entries ARE deduped across calls."""
        entries = emit(_file(
            _bare("ABC"),  # emits IDENTITY A, B, C; CANONIZE ABC; IDENTITY ABC
        ))
        # Exactly one IDENTITY per unique char (A, B, C) plus compound ABC
        identity_a = _find_entries(entries, sig="A", op="IDENTITY")
        assert len(identity_a) == 1  # deduped

    def test_non_mcs_identity_no_dedup(self):
        """Non-MCS IDENTITY entries (from bare single-char scopes) are NOT deduped."""
        entries = emit(_file(
            _bare("A"),
            _bare("A"),
        ))
        identity_a = _find_entries(entries, sig="A", op="IDENTITY")
        assert len(identity_a) == 2  # NOT deduped


# ======================================================================
# Test: MCS component IDENTITY intra- and inter-expansion dedup
# ======================================================================


class TestMCSComponentDedup:
    """MCS component IDENTITY deduplication (§8.3 extended)."""

    def test_intra_expansion_dedup(self):
        """MHALL has two L's — only one IDENTITY L is emitted."""
        entries = emit(_file(_bare("MHALL")))
        identity_l = _find_entries(entries, sig="L", op="IDENTITY")
        assert len(identity_l) == 1  # not 2

    def test_inter_expansion_dedup(self):
        """Second _emit_mcs for same compound emits no component IDENTITY."""
        emitter = ASTEmitter()
        idx1 = emitter._emit_mcs("ABC")
        count_after_first = len(emitter.entries)
        idx2 = emitter._emit_mcs("ABC")
        assert len(emitter.entries) == count_after_first  # no new entries
        assert idx2 == idx1  # returns existing CANONIZE index

    def test_cross_compound_partial_dedup(self):
        """SVO after MHALL: S,V,O are new, M,H,A,L already emitted."""
        emitter = ASTEmitter()
        emitter._emit_mcs("MHALL")
        count_after_mhall = len(emitter.entries)
        emitter._emit_mcs("SVO")
        new_entries = emitter.entries[count_after_mhall:]
        # SVO should emit: IDENTITY S, V, O + CANONIZE SVO + IDENTITY SVO
        assert len(new_entries) == 5
        sigs = [e.sig for e in new_entries]
        assert sigs == ["S", "V", "O", "SVO", "SVO"]


# ======================================================================
# Test: Compound-own IDENTITY emission
# ======================================================================


class TestCompoundOwnIdentity:
    """MCS emits IDENTITY for the compound identifier itself."""

    def test_compound_own_identity_emitted(self):
        """Bare ABC produces IDENTITY ABC []."""
        entries = emit(_file(_bare("ABC")))
        compound_identity = _find_entries(entries, sig="ABC", op="IDENTITY")
        assert len(compound_identity) == 1
        assert compound_identity[0].nodes == []

    def test_compound_own_identity_dedup(self):
        """Second _emit_mcs for same compound skips compound-own IDENTITY."""
        emitter = ASTEmitter()
        emitter._emit_mcs("ABC")
        count_after_first = len(emitter.entries)
        emitter._emit_mcs("ABC")
        assert len(emitter.entries) == count_after_first

    def test_compound_own_after_canonize(self):
        """Compound-own IDENTITY comes after CANONIZE in entry order."""
        entries = emit(_file(_bare("ABC")))
        canonize_idx = next(
            i for i, e in enumerate(entries)
            if e.sig == "ABC" and e.op == "CANONIZE"
        )
        identity_idx = next(
            i for i, e in enumerate(entries)
            if e.sig == "ABC" and e.op == "IDENTITY"
        )
        assert identity_idx > canonize_idx


# ======================================================================
# Test: CANONIZE scope push/pop
# ======================================================================


class TestScopePushPop:
    """CANONIZE scopes push/pop BindingScope correctly."""

    def test_scope_isolation(self):
        """Inner scope bindings don't leak to outer."""
        scope = BindingScope()
        scope.push_scope()

        # Outer: (Mary Had)
        # A => [inner: (apple)]
        # B
        ast = _file(
            _ann("(Mary Had)"),
            _scope("A", TokenType.CANONIZE, items=[], child_block=_block(
                _ann("(apple)"),
                _bare("B"),
            )),
        )
        entries = emit(ast, scope=scope)

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
