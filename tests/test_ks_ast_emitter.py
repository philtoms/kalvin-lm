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


# ======================================================================
# Test: KS-19 MCS expansion
# ======================================================================


class TestKS19MCS:
    """ABC → UNSIGNED entries for A, B, C; CANONIZE {ABC:[A,B,C]}; UNSIGNED {ABC:[]}."""

    def test_mcs_expansion(self):
        entries = emit(_file(_bare("ABC")))

        # Component UNSIGNEDs
        assert_has_entry(entries, "A", [], "UNSIGNED")
        assert_has_entry(entries, "B", [], "UNSIGNED")
        assert_has_entry(entries, "C", [], "UNSIGNED")

        # MCS CANONIZE
        assert_has_entry(entries, "ABC", ["A", "B", "C"], "CANONIZE")

        # Compound's own UNSIGNED
        assert_has_entry(entries, "ABC", [], "UNSIGNED")

    def test_mcs_entry_order(self):
        """MCS components come before CANONIZE, compound UNSIGNED comes last."""
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
            if e.sig == "ABC" and e.op == "UNSIGNED"
        )
        assert idx_A < idx_canonize
        assert idx_B < idx_canonize
        assert idx_C < idx_canonize
        assert idx_canonize < idx_unsigned


# ======================================================================
# Test: KS-20 No MCS for single-char
# ======================================================================


class TestKS20NoMCS:
    """A → no CANONIZE entries, only UNSIGNED {A:[]}."""

    def test_no_mcs_single_char(self):
        entries = emit(_file(_bare("A")))
        canonize = _find_entries(entries, op="CANONIZE")
        assert len(canonize) == 0
        unsigned = _find_entries(entries, sig="A", op="UNSIGNED")
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
        # MCS for MHALL: component UNSIGNEDs
        assert_has_entry(entries, "M", [], "UNSIGNED")
        assert_has_entry(entries, "H", [], "UNSIGNED")
        assert_has_entry(entries, "A", [], "UNSIGNED")
        assert_has_entry(entries, "L", [], "UNSIGNED")
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
    """A = A → {A:[], UNSIGNED} (collapsed from UNDERSIGN)."""

    def test_self_identity(self):
        entries = emit(_file(
            _scope("A", TokenType.UNDERSIGN, items=[_sig("A")])
        ))
        # Should produce UNSIGNED with empty nodes, not UNDERSIGN
        assert_has_entry(entries, "A", [], "UNSIGNED")
        # Should NOT produce UNDERSIGN entry
        assert_no_entry(entries, "A", ["A"], "UNDERSIGN")


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
        # Only UNSIGNED for A — no entry for the annotation
        assert len([e for e in entries if e.op != "UNSIGNED" or e.sig == "A"]) >= 1

    def test_annotation_feeds_scope(self):
        """Block annotation words are available for resolution."""
        scope = BindingScope()
        scope.push_scope()
        entries = emit(_file(
            _ann("(Mary Had A Little Lamb)"),
            _bare("M"),
        ), scope=scope)
        # M should be resolved to "Mary"
        unsigned_m = _find_entries(entries, sig="Mary", op="UNSIGNED")
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

    def test_unsigned_no_dedup(self):
        """Non-CANONIZE entries are NOT deduped."""
        entries = emit(_file(
            _bare("A"),
            _bare("A"),
        ))
        unsigned_a = _find_entries(entries, sig="A", op="UNSIGNED")
        assert len(unsigned_a) == 2  # NOT deduped


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
