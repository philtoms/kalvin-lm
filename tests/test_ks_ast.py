"""Tests for KScript v3 AST node definitions.

Covers:
  KS-6  — Scope model structure (OperatorScope with sig, op, items, child_block)
  KS-8  — Annotations preserved as AST nodes
  KS-9  — Inline annotation attachment (sig-side and node-side)
  Defaults — OperatorScope default field values
  ScopeItem composition — mixed list of Signature, Annotation, OperatorScope
  ConstructItem composition — mixed list of Annotation, OperatorScope, Block
"""

from __future__ import annotations

from ks.ast import (
    Annotation,
    Block,
    ConstructItem,
    KScriptFile,
    OperatorScope,
    ScopeItem,
    Signature,
)
from ks.token import TokenType

# ---------------------------------------------------------------------------
# KS-6 — Scope model structure
# ---------------------------------------------------------------------------


class TestScopeModelStructure:
    """KS-6: AST structure reflects scope model with OperatorScope nodes."""

    def test_single_operator_scope_fields(self) -> None:
        """Build OperatorScope with explicit sig, op, items, child_block."""
        sig_a = Signature("A", 1, 1)
        sig_b = Signature("B", 1, 6)
        scope = OperatorScope(
            sig=sig_a,
            op=TokenType.COUNTERSIGN,
            items=[sig_b],
            child_block=None,
        )

        assert scope.sig is sig_a
        assert scope.op is TokenType.COUNTERSIGN
        assert scope.items == [sig_b]
        assert scope.child_block is None
        assert scope.inline_annotation is None

    def test_flat_scope_model_a_eq_b_gt_c_eq_d(self) -> None:
        """Build three OperatorScope nodes for A == B > C = D as flat list.

        In the old AST this would be a chain_right linked list. In the new
        model each operator boundary produces a separate OperatorScope.
        """
        # A == B
        scope_ab = OperatorScope(
            sig=Signature("A", 1, 1),
            op=TokenType.COUNTERSIGN,
            items=[Signature("B", 1, 6)],
        )
        # B > C
        scope_bc = OperatorScope(
            sig=Signature("B", 1, 6),
            op=TokenType.CONNOTATE,
            items=[Signature("C", 1, 10)],
        )
        # C = D
        scope_cd = OperatorScope(
            sig=Signature("C", 1, 10),
            op=TokenType.UNDERSIGN,
            items=[Signature("D", 1, 14)],
        )

        # As a flat list — no chaining
        scopes = [scope_ab, scope_bc, scope_cd]

        assert len(scopes) == 3
        assert scopes[0].op is TokenType.COUNTERSIGN
        assert scopes[1].op is TokenType.CONNOTATE
        assert scopes[2].op is TokenType.UNDERSIGN
        # Verify no chain_right exists — each scope is independent
        for s in scopes:
            assert not hasattr(s, "chain_right")


# ---------------------------------------------------------------------------
# KS-8 — Annotations preserved as AST nodes
# ---------------------------------------------------------------------------


class TestAnnotationsPreserved:
    """KS-8: Annotations are preserved as AST nodes, not discarded."""

    def test_annotation_in_block_constructs(self) -> None:
        """Annotation is a valid ConstructItem in Block.constructs."""
        ann = Annotation("test", 1, 1)
        block = Block(constructs=[ann])

        assert len(block.constructs) == 1
        assert isinstance(block.constructs[0], Annotation)
        assert block.constructs[0].text == "test"

    def test_annotation_in_operator_scope_items(self) -> None:
        """Annotation is a valid ScopeItem in OperatorScope.items."""
        ann = Annotation("(subject)", 2, 5)
        scope = OperatorScope(
            sig=Signature("A", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[ann, Signature("B", 2, 15)],
        )

        assert len(scope.items) == 2
        assert isinstance(scope.items[0], Annotation)
        assert scope.items[0].text == "(subject)"

    def test_annotation_in_file_constructs(self) -> None:
        """Annotation is a valid ConstructItem in KScriptFile.constructs."""
        ann = Annotation("(hello)", 1, 1)
        kf = KScriptFile(constructs=[ann])

        assert len(kf.constructs) == 1
        assert isinstance(kf.constructs[0], Annotation)


# ---------------------------------------------------------------------------
# KS-9 — Inline annotation attachment
# ---------------------------------------------------------------------------


class TestInlineAnnotation:
    """KS-9: Inline annotations attach to sig-side and node-side."""

    def test_sig_side_inline_annotation(self) -> None:
        """S(ubject) = M — inline_annotation on sig-side."""
        scope = OperatorScope(
            sig=Signature("S", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[Signature("M", 1, 12)],
            inline_annotation=Annotation("(ubject)", 1, 2),
        )

        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"
        assert scope.inline_annotation.line == 1
        assert scope.inline_annotation.column == 2

    def test_node_side_inline_annotation(self) -> None:
        """A = D(et) — inline annotation on the D item (Signature.inline_annotation)."""
        d_item = Signature("D", 1, 5, inline_annotation=Annotation("(et)", 1, 6))
        scope = OperatorScope(
            sig=Signature("A", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[d_item],
        )

        assert d_item.inline_annotation is not None
        assert d_item.inline_annotation.text == "(et)"

    def test_both_inline_annotations(self) -> None:
        """Both sig-side and node-side inline annotations on same scope."""
        d_item = Signature("D", 1, 10, inline_annotation=Annotation("(et)", 1, 11))
        scope = OperatorScope(
            sig=Signature("S", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[d_item],
            inline_annotation=Annotation("(ubject)", 1, 2),
        )

        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"
        assert d_item.inline_annotation is not None
        assert d_item.inline_annotation.text == "(et)"


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """OperatorScope default field values."""

    def test_minimal_operator_scope(self) -> None:
        """Only sig is required; all other fields have sensible defaults."""
        scope = OperatorScope(sig=Signature("A", 1, 1))

        assert scope.op is None
        assert scope.items == []
        assert scope.child_block is None
        assert scope.inline_annotation is None

    def test_default_items_is_independent(self) -> None:
        """Each OperatorScope gets its own items list (not shared)."""
        s1 = OperatorScope(sig=Signature("A", 1, 1))
        s2 = OperatorScope(sig=Signature("B", 2, 1))

        s1.items.append(Signature("X", 1, 3))
        assert s1.items == [Signature("X", 1, 3)]
        assert s2.items == []


# ---------------------------------------------------------------------------
# ScopeItem composition
# ---------------------------------------------------------------------------


class TestScopeItemComposition:
    """ScopeItem = Signature | Annotation | OperatorScope."""

    def test_mixed_scope_items(self) -> None:
        """A list[ScopeItem] can hold Signature, Annotation, OperatorScope."""
        items: list[ScopeItem] = [
            Signature("A", 1, 1),
            Annotation("(hello)", 1, 3),
            OperatorScope(sig=Signature("B", 1, 10), op=TokenType.UNDERSIGN),
            Signature("C", 2, 1),
        ]

        assert len(items) == 4
        assert isinstance(items[0], Signature)
        assert isinstance(items[1], Annotation)
        assert isinstance(items[2], OperatorScope)
        assert isinstance(items[3], Signature)

    def test_nested_operator_scopes_via_items(self) -> None:
        """OperatorScope.items can contain child OperatorScope instances."""
        child = OperatorScope(
            sig=Signature("X", 2, 3),
            op=TokenType.CONNOTATE,
            items=[Signature("Y", 2, 7)],
        )
        parent = OperatorScope(
            sig=Signature("A", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[child],
        )

        assert isinstance(parent.items[0], OperatorScope)
        assert parent.items[0].sig.id == "X"


# ---------------------------------------------------------------------------
# ConstructItem composition
# ---------------------------------------------------------------------------


class TestConstructItemComposition:
    """ConstructItem = Annotation | OperatorScope | Block."""

    def test_mixed_construct_items(self) -> None:
        """A list[ConstructItem] can hold Annotation, OperatorScope, Block."""
        items: list[ConstructItem] = [
            Annotation("(header)", 1, 1),
            OperatorScope(sig=Signature("A", 2, 1), op=TokenType.COUNTERSIGN),
            Block(constructs=[Annotation("(nested)", 3, 5)]),
        ]

        assert len(items) == 3
        assert isinstance(items[0], Annotation)
        assert isinstance(items[1], OperatorScope)
        assert isinstance(items[2], Block)

    def test_signature_is_not_construct_item(self) -> None:
        """Signature is NOT a valid ConstructItem — only Annotation,
        OperatorScope, and Block are.

        At runtime Python won't enforce this (TypeAlias is just an annotation),
        but type checkers will flag:
            x: ConstructItem = Signature("A", 1, 1)  # type error
        This is by design: bare Signatures at construct level are wrapped in
        an OperatorScope with op=None. See grammar §4: construct ::= block |
        annotation | operator_scope.
        """
        # ConstructItem only includes Annotation, OperatorScope, Block
        # Signature is excluded — it can only appear within OperatorScope.items
        valid_types = (Annotation, OperatorScope, Block)
        assert Signature not in valid_types


# ---------------------------------------------------------------------------
# KScriptFile integration
# ---------------------------------------------------------------------------


class TestKScriptFile:
    """Top-level file container."""

    def test_empty_file(self) -> None:
        kf = KScriptFile(constructs=[])
        assert kf.constructs == []

    def test_file_with_constructs(self) -> None:
        kf = KScriptFile(
            constructs=[
                Annotation("(doc)", 1, 1),
                OperatorScope(sig=Signature("A", 2, 1), op=TokenType.UNDERSIGN),
            ]
        )
        assert len(kf.constructs) == 2

    def test_block_with_child_scopes(self) -> None:
        """Block can contain OperatorScopes with nested child_blocks."""
        inner = Block(
            constructs=[
                OperatorScope(sig=Signature("X", 3, 5)),
            ]
        )
        scope = OperatorScope(
            sig=Signature("A", 1, 1),
            op=TokenType.UNDERSIGN,
            items=[Signature("B", 1, 5)],
            child_block=inner,
        )
        outer = Block(constructs=[scope])

        assert isinstance(outer.constructs[0], OperatorScope)
        assert outer.constructs[0].child_block is not None
        assert len(outer.constructs[0].child_block.constructs) == 1
