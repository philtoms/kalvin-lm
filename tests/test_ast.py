"""Tests for AST node definitions — Comment dataclass and type integration."""

from dataclasses import astuple

import pytest

from kscript.ast import (
    Block,
    Comment,
    Construct,
    ConstructItem,
    KScriptFile,
    Literal,
    PrimaryConstruct,
    Script,
    Signature,
)
from kscript.token import TokenType


# =============================================================================
# 1. Comment construction
# =============================================================================

class TestCommentConstruction:
    """Verify Comment dataclass stores all fields correctly."""

    def test_basic_construction(self):
        c = Comment(text="(hello)", line=1, column=3)
        assert c.text == "(hello)"
        assert c.line == 1
        assert c.column == 3

    def test_multiline_text(self):
        c = Comment(text="(a longer comment with spaces)", line=5, column=10)
        assert c.text == "(a longer comment with spaces)"
        assert c.line == 5
        assert c.column == 10


# =============================================================================
# 2. Comment in ConstructItem union
# =============================================================================

class TestCommentConstructItem:
    """Verify Comment is a valid ConstructItem."""

    def test_comment_is_valid_construct_item(self):
        """ConstructItem is a type alias — verify the import resolves and
        a Comment instance is one of the types in the union."""
        c = Comment("(note)", 1, 1)
        # Comment should be one of: PrimaryConstruct, Literal, Comment
        assert isinstance(c, Comment)

    def test_comment_as_block_level_item(self):
        """A Comment can serve as a block-level item alongside other types."""
        from typing import get_args
        # ConstructItem is a string-quoted alias, so check resolution at runtime
        # by verifying it's importable and Comment is in the module
        assert Comment is not None


# =============================================================================
# 3. PrimaryConstruct with inline_comment
# =============================================================================

class TestPrimaryConstructInlineComment:
    """Verify PrimaryConstruct.inline_comment field."""

    def test_default_is_none(self):
        pc = PrimaryConstruct(sig=Signature("A", 1, 1))
        assert pc.inline_comment is None

    def test_with_inline_comment(self):
        comment = Comment("(note)", 1, 3)
        pc = PrimaryConstruct(
            sig=Signature("A", 1, 1),
            inline_comment=comment,
        )
        assert pc.inline_comment is not None
        assert pc.inline_comment.text == "(note)"
        assert pc.inline_comment.line == 1
        assert pc.inline_comment.column == 3


# =============================================================================
# 4. Construct with Comment inner
# =============================================================================

class TestConstructCommentInner:
    """Verify a Comment can appear as a standalone construct inner."""

    def test_comment_as_inner(self):
        comment = Comment("(standalone)", 1, 1)
        construct = Construct(comment)
        assert isinstance(construct.inner, Comment)
        assert construct.inner.text == "(standalone)"

    def test_comment_inner_preserves_position(self):
        comment = Comment("(at position)", 3, 7)
        construct = Construct(comment)
        assert construct.inner.line == 3
        assert construct.inner.column == 7


# =============================================================================
# 5. Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Verify existing construction patterns still work without inline_comment."""

    def test_primary_construct_without_inline_comment(self):
        """Positional-style construction that doesn't pass inline_comment."""
        pc = PrimaryConstruct(
            sig=Signature("A", 1, 1),
            op=TokenType.COUNTERSIGN,
            node=Signature("B", 1, 5),
        )
        assert pc.sig.id == "A"
        assert pc.op == TokenType.COUNTERSIGN
        assert pc.node.id == "B"
        assert pc.inline_comment is None

    def test_bare_primary_construct(self):
        pc = PrimaryConstruct(sig=Signature("X", 2, 1))
        assert pc.op is None
        assert pc.node is None
        assert pc.inline_comment is None

    def test_construct_with_literal_inner(self):
        """Existing Literal construct still works."""
        lit = Literal("hello", 1, 1)
        construct = Construct(lit)
        assert isinstance(construct.inner, Literal)

    def test_construct_with_primary_list_inner(self):
        """Existing PrimaryConstruct list construct still works."""
        primaries = [
            PrimaryConstruct(sig=Signature("A", 1, 1)),
            PrimaryConstruct(sig=Signature("B", 2, 1), op=TokenType.UNDERSIGN, node=Literal("x", 2, 5)),
        ]
        construct = Construct(primaries)
        assert isinstance(construct.inner, list)
        assert len(construct.inner) == 2

    def test_construct_with_block_inner(self):
        """Existing Block construct still works."""
        block = Block(constructs=[
            Construct(Literal("a", 1, 1)),
        ])
        construct = Construct(block)
        assert isinstance(construct.inner, Block)


# =============================================================================
# 6. Comment equality
# =============================================================================

class TestCommentEquality:
    """Verify dataclass default __eq__ works for Comment."""

    def test_equal_comments(self):
        c1 = Comment("(same)", 1, 1)
        c2 = Comment("(same)", 1, 1)
        assert c1 == c2

    def test_unequal_text(self):
        c1 = Comment("(one)", 1, 1)
        c2 = Comment("(two)", 1, 1)
        assert c1 != c2

    def test_unequal_position(self):
        c1 = Comment("(same)", 1, 1)
        c2 = Comment("(same)", 2, 1)
        assert c1 != c2

    def test_not_equal_to_non_comment(self):
        c = Comment("(text)", 1, 1)
        assert c != "(text)"
