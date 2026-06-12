"""Tests for KScript v3 parser (KB-187).

Covers:
  KS-6  — OperatorScope nodes with sig, op, items (nested scope model)
  KS-7  — Block parsing: INDENT/DEDENT creates Block nodes
  KS-8  — Annotations preserved as AST nodes
  KS-9  — Inline annotation attachment (sig-side and node-side)
  KS-10 — Empty source produces empty constructs
  Additional coverage: bare signature, CANONIZE, multi-item scopes,
  child blocks, annotations mixed with operators, parse errors.
"""

from __future__ import annotations

import pytest

from ks.ast import Annotation, Block, KScriptFile, OperatorScope, Signature
from ks.parser import ParseError, Parser
from ks.token import Token, TokenType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tok(type_: TokenType, value: str = "", line: int = 1, column: int = 1) -> Token:
    """Shorthand for building a single Token."""
    return Token(type_, value, line, column)


def tokens(*specs: tuple[TokenType, str], eof_line: int = 1, eof_col: int = 0) -> list[Token]:
    """Build a token list with line 1 and auto-incrementing columns.

    Usage: tokens((SIGNATURE, "A"), (COUNTERSIGN, "=="), (SIGNATURE, "B"))
    """
    result: list[Token] = []
    col = 1
    for type_, value in specs:
        result.append(Token(type_, value, 1, col))
        col += len(value) + 1  # space between tokens
    result.append(Token(TokenType.EOF, "", eof_line, eof_col))
    return result


def parse(*specs: tuple[TokenType, str], eof_line: int = 1, eof_col: int = 0) -> KScriptFile:
    """Build tokens from specs and parse them in one step."""
    return Parser(tokens(*specs, eof_line=eof_line, eof_col=eof_col)).parse()


def parse_tokens(tok_list: list[Token]) -> KScriptFile:
    """Parse a pre-built token list."""
    return Parser(tok_list).parse()


# ---------------------------------------------------------------------------
# KS-10 — Empty source
# ---------------------------------------------------------------------------

class TestKS10EmptySource:
    """KS-10: Empty source produces empty script (no error)."""

    def test_eof_only(self) -> None:
        """A single EOF token produces KScriptFile(constructs=[])."""
        result = parse_tokens([tok(TokenType.EOF)])
        assert result == KScriptFile(constructs=[])

    def test_whitespace_only_newlines(self) -> None:
        """Only NEWLINE tokens produce empty constructs."""
        result = parse_tokens([
            tok(TokenType.NEWLINE, "\n", 1, 1),
            tok(TokenType.NEWLINE, "\n", 2, 1),
            tok(TokenType.EOF, "", 3, 1),
        ])
        assert result == KScriptFile(constructs=[])


# ---------------------------------------------------------------------------
# KS-6 — OperatorScope structure (chained scopes)
# ---------------------------------------------------------------------------

class TestKS6ChainedScopes:
    """KS-6: AST structure reflects scope model with nested OperatorScope nodes."""

    def test_a_counter_b_connotate_c_undersign_d(self) -> None:
        """Parse 'A == B > C = D' — three nested OperatorScope nodes.

        Structure (nested model per grammar §4 item ::= operator_scope):
            constructs[0]         → OperatorScope(sig=A, op=COUNTERSIGN)
            constructs[0].items[0] → OperatorScope(sig=B, op=CONNOTATE)
            constructs[0].items[0].items[0] → OperatorScope(sig=C, op=UNDERSIGN, items=[Signature(D)])

        Scope rules S2/S3: B is a node in scope A and the signature of scope B.
        C is a node in scope B and the signature of scope C.  D is a node in scope C.
        """
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.COUNTERSIGN, "==", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 6),
            tok(TokenType.CONNOTATE, ">", 1, 8),
            tok(TokenType.SIGNATURE, "C", 1, 10),
            tok(TokenType.UNDERSIGN, "=", 1, 12),
            tok(TokenType.SIGNATURE, "D", 1, 14),
            tok(TokenType.EOF, "", 1, 15),
        ])

        assert len(result.constructs) == 1

        # Scope 1: A == B (B carried forward as nested scope)
        scope_ab = result.constructs[0]
        assert isinstance(scope_ab, OperatorScope)
        assert scope_ab.sig.id == "A"
        assert scope_ab.op is TokenType.COUNTERSIGN
        assert len(scope_ab.items) == 1

        # Scope 2: B > C (nested as item of scope 1)
        scope_bc = scope_ab.items[0]
        assert isinstance(scope_bc, OperatorScope)
        assert scope_bc.sig.id == "B"
        assert scope_bc.op is TokenType.CONNOTATE
        assert len(scope_bc.items) == 1

        # Scope 3: C = D (nested as item of scope 2)
        scope_cd = scope_bc.items[0]
        assert isinstance(scope_cd, OperatorScope)
        assert scope_cd.sig.id == "C"
        assert scope_cd.op is TokenType.UNDERSIGN
        assert len(scope_cd.items) == 1
        assert isinstance(scope_cd.items[0], Signature)
        assert scope_cd.items[0].id == "D"

    def test_signatures_carry_forward_s2_s3(self) -> None:
        """Verify scope rules S2 (preceding id = sig) and S3 (succeeding id = node)."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "X", 1, 1),
            tok(TokenType.UNDERSIGN, "=", 1, 3),
            tok(TokenType.SIGNATURE, "Y", 1, 5),
            tok(TokenType.CONNOTATE, ">", 1, 7),
            tok(TokenType.SIGNATURE, "Z", 1, 9),
            tok(TokenType.EOF, "", 1, 10),
        ])

        # S2: X precedes = → X is sig of scope 1
        scope1 = result.constructs[0]
        assert scope1.sig.id == "X"

        # S3: Y succeeds = → Y is node in scope 1
        # S2: Y precedes > → Y is also sig of scope 2
        assert isinstance(scope1.items[0], OperatorScope)
        scope2 = scope1.items[0]
        assert scope2.sig.id == "Y"

        # S3: Z succeeds > → Z is node in scope 2
        assert isinstance(scope2.items[0], Signature)
        assert scope2.items[0].id == "Z"


# ---------------------------------------------------------------------------
# KS-7 — Block parsing
# ---------------------------------------------------------------------------

class TestKS7BlockParsing:
    """KS-7: INDENT/DEDENT creates Block nodes in child_block."""

    def test_canonize_with_child_block(self) -> None:
        """Parse 'A =>\\n  B\\n  C' — child_block with Block node."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.CANONIZE, "=>", 1, 3),
            tok(TokenType.NEWLINE, "\n", 1, 5),
            tok(TokenType.INDENT, "", 2, 1),
            tok(TokenType.SIGNATURE, "B", 2, 3),
            tok(TokenType.NEWLINE, "\n", 2, 4),
            tok(TokenType.SIGNATURE, "C", 3, 3),
            tok(TokenType.DEDENT, "", 4, 1),
            tok(TokenType.EOF, "", 4, 1),
        ])

        assert len(result.constructs) == 1
        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op is TokenType.CANONIZE
        assert scope.items == []  # no items on same line

        # child_block is a Block with two bare-sig constructs
        assert scope.child_block is not None
        block = scope.child_block
        assert isinstance(block, Block)
        assert len(block.constructs) == 2

        b_scope = block.constructs[0]
        assert isinstance(b_scope, OperatorScope)
        assert b_scope.sig.id == "B"
        assert b_scope.op is None  # bare signature

        c_scope = block.constructs[1]
        assert isinstance(c_scope, OperatorScope)
        assert c_scope.sig.id == "C"
        assert c_scope.op is None

    def test_nested_child_blocks(self) -> None:
        """Two levels of indentation produce nested Blocks."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.CANONIZE, "=>", 1, 3),
            tok(TokenType.NEWLINE, "\n", 1, 5),
            tok(TokenType.INDENT, "", 2, 1),
            tok(TokenType.SIGNATURE, "B", 2, 3),
            tok(TokenType.UNDERSIGN, "=", 2, 5),
            tok(TokenType.SIGNATURE, "C", 2, 7),
            tok(TokenType.NEWLINE, "\n", 2, 8),
            tok(TokenType.INDENT, "", 3, 1),
            tok(TokenType.SIGNATURE, "D", 3, 5),
            tok(TokenType.DEDENT, "", 4, 1),
            tok(TokenType.DEDENT, "", 4, 1),
            tok(TokenType.EOF, "", 4, 1),
        ])

        scope_a = result.constructs[0]
        assert scope_a.child_block is not None
        assert len(scope_a.child_block.constructs) == 1

        scope_b = scope_a.child_block.constructs[0]
        assert isinstance(scope_b, OperatorScope)
        assert scope_b.sig.id == "B"
        assert scope_b.op is TokenType.UNDERSIGN
        assert scope_b.items[0].id == "C"
        assert scope_b.child_block is not None
        assert len(scope_b.child_block.constructs) == 1

        scope_d = scope_b.child_block.constructs[0]
        assert isinstance(scope_d, OperatorScope)
        assert scope_d.sig.id == "D"
        assert scope_d.op is None

    def test_items_with_child_block(self) -> None:
        """'A == B\\n  C\\n  D' — items on same line plus child block."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.COUNTERSIGN, "==", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 6),
            tok(TokenType.NEWLINE, "\n", 1, 7),
            tok(TokenType.INDENT, "", 2, 1),
            tok(TokenType.SIGNATURE, "C", 2, 3),
            tok(TokenType.NEWLINE, "\n", 2, 4),
            tok(TokenType.SIGNATURE, "D", 3, 3),
            tok(TokenType.DEDENT, "", 4, 1),
            tok(TokenType.EOF, "", 4, 1),
        ])

        scope = result.constructs[0]
        assert scope.sig.id == "A"
        assert scope.op is TokenType.COUNTERSIGN
        # B on same line
        assert len(scope.items) == 1
        assert scope.items[0].id == "B"
        # C, D in child block
        assert scope.child_block is not None
        assert len(scope.child_block.constructs) == 2


# ---------------------------------------------------------------------------
# KS-8 — Annotations preserved
# ---------------------------------------------------------------------------

class TestKS8AnnotationsPreserved:
    """KS-8: Annotations are preserved as AST nodes, not discarded."""

    def test_standalone_annotation_before_scope(self) -> None:
        """(Mary Had A Little Lamb)\\nMHALL == SVO"""
        result = parse_tokens([
            tok(TokenType.ANNOTATION, "(Mary Had A Little Lamb)", 1, 1),
            tok(TokenType.NEWLINE, "\n", 1, 24),
            tok(TokenType.SIGNATURE, "MHALL", 2, 1),
            tok(TokenType.COUNTERSIGN, "==", 2, 7),
            tok(TokenType.SIGNATURE, "SVO", 2, 10),
            tok(TokenType.EOF, "", 2, 13),
        ])

        assert len(result.constructs) == 2

        # First construct is the Annotation
        ann = result.constructs[0]
        assert isinstance(ann, Annotation)
        assert ann.text == "(Mary Had A Little Lamb)"

        # Second construct is the OperatorScope
        scope = result.constructs[1]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "MHALL"

    def test_annotation_in_child_block(self) -> None:
        """Annotation inside a Block is preserved as a construct."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.CANONIZE, "=>", 1, 3),
            tok(TokenType.NEWLINE, "\n", 1, 5),
            tok(TokenType.INDENT, "", 2, 1),
            tok(TokenType.ANNOTATION, "(hello)", 2, 3),
            tok(TokenType.NEWLINE, "\n", 2, 10),
            tok(TokenType.SIGNATURE, "B", 3, 3),
            tok(TokenType.DEDENT, "", 4, 1),
            tok(TokenType.EOF, "", 4, 1),
        ])

        scope = result.constructs[0]
        assert scope.child_block is not None
        assert len(scope.child_block.constructs) == 2
        assert isinstance(scope.child_block.constructs[0], Annotation)
        assert scope.child_block.constructs[0].text == "(hello)"

    def test_annotation_as_scope_item(self) -> None:
        """Annotation between items is preserved in OperatorScope.items."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.UNDERSIGN, "=", 1, 3),
            tok(TokenType.ANNOTATION, "(note)", 1, 5),
            tok(TokenType.SIGNATURE, "B", 1, 12),
            tok(TokenType.EOF, "", 1, 13),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert len(scope.items) == 2
        assert isinstance(scope.items[0], Annotation)
        assert scope.items[0].text == "(note)"
        assert isinstance(scope.items[1], Signature)
        assert scope.items[1].id == "B"


# ---------------------------------------------------------------------------
# KS-9 — Inline annotation attachment
# ---------------------------------------------------------------------------

class TestKS9InlineAnnotation:
    """KS-9: Inline annotations attach to sig-side and node-side."""

    def test_sig_side_inline_annotation(self) -> None:
        """S(ubject) = M — ANNOTATION after SIGNATURE before operator."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "S", 1, 1),
            tok(TokenType.ANNOTATION, "(ubject)", 1, 2),
            tok(TokenType.UNDERSIGN, "=", 1, 11),
            tok(TokenType.SIGNATURE, "M", 1, 13),
            tok(TokenType.EOF, "", 1, 14),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "S"
        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"
        assert scope.op is TokenType.UNDERSIGN
        assert scope.items[0].id == "M"
        assert scope.node_inline_annotation is None

    def test_node_side_inline_annotation(self) -> None:
        """A = D(et) — ANNOTATION after SIGNATURE that is an item."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.UNDERSIGN, "=", 1, 3),
            tok(TokenType.SIGNATURE, "D", 1, 5),
            tok(TokenType.ANNOTATION, "(et)", 1, 6),
            tok(TokenType.EOF, "", 1, 10),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.inline_annotation is None
        assert scope.node_inline_annotation is not None
        assert scope.node_inline_annotation.text == "(et)"
        assert scope.items[0].id == "D"

    def test_both_inline_annotations(self) -> None:
        """S(ubject) = D(et) — both sig-side and node-side on same scope."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "S", 1, 1),
            tok(TokenType.ANNOTATION, "(ubject)", 1, 2),
            tok(TokenType.UNDERSIGN, "=", 1, 11),
            tok(TokenType.SIGNATURE, "D", 1, 13),
            tok(TokenType.ANNOTATION, "(et)", 1, 14),
            tok(TokenType.EOF, "", 1, 18),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"
        assert scope.node_inline_annotation is not None
        assert scope.node_inline_annotation.text == "(et)"

    def test_sig_side_inline_on_bare_signature(self) -> None:
        """S(ubject) without operator — bare sig with inline annotation."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "S", 1, 1),
            tok(TokenType.ANNOTATION, "(ubject)", 1, 2),
            tok(TokenType.EOF, "", 1, 10),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "S"
        assert scope.op is None  # bare
        assert scope.inline_annotation is not None
        assert scope.inline_annotation.text == "(ubject)"
        assert scope.items == []


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------

class TestBareSignature:
    """Single bare signature produces OperatorScope with op=None."""

    def test_single_bare_sig(self) -> None:
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.EOF, "", 1, 2),
        ])
        assert len(result.constructs) == 1
        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op is None
        assert scope.items == []
        assert scope.child_block is None


class TestCanonizeWithItems:
    """CANONIZE operator with multiple items on the same line."""

    def test_canonize_multiple_items(self) -> None:
        """A => B C D — three items aggregated under CANONIZE."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.CANONIZE, "=>", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 6),
            tok(TokenType.SIGNATURE, "C", 1, 8),
            tok(TokenType.SIGNATURE, "D", 1, 10),
            tok(TokenType.EOF, "", 1, 11),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op is TokenType.CANONIZE
        assert len(scope.items) == 3
        assert [s.id for s in scope.items] == ["B", "C", "D"]
        assert scope.child_block is None


class TestOperatorWithMultipleItems:
    """Operators with multiple items where last carries into next scope."""

    def test_undersign_multiple_items(self) -> None:
        """A = B C D — three items under UNDERSIGN."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.UNDERSIGN, "=", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 5),
            tok(TokenType.SIGNATURE, "C", 1, 7),
            tok(TokenType.SIGNATURE, "D", 1, 9),
            tok(TokenType.EOF, "", 1, 10),
        ])

        scope = result.constructs[0]
        assert isinstance(scope, OperatorScope)
        assert scope.sig.id == "A"
        assert scope.op is TokenType.UNDERSIGN
        assert len(scope.items) == 3
        assert [s.id for s in scope.items] == ["B", "C", "D"]

    def test_mixed_chain_with_items(self) -> None:
        """A == B C > D — two items in first scope, then nested scope."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.COUNTERSIGN, "==", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 6),
            tok(TokenType.SIGNATURE, "C", 1, 8),
            tok(TokenType.CONNOTATE, ">", 1, 10),
            tok(TokenType.SIGNATURE, "D", 1, 12),
            tok(TokenType.EOF, "", 1, 13),
        ])

        scope_ab = result.constructs[0]
        assert scope_ab.sig.id == "A"
        assert scope_ab.op is TokenType.COUNTERSIGN
        # B and C are items; C is also the sig of the nested scope
        assert len(scope_ab.items) == 2
        assert isinstance(scope_ab.items[0], Signature)
        assert scope_ab.items[0].id == "B"
        assert isinstance(scope_ab.items[1], OperatorScope)
        assert scope_ab.items[1].sig.id == "C"
        assert scope_ab.items[1].op is TokenType.CONNOTATE
        assert scope_ab.items[1].items[0].id == "D"


class TestIndentedChildBlock:
    """Indented child blocks under various operators."""

    def test_canonize_with_items_and_child_block(self) -> None:
        """A => B\\n  C — B on same line, C in child block."""
        result = parse_tokens([
            tok(TokenType.SIGNATURE, "A", 1, 1),
            tok(TokenType.CANONIZE, "=>", 1, 3),
            tok(TokenType.SIGNATURE, "B", 1, 6),
            tok(TokenType.NEWLINE, "\n", 1, 7),
            tok(TokenType.INDENT, "", 2, 1),
            tok(TokenType.SIGNATURE, "C", 2, 3),
            tok(TokenType.DEDENT, "", 3, 1),
            tok(TokenType.EOF, "", 3, 1),
        ])

        scope = result.constructs[0]
        assert scope.sig.id == "A"
        assert scope.op is TokenType.CANONIZE
        assert len(scope.items) == 1
        assert scope.items[0].id == "B"
        assert scope.child_block is not None
        assert len(scope.child_block.constructs) == 1
        assert scope.child_block.constructs[0].sig.id == "C"


class TestAnnotationsMixedWithOperators:
    """Standalone annotations and operators on separate lines."""

    def test_annotation_between_scopes(self) -> None:
        """(doc)\\nA == B\\n(note)\\nC > D"""
        result = parse_tokens([
            tok(TokenType.ANNOTATION, "(doc)", 1, 1),
            tok(TokenType.NEWLINE, "\n", 1, 6),
            tok(TokenType.SIGNATURE, "A", 2, 1),
            tok(TokenType.COUNTERSIGN, "==", 2, 3),
            tok(TokenType.SIGNATURE, "B", 2, 6),
            tok(TokenType.NEWLINE, "\n", 2, 7),
            tok(TokenType.ANNOTATION, "(note)", 3, 1),
            tok(TokenType.NEWLINE, "\n", 3, 7),
            tok(TokenType.SIGNATURE, "C", 4, 1),
            tok(TokenType.CONNOTATE, ">", 4, 3),
            tok(TokenType.SIGNATURE, "D", 4, 5),
            tok(TokenType.EOF, "", 4, 6),
        ])

        assert len(result.constructs) == 4

        assert isinstance(result.constructs[0], Annotation)
        assert result.constructs[0].text == "(doc)"

        scope_ab = result.constructs[1]
        assert isinstance(scope_ab, OperatorScope)
        assert scope_ab.sig.id == "A"
        assert scope_ab.op is TokenType.COUNTERSIGN

        assert isinstance(result.constructs[2], Annotation)
        assert result.constructs[2].text == "(note)"

        scope_cd = result.constructs[3]
        assert isinstance(scope_cd, OperatorScope)
        assert scope_cd.sig.id == "C"
        assert scope_cd.op is TokenType.CONNOTATE


class TestParseErrors:
    """Parser raises ParseError for unexpected tokens."""

    def test_operator_without_signature(self) -> None:
        """Operator at start of input raises ParseError."""
        with pytest.raises(ParseError, match="Unexpected token"):
            parse_tokens([
                tok(TokenType.COUNTERSIGN, "==", 1, 1),
                tok(TokenType.EOF, "", 1, 3),
            ])

    def test_double_operator(self) -> None:
        """Two operators in a row raises ParseError."""
        with pytest.raises(ParseError):
            parse_tokens([
                tok(TokenType.SIGNATURE, "A", 1, 1),
                tok(TokenType.COUNTERSIGN, "==", 1, 3),
                tok(TokenType.UNDERSIGN, "=", 1, 6),
                tok(TokenType.EOF, "", 1, 7),
            ])

    def test_unexpected_dedent(self) -> None:
        """DEDENT without matching INDENT raises ParseError."""
        with pytest.raises(ParseError, match="Unexpected token"):
            parse_tokens([
                tok(TokenType.DEDENT, "", 1, 1),
                tok(TokenType.EOF, "", 1, 1),
            ])
