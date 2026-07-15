"""Comprehensive tests for the KScript v3 lexer.

Covers all acceptance criteria:
- KS-1: All token types recognized
- KS-2: Multi-char operator priority
- KS-3: ANNOTATION with nested parens
- KS-4: INDENT/DEDENT tracking
- KS-5: Edge cases
- Token position tracking
- Operator chains
"""

import pytest

from src.ks.lexer import Lexer, LexerError
from src.ks.token import Token, TokenType

# Shorthand aliases
CS = TokenType.COUNTERSIGNS
CZ = TokenType.CANONIZES
CT = TokenType.CONNOTES
US = TokenType.DENOTES
SG = TokenType.SIGNATURE
AN = TokenType.ANNOTATION
NL = TokenType.NEWLINE
IN = TokenType.INDENT
DD = TokenType.DEDENT
EOF = TokenType.EOF


def _types(source: str) -> list[TokenType]:
    """Helper: extract token types from source, excluding EOF."""
    return [t.type for t in Lexer(source).tokenize()[:-1]]


def _tokens(source: str) -> list[Token]:
    """Helper: get all tokens including EOF."""
    return Lexer(source).tokenize()


# ── KS-1: All token types recognized ───────────────────────────────


class TestAllTokenTypes:
    """KS-1: Source containing every token type produces correct sequence."""

    def test_all_types_present(self):
        source = "A == B\n  C(inline) => D > E = F\n(word)\n  G\nH"
        tokens = _tokens(source)
        type_names = {t.type.name for t in tokens}
        expected = {
            "COUNTERSIGNS",
            "CANONIZES",
            "CONNOTES",
            "DENOTES",
            "SIGNATURE",
            "ANNOTATION",
            "NEWLINE",
            "INDENT",
            "DEDENT",
            "EOF",
        }
        assert expected.issubset(type_names), f"Missing types: {expected - type_names}"

    def test_all_types_in_correct_order(self):
        source = "A == B => C > D = E"
        types = _types(source)
        assert types == [
            SG,
            CS,
            SG,
            CZ,
            SG,
            CT,
            SG,
            US,
            SG,
        ]


# ── KS-2: Multi-char operator priority ─────────────────────────────


class TestOperatorPriority:
    """KS-2: Multi-char operators (==, =>) matched before single-char (=, >)."""

    def test_countersign_not_two_denotes(self):
        """'A == B' → SIGNATURE, COUNTERSIGNS, SIGNATURE (not two DENOTES)."""
        tokens = _tokens("A == B")
        assert tokens[1].type == CS
        assert tokens[1].value == "=="
        assert len([t for t in tokens if t.type == US]) == 0

    def test_canonize_not_denote_connote(self):
        """'A => B' → SIGNATURE, CANONIZES, SIGNATURE (not =, >)."""
        tokens = _tokens("A => B")
        assert tokens[1].type == CZ
        assert tokens[1].value == "=>"
        assert len([t for t in tokens if t.type == US]) == 0
        assert len([t for t in tokens if t.type == CT]) == 0

    def test_single_char_operators_when_not_multi(self):
        """'A = B > C' → SIGNATURE, DENOTES, SIGNATURE, CONNOTES, SIGNATURE."""
        types = _types("A = B > C")
        assert types == [SG, US, SG, CT, SG]

    def test_operator_chain(self):
        """'A == B > C = D' produces correct chain of operators."""
        types = _types("A == B > C = D")
        assert types == [SG, CS, SG, CT, SG, US, SG]


# ── KS-3: ANNOTATION with nested parens ────────────────────────────


class TestAnnotation:
    """KS-3: ANNOTATION tokens with nested parens, inline, and multi-line."""

    def test_nested_parens_standalone(self):
        """'(Mary Had A (Little) Lamb)' → single ANNOTATION token."""
        tokens = _tokens("(Mary Had A (Little) Lamb)")
        assert len(tokens) == 2  # ANNOTATION + EOF
        assert tokens[0].type == AN
        assert tokens[0].value == "(Mary Had A (Little) Lamb)"

    def test_simple_standalone_annotation(self):
        """'(word)' → ANNOTATION."""
        tokens = _tokens("(word)")
        assert tokens[0].type == AN
        assert tokens[0].value == "(word)"

    def test_inline_annotation(self):
        """'S(ubject)' → SIGNATURE('S') followed by ANNOTATION('(ubject)')."""
        tokens = _tokens("S(ubject)")
        assert tokens[0].type == SG
        assert tokens[0].value == "S"
        assert tokens[1].type == AN
        assert tokens[1].value == "(ubject)"

    def test_multiline_annotation(self):
        """Multi-line annotation spanning newlines → single ANNOTATION token."""
        source = "(multi\nline)"
        tokens = _tokens(source)
        assert tokens[0].type == AN
        assert tokens[0].value == "(multi\nline)"

    def test_annotation_with_deep_nesting(self):
        """Annotations with multiple nesting levels."""
        source = "(outer (middle (inner) middle) outer)"
        tokens = _tokens(source)
        assert tokens[0].type == AN
        assert tokens[0].value == "(outer (middle (inner) middle) outer)"

    def test_inline_annotation_with_operators(self):
        """Inline annotation followed by operators."""
        source = "S(ubject) = M"
        types = _types(source)
        assert types == [SG, AN, US, SG]

    def test_empty_annotation(self):
        """Empty parentheses → ANNOTATION."""
        tokens = _tokens("()")
        assert tokens[0].type == AN
        assert tokens[0].value == "()"


# ── KS-4: INDENT/DEDENT tracking ───────────────────────────────────


class TestIndentDedent:
    """KS-4: Python-style INDENT/DEDENT tokens based on leading whitespace."""

    def test_simple_indent(self):
        """Simple indent produces INDENT + content + DEDENT."""
        source = "A\n  B\nC"
        types = _types(source)
        assert types == [SG, NL, IN, SG, NL, DD, SG]

    def test_nested_indent(self):
        """Nested indent produces INDENT, INDENT, content, DEDENT, DEDENT."""
        source = "A\n  B\n    C\n  D\nE"
        types = _types(source)
        assert types == [
            SG,
            NL,
            IN,
            SG,
            NL,
            IN,
            SG,
            NL,
            DD,
            SG,
            NL,
            DD,
            SG,
        ]

    def test_multiple_dedent_levels(self):
        """Multiple dedent levels emit multiple DEDENT tokens in sequence."""
        source = "A\n  B\n    C\nD"
        types = _types(source)
        assert types == [
            SG,
            NL,
            IN,
            SG,
            NL,
            IN,
            SG,
            NL,
            DD,
            DD,
            SG,
        ]

    def test_same_level_continuation(self):
        """Same-level continuation produces no INDENT/DEDENT."""
        source = "A\nB\nC"
        types = _types(source)
        assert types == [SG, NL, SG, NL, SG]

    def test_indent_stack_at_eof(self):
        """Remaining indent levels are closed with DEDENT at EOF."""
        source = "A\n  B"
        tokens = _tokens(source)
        # Should end with DEDENT, EOF
        assert tokens[-2].type == DD
        assert tokens[-1].type == EOF

    def test_multiple_indent_same_block(self):
        """Two lines at same indent level → INDENT only once."""
        source = "A\n  B\n  C\nD"
        types = _types(source)
        assert types == [SG, NL, IN, SG, NL, SG, NL, DD, SG]


# ── KS-5: Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    """KS-5: Edge cases for empty input, errors, and invalid identifiers."""

    def test_empty_string(self):
        """Empty string → [EOF]."""
        tokens = _tokens("")
        assert len(tokens) == 1
        assert tokens[0].type == EOF

    def test_whitespace_only(self):
        """Whitespace-only → [EOF]."""
        tokens = _tokens("   \n  ")
        assert len(tokens) == 1
        assert tokens[0].type == EOF

    def test_unknown_character_error(self):
        """'@' → LexerError."""
        with pytest.raises(LexerError) as exc_info:
            _tokens("@")
        assert exc_info.value.line == 1
        assert exc_info.value.column == 1

    def test_less_than_error(self):
        """'<' → LexerError."""
        with pytest.raises(LexerError) as exc_info:
            _tokens("A < B")
        assert exc_info.value.line == 1
        assert exc_info.value.column == 3

    def test_identifier_with_digits_accepted(self):
        """'AB3' → SIGNATURE (digits after first char are valid)."""
        toks = _tokens("AB3")
        assert toks[0].type == TokenType.SIGNATURE
        assert toks[0].value == "AB3"

    def test_lowercase_identifier_accepted(self):
        """'abc' → SIGNATURE (identifiers are case-insensitive)."""
        toks = _tokens("abc")
        assert toks[0].type == TokenType.SIGNATURE
        assert toks[0].value == "abc"

    def test_tabs_as_indent(self):
        """Tabs count as indent units."""
        source = "A\n\tB\nC"
        types = _types(source)
        assert types == [SG, NL, IN, SG, NL, DD, SG]

    def test_mixed_tabs_spaces_indent(self):
        """Mixed tabs and spaces count towards indent level."""
        source = "A\n\t B\nC"
        types = _types(source)
        assert types == [SG, NL, IN, SG, NL, DD, SG]

    def test_multi_char_signature(self):
        """Multi-char uppercase identifier → SIGNATURE."""
        tokens = _tokens("ABC")
        assert tokens[0].type == SG
        assert tokens[0].value == "ABC"

    def test_consecutive_operators(self):
        """'A = =' → SIGNATURE, DENOTES, DENOTES."""
        types = _types("A = =")
        assert types == [SG, US, US]


# ── Token position tracking ────────────────────────────────────────


class TestPositionTracking:
    """Tokens have correct line and column values (1-based)."""

    def test_single_line_positions(self):
        """Tokens on a single line have correct columns."""
        tokens = _tokens("A == B")
        assert tokens[0].line == 1 and tokens[0].column == 1
        assert tokens[1].line == 1 and tokens[1].column == 3
        assert tokens[2].line == 1 and tokens[2].column == 6

    def test_newline_updates_line(self):
        """NEWLINE token has correct position, next token on new line."""
        tokens = _tokens("A\n  B")
        # A at (1,1)
        assert tokens[0].line == 1 and tokens[0].column == 1
        # NEWLINE at (1,2)
        assert tokens[1].line == 1 and tokens[1].column == 2
        # INDENT at (2,1)
        assert tokens[2].line == 2 and tokens[2].column == 1
        # B at (2,3)
        assert tokens[3].line == 2 and tokens[3].column == 3

    def test_multiline_annotation_positions(self):
        """Multi-line annotation updates line tracking."""
        source = "A(multi\nline)\nB"
        tokens = _tokens(source)
        # A at (1,1)
        assert tokens[0].line == 1 and tokens[0].column == 1
        # ANNOTATION starting at (1,2)
        assert tokens[1].line == 1 and tokens[1].column == 2
        # NEWLINE at (2,6) — after 'line)'
        # B at (3,1)
        assert tokens[3].line == 3 and tokens[3].column == 1

    def test_indent_token_position(self):
        """INDENT/DEDENT tokens have correct position at line start."""
        tokens = _tokens("A\n  B\nC")
        # INDENT at (2,1)
        indents = [t for t in tokens if t.type == IN]
        assert len(indents) == 1
        assert indents[0].line == 2
        # DEDENT at (3,1)
        dedents = [t for t in tokens if t.type == DD]
        assert len(dedents) == 1
        assert dedents[0].line == 3

    def test_eof_position(self):
        """EOF token has position at end of input."""
        tokens = _tokens("A\nB")
        eof = tokens[-1]
        assert eof.type == EOF
        assert eof.line == 2


# ── LexerError attributes ──────────────────────────────────────────


class TestLexerError:
    """LexerError has message, line, and column attributes."""

    def test_error_has_message(self):
        with pytest.raises(LexerError) as exc_info:
            _tokens("@")
        assert exc_info.value.message
        assert "@" in exc_info.value.message

    def test_error_has_line_column(self):
        with pytest.raises(LexerError) as exc_info:
            _tokens("A < B")
        assert exc_info.value.line == 1
        assert exc_info.value.column == 3

    def test_error_inherits_exception(self):
        with pytest.raises(LexerError) as exc_info:
            _tokens("3abc")
        assert isinstance(exc_info.value, Exception)

    def test_error_string_includes_position(self):
        with pytest.raises(LexerError) as exc_info:
            _tokens("A\n@")
        error_str = str(exc_info.value)
        assert "Line 2" in error_str
        assert "column 1" in error_str


# ── Complex integration scenarios ──────────────────────────────────


class TestComplexScenarios:
    """Integration tests combining multiple features."""

    def test_canonize_with_indented_block(self):
        """A =>\\n  B\\n  C → correct structure with INDENT/DEDENT."""
        source = "A =>\n  B\n  C"
        types = _types(source)
        assert types == [SG, CZ, NL, IN, SG, NL, SG, DD]

    def test_full_construct_with_inline_annotation(self):
        """Full construct: MHALL == SVO with annotations."""
        source = "MHALL == SVO"
        tokens = _tokens(source)
        assert tokens[0].type == SG and tokens[0].value == "MHALL"
        assert tokens[1].type == CS
        assert tokens[2].type == SG and tokens[2].value == "SVO"

    def test_annotation_between_constructs(self):
        """Standalone annotation between constructs."""
        source = "A(word)\nB"
        types = _types(source)
        # SIGNATURE("A"), ANNOTATION("(word)"), NEWLINE, SIGNATURE("B")
        assert types == [SG, AN, NL, SG]

    def test_nested_indented_blocks(self):
        """Multiple levels of indentation."""
        source = "A\n  B\n    C\n      D\nE"
        types = _types(source)
        # A NL IN B NL IN C NL IN D NL DD DD DD E
        assert types == [
            SG,
            NL,
            IN,
            SG,
            NL,
            IN,
            SG,
            NL,
            IN,
            SG,
            NL,
            DD,
            DD,
            DD,
            SG,
        ]

    def test_inline_annotation_in_indented_block(self):
        """Inline annotation inside an indented block."""
        source = "A =>\n  S(ubject) = M"
        types = _types(source)
        assert types == [SG, CZ, NL, IN, SG, AN, US, SG, DD]

    def test_multiline_annotation_does_not_break_indent(self):
        """Multi-line annotation spanning lines doesn't trigger INDENT/DEDENT."""
        source = "A\n(multi\nline)\nB"
        types = _types(source)
        assert types == [SG, NL, AN, NL, SG]
