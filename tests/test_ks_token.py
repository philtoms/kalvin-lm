"""Tests for KScript v3 token types and Token dataclass."""

import pytest

from ks.token import Token, TokenType


class TestTokenType:
    """Tests for the TokenType enum."""

    EXPECTED_NAMES = {
        "COUNTERSIGN",
        "CANONIZE",
        "CONNOTATE",
        "UNDERSIGN",
        "SIGNATURE",
        "ANNOTATION",
        "NEWLINE",
        "INDENT",
        "DEDENT",
        "EOF",
    }

    def test_all_ten_token_types_exist(self):
        """TokenType must have exactly the 10 specified members."""
        actual_names = {t.name for t in TokenType}
        assert actual_names == self.EXPECTED_NAMES

    def test_exactly_ten_members(self):
        """TokenType must have exactly 10 members — no more, no fewer."""
        assert len(TokenType) == 10

    def test_all_values_distinct(self):
        """No two token types may share the same value (no aliasing)."""
        values = [t.value for t in TokenType]
        assert len(values) == len(set(values))

    def test_annotation_exists(self):
        """ANNOTATION replaces the old COMMENT token type."""
        assert hasattr(TokenType, "ANNOTATION")
        assert isinstance(TokenType.ANNOTATION, TokenType)

    def test_comment_does_not_exist(self):
        """COMMENT must NOT exist on the new enum."""
        assert not hasattr(TokenType, "COMMENT")


class TestToken:
    """Tests for the Token frozen dataclass."""

    def test_field_access(self):
        """Token fields return correct values."""
        token = Token(type=TokenType.COUNTERSIGN, value="==", line=3, column=7)
        assert token.type is TokenType.COUNTERSIGN
        assert token.value == "=="
        assert token.line == 3
        assert token.column == 7

    def test_frozen(self):
        """Token must be frozen — attribute assignment raises FrozenInstanceError."""
        token = Token(type=TokenType.CANONIZE, value="=>", line=1, column=1)
        with pytest.raises(AttributeError):
            token.value = "!="
        with pytest.raises(AttributeError):
            token.line = 99

    def test_hashable(self):
        """Token must be hashable so it can be used in sets and dicts."""
        t1 = Token(type=TokenType.EOF, value="", line=1, column=1)
        t2 = Token(type=TokenType.NEWLINE, value="\n", line=2, column=1)
        token_set = {t1, t2}
        assert len(token_set) == 2
        assert t1 in token_set
        assert t2 in token_set

    def test_equality(self):
        """Two tokens with identical fields are equal."""
        t1 = Token(type=TokenType.INDENT, value="", line=4, column=1)
        t2 = Token(type=TokenType.INDENT, value="", line=4, column=1)
        assert t1 == t2

    def test_inequality(self):
        """Tokens differing in any field are not equal."""
        t1 = Token(type=TokenType.INDENT, value="", line=4, column=1)
        t2 = Token(type=TokenType.DEDENT, value="", line=4, column=1)
        assert t1 != t2
