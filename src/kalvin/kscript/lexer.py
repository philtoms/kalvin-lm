"""Lexer for KScript source code."""

from typing import Generator

from .tokens import Token, TokenType

# Reserved keywords that cannot be used as identifiers
KEYWORDS = {
    "load": TokenType.LOAD,
    "save": TokenType.SAVE,
}

# Multi-character operators (must check before single-char)
MULTI_CHAR_OPS = {
    "=>": TokenType.S2,
    "!=": TokenType.S4,
}

# Single-character operators
SINGLE_CHAR_OPS = {
    "=": TokenType.S1,
    ">": TokenType.S3_FORWARD,
    "<": TokenType.S3_BACKWARD,
    "?": TokenType.ATTENTION,
}


class LexerError(Exception):
    """Lexer error with position information."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Line {line}, Column {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """
    Tokenizer for KScript source code.

    Uses a generator pattern for memory-efficient streaming tokenization.
    Comments in parentheses can appear anywhere, even mid-identifier.
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self._tokens: list[Token] | None = None

    def tokenize(self) -> list[Token]:
        """Tokenize entire source and return list of tokens."""
        if self._tokens is None:
            self._tokens = list(self._generate_tokens())
        return self._tokens

    def __iter__(self) -> Generator[Token, None, None]:
        """Iterate over tokens lazily."""
        if self._tokens is not None:
            yield from self._tokens
        else:
            yield from self._generate_tokens()

    def _generate_tokens(self) -> Generator[Token, None, None]:
        """Generate tokens from source."""
        while not self._at_end():
            # Skip whitespace (but not newlines) and comments
            self._skip_whitespace_and_comments()
            if self._at_end():
                break

            # Check for newline (statement separator)
            if self._current() == "\n":
                start_line, start_col = self.line, self.column
                self._advance()
                # Skip any additional newlines
                while self._current() == "\n":
                    self._advance()
                yield Token(TokenType.NEWLINE, "\n", start_line, start_col)
                continue

            # Try multi-character operators first
            if token := self._try_multi_char_op():
                yield token
            # Try single-character operators
            elif token := self._try_single_char_op():
                yield token
            # Try identifier or keyword
            elif token := self._try_identifier():
                yield token
            else:
                raise LexerError(
                    f"Unexpected character: '{self._current()}'",
                    self.line,
                    self.column,
                )

        yield Token(TokenType.EOF, "", self.line, self.column)

    def _current(self) -> str:
        """Get current character or empty string at end."""
        return self.source[self.pos] if not self._at_end() else ""

    def _peek(self, offset: int = 1) -> str:
        """Peek at character ahead."""
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else ""

    def _at_end(self) -> bool:
        """Check if at end of source."""
        return self.pos >= len(self.source)

    def _advance(self) -> str:
        """Advance position and return current character."""
        char = self._current()
        self.pos += 1
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace (except newlines) and comments."""
        while not self._at_end():
            c = self._current()
            if c in " \t\r":  # Skip whitespace but not newlines
                self._advance()
            elif c == "(":
                self._skip_comment()
            else:
                break

    def _skip_comment(self) -> None:
        """Skip a comment in parentheses: (anything)."""
        # We're at '('
        self._advance()  # consume '('

        # Find matching ')'
        depth = 1
        while not self._at_end() and depth > 0:
            c = self._current()
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            self._advance()

        # If depth > 0, we hit EOF without closing
        if depth > 0:
            raise LexerError("Unterminated comment", self.line, self.column)

    def _try_multi_char_op(self) -> Token | None:
        """Try to match a multi-character operator."""
        for op, token_type in MULTI_CHAR_OPS.items():
            if self.source[self.pos : self.pos + len(op)] == op:
                start_line, start_col = self.line, self.column
                for _ in op:
                    self._advance()
                return Token(token_type, op, start_line, start_col)
        return None

    def _try_single_char_op(self) -> Token | None:
        """Try to match a single-character operator."""
        c = self._current()
        if c in SINGLE_CHAR_OPS:
            start_line, start_col = self.line, self.column
            self._advance()
            return Token(SINGLE_CHAR_OPS[c], c, start_line, start_col)
        return None

    def _try_identifier(self) -> Token | None:
        """
        Try to match an identifier or keyword.

        Identifiers are unquoted and can include:
        - Alphanumeric characters
        - Underscores
        - Hyphens
        - Forward slashes (for file paths)
        - Dots (for file extensions)

        Comments in parentheses can appear mid-identifier and are spliced out.
        Example: V(erb) -> identifier "V"
        Example: hel(comment)lo -> identifier "hello"
        """
        c = self._current()

        # Identifiers must start with alphanumeric or allowed special chars
        if not (c.isalnum() or c in "_/-."):
            return None

        start_line, start_col = self.line, self.column
        chars: list[str] = []

        while not self._at_end():
            c = self._current()

            # Skip inline comments (they can appear mid-identifier)
            if c == "(":
                self._skip_comment()
                continue

            # Allow identifier characters
            if c.isalnum() or c in "_-./":
                chars.append(self._advance())
            else:
                break

        value = "".join(chars)

        # Check if it's a keyword
        if value in KEYWORDS:
            return Token(KEYWORDS[value], value, start_line, start_col)

        return Token(TokenType.IDENTIFIER, value, start_line, start_col)
