"""Lexer for KScript source code with indentation support."""

from typing import Generator

from .tokens import Token, TokenType

# Reserved keywords that cannot be used as identifiers
KEYWORDS = {
    "load": TokenType.LOAD,
    "save": TokenType.SAVE,
}

# Multi-character operators (must check before single_char)
MULTI_CHAR_OPS = {
    "=>": TokenType.S2,
    "!=": TokenType.S4,
}

# Single-character operators
SINGLE_CHAR_OPS = {
    "=": TokenType.S1,
    ">": TokenType.S3_FORWARD,
    "<": TokenType.S3_BACKWARD,
}


class LexerError(Exception):
    """Lexer error with position information."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Line {line}, Column {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """
    Tokenizer for KScript source code with Python-style indentation.

    Uses indentation tracking to support multi-line KLines:
        MHALL = SVO =>
            S < M
            V < H
            O < ALL

    Indentation rules:
    - Consistent indentation within a block
    - INDENT token when indentation increases
    - DEDENT token(s) when indentation decreases
    - Blank lines and comment-only lines are ignored
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self._tokens: list[Token] | None = None
        # Indentation tracking
        self._indent_stack: list[int] = [0]  # Start at indentation level 0
        self._at_start_of_line = True
        self._pending_dedents = 0  # DEDENT tokens to emit
        self._in_block = False  # Track if we're in an indented block

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
        """Generate tokens from source with indentation tracking."""
        while not self._at_end():
            # Handle pending DEDENTs first
            if self._pending_dedents > 0:
                self._pending_dedents -= 1
                self._indent_stack.pop()
                yield Token(TokenType.DEDENT, "", self.line, 1)
                continue

            # At start of line: check indentation
            if self._at_start_of_line:
                self._at_start_of_line = False
                indent = self._count_indent()

                # Skip blank lines and comment-only lines
                if self._is_blank_or_comment_line():
                    self._skip_line()
                    self._at_start_of_line = True
                    continue

                # Compare with current indentation level
                current_indent = self._indent_stack[-1]

                if indent > current_indent:
                    # Increased indentation - emit INDENT
                    self._indent_stack.append(indent)
                    yield Token(TokenType.INDENT, " " * indent, self.line, 1)
                elif indent < current_indent:
                    # Decreased indentation - emit DEDENT(s)
                    while self._indent_stack and self._indent_stack[-1] > indent:
                        self._indent_stack.pop()
                        self._pending_dedents += 1

                    # Validate indentation matches a previous level
                    if not self._indent_stack or self._indent_stack[-1] != indent:
                        raise LexerError(
                            f"Unindent does not match any outer indentation level",
                            self.line,
                            1,
                        )

                    # Emit first DEDENT
                    self._pending_dedents -= 1
                    yield Token(TokenType.DEDENT, "", self.line, 1)
                    continue

                # Now skip the indent whitespace we counted
                for _ in range(indent):
                    self._advance()

            # Skip whitespace (but not newlines) and comments
            self._skip_whitespace_and_comments()
            if self._at_end():
                break

            # Check for newline (statement/line separator)
            if self._current() == "\n":
                start_line, start_col = self.line, self.column
                self._advance()
                # Skip any additional blank lines
                while self._current() == "\n":
                    # Check if next line is blank/comment
                    saved_pos = self.pos
                    saved_line = self.line
                    saved_col = self.column
                    self._advance()
                    indent = self._count_indent()
                    if self._is_blank_or_comment_line():
                        self._skip_line()
                    else:
                        # Restore position - this line has content
                        self.pos = saved_pos
                        self.line = saved_line
                        self.column = saved_col
                        break

                self._at_start_of_line = True
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

        # Emit remaining DEDENTs at EOF
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            yield Token(TokenType.DEDENT, "", self.line, 1)

        yield Token(TokenType.EOF, "", self.line, self.column)

    def _count_indent(self) -> int:
        """Count indentation at current position (spaces/tabs)."""
        indent = 0
        pos = self.pos
        while pos < len(self.source):
            c = self.source[pos]
            if c == " ":
                indent += 1
                pos += 1
            elif c == "\t":
                indent += 4  # Treat tab as 4 spaces
                pos += 1
            else:
                break
        return indent

    def _is_blank_or_comment_line(self) -> bool:
        """Check if current line is blank or comment-only."""
        pos = self.pos
        while pos < len(self.source):
            c = self.source[pos]
            if c in " \t":
                pos += 1
            elif c == "(":
                # Comment - skip to end of comment
                depth = 1
                pos += 1
                while pos < len(self.source) and depth > 0:
                    if self.source[pos] == "(":
                        depth += 1
                    elif self.source[pos] == ")":
                        depth -= 1
                    pos += 1
            elif c in "\n\r":
                # End of line - it was blank/comment-only
                return True
            else:
                # Found actual content
                return False
        return True  # EOF - consider it blank

    def _skip_line(self) -> None:
        """Skip to the end of the current line."""
        while not self._at_end() and self._current() != "\n":
            if self._current() == "(":
                self._skip_comment()
            else:
                self._advance()
        if self._current() == "\n":
            self._advance()
            self._at_start_of_line = True

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
