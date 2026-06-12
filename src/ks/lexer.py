"""Lexer for KScript v3 — converts source text into a flat list of Token objects.

Handles operator precedence (multi-char before single-char), SIGNATURE
identification, ANNOTATION tokens with nested parentheses, and Python-style
INDENT/DEDENT tracking.

See specs/kscript.md §2.1–2.4.
"""

from .token import Token, TokenType


class LexerError(Exception):
    """Error during lexing with position tracking.

    Attributes:
        message: Human-readable error description
        line: 1-based line number where the error occurred
        column: 1-based column number where the error occurred
    """

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Line {line}, column {column}: {message}")
        self.message = message
        self.line = line
        self.column = column


class Lexer:
    """Tokenizes KScript v3 source code with indentation tracking.

    The lexer produces a flat list of Token objects, handling:
    - Multi-character operators (==, =>) matched before single-char (=, >)
    - SIGNATURE tokens for [A-Z][A-Z0-9]* identifiers (digits → error)
    - ANNOTATION tokens for parenthesised content, with nesting and multi-line
    - Inline ANNOTATION tokens emitted after their preceding SIGNATURE
    - Python-style INDENT/DEDENT tokens based on leading whitespace
    """

    def __init__(self, source: str):
        """Initialize lexer with source string.

        Args:
            source: The KScript v3 source code to tokenize
        """
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack: list[int] = [0]
        self.pending_tokens: list[Token] = []
        self.at_line_start = True

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source and return list of tokens.

        Returns:
            List of Token objects ending with an EOF token.
        """
        # Empty or whitespace-only input produces only EOF
        if not self.source or not self.source.strip():
            return [Token(TokenType.EOF, "", 1, 1)]

        tokens: list[Token] = []

        while self.pos < len(self.source) or self.pending_tokens:
            # Emit any pending tokens (INDENT/DEDENT/inline ANNOTATION)
            if self.pending_tokens:
                tokens.append(self.pending_tokens.pop(0))
                continue

            token = self._next_token()
            if token is not None:
                tokens.append(token)

        # Close all remaining indent levels with DEDENT tokens
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(Token(TokenType.DEDENT, "", self.line, self.column))

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens

    def _next_token(self) -> Token | None:
        """Get the next token, handling indentation at line start."""
        # Handle indentation at line start
        if self.at_line_start:
            self.at_line_start = False
            indent, saved_line, saved_col = self._count_indent()
            indent_token = self._handle_indent(indent, saved_line, saved_col)
            if indent_token is not None:
                return indent_token
            # Same indent level — fall through to process content

        # Skip whitespace (not newlines)
        while self.pos < len(self.source) and self.source[self.pos] in " \t":
            self._advance()

        if self.pos >= len(self.source):
            return None

        ch = self.source[self.pos]

        # Newline
        if ch == "\n":
            return self._read_newline()

        # Multi-char operators (matched before single-char)
        if self.pos + 1 < len(self.source):
            two_char = self.source[self.pos : self.pos + 2]
            if two_char == "==":
                return self._make_token(TokenType.COUNTERSIGN, "==")
            if two_char == "=>":
                return self._make_token(TokenType.CANONIZE, "=>")

        # Single-char operators
        if ch == "=":
            return self._make_token(TokenType.UNDERSIGN, "=")
        if ch == ">":
            return self._make_token(TokenType.CONNOTATE, ">")

        # '<' is explicitly invalid
        if ch == "<":
            raise LexerError(
                f"Unexpected character: {ch!r}", self.line, self.column
            )

        # Identifier: must start with uppercase letter [A-Z]
        if ch.isupper():
            return self._read_identifier()

        # Lowercase letter — not a valid identifier start
        if ch.islower():
            raise LexerError(
                f"Unexpected character: {ch!r}", self.line, self.column
            )

        # Standalone annotation: (...) with nested parens
        if ch == "(":
            return self._read_annotation()

        # Any other unknown character
        raise LexerError(
            f"Unexpected character: {ch!r}", self.line, self.column
        )

    # ── Indentation handling ──────────────────────────────────────────

    def _count_indent(self) -> tuple[int, int, int]:
        """Count leading whitespace at line start (spaces and tabs).

        Returns:
            Tuple of (indent_count, saved_line, saved_column) where
            saved positions are from the start of the line before advancing.
        """
        indent = 0
        saved_line, saved_column = self.line, self.column
        while self.pos < len(self.source) and self.source[self.pos] in " \t":
            indent += 1
            self._advance()
        return indent, saved_line, saved_column

    def _handle_indent(
        self, indent: int, line: int, column: int
    ) -> Token | None:
        """Handle indentation changes, returning INDENT/DEDENT or None."""
        current = self.indent_stack[-1]

        if indent > current:
            self.indent_stack.append(indent)
            return Token(TokenType.INDENT, "", line, column)

        if indent < current:
            # Emit one or more DEDENT tokens to close levels
            while (
                len(self.indent_stack) > 1 and self.indent_stack[-1] > indent
            ):
                self.indent_stack.pop()
                self.pending_tokens.append(
                    Token(TokenType.DEDENT, "", line, column)
                )
            # Verify the dedent lands on a valid level
            if self.indent_stack[-1] != indent:
                raise LexerError(
                    f"Inconsistent indentation: level {indent} "
                    f"does not match any previous level",
                    line,
                    column,
                )
            return self.pending_tokens.pop(0) if self.pending_tokens else None

        # Same indentation — no structural token
        return None

    # ── Token readers ─────────────────────────────────────────────────

    def _read_newline(self) -> Token:
        """Read a newline token and enter line-start mode."""
        line, col = self.line, self.column
        self._advance()
        self.line += 1
        self.column = 1
        self.at_line_start = True
        return Token(TokenType.NEWLINE, "\n", line, col)

    def _read_identifier(self) -> Token:
        """Read an identifier [A-Z][A-Z0-9]* and classify it.

        - All uppercase alpha → SIGNATURE
        - Contains digits → LexerError
        """
        start_line, start_col = self.line, self.column
        name = ""

        while (
            self.pos < len(self.source)
            and self.source[self.pos].isupper()
        ):
            name += self._advance()

        # Check for digit continuation (allowed by pattern but invalid)
        while (
            self.pos < len(self.source)
            and self.source[self.pos].isdigit()
        ):
            name += self._advance()

        if not name.isalpha():
            raise LexerError(
                f"Invalid identifier '{name}': "
                f"identifiers must be all uppercase letters (no digits)",
                start_line,
                start_col,
            )

        # Inline annotation: if '(' immediately follows, queue ANNOTATION
        if self.pos < len(self.source) and self.source[self.pos] == "(":
            self.pending_tokens.append(self._read_annotation())

        return Token(TokenType.SIGNATURE, name, start_line, start_col)

    def _read_annotation(self) -> Token:
        """Read an annotation (...) with nested parentheses and multi-line.

        The value includes the outer parentheses.
        """
        start_line, start_col = self.line, self.column
        value = self._advance()  # consume opening '('
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            ch = self.source[self.pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1

            if ch == "\n":
                self._advance()
                self.line += 1
                self.column = 1
                value += "\n"
                continue

            value += self._advance()

        if depth > 0:
            raise LexerError(
                "Unterminated annotation: missing closing ')'",
                start_line,
                start_col,
            )

        return Token(TokenType.ANNOTATION, value, start_line, start_col)

    # ── Helpers ────────────────────────────────────────────────────────

    def _make_token(self, token_type: TokenType, value: str) -> Token:
        """Create a token and advance past its value."""
        line, col = self.line, self.column
        for _ in value:
            self._advance()
        return Token(token_type, value, line, col)

    def _advance(self) -> str:
        """Advance position by one character and return it."""
        ch = self.source[self.pos]
        self.pos += 1
        self.column += 1
        return ch
