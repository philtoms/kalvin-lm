"""Lexer for KScript language with indentation support."""

from .token import Token, TokenType


class LexerError(Exception):
    """Error during lexing."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"Line {line}, column {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """Tokenizes KScript source code with indentation tracking.

    The lexer handles:
    - Multi-character operators (==, =>, <=) before single-char
    - Signatures [A-Z]+ with optional inline comment
    - String literals "..." with escape support
    - Number literals [0-9]+
    - Comments (...) with nested paren handling
    - Python-style INDENT/DEDENT tokens
    """

    def __init__(self, source: str):
        """Initialize lexer with source string.

        Args:
            source: The KScript source code to tokenize
        """
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack: list[int] = [0]  # Track indentation levels
        self.pending_tokens: list[Token] = []  # INDENT/DEDENT to emit
        self.at_line_start = True

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source and return list of tokens.

        Returns:
            List of Token objects ending with EOF token
        """
        tokens: list[Token] = []

        while self.pos < len(self.source) or self.pending_tokens:
            # Emit any pending INDENT/DEDENT tokens first
            if self.pending_tokens:
                tokens.append(self.pending_tokens.pop(0))
                continue

            token = self._next_token()
            if token:
                tokens.append(token)

        # Emit remaining DEDENTs at EOF
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(
                Token(TokenType.DEDENT, "", self.line, self.column)
            )

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens

    def _next_token(self) -> Token | None:
        """Get the next token, handling indentation at line start."""
        # Handle indentation at line start
        if self.at_line_start:
            self.at_line_start = False
            indent = self._count_indent()
            return self._handle_indent(indent)

        # Skip whitespace (not newlines)
        while self.pos < len(self.source) and self.source[self.pos] in " \t":
            self._advance()

        if self.pos >= len(self.source):
            return None

        ch = self.source[self.pos]

        # Newline
        if ch == "\n":
            return self._read_newline()

        # Multi-char operators (check before single-char)
        if self.pos + 1 < len(self.source):
            two_char = self.source[self.pos : self.pos + 2]
            if two_char == "==":
                return self._make_token(TokenType.COUNTERSIGN, "==")
            if two_char == "=>":
                return self._make_token(TokenType.CANONIZE_FWD, "=>")
            if two_char == "<=":
                return self._make_token(TokenType.CANONIZE_BWD, "<=")

        # Single-char operators
        if ch == "=":
            return self._make_token(TokenType.UNDERSIGN, "=")
        if ch == ">":
            return self._make_token(TokenType.CONNOTATE_FWD, ">")
        if ch == "<":
            return self._make_token(TokenType.CONNOTATE_BWD, "<")

        # Signature [A-Z]+
        if ch.isupper():
            return self._read_signature()

        # Number [0-9]+
        if ch.isdigit():
            return self._read_number()

        # String "..."
        if ch == '"':
            return self._read_string()

        # Comment (...)
        if ch == "(":
            return self._read_comment()

        # Unknown character - skip it
        self._advance()
        return None

    def _count_indent(self) -> int:
        """Count indentation at line start (spaces and tabs)."""
        indent = 0
        while self.pos < len(self.source) and self.source[self.pos] in " \t":
            indent += 1
            self._advance()
        return indent

    def _handle_indent(self, indent: int) -> Token | None:
        """Handle indentation changes, emitting INDENT/DEDENT tokens."""
        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            # Increased indentation
            self.indent_stack.append(indent)
            return Token(TokenType.INDENT, "", self.line, self.column)

        if indent < current_indent:
            # Decreased indentation - may need multiple DEDENTs
            while (
                len(self.indent_stack) > 1
                and self.indent_stack[-1] > indent
            ):
                self.indent_stack.pop()
                self.pending_tokens.append(
                    Token(TokenType.DEDENT, "", self.line, self.column)
                )
            return self.pending_tokens.pop(0) if self.pending_tokens else None

        # Same indentation - no token needed
        return None

    def _read_newline(self) -> Token:
        """Read a newline token."""
        line, col = self.line, self.column
        self._advance()
        self.line += 1
        self.column = 1
        self.at_line_start = True
        return Token(TokenType.NEWLINE, "\n", line, col)

    def _read_signature(self) -> Token:
        """Read a signature [A-Z]+ with optional inline comment."""
        start_line, start_col = self.line, self.column
        name = ""

        while self.pos < len(self.source) and self.source[self.pos].isupper():
            name += self._advance()

        # Check for inline comment - consume but don't attach
        if self.pos < len(self.source) and self.source[self.pos] == "(":
            self._read_comment()  # Consumes and discards

        return Token(
            TokenType.SIGNATURE,
            name,
            start_line,
            start_col,
        )

    def _read_number(self) -> Token:
        """Read a number [0-9]+."""
        start_line, start_col = self.line, self.column
        value = ""

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            value += self._advance()

        return Token(TokenType.NUMBER, value, start_line, start_col)

    def _read_string(self) -> Token:
        """Read a string literal "..." with escape support."""
        start_line, start_col = self.line, self.column
        value = self._advance()  # opening "

        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == "\\" and self.pos + 1 < len(self.source):
                # Escape sequence
                value += self._advance()
                value += self._advance()
            elif ch == '"':
                value += self._advance()  # closing "
                break
            elif ch == "\n":
                # Unterminated string - stop at newline
                break
            else:
                value += self._advance()

        return Token(TokenType.STRING, value, start_line, start_col)

    def _read_comment(self) -> Token:
        """Read a comment (...) - greedy match handling nested parens."""
        start_line, start_col = self.line, self.column
        value = self._advance()  # opening (
        depth = 1  # Track nesting depth

        while self.pos < len(self.source) and depth > 0:
            ch = self.source[self.pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "\n":
                # Unterminated comment - stop at newline
                break
            value += self._advance()

        return Token(TokenType.COMMENT, value, start_line, start_col)

    def _make_token(self, token_type: TokenType, value: str) -> Token:
        """Create a token and advance past its value."""
        line, col = self.line, self.column
        for _ in value:
            self._advance()
        return Token(token_type, value, line, col)

    def _advance(self) -> str:
        """Advance position and return the character."""
        ch = self.source[self.pos]
        self.pos += 1
        self.column += 1
        return ch
