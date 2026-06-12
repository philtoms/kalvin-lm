"""KScript v3 lexer — tokenizes source code into a stream of Token objects.

Handles:
  - Multi-character operators (==, =>) before single-char (=, >)
  - Signatures [A-Z]+ with optional inline annotation
  - Annotations (...) with nested paren handling
  - Python-style INDENT/DEDENT tokens
  - Unknown characters raise LexerError

Key difference from v2: COMMENT tokens are now ANNOTATION tokens,
reflecting their semantic purpose in BPE encoding.

Spec ref: @specs/kscript.md §2 (Lexical Analysis)
"""

from __future__ import annotations

from .token import Token, TokenType


class LexerError(Exception):
    """Error during lexing.

    Attributes:
        line: 1-based line number where the error occurred.
        column: 1-based column number.
    """

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"Line {line}, column {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """Tokenizes KScript source code with indentation tracking.

    The lexer produces a flat list of Token objects from a source string.
    Only SIGNATURE tokens ([A-Z]+) can be construct owners in the grammar.

    Usage::

        tokens = Lexer("A == B").tokenize()
        # [SIGNATURE("A"), COUNTERSIGN("=="), SIGNATURE("B"), EOF("")]
    """

    def __init__(self, source: str) -> None:
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
            List of Token objects ending with EOF.
        """
        tokens: list[Token] = []

        while self.pos < len(self.source) or self.pending_tokens:
            if self.pending_tokens:
                tokens.append(self.pending_tokens.pop(0))
                continue

            token = self._next_token()
            if token:
                tokens.append(token)

        # Emit remaining DEDENTs at EOF
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(Token(TokenType.DEDENT, "", self.line, self.column))

        tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return tokens

    def _next_token(self) -> Token | None:
        """Get the next token, handling indentation at line start."""
        if self.at_line_start:
            self.at_line_start = False
            indent = self._count_indent()

            # Blank lines (only whitespace, no content) must not affect
            # indent state — skip them entirely.  Spec ref: KS-5.
            if self.pos >= len(self.source) or self.source[self.pos] == "\n":
                if self.pos < len(self.source) and self.source[self.pos] == "\n":
                    self.pos += 1
                    self.line += 1
                    self.column = 1
                self.at_line_start = True
                return None

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
                return self._make_token(TokenType.CANONIZE, "=>")

        # Single-char operators
        if ch == "=":
            return self._make_token(TokenType.UNDERSIGN, "=")
        if ch == ">":
            return self._make_token(TokenType.CONNOTATE, ">")
        if ch == "<":
            raise LexerError(f"Unexpected character: {ch!r}", self.line, self.column)

        # Identifier
        if ch.isalpha():
            return self._read_identifier()

        # Annotation (...)
        if ch == "(":
            return self._read_annotation()

        # Unknown character
        raise LexerError(f"Unexpected character: {ch!r}", self.line, self.column)

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
            self.indent_stack.append(indent)
            return Token(TokenType.INDENT, "", self.line, self.column)

        if indent < current_indent:
            while len(self.indent_stack) > 1 and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                self.pending_tokens.append(
                    Token(TokenType.DEDENT, "", self.line, self.column)
                )
            return self.pending_tokens.pop(0) if self.pending_tokens else None

        return None  # Same indentation — no token needed

    def _read_newline(self) -> Token:
        """Read a newline token."""
        line, col = self.line, self.column
        self._advance()
        self.line += 1
        self.column = 1
        self.at_line_start = True
        return Token(TokenType.NEWLINE, "\n", line, col)

    def _read_identifier(self) -> Token:
        """Read an identifier [a-zA-Z][a-zA-Z0-9]*.

        Returns SIGNATURE if all uppercase alpha.
        Raises LexerError for mixed/lowercase identifiers.

        When '(' immediately follows the identifier (e.g., S(ubject)),
        reads the annotation and queues it as a pending ANNOTATION token.
        """
        start_line, start_col = self.line, self.column
        name = ""

        while self.pos < len(self.source) and self._is_ident_char(self.source[self.pos]):
            name += self._advance()

        # Check for inline annotation — queue as pending token
        if self.pos < len(self.source) and self.source[self.pos] == "(":
            self.pending_tokens.append(self._read_annotation())

        # All uppercase alpha → SIGNATURE
        if name.isupper() and name.isalpha():
            return Token(TokenType.SIGNATURE, name, start_line, start_col)

        # Non-uppercase identifiers are not valid
        raise LexerError(
            f"Invalid identifier '{name}': identifiers must be all uppercase "
            f"(signatures). Use a quoted string for non-signature values.",
            start_line,
            start_col,
        )

    def _is_ident_char(self, ch: str) -> bool:
        """Check if character is valid in an identifier (alphanumeric)."""
        return ch.isalnum()

    def _read_annotation(self) -> Token:
        """Read an annotation (...) — multi-line, handles nested parens.

        Returns an ANNOTATION token with the full text including parens.
        """
        start_line, start_col = self.line, self.column
        value = self._advance()  # opening (
        depth = 1

        while self.pos < len(self.source) and depth > 0:
            ch = self.source[self.pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "\n":
                self._advance()
                self.line += 1
                self.column = 1
                value += "\n"
                continue
            value += self._advance()

        return Token(TokenType.ANNOTATION, value, start_line, start_col)

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
