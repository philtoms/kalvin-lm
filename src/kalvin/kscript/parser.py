"""Recursive descent parser for KScript."""

from .ast import (
    Identifier,
    KLineExpr,
    KNodeRef,
    KScript,
    KScriptStatement,
    LoadStatement,
    SaveStatement,
    SignificanceType,
)
from .lexer import Lexer
from .tokens import Token, TokenType


class ParseError(Exception):
    """Parser error with position information."""

    def __init__(self, message: str, token: Token):
        super().__init__(f"Line {token.line}, Column {token.column}: {message}")
        self.token = token


class Parser:
    """
    Recursive descent parser for KScript.

    Grammar (in pseudo-BNF):
        script      ::= statement*
        statement   ::= load_stmt | save_stmt | kline_expr
        load_stmt   ::= "load" IDENTIFIER
        save_stmt   ::= "save" [IDENTIFIER]
        kline_expr  ::= ksig [significance nodes] [attention]
        ksig        ::= IDENTIFIER | kline_expr
        nodes       ::= IDENTIFIER+
        significance ::= "=" | "=>" | ">" | "<" | "!="
        attention   ::= "?"
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def from_source(cls, source: str) -> "Parser":
        """Create parser from source string."""
        lexer = Lexer(source)
        return cls(lexer.tokenize())

    def parse(self) -> KScript:
        """Parse the token stream into a KScript AST."""
        statements: list[KScriptStatement] = []

        # Skip leading newlines
        while self._match(TokenType.NEWLINE):
            pass

        while not self._check(TokenType.EOF):
            if stmt := self._parse_statement():
                statements.append(stmt)

            # Expect newline or EOF after each statement
            if not self._check(TokenType.EOF):
                # Skip newlines (multiple blank lines are OK)
                while self._match(TokenType.NEWLINE):
                    pass

        return KScript(statements=statements)

    def _current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos]

    def _check(self, *types: TokenType) -> bool:
        """Check if current token is one of the given types."""
        return self._current().type in types

    def _advance(self) -> Token:
        """Advance and return the current token."""
        token = self._current()
        if not self._check(TokenType.EOF):
            self.pos += 1
        return token

    def _match(self, *types: TokenType) -> Token | None:
        """If current token matches, consume and return it."""
        if self._check(*types):
            return self._advance()
        return None

    def _expect(self, token_type: TokenType, message: str) -> Token:
        """Expect a specific token type or raise error."""
        if self._check(token_type):
            return self._advance()
        raise ParseError(message, self._current())

    def _parse_statement(self) -> KScriptStatement | None:
        """Parse a single statement."""
        if self._check(TokenType.LOAD):
            return self._parse_load()
        elif self._check(TokenType.SAVE):
            return self._parse_save()
        else:
            return self._parse_kline_expr()

    def _parse_load(self) -> LoadStatement:
        """Parse: load <path>"""
        self._expect(TokenType.LOAD, "Expected 'load'")
        path_token = self._expect(TokenType.IDENTIFIER, "Expected file path after 'load'")
        return LoadStatement(path=Identifier(path_token.value, path_token.line, path_token.column))

    def _parse_save(self) -> SaveStatement:
        """Parse: save [path]"""
        self._expect(TokenType.SAVE, "Expected 'save'")

        # Path is optional
        if path_token := self._match(TokenType.IDENTIFIER):
            return SaveStatement(
                path=Identifier(path_token.value, path_token.line, path_token.column)
            )

        return SaveStatement()

    def _parse_kline_expr(self) -> KLineExpr:
        """
        Parse a KLine expression.

        Forms:
        - identifier
        - identifier significance nodes
        - identifier attention
        - identifier significance nodes attention
        """
        start_token = self._current()
        sig = self._parse_ksig()

        significance: SignificanceType | None = None
        nodes: list[KNodeRef] = []
        attention = False

        # Check for significance operator
        if token := self._match(TokenType.S1):
            significance = SignificanceType.S1
        elif token := self._match(TokenType.S2):
            significance = SignificanceType.S2
        elif token := self._match(TokenType.S3_FORWARD):
            significance = SignificanceType.S3_FORWARD
        elif token := self._match(TokenType.S3_BACKWARD):
            significance = SignificanceType.S3_BACKWARD
        elif token := self._match(TokenType.S4):
            significance = SignificanceType.S4

        # Parse nodes if significance was found
        if significance is not None:
            nodes = self._parse_nodes()

        # Check for attention marker
        if self._match(TokenType.ATTENTION):
            attention = True

        return KLineExpr(
            sig=sig,
            significance=significance,
            nodes=nodes,
            attention=attention,
            line=start_token.line,
            column=start_token.column,
        )

    def _parse_ksig(self) -> Identifier:
        """
        Parse a KSig (identifier for now).

        For simplicity in the first pass, we only allow identifiers.
        Nested KLineExpr support can be added later.
        """
        token = self._expect(TokenType.IDENTIFIER, "Expected identifier for KLine")
        return Identifier(token.value, token.line, token.column)

    def _parse_nodes(self) -> list[KNodeRef]:
        """Parse one or more KNode references (stops at NEWLINE)."""
        nodes: list[KNodeRef] = []

        # At least one node required after significance
        first = self._expect(
            TokenType.IDENTIFIER, "Expected at least one node after significance operator"
        )
        nodes.append(KNodeRef(Identifier(first.value, first.line, first.column)))

        # Collect additional nodes (stop at NEWLINE or other non-identifier)
        while not self._check(TokenType.NEWLINE, TokenType.EOF):
            if token := self._match(TokenType.IDENTIFIER):
                nodes.append(KNodeRef(Identifier(token.value, token.line, token.column)))
            else:
                break

        return nodes
