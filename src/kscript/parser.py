"""Recursive descent parser for KScript with multi-line support."""

from .ast import (
    Identifier,
    KLineExpr,
    KLineRelationship,
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
    Recursive descent parser for KScript with indentation support.

    Grammar:
        script      ::= statement*
        statement   ::= load_stmt | save_stmt | kline_expr
        load_stmt   ::= "load" IDENTIFIER
        save_stmt   ::= "save" [IDENTIFIER]
        kline_expr  ::= KSig kline_tail*
        kline_tail  ::= significance nodes
        nodes       ::= inline_nodes | indented_klines
        inline_nodes ::= IDENTIFIER+
        indented_klines ::= INDENT kline_expr+ DEDENT
        significance ::= "=" | "=>" | ">" | "<" | "!="

    Multi-line example:
        MHALL = SVO =>
            S < M
            V < H
            O < ALL

    This parses as:
        MHALL
        = SVO          (S1 relationship to SVO)
        => [S<M, V<H, O<ALL]  (S2 relationship to indented KLines)
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

        # Skip leading newlines and indents
        while self._match(TokenType.NEWLINE, TokenType.INDENT):
            pass

        while not self._check(TokenType.EOF, TokenType.DEDENT):
            if stmt := self._parse_statement():
                statements.append(stmt)

            # Skip newlines and indents between statements
            while self._match(TokenType.NEWLINE, TokenType.INDENT):
                pass

        return KScript(statements=statements)

    def _current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

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
        Parse a KLine expression with optional chained significance relationships.

        Forms:
        - identifier
        - identifier significance nodes
        - identifier significance nodes significance nodes ...
        """
        start_token = self._current()
        sig = self._parse_ksig()

        # Collect all chained relationships
        # e.g., MHALL = SVO => [indented] means:
        #   MHALL with S1->SVO and S2->[indented klines]
        chained_relationships: list[tuple[SignificanceType, list[KNodeRef]]] = []

        while self._check(
            TokenType.S1, TokenType.S2, TokenType.S3_FORWARD, TokenType.S3_BACKWARD, TokenType.S4
        ):
            # Parse significance operator
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
            else:
                break

            # Parse nodes (inline or indented)
            nodes = self._parse_nodes()
            chained_relationships.append((significance, nodes))

        # Build relationships list from chained relationships
        relationships = [
            KLineRelationship(significance=sig, nodes=nodes) for sig, nodes in chained_relationships
        ]

        return KLineExpr(
            sig=sig,
            relationships=relationships,
            line=start_token.line,
            column=start_token.column,
        )

    def _parse_ksig(self) -> Identifier:
        """Parse a KSig (identifier)."""
        token = self._expect(TokenType.IDENTIFIER, "Expected identifier for KLine")
        return Identifier(token.value, token.line, token.column)

    def _parse_nodes(self) -> list[KNodeRef]:
        """
        Parse nodes after a significance operator.

        Nodes can be:
        1. Inline identifiers on the same line: = hello world
        2. Indented KLine expressions on following lines:
           =>
               S < M
               V < H

        For indented KLines, we create KNodeRefs from the KLine's sig identifier.
        """
        # Skip any trailing whitespace/newlines before checking for indented block
        # This handles the case: "MHALL = SVO =>\n    S < M"
        self._match(TokenType.NEWLINE)

        # Check for indented block
        if self._check(TokenType.INDENT):
            return self._parse_indented_klines_as_nodes()

        # Otherwise, parse inline identifiers
        return self._parse_inline_nodes()

    def _parse_inline_nodes(self) -> list[KNodeRef]:
        """Parse inline identifiers as nodes."""
        nodes: list[KNodeRef] = []

        # At least one node required after significance
        first = self._expect(
            TokenType.IDENTIFIER, "Expected at least one node after significance operator"
        )
        nodes.append(KNodeRef(Identifier(first.value, first.line, first.column)))

        # Collect additional inline nodes
        while self._check(TokenType.IDENTIFIER):
            token = self._advance()
            nodes.append(KNodeRef(Identifier(token.value, token.line, token.column)))

        return nodes

    def _parse_indented_klines_as_nodes(self) -> list[KNodeRef]:
        """
        Parse an indented block of KLine expressions as nodes.

        Each KLine in the block becomes a node reference with the full KLine
        expression stored for compilation.

        Example:
            =>
                S < M      -> node "S" (KLine with S3_BACKWARD to M)
                V < H      -> node "V" (KLine with S3_BACKWARD to H)

        The nested KLines will be compiled and their sig identifiers used as
        node references.
        """
        nodes: list[KNodeRef] = []

        # Consume INDENT
        indent_token = self._expect(TokenType.INDENT, "Expected INDENT")

        # Skip any newlines after INDENT
        while self._match(TokenType.NEWLINE):
            pass

        # Parse KLines until DEDENT
        while not self._check(TokenType.DEDENT, TokenType.EOF):
            kline = self._parse_kline_expr()
            # Create node reference from the KLine's sig identifier
            # and store the full KLine for compilation
            if isinstance(kline.sig, Identifier):
                nodes.append(KNodeRef(kline.sig, nested_kline=kline))
            # If sig is a KLineExpr, we need to handle that too
            else:
                # For nested KLineExpr as sig, just use a placeholder
                # This is a more complex case
                nodes.append(KNodeRef(Identifier("", kline.line, kline.column), nested_kline=kline))

            # Skip newlines between indented KLines
            while self._match(TokenType.NEWLINE):
                pass

        # Consume DEDENT
        self._expect(TokenType.DEDENT, "Expected DEDENT to close indented block")

        return nodes
