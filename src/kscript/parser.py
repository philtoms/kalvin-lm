"""Recursive descent parser for KScript language."""

from .ast import (
    Construct,
    ConstructType,
    KScriptFile,
    Node,
    NumberLiteral,
    Script,
    Signature,
    StringLiteral,
)
from .token import Token, TokenType


class ParseError(Exception):
    """Error during parsing."""

    def __init__(self, message: str, token: Token):
        super().__init__(f"Line {token.line}, column {token.column}: {message}")
        self.token = token


class Parser:
    """Recursive descent parser for KScript.

    Handles:
    - Multiple top-level scripts (column 1 signatures)
    - Construct parsing with operator detection
    - Immediate binding for chained constructs
    - Backward canonize with leading nodes detection
    - Recursive subscript parsing with INDENT/DEDENT
    """

    def __init__(self, tokens: list[Token]):
        """Initialize parser with token list.

        Args:
            tokens: List of tokens from lexer (ending with EOF)
        """
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> KScriptFile:
        """Parse tokens into a KScriptFile AST.

        Returns:
            KScriptFile containing all top-level scripts
        """
        scripts: list[Script] = []

        while not self._at_end():
            # Skip newlines and comments at top level
            if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                self._advance()
                continue

            # Parse top-level script (starts at column 1)
            if self._check(TokenType.SIGNATURE):
                script = self._parse_script(is_top_level=True)
                scripts.append(script)
            else:
                # Skip unexpected tokens at top level
                self._advance()

        return KScriptFile(scripts)

    def _parse_script(self, is_top_level: bool = False) -> Script:
        """Parse a script: signature followed by constructs and optional subscripts.

        Args:
            is_top_level: True if this is a top-level script (column 1)

        Returns:
            Script AST node
        """
        sig_token = self._expect(TokenType.SIGNATURE)
        signature = self._make_signature(sig_token)

        # Parse constructs on the same line
        constructs: list[Construct] = []

        while not self._at_end() and not self._check(TokenType.NEWLINE):
            # Check for backward canonize pattern: nodes before <=
            leading_nodes = self._try_collect_leading_nodes()
            if leading_nodes and self._check(TokenType.CANONIZE_BWD):
                # Backward canonize with leading nodes: B C D <= A
                construct = self._parse_backward_canonize_with_leading(leading_nodes)
                if construct:
                    constructs.append(construct)
                continue

            construct = self._try_parse_construct()
            if construct:
                constructs.append(construct)
            else:
                break

        # Consume newline if present
        if self._check(TokenType.NEWLINE):
            self._advance()

        # Parse subscripts (indented scripts)
        subscripts: list[Script] = []
        while self._check(TokenType.INDENT):
            self._advance()  # consume INDENT
            # Parse subscript scripts
            while (
                not self._at_end()
                and not self._check(TokenType.DEDENT)
                and not self._check(TokenType.EOF)
            ):
                # Skip newlines and comments within subscript block
                if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                    self._advance()
                    continue

                if self._check(TokenType.SIGNATURE):
                    subscript = self._parse_script(is_top_level=False)
                    subscripts.append(subscript)
                else:
                    # Unexpected token - skip
                    self._advance()

            # Consume DEDENT if present
            if self._check(TokenType.DEDENT):
                self._advance()

        return Script(signature, constructs, subscripts, signature.line)

    def _try_collect_leading_nodes(self) -> list[Node]:
        """Collect nodes (signatures) that appear before a backward canonize operator.

        Look ahead to see if there are signatures followed by <= without consuming them.
        Returns empty list if no leading nodes found or next token is not <=.
        """
        # Save position for lookahead
        saved_pos = self.pos
        nodes: list[Node] = []

        # Collect signatures until we hit an operator or end
        while self._check(TokenType.SIGNATURE):
            token = self._advance()
            nodes.append(self._make_signature(token))

        # Check if next token is <= (backward canonize)
        if nodes and self._check(TokenType.CANONIZE_BWD):
            # Found backward canonize pattern - keep the nodes and position
            return nodes

        # Not a backward canonize pattern - restore position and return empty
        self.pos = saved_pos
        return []

    def _parse_backward_canonize_with_leading(self, leading_nodes: list[Node]) -> Construct | None:
        """Parse backward canonize with leading nodes already collected."""
        if not self._check(TokenType.CANONIZE_BWD):
            return None

        line = self._peek().line
        self._advance()  # consume <=

        # Parse trailing nodes
        trailing_nodes = self._parse_nodes(multi=True)

        # Combine: leading nodes + trailing nodes
        all_nodes = leading_nodes + trailing_nodes

        return Construct(ConstructType.CANONIZE_BWD, all_nodes, line, has_leading_nodes=True)

    def _try_parse_construct(self) -> Construct | None:
        """Try to parse a construct. Returns None if no operator found."""
        op_type = self._try_parse_operator()
        if not op_type:
            return None

        line = self._peek().line

        # Parse nodes
        nodes: list[Node] = []

        # Canonize constructs can have multiple nodes
        if op_type in (ConstructType.CANONIZE_FWD, ConstructType.CANONIZE_BWD):
            nodes = self._parse_nodes(multi=True)
        else:
            nodes = self._parse_nodes(multi=False)

        # Create construct even with no nodes - compiler will use subscript signatures
        return Construct(op_type, nodes, line)

    def _try_parse_operator(self) -> ConstructType | None:
        """Try to parse a construct operator. Returns None if not found."""
        if self._check(TokenType.COUNTERSIGN):
            self._advance()
            return ConstructType.COUNTERSIGN

        if self._check(TokenType.CANONIZE_FWD):
            self._advance()
            return ConstructType.CANONIZE_FWD

        if self._check(TokenType.CANONIZE_BWD):
            self._advance()
            return ConstructType.CANONIZE_BWD

        if self._check(TokenType.CONNOTATE_FWD):
            self._advance()
            return ConstructType.CONNOTATE_FWD

        if self._check(TokenType.CONNOTATE_BWD):
            self._advance()
            return ConstructType.CONNOTATE_BWD

        if self._check(TokenType.UNDERSIGN):
            self._advance()
            return ConstructType.UNDERSIGN

        return None

    def _parse_nodes(self, multi: bool) -> list[Node]:
        """Parse one or more nodes.

        Args:
            multi: If True, parse multiple nodes; if False, parse single node
        """
        nodes: list[Node] = []

        while not self._at_end():
            node = self._try_parse_node()
            if node:
                nodes.append(node)
                if not multi:
                    break  # Single node only
            else:
                break

        return nodes

    def _try_parse_node(self) -> Node | None:
        """Try to parse a single node (signature, string, or number)."""
        if self._check(TokenType.SIGNATURE):
            token = self._advance()
            return self._make_signature(token)

        if self._check(TokenType.STRING):
            token = self._advance()
            return StringLiteral(token.value, token.line, token.column)

        if self._check(TokenType.NUMBER):
            token = self._advance()
            return NumberLiteral(token.value, token.line, token.column)

        return None

    def _make_signature(self, token: Token) -> Signature:
        """Create a Signature from a token."""
        return Signature(token.value, None, token.line, token.column)

    def _peek(self) -> Token:
        """Return current token without advancing."""
        return self.tokens[self.pos]

    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._at_end():
            return False
        return self._peek().type == token_type

    def _advance(self) -> Token:
        """Advance and return current token."""
        if not self._at_end():
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return self.tokens[-1]  # EOF

    def _expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type, raise error if not found."""
        if self._check(token_type):
            return self._advance()
        raise ParseError(f"Expected {token_type.name}", self._peek())

    def _at_end(self) -> bool:
        """Check if at end of tokens."""
        return self._peek().type == TokenType.EOF
