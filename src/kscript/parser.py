"""Recursive descent parser for KScript.

Grammar (left recursion eliminated):

    script ::= construct+
    construct ::= block | primary_construct+ ( ( "=>" | "<=" | "<" ) construct )?
    block ::= <INDENT> construct+ <DEDENT>
    primary_construct ::= sig ( ( "==" | ">" | "=" ) node )?
    node ::= sig | literal
    sig ::= [A-Z]+
    literal ::= ![A-Z]+

NEWLINE and COMMENT tokens are treated as insignificant whitespace
and skipped between constructs and at construct boundaries.
"""

from dataclasses import dataclass
from typing import TypeAlias

from .token import Token, TokenType


# =============================================================================
# AST Nodes
# =============================================================================

@dataclass
class Signature:
    """sig ::= [A-Z]+"""
    id: str
    line: int
    column: int


@dataclass
class Literal:
    """literal ::= ![A-Z]+"""
    id: str
    line: int
    column: int


Node: TypeAlias = Signature | Literal


@dataclass
class PrimaryConstruct:
    """primary_construct ::= sig ( ( "==" | ">" | "=" ) node )?

    If op is None, this is an identity (bare signature).
    """
    sig: Signature
    op: TokenType | None = None  # COUNTERSIGN, CONNOTATE_FWD, UNDERSIGN, or None
    node: Node | None = None


@dataclass
class Block:
    """block ::= <INDENT> construct+ <DEDENT>"""
    constructs: list["Construct"]


@dataclass
class Construct:
    """construct ::= block | primary_construct+ ( ( "=>" | "<=" | "<" ) construct )?

    inner:      The block or list of primary_constructs (first alternative matched).
    chain_op:   CANONIZE_FWD, CANONIZE_BWD, CONNOTATE_BWD, or None if no chain.
    chain_right: The right-hand construct of the chain, if any.
    """
    inner: Block | list[PrimaryConstruct]
    chain_op: TokenType | None = None
    chain_right: "Construct | None" = None


@dataclass
class Script:
    """script ::= construct+"""
    constructs: list[Construct]


@dataclass
class KScriptFile:
    """Top-level file container (one script per file)."""
    scripts: list[Script]


# =============================================================================
# Parser
# =============================================================================

class ParseError(Exception):
    """Error during parsing."""

    def __init__(self, message: str, token: Token):
        super().__init__(f"Line {token.line}, column {token.column}: {message}")
        self.token = token


class Parser:
    """Recursive descent parser matching the grammar exactly."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> KScriptFile:
        """Parse all tokens into a KScriptFile."""
        self._skip_insignificant()
        if self._at_end():
            return KScriptFile([Script([])])
        script = self._parse_script()
        return KScriptFile([script])

    # -- Grammar rules --------------------------------------------------------

    # script ::= construct+
    def _parse_script(self) -> Script:
        constructs = self._parse_constructs_until(TokenType.EOF)
        return Script(constructs)

    # construct ::= block | primary_construct+ ( ( "=>" | "<=" | "<" ) construct )?
    def _parse_construct(self) -> Construct:
        self._skip_insignificant()

        # block: <INDENT> construct+ <DEDENT>
        if self._check(TokenType.INDENT):
            return self._parse_block()

        # primary_construct+ (one or more at current => indent)
        indent = self._peek().column
        primaries = [self._parse_primary_construct()]
        while self._is_primary_construct_start() and self._peek().column >= indent:
            c = self._peek().column
            primaries.append(self._parse_primary_construct())

        # ( ( "=>" | "<=" | "<" ) construct )?
        chain_op = self._try_chain_op()
        if chain_op is not None:
            right = self._parse_construct()
            return Construct(primaries, chain_op, right)

        return Construct(primaries)

    # block ::= <INDENT> construct+ <DEDENT>
    def _parse_block(self) -> Construct:
        self._expect(TokenType.INDENT)
        constructs = self._parse_constructs_until(TokenType.DEDENT)
        self._expect(TokenType.DEDENT)
        return Construct(Block(constructs))

    # primary_construct ::= sig ( ( "==" | ">" | "=" ) node )?
    def _parse_primary_construct(self) -> PrimaryConstruct:
        sig = self._parse_sig()

        op = self._try_inline_op()
        if op is not None:
            node = self._parse_node()
            return PrimaryConstruct(sig, op, node)

        return PrimaryConstruct(sig)

    # node ::= sig | literal
    def _parse_node(self) -> Node:
        if self._check(TokenType.SIGNATURE):
            return self._parse_sig()
        if self._check(TokenType.LITERAL):
            return self._parse_literal()
        raise ParseError("Expected signature or literal", self._peek())

    # sig ::= [A-Z]+
    def _parse_sig(self) -> Signature:
        token = self._expect(TokenType.SIGNATURE)
        return Signature(token.value, token.line, token.column)

    # literal ::= ![A-Z]+
    def _parse_literal(self) -> Literal:
        token = self._expect(TokenType.LITERAL)
        return Literal(token.value, token.line, token.column)

    # -- Helpers --------------------------------------------------------------

    def _parse_constructs_until(self, sentinel: TokenType) -> list[Construct]:
        """Parse construct+ until sentinel (DEDENT or EOF)."""
        constructs: list[Construct] = []
        while not self._at_end() and not self._check(sentinel):
            self._skip_insignificant()
            if self._at_end() or self._check(sentinel):
                break
            constructs.append(self._parse_construct())
        return constructs

    def _try_inline_op(self) -> TokenType | None:
        """Try to match COUNTERSIGN | CONNOTATE_FWD | UNDERSIGN."""
        for tt in (TokenType.COUNTERSIGN, TokenType.CONNOTATE_FWD, TokenType.UNDERSIGN):
            if self._check(tt):
                self._advance()
                return tt
        return None

    def _is_primary_construct_start(self) -> bool:
        """Check if current token can start a primary_construct (SIGNATURE)."""
        self._skip_insignificant()
        return self._check(TokenType.SIGNATURE)

    def _try_chain_op(self) -> TokenType | None:
        """Try to match CANONIZE_FWD | CANONIZE_BWD | CONNOTATE_BWD."""
        for tt in (TokenType.CANONIZE_FWD, TokenType.CANONIZE_BWD, TokenType.CONNOTATE_BWD):
            if self._check(tt):
                self._advance()
                return tt
        return None

    def _skip_insignificant(self) -> None:
        """Skip NEWLINE and COMMENT tokens."""
        while not self._at_end() and (
            self._peek().type == TokenType.NEWLINE
            or self._peek().type == TokenType.COMMENT
        ):
            self._advance()

    # -- Token stream ---------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _check(self, tt: TokenType) -> bool:
        return not self._at_end() and self._peek().type == tt

    def _advance(self) -> Token:
        if not self._at_end():
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return self.tokens[-1]  # EOF

    def _expect(self, tt: TokenType) -> Token:
        if self._check(tt):
            return self._advance()
        raise ParseError(f"Expected {tt.name}", self._peek())

    def _at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

