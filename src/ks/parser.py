"""KScript v3 parser — transforms a token stream into a scope-model AST.

Grammar (spec §4)::

    script          ::= construct*
    construct       ::= block | annotation | operator_scope
    block           ::= INDENT construct+ DEDENT
    annotation      ::= ANNOTATION
    operator_scope  ::= sig ( operator items )?
    items           ::= item*
    item            ::= sig | annotation | operator_scope
    sig             ::= SIGNATURE
    operator        ::= COUNTERSIGNS | CANONIZES | CONNOTES | DENOTES

Scope rules enforced (spec §3):

    S2  Preceding identifier is the signature.
    S3  Succeeding identifiers are nodes (items).
    S4  INDENT creates child scope → stored in OperatorScope.child_block.
    S5  DEDENT closes child scope.

Inline annotations (spec §5.1):

    Sig-side   S(ubject) = M   →  OperatorScope.inline_annotation
    Node-side  A = D(et)       →  Signature.inline_annotation (per-item)

NEWLINE tokens are insignificant — skipped between constructs.
Bare signature (no operator) produces OperatorScope with op=None.
Empty source produces KScriptFile(constructs=[]).

The parser produces **nested** OperatorScope nodes when an operator chain
appears on a single line (e.g. ``A == B > C``).  The grammar rule
``item ::= operator_scope`` means each scope's items may contain child
OperatorScope instances whose own sig carries forward as both the
parent scope's last node and the child scope's signature.
"""

from __future__ import annotations

from ks.ast import (
    Annotation,
    Block,
    ConstructItem,
    KScriptFile,
    OperatorScope,
    ScopeItem,
    Signature,
)
from ks.token import Token, TokenType

# Operator token types that create scope boundaries
_OPERATOR_TYPES: frozenset[TokenType] = frozenset(
    {
        TokenType.COUNTERSIGNS,
        TokenType.CANONIZES,
        TokenType.CONNOTES,
        TokenType.DENOTES,
    }
)


class ParseError(Exception):
    """Error raised when the parser encounters an unexpected token.

    Attributes:
        message: Human-readable description of the error.
        token: The offending Token.
    """

    def __init__(self, message: str, token: Token) -> None:
        super().__init__(f"Line {token.line}, column {token.column}: {message}")
        self.message = message
        self.token = token


class Parser:
    """Recursive-descent parser for KScript v3.

    Takes a list of Token objects (as produced by the lexer) and returns
    a KScriptFile AST node.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos: int = 0

    # Public API

    def parse(self) -> KScriptFile:
        """Parse the token stream and return a KScriptFile.

        Empty input (only an EOF token) produces KScriptFile(constructs=[]).
        """
        self._skip_newlines()
        if self._at_end():
            return KScriptFile(constructs=[])
        constructs = self._parse_constructs_until(TokenType.EOF)
        return KScriptFile(constructs=constructs)

    # Construct-level parsing

    def _parse_constructs_until(self, *stop: TokenType) -> list[ConstructItem]:
        """Parse zero or more constructs until a stop token is seen."""
        constructs: list[ConstructItem] = []
        while not self._at_end() and self._peek().type not in stop:
            self._skip_newlines()
            if self._at_end() or self._peek().type in stop:
                break
            constructs.append(self._parse_construct())
            self._skip_newlines()
        return constructs

    def _parse_construct(self) -> ConstructItem:
        """Dispatch to the correct construct parser based on the next token."""
        tok = self._peek()
        if tok.type == TokenType.INDENT:
            return self._parse_block()
        if tok.type == TokenType.ANNOTATION:
            return self._parse_annotation()
        if tok.type == TokenType.SIGNATURE:
            return self._parse_operator_scope()
        raise ParseError(
            f"Unexpected token {tok.type.name}; expected INDENT, ANNOTATION, or SIGNATURE",
            tok,
        )

    # Block  (INDENT construct+ DEDENT)

    def _parse_block(self) -> Block:
        """Parse INDENT construct+ DEDENT → Block."""
        self._expect(TokenType.INDENT)
        constructs = self._parse_constructs_until(TokenType.DEDENT)
        self._skip_newlines()
        self._expect(TokenType.DEDENT)
        return Block(constructs=constructs)

    # Annotation

    def _parse_annotation(self) -> Annotation:
        """Consume an ANNOTATION token and return an Annotation node."""
        tok = self._advance()
        return Annotation(text=tok.value, line=tok.line, column=tok.column)

    # OperatorScope  (sig (operator items)?)

    def _parse_operator_scope(self) -> OperatorScope:
        """Parse an operator scope: SIGNATURE [inline_ann] [operator items] [child_block]."""
        sig_token = self._expect(TokenType.SIGNATURE)
        sig = Signature(id=sig_token.value, line=sig_token.line, column=sig_token.column)

        # Sig-side inline annotation: S(ubject) = M
        inline_ann: Annotation | None = None
        if not self._at_end() and self._peek().type == TokenType.ANNOTATION:
            ann_tok = self._advance()
            inline_ann = Annotation(
                text=ann_tok.value,
                line=ann_tok.line,
                column=ann_tok.column,
            )

        # Operator (optional — bare signature if absent)
        op: TokenType | None = None
        if not self._at_end() and self._peek().type in _OPERATOR_TYPES:
            op = self._advance().type

        if op is None:
            # Bare unsigned signature
            return OperatorScope(sig=sig, op=None, items=[], inline_annotation=inline_ann)

        # Items on the same line as the operator
        items = self._parse_items()

        # Child block on subsequent indented lines
        self._skip_newlines()
        child_block: Block | None = None
        if not self._at_end() and self._peek().type == TokenType.INDENT:
            child_block = self._parse_block()

        return OperatorScope(
            sig=sig,
            op=op,
            items=items,
            child_block=child_block,
            inline_annotation=inline_ann,
        )

    # Items  (same-line nodes within an operator scope)

    def _parse_items(self) -> list[ScopeItem]:
        """Collect items on the same line as the operator.

        Items stop at NEWLINE, INDENT, DEDENT, or EOF.
        A SIGNATURE followed (skipping ANNOTATIONs) by an operator is
        parsed as a nested OperatorScope per grammar rule
        ``item ::= operator_scope``.

        An ANNOTATION immediately following a bare Signature item attaches to
        that item as its ``inline_annotation`` (Word Binding: inline annotations
        bind unconditionally to their item), rather than becoming a loose item.
        """
        items: list[ScopeItem] = []

        while not self._at_end():
            tok = self._peek()

            # End-of-line / end-of-scope markers
            if tok.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
                break

            if tok.type == TokenType.SIGNATURE:
                if self._sig_followed_by_operator():
                    # Nested operator_scope item. If it consumes a child block
                    # (multi-line), the parent's same-line items end here — a
                    # nested scope crossing a newline cannot be followed by more
                    # items on the parent's original line.
                    nested = self._parse_operator_scope()
                    items.append(nested)
                    if nested.child_block is not None:
                        break
                else:
                    # Bare Signature item
                    sig_tok = self._advance()
                    sig_item = Signature(
                        id=sig_tok.value,
                        line=sig_tok.line,
                        column=sig_tok.column,
                    )
                    items.append(sig_item)

                    # Inline annotation on this item: D(et) — attach to the
                    # Signature (bound unconditionally to it per Word Binding).
                    if not self._at_end() and self._peek().type == TokenType.ANNOTATION:
                        ann_tok = self._advance()
                        sig_item.inline_annotation = Annotation(
                            text=ann_tok.value,
                            line=ann_tok.line,
                            column=ann_tok.column,
                        )

            elif tok.type == TokenType.ANNOTATION:
                # An annotation not following a Signature (e.g. a leading word
                # list) stays a loose item.
                items.append(self._parse_annotation())

            else:
                # Unexpected token → stop item collection
                break

        return items

    def _sig_followed_by_operator(self) -> bool:
        """True if the current SIGNATURE is followed (skipping ANNOTATIONs) by an operator.

        Does NOT consume any tokens.
        """
        i = self.pos + 1  # past the SIGNATURE at self.pos
        while i < len(self.tokens) and self.tokens[i].type == TokenType.ANNOTATION:
            i += 1
        return i < len(self.tokens) and self.tokens[i].type in _OPERATOR_TYPES

    # Token-level helpers

    def _peek(self) -> Token:
        """Return the current token without advancing."""
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _at_end(self) -> bool:
        """True if at or past the EOF token."""
        return self.pos >= len(self.tokens) or self.tokens[self.pos].type == TokenType.EOF

    def _skip_newlines(self) -> None:
        """Advance past any NEWLINE tokens."""
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == TokenType.NEWLINE:
            self.pos += 1

    def _expect(self, token_type: TokenType) -> Token:
        """Consume the next token, raising ParseError if it doesn't match."""
        if self.pos >= len(self.tokens):
            raise ParseError(
                f"Expected {token_type.name} but reached end of input",
                self.tokens[-1],
            )
        tok = self.tokens[self.pos]
        if tok.type != token_type:
            raise ParseError(
                f"Expected {token_type.name} but got {tok.type.name}",
                tok,
            )
        self.pos += 1
        return tok
