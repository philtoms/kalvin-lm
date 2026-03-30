"""CLN-based parser for KScript language.

Key insights:
- CLNs (Construct Level Nodes) are nodes collected BETWEEN same level construct operators
- CLN collection is construct-type agnostic
- Construct parsing is 2 step and recursive
- BWD operators act between constructs on RHS signature and LHS construct CLNs
- BWD constructs require previously collected CLNs plus next signature look ahead

Grammar:
    script ::= construct+
    construct ::=
      | sig                              -- identity (S4)
      | sig == node                      -- countersign (S1)
      | sig > node                       -- connotate fwd (S3)
      | sig = node                       -- undersign (S4)
      | sig => node+                     -- canonize fwd (S2 right-assoc)
      | construct <= construct           -- canonize bwd (S2, ALL CLNs)
      | construct < construct            -- connotate bwd (S3, CLNs[-1])
      | construct construct*             -- sequence
      | <INDENT> construct+ <DEDENT>     -- subscript

    sig ::= [A-Z]+
    node ::= sig | literal
    literal ::= ![A-Z]+
"""

from dataclasses import dataclass, field
from typing import TypeAlias

from .token import Token, TokenType


# =============================================================================
# AST Node Types
# =============================================================================

@dataclass
class Signature:
    """A signature identifier [A-Z]+."""
    id: str
    line: int
    column: int


@dataclass
class Literal:
    """A literal value (anything not [A-Z]+)."""
    id: str
    line: int
    column: int


Node: TypeAlias = Signature | Literal


# =============================================================================
# Construct Types
# =============================================================================

class ConstructType:
    """Types of construct operators with significance levels."""
    IDENTITY = "S4"       # Just a signature {sig: null}
    COUNTERSIGN = "S1"    # == Bidirectional {A:B} AND {B:A}
    CANONIZE_FWD = "S2"   # => Forward multi-node {A:[B,C,...]}
    CANONIZE_BWD = "S2"   # <= Backward ALL nodes
    CONNOTATE_FWD = "S3"  # > Forward single-node {A:[B]} AND {B:null}
    CONNOTATE_BWD = "S3"  # < Backward CLOSEST node
    UNDERSIGN = "S4"      # = Unidirectional {A:B} AND {B:null}


@dataclass
class Construct:
    """A parsed construct ready for compilation.

    Attributes:
        sig: The signature that owns this construct
        op: Operator type (identity, countersign, etc.)
        clns: Construct Level Nodes collected for this construct
        bwd: Optional BWD construct (sig, op, clns to bind)
        subscripts: List of subscript constructs to process recursively
        line: Source line number
    """
    sig: Signature
    op: str
    clns: list[Node] = field(default_factory=list)
    bwd: tuple[Signature, str, list[Node]] | None = None  # (sig, op, clns_to_bind)
    subscripts: list["Construct"] = field(default_factory=list)
    line: int = 0


@dataclass
class Script:
    """A script: primary signature with constructs."""
    sig: Signature
    constructs: list[Construct]
    line: int


@dataclass
class KScriptFile:
    """A complete KScript file with multiple top-level scripts."""
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
    """CLN-based parser for KScript.

    Key design:
    - CLNs are collected between operators (construct-type agnostic)
    - Eager emit: constructs are complete when we see next operator or newline
    - BWD operators use already-collected CLNs for binding
    - Subscripts processed recursively in step 2
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> KScriptFile:
        """Parse tokens into a KScriptFile."""
        scripts: list[Script] = []

        while not self._at_end():
            # Skip newlines and comments at top level
            if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                self._advance()
                continue

            # Skip literals at top level (error recovery)
            if self._check(TokenType.LITERAL):
                self._advance()
                continue

            # Parse top-level script
            if self._check(TokenType.SIGNATURE):
                script = self._parse_script()
                scripts.append(script)
            else:
                self._advance()

        return KScriptFile(scripts)

    def _parse_script(self) -> Script:
        """Parse a script starting with a signature."""
        start_token = self._expect(TokenType.SIGNATURE)
        primary_sig = self._make_signature(start_token)

        constructs: list[Construct] = []

        # Phase 1: Parse inline constructs
        self._parse_inline_constructs(primary_sig, constructs)

        # Consume newline if present
        if self._check(TokenType.NEWLINE):
            self._advance()

        # Phase 2: Parse subscripts recursively
        self._parse_subscripts(constructs)

        return Script(primary_sig, constructs, primary_sig.line)

    def _parse_inline_constructs(
        self,
        first_sig: Signature,
        constructs: list[Construct]
    ) -> None:
        """Parse inline constructs until newline or end.

        CLNs are collected between operators. When we see an operator,
        we emit a construct using the collected CLNs.
        """
        current_sig = first_sig
        current_clns: list[Node] = []

        while not self._at_end() and not self._check(TokenType.NEWLINE):
            # Check for BWD operators first
            if self._check_bwd_op():
                bwd_op = self._parse_bwd_op()
                if bwd_op:
                    # BWD uses current sig + CLNs for binding
                    clns_to_bind = [first_sig] + current_clns
                    if bwd_op == "<" and clns_to_bind:
                        clns_to_bind = [clns_to_bind[-1]]

                    # Next signature becomes BWD owner
                    if self._check(TokenType.SIGNATURE):
                        bwd_sig = self._make_signature(self._advance())
                        constructs.append(Construct(
                            sig=current_sig,
                            op="identity",
                            clns=current_clns,
                            bwd=(bwd_sig, bwd_op, clns_to_bind),
                            line=current_sig.line
                        ))
                        current_sig = bwd_sig
                        current_clns = []
                        continue
                    else:
                        # Literal in sig position - invalid BWD
                        # Treat current sig as identity
                        constructs.append(Construct(
                            sig=current_sig,
                            op="identity",
                            clns=[],
                            line=current_sig.line
                        ))
                        break

            # Check for FWD operators
            fwd_op = self._try_parse_fwd_op()
            if fwd_op:
                # Collect CLNs for this construct
                clns = self._parse_clns()
                current_clns = clns

                # Check for BWD operator BEFORE emitting
                bwd_info = None
                if self._check_bwd_op():
                    bwd_op = self._parse_bwd_op()
                    if bwd_op and self._check(TokenType.SIGNATURE):
                        bwd_sig = self._make_signature(self._advance())
                        # BWD binding: S2 (<=) binds ALL, S3 (<) binds CLNs[-1]
                        clns_to_bind = clns if bwd_op == "<=" else [clns[-1]] if clns else []
                        bwd_info = (bwd_sig, bwd_op, clns_to_bind)

                # Emit construct (with BWD if present)
                constructs.append(Construct(
                    sig=current_sig,
                    op=fwd_op,
                    clns=clns,
                    bwd=bwd_info,
                    line=current_sig.line
                ))

                # For => operator, last signature becomes new owner (right-assoc)
                if fwd_op == "=>" and clns:
                    last = clns[-1]
                    if isinstance(last, Signature):
                        current_sig = last
                # For == with signature, switch owner
                elif fwd_op == "==" and clns:
                    first = clns[0]
                    if isinstance(first, Signature):
                        current_sig = first
                continue

            # Check for sequence (signature without operator)
            if self._check(TokenType.SIGNATURE):
                sig = self._make_signature(self._advance())
                # Emit identity for previous sig if it had no construct
                if not current_clns:
                    constructs.append(Construct(
                        sig=current_sig,
                        op="identity",
                        clns=[],
                        line=current_sig.line
                    ))
                current_sig = sig
                current_clns = [sig]
                continue

            # Check for literal in node position
            if self._check(TokenType.LITERAL):
                node = self._make_literal(self._advance())
                current_clns.append(node)
                continue

            # Nothing recognized - break
            break

        # Emit final identity if no constructs were created (standalone signature)
        if not constructs:
            constructs.append(Construct(
                sig=first_sig,
                op="identity",
                clns=[],
                line=first_sig.line
            ))

    def _parse_subscripts(self, constructs: list[Construct]) -> None:
        """Parse subscripts recursively for all constructs.

        Subscript signatures become CLNs for the parent construct.
        Subscripts with operators also create their own constructs.
        """
        for construct in constructs:
            if self._check(TokenType.INDENT):
                self._advance()  # consume INDENT

                # Collect subscript signatures as CLNs for parent
                subscript_clns: list[Node] = []
                subscript_constructs: list[Construct] = []

                while not self._at_end() and not self._check(TokenType.DEDENT):
                    if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                        self._advance()
                        continue

                    # Nested INDENT - break to let recursive call handle it
                    if self._check(TokenType.INDENT):
                        break

                    if self._check(TokenType.SIGNATURE):
                        sig = self._make_signature(self._advance())
                        # Add signature to parent's CLNs
                        subscript_clns.append(sig)

                        # Check for FWD operator
                        fwd_op = self._try_parse_fwd_op()
                        if fwd_op:
                            clns = self._parse_clns()
                            subscript_constructs.append(Construct(
                                sig=sig,
                                op=fwd_op,
                                clns=clns,
                                line=sig.line
                            ))
                        elif self._check_bwd_op():
                            # BWD operator
                            bwd_op = self._parse_bwd_op()
                            if bwd_op and self._check(TokenType.SIGNATURE):
                                bwd_sig = self._make_signature(self._advance())
                                subscript_constructs.append(Construct(
                                    sig=sig,
                                    op="identity",
                                    clns=[],
                                    bwd=(bwd_sig, bwd_op, [sig]),
                                    line=sig.line
                                ))
                        # Identity subscripts don't create constructs - just CLNs

                    elif self._check(TokenType.LITERAL):
                        node = self._make_literal(self._advance())
                        subscript_clns.append(node)

                # Add subscript CLNs to parent construct
                construct.clns.extend(subscript_clns)

                # Attach subscript constructs
                construct.subscripts = subscript_constructs

                # Recursively parse nested subscripts (handles nested INDENT)
                self._parse_subscripts(subscript_constructs)

                # Now consume DEDENT after nested subscripts are processed
                if self._check(TokenType.DEDENT):
                    self._advance()

    def _parse_clns(self) -> list[Node]:
        """Parse CLNs (nodes) until operator, newline, or end."""
        clns: list[Node] = []

        while not self._at_end():
            if self._check(TokenType.SIGNATURE):
                sig = self._make_signature(self._advance())
                clns.append(sig)
                # Right-assoc: if => follows, this sig starts new construct
                if self._check(TokenType.CANONIZE_FWD):
                    break
            elif self._check(TokenType.LITERAL):
                node = self._make_literal(self._advance())
                clns.append(node)
            elif self._check_bwd_op() or self._check(TokenType.NEWLINE):
                break
            else:
                break

        return clns

    def _try_parse_fwd_op(self) -> str | None:
        """Try to parse a FWD operator. Returns op string or None."""
        if self._check(TokenType.COUNTERSIGN):
            self._advance()
            return "=="
        if self._check(TokenType.CANONIZE_FWD):
            self._advance()
            return "=>"
        if self._check(TokenType.CONNOTATE_FWD):
            self._advance()
            return ">"
        if self._check(TokenType.UNDERSIGN):
            self._advance()
            return "="
        return None

    def _check_bwd_op(self) -> bool:
        """Check if current token is a BWD operator."""
        return self._check(TokenType.CANONIZE_BWD) or self._check(TokenType.CONNOTATE_BWD)

    def _parse_bwd_op(self) -> str | None:
        """Parse a BWD operator. Returns op string or None."""
        if self._check(TokenType.CANONIZE_BWD):
            self._advance()
            return "<="
        if self._check(TokenType.CONNOTATE_BWD):
            self._advance()
            return "<"
        return None

    def _make_signature(self, token: Token) -> Signature:
        """Create a Signature from a token."""
        return Signature(token.value, token.line, token.column)

    def _make_literal(self, token: Token) -> Literal:
        """Create a Literal from a token."""
        return Literal(token.value, token.line, token.column)

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
