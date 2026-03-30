"""Minimal KScript parser test using simplified semantics."""

from dataclasses import dataclass, field
from typing import TypeAlias

from .token import Token, TokenType
from .lexer import Lexer


# =============================================================================
# Simplified AST
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


@dataclass
class Construct:
    """A construct with simplified semantics.

    Attributes:
        sig: The owner signature
        op: Operator string ("identity", "==", "=>", "<=", ">", "<", "=")
        clns: List of child nodes
        bwd: Optional backward clause as (bwd_sig, bwd_op, bwd_clns)
        subscripts: Nested constructs from indentation
        line: Line number
    """
    sig: Signature
    op: str = "identity"
    clns: list[Node] = field(default_factory=list)
    bwd: tuple[Signature, str, list[Node]] | None = None
    subscripts: list["Construct"] = field(default_factory=list)
    line: int = 0


@dataclass
class Script:
    """A script starting from a primary signature."""
    first_sig: Signature
    constructs: list[Construct]
    line: int


@dataclass
class KScriptFile:
    """Complete KScript file."""
    scripts: list[Script]


# =============================================================================
# Simplified Parser
# =============================================================================

class Parser:
    """Simplified KScript parser with immediate binding semantics."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> KScriptFile:
        scripts = []
        while not self._at_end():
            if self._check(TokenType.SIGNATURE):
                scripts.append(self._parse_script())
            else:
                self._advance()  # skip garbage
        return KScriptFile(scripts)

    def _parse_script(self) -> Script:
        # Step 1: Collect signature and CLNs
        sig = self._make_signature(self._expect(TokenType.SIGNATURE))
        constructs = []

        while not self._at_end() and not self._check(TokenType.NEWLINE):

        self._parse_inline_chain(sig, constructs)

        self._consume_newlines()

        # Step 2: Parse subscripts (attaches CLNs, creates independent constructs)
        if constructs:
            self._parse_subscripts(constructs[-1])

        return Script(sig, constructs, sig.line)

    def _parse_inline_chain(self, first_sig: Signature, constructs: list) -> None:
        """Parse chain of constructs (handles right-assoc =>)."""
        current_sig = first_sig

        while not self._at_end() and not self._check(TokenType.NEWLINE):
            construct = self._parse_single_construct(current_sig, constructs)

            # Right-associativity: check for => operator to continue chain
            if self._check(TokenType.CANONIZE_FWD) and construct.clns:
                last = construct.clns[-1]
                if isinstance(last, Signature) and not construct.bwd:
                    self._advance()  # consume =>
                    current_sig = last
                    continue

            break  # No continuation

    def _parse_single_construct(self, owner: Signature, constructs: list[Construct]) -> Construct:
        """Parse: sig (FWD_OP nodes)? (BWD_OP sig)?"""
        op = "identity"
        clns = []
        bwd = None

        # Phase 1: FWD clause?
        if self._check_fwd_op():
            op = self._parse_fwd_op()
            clns = self._parse_nodes()

        # Phase 2: BWD clause?
        if self._check_bwd_op():
            bwd_op = self._parse_bwd_op()
            if self._check(TokenType.SIGNATURE):
                bwd_sig = self._make_signature(self._advance())
                # Bind: ALL for <=, LAST for <
                if bwd_op == "<=":
                    bwd_clns = clns if clns else [owner]
                else:
                    bwd_clns = [clns[-1]] if clns else [owner]
                bwd = (bwd_sig, bwd_op, bwd_clns)
                construct = Construct(sig=bwd_sig, op=bwd_op, clns=bwd_clns, bwd=bwd, line=owner.line)
                constructs.append(construct)

        construct = Construct(sig=owner, op=op, clns=clns, bwd=bwd, line=owner.line)
        constructs.append(construct)
        return construct

    def _parse_nodes(self) -> list[Node]:
        """Collect nodes until operator/newline/EOF."""
        nodes = []
        while not self._at_end():
            if self._check(TokenType.SIGNATURE):
                nodes.append(self._make_signature(self._advance()))
            elif self._check(TokenType.LITERAL):
                nodes.append(self._make_literal(self._advance()))
            elif self._check_fwd_op() or self._check_bwd_op() or self._check(TokenType.NEWLINE):
                break
            else:
                break
        return nodes

    def _parse_subscripts(self, parent: Construct) -> None:
        """Parse subscripts: sigs become CLNs, constructs are independent."""
        if not self._check(TokenType.INDENT):
            return

        self._advance()  # consume INDENT
        subscript_constructs = []

        while not self._at_end() and not self._check(TokenType.DEDENT):
            if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                self._advance()
                continue

            if self._check(TokenType.SIGNATURE):
                sig = self._make_signature(self._advance())
                parent.clns.append(sig)  # sig becomes parent CLN

                # Parse construct(s) for this subscript (handles => continuation)
                constructs = []
                self._parse_inline_chain(sig, constructs)

                # Consume newlines before checking for nested subscripts
                self._consume_newlines()

                if constructs:
                    subscript_constructs.append(constructs[0])
                    # Handle any chained constructs
                    for c in constructs[1:]:
                        subscript_constructs.append(c)

                    # Recurse for nested subscripts on the last construct
                    self._parse_subscripts(constructs[-1])

            elif self._check(TokenType.LITERAL):
                parent.clns.append(self._make_literal(self._advance()))

            else:
                # Skip unexpected tokens to avoid infinite loop
                self._advance()

        parent.subscripts = subscript_constructs

        if self._check(TokenType.DEDENT):
            self._advance()

    # --- Helper methods ---

    def _check_fwd_op(self) -> bool:
        return self._check(TokenType.COUNTERSIGN, TokenType.CANONIZE_FWD,
                          TokenType.CONNOTATE_FWD, TokenType.UNDERSIGN)

    def _check_bwd_op(self) -> bool:
        return self._check(TokenType.CANONIZE_BWD, TokenType.CONNOTATE_BWD)

    def _parse_fwd_op(self) -> str:
        tok = self._advance()
        return {
            TokenType.COUNTERSIGN: "==",
            TokenType.CANONIZE_FWD: "=>",
            TokenType.CONNOTATE_FWD: ">",
            TokenType.UNDERSIGN: "=",
        }.get(tok.type, "identity")

    def _parse_bwd_op(self) -> str:
        tok = self._advance()
        return {
            TokenType.CANONIZE_BWD: "<=",
            TokenType.CONNOTATE_BWD: "<",
        }.get(tok.type, "")

    def _make_signature(self, tok: Token) -> Signature:
        return Signature(id=tok.value, line=tok.line, column=tok.column)

    def _make_literal(self, tok: Token) -> Literal:
        return Literal(id=tok.value, line=tok.line, column=tok.column)

    def _consume_newlines(self) -> None:
        while self._check(TokenType.NEWLINE):
            self._advance()

    def _check(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _check_node(self) -> bool:
        return self._peek().type == TokenType.SIGNATURE or type == TokenType.LITERAL

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        if self._check(ttype):
            return self._advance()
        raise SyntaxError(f"Expected {ttype}, got {self._peek().type} at line {self._peek().line}")

    def _at_end(self) -> bool:
        return self._check(TokenType.EOF)


# =============================================================================
# Pretty Printer
# =============================================================================

def indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.split("\n"))


def format_node(node: Node) -> str:
    if isinstance(node, Signature):
        return f"Sig({node.id})"
    return f"Lit({node.id})"


def format_construct(c: Construct, depth: int = 0) -> str:
    parts = [f"Construct({c.sig.id} {c.op}"]
    if c.clns:
        parts.append(" [" + ", ".join(format_node(n) for n in c.clns) + "]")
    else:
        parts.append(" []")
    if c.bwd:
        bwd_sig, bwd_op, bwd_clns = c.bwd
        parts.append(f" {bwd_op} ")
        parts.append(f"{bwd_sig.id}:[")
        parts.append(", ".join(format_node(n) for n in bwd_clns))
        parts.append("]")
    parts.append(")")

    if c.subscripts:
        parts.append(" {\n")
        for sub in c.subscripts:
            parts.append(indent(format_construct(sub, depth + 1), 2))
            parts.append("\n")
        parts.append("}")

    return "".join(parts)


def format_script(s: Script) -> str:
    lines = [f"Script(line={s.line}, first_sig={s.first_sig.id})"]
    for c in s.constructs:
        lines.append("  " + format_construct(c))
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def parse_source(source: str) -> KScriptFile:
    """Parse KScript source and return AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


if __name__ == "__main__":
    SOURCE = """
MHALL = SVO =>
  S = M
  V = H
  O = ALL =>
    A = D
    L = M
    L > O < L
"""

    print("=" * 60)
    print("SOURCE:")
    print("=" * 60)
    print(SOURCE)
    print("=" * 60)
    print("AST OUTPUT:")
    print("=" * 60)

    result = parse_source(SOURCE)
    for script in result.scripts:
        print(format_script(script))
