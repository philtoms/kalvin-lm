"""AST node definitions for KScript language.

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

from .token import TokenType


# =============================================================================
# Leaf Nodes
# =============================================================================

@dataclass
class Signature:
    """A signature identifier [A-Z]+.

    Signatures can be construct owners (appear in signature position).

    Attributes:
        id: The uppercase identifier (e.g., "MHALL", "S")
        line: 1-based line number
        column: 1-based column number
    """
    id: str
    line: int
    column: int


@dataclass
class Literal:
    """A literal value (anything not [A-Z]+).

    Literals can only appear in node positions, never as construct owners.

    Attributes:
        id: The literal value (e.g., "hello", "42", '"quoted"')
        line: 1-based line number
        column: 1-based column number
    """
    id: str
    line: int
    column: int


# Union type for all node types
Node: TypeAlias = Signature | Literal


# =============================================================================
# Construct Nodes
# =============================================================================

@dataclass
class PrimaryConstruct:
    """primary_construct ::= sig ( ( "==" | ">" | "=" ) node )?

    A primary construct with optional inline operator.

    If op is None, this is an identity (bare signature).

    Attributes:
        sig: The signature that owns this construct
        op: The inline operator (COUNTERSIGN, CONNOTATE_FWD, UNDERSIGN, IDENTITY), or None
        node: The node on the right side of the operator, if any
    """
    sig: Signature
    op: TokenType | None = None
    node: Node | None = None


@dataclass
class Block:
    """block ::= <INDENT> construct+ <DEDENT>

    A block of indented constructs.

    Attributes:
        constructs: List of constructs in this block
    """
    constructs: list["Construct"]


@dataclass
class Construct:
    """construct ::= block | primary_construct+ ( ( "=>" | "<=" | "<" ) construct )?

    A construct is either a block or a sequence of primary constructs
    with an optional chain operator.

    Attributes:
        inner: Either a Block or a list of PrimaryConstruct
        chain_op: The chain operator (CANONIZE_FWD, CANONIZE_BWD, CONNOTATE_BWD), or None
        chain_right: The right-hand construct of the chain, if any
    """
    inner: Block | list[PrimaryConstruct]
    chain_op: TokenType | None = None
    chain_right: "Construct | None" = None


@dataclass
class Script:
    """script ::= construct+

    A script is a sequence of constructs.

    Attributes:
        constructs: List of constructs in this script
    """
    constructs: list[Construct]


@dataclass
class KScriptFile:
    """Top-level file container (one script per file).

    Attributes:
        scripts: List of scripts (typically just one)
    """
    scripts: list[Script]


# =============================================================================
# Re-exports for backwards compatibility
# =============================================================================

__all__ = [
    "Signature",
    "Literal",
    "Node",
    "PrimaryConstruct",
    "Block",
    "Construct",
    "Script",
    "KScriptFile",
]
