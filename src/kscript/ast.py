"""AST node definitions for KScript language."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


# =============================================================================
# Node Types (leaf nodes in constructs)
# =============================================================================

@dataclass
class Signature:
    """A signature identifier [A-Z]+ with optional comment.

    Attributes:
        id: The uppercase identifier (e.g., "MHALL", "S")
        comment: Optional parenthesized comment (e.g., "(ubject)")
        line: 1-based line number
        column: 1-based column number
    """
    id: str
    comment: str | None
    line: int
    column: int


@dataclass
class StringLiteral:
    """A string literal "..." including the quotes.

    Attributes:
        id: The string value including quotes (e.g., '"hello"')
        line: 1-based line number
        column: 1-based column number
    """
    id: str
    line: int
    column: int


@dataclass
class NumberLiteral:
    """A number literal [0-9]+.

    Attributes:
        id: The numeric string (e.g., "42")
        line: 1-based line number
        column: 1-based column number
    """
    id: str
    line: int
    column: int


# Union type for all node types
Node: TypeAlias = Signature | StringLiteral | NumberLiteral


# =============================================================================
# Construct Types (operators)
# =============================================================================

class ConstructType(Enum):
    """Types of construct operators.

    Each operator creates different KLine relationships:
        COUNTERSIGN:   ==  Bidirectional signature link {A:B} AND {B:A}
        CANONIZE_FWD:  =>  Forward multi-node composition {A:[B,C,...]}
        CANONIZE_BWD:  <=  Backward multi-node composition
        CONNOTATE_FWD: >   Forward single-node annotation {A:[B]} AND {B:null}
        CONNOTATE_BWD: <   Backward single-node annotation {B:[A]} AND {A:null}
        UNDERSIGN:     =   Unidirectional signature link {A:B} AND {B:null}
    """
    COUNTERSIGN = "=="
    CANONIZE_FWD = "=>"
    CANONIZE_BWD = "<="
    CONNOTATE_FWD = ">"
    CONNOTATE_BWD = "<"
    UNDERSIGN = "="


# =============================================================================
# Construct (operator + nodes)
# =============================================================================

@dataclass
class Construct:
    """A construct: operator followed by nodes.

    Attributes:
        type: The ConstructType (operator)
        nodes: List of Node objects (signatures or literals)
        line: 1-based line number
        has_leading_nodes: True if nodes appeared before the operator
                          (only for CANONIZE_BWD patterns like "B C D <= A")
    """
    type: ConstructType
    nodes: list[Node]
    line: int
    has_leading_nodes: bool = False


# =============================================================================
# Script (signature + constructs + subscripts)
# =============================================================================

@dataclass
class Script:
    """A script: signature with constructs and optional subscripts.

    A script represents a KLine definition:
        SIGNATURE [CONSTRUCT...] [SUBSCRIPT...]

    Examples:
        A                    → identity script
        A == B               → countersign construct
        A => B C             → canonize construct
        A =>                 → canonize with subscript nodes
          B
          C
    """
    signature: Signature
    constructs: list[Construct]
    subscripts: list["Script"]
    line: int


# =============================================================================
# File (top-level container)
# =============================================================================

@dataclass
class KScriptFile:
    """A complete KScript file with multiple top-level scripts.

    Each script starts at column 1 (no indentation).
    """
    scripts: list[Script]
