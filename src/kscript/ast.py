"""AST node definitions for KScript language.

Grammar:
    script ::= construct+
    construct ::=
      | sig                              -- identity
      | sig == node                      -- countersign
      | sig > node                       -- connotate fwd
      | sig = node                       -- undersign
      | sig => construct                 -- canonize fwd (right-assoc)
      | construct <= construct           -- canonize bwd
      | construct < construct            -- connotate bwd
      | construct construct*             -- sequence

    sig ::= [A-Z]+
    node ::= sig | literal
    literal ::= ![A-Z]+

Key insight: Only signatures ([A-Z]+) can be construct owners.
Literals can only appear in node positions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


# =============================================================================
# Node Types (leaf nodes in constructs)
# =============================================================================

@dataclass
class Signature:
    """A signature identifier [A-Z]+ with optional comment.

    Signatures can be construct owners (appear in signature position).

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
# Construct Types (operators)
# =============================================================================

class ConstructType(Enum):
    """Types of construct operators.

    Each operator creates different KLine relationships:
        COUNTERSIGN:   ==  Bidirectional signature link {A:B} AND {B:A}
        CANONIZE_FWD:  =>  Forward multi-node composition {A:[B,C,...]}
        CANONIZE_BWD:  <=  Backward: RIGHT sig points to ALL LEFT nodes
        CONNOTATE_FWD: >   Forward single-node annotation {A:[B]} AND {B:null}
        CONNOTATE_BWD: <   Backward: RIGHT sig points to CLOSEST LEFT node
        UNDERSIGN:     =   Unidirectional signature link {A:B} AND {B:null}
        IDENTITY:      -   Just a signature {sig: null}
    """
    IDENTITY = ""
    COUNTERSIGN = "=="
    CANONIZE_FWD = "=>"
    CANONIZE_BWD = "<="
    CONNOTATE_FWD = ">"
    CONNOTATE_BWD = "<"
    UNDERSIGN = "="


# =============================================================================
# Construct (operator + owner + nodes)
# =============================================================================

@dataclass
class Construct:
    """A construct with a signature owner.

    Key insight: Every construct has an OWNER signature.
    - For FWD operators: owner is the signature on the left
    - For BWD operators: owner is the signature on the RIGHT side

    Attributes:
        owner: The Signature that owns this construct
        type: The ConstructType (operator)
        nodes: List of Node objects (signatures or literals)
        line: 1-based line number
    """
    owner: Signature
    type: ConstructType
    nodes: list[Node] = field(default_factory=list)
    line: int = 0


# =============================================================================
# Script (signature + constructs, no subscripts)
# =============================================================================

@dataclass
class Script:
    """A script: primary signature with constructs.

    A script represents one or more constructs starting from a primary signature.
    Subscripts are normalized to inline constructs during parsing.

    Examples:
        A                    → Script(sig=A, constructs=[identity])
        A => B               → Script(sig=A, constructs=[A=>B])
        A => B <= C => D     → Script(sig=A, constructs=[A=>B, C<=B, C=>D])
    """
    signature: Signature
    constructs: list[Construct]
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
