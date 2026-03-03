"""AST node dataclasses for KScript parser."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class SignificanceType(Enum):
    """Significance level for KLine relationships."""

    S1 = "="  # Prefix match
    S2 = "=>"  # Partial positional
    S3_FORWARD = ">"  # Unordered (forward)
    S3_BACKWARD = "<"  # Unordered (backward)
    S4 = "!="  # No match


@dataclass
class Identifier:
    """An unquoted string identifier (KNode name or file path)."""

    name: str
    line: int
    column: int


@dataclass
class LoadStatement:
    """Load command: load <path>"""

    path: Identifier


@dataclass
class SaveStatement:
    """Save command: save [path]"""

    path: Identifier | None = None


# Forward reference for recursive types
KScriptStatement = Union["LoadStatement", "SaveStatement", "KLineExpr"]


@dataclass
class KScript:
    """Root node containing a sequence of statements."""

    statements: list[KScriptStatement] = field(default_factory=list)


@dataclass
class KNodeRef:
    """Reference to a KNode by identifier name."""

    identifier: Identifier


# KSig can be either an identifier or a nested KLine expression
KSig = Union[Identifier, "KLineExpr"]


@dataclass
class KLineExpr:
    """
    KLine expression with optional significance relationship.

    Forms:
    - `name` -> Simple KLine with just a name
    - `name = node` -> KLine with S1 relationship to node
    - `name => node1 node2` -> KLine with S2 relationship to nodes
    - `name > node1 node2` -> KLine with S3 forward relationship
    - `name < node1 node2` -> KLine with S3 backward relationship
    - `name != node` -> KLine with S4 relationship
    - `name ?` -> KLine with attention marker
    - `kline1 > nodes` -> KLine referencing another KLine
    """

    sig: KSig  # Identifier or nested KLineExpr
    significance: SignificanceType | None = None
    nodes: list[KNodeRef] = field(default_factory=list)
    attention: bool = False
    line: int = 0
    column: int = 0
