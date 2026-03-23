"""Token types and Token dataclass for KScript lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript language."""

    # Construct operators
    COUNTERSIGN = auto()  # ==
    CANONIZE_FWD = auto()  # =>
    CANONIZE_BWD = auto()  # <=
    CONNOTATE_FWD = auto()  # >
    CONNOTATE_BWD = auto()  # <
    UNDERSIGN = auto()  # =

    # Literals
    SIGNATURE = auto()  # [A-Z]+
    STRING = auto()  # "..."
    NUMBER = auto()  # [0-9]+
    COMMENT = auto()  # (...)

    # Structure
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A single token from the lexer."""

    type: TokenType
    value: str
    line: int
    column: int
