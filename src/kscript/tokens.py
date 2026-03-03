"""Token types and Token dataclass for KScript lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript."""

    # Keywords
    LOAD = auto()  # "load"
    SAVE = auto()  # "save"

    # Significance operators
    S1 = auto()  # "="
    S2 = auto()  # "=>"
    S3_FORWARD = auto()  # ">"
    S3_BACKWARD = auto()  # "<"
    S4 = auto()  # "!="

    # Attention operator
    ATTENTION = auto()  # "?"

    # Identifiers (unquoted strings)
    IDENTIFIER = auto()  # e.g., "hello", "world", "/path/to/file"

    # Statement separator
    NEWLINE = auto()  # newline character

    # Indentation tracking (Python-style)
    INDENT = auto()  # increase in indentation
    DEDENT = auto()  # decrease in indentation

    # End of file
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """Immutable token with type, value, and position information."""

    type: TokenType
    value: str
    line: int
    column: int

    @property
    def is_significance(self) -> bool:
        """Check if this token is a significance operator."""
        return self.type in (
            TokenType.S1,
            TokenType.S2,
            TokenType.S3_FORWARD,
            TokenType.S3_BACKWARD,
            TokenType.S4,
        )
