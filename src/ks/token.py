"""Token types and Token dataclass for KScript v3 lexer.

See specs/kscript.md §2.1 for the token type reference table.
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript v3 language.

    Construct operators (relationship each token declares):
        COUNTERSIGNS: ==  (bidirectional link)
        CANONIZES:    =>  (aggregation)
        CONNOTES:     >   (connotation)
        DENOTES:      =   (denotation)

    Node types:
        SIGNATURE: [a-zA-Z][a-zA-Z0-9]*    (identifier — case-insensitive; can be construct owner)

    Structure:
        ANNOTATION: (...)    (parenthesized annotation)
        NEWLINE:    \\n       (line ending)
        INDENT:     -        (increased indentation)
        DEDENT:     -        (decreased indentation)
        EOF:        -        (end of file)
    """

    # Construct operators (relationship each token declares)
    COUNTERSIGNS = auto()  # ==
    CANONIZES = auto()  # =>
    CONNOTES = auto()  # >
    DENOTES = auto()  # =

    # Node types
    SIGNATURE = auto()  # [a-zA-Z][a-zA-Z0-9]*

    # Structure
    ANNOTATION = auto()  # (...)
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A single token from the lexer.

    Attributes:
        type: The TokenType of this token
        value: The raw string value from source
        line: 1-based line number
        column: 1-based column number
    """

    type: TokenType
    value: str
    line: int
    column: int
