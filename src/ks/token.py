"""Token types and Token dataclass for KScript v3 lexer.
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript v3 language.

    Construct operators (relationship each token declares):
        COUNTERSIGNS: ==  (goal-targeted training: entry ask + implied goal)
        CANONICALISES:    =>  (aggregation)
        CONNOTES:     >   (connotation)
    RCONNOTES:    <   (connotation, reversed direction)
        DENOTES:      =   (denotation)

    Node types:
        SIGNATURE: [A-Za-z0-9-._']+ continues [A-Za-z0-9-._']    (identifier — case frames the reading; can be construct owner)

    Structure:
        ANNOTATION: (...)    (parenthesized annotation)
        NEWLINE:    \\n       (line ending)
        INDENT:     -        (increased indentation)
        DEDENT:     -        (decreased indentation)
        EOF:        -        (end of file)
    """

    # Construct operators (relationship each token declares)
    COUNTERSIGNS = auto()  # ==
    CANONICALISES = auto()  # =>
    CONNOTES = auto()  # >
    RCONNOTES = auto()  # <
    DENOTES = auto()  # =

    # Node types
    SIGNATURE = auto()  # [A-Za-z0-9-._'] (operators/structure =><()# excluded)

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
