"""Token types and Token dataclass for KScript lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript language.

    Operators (construct types):
        COUNTERSIGN:  ==  (bidirectional link)
        CANONIZE:     =>  (canonization)
        CONNOTATE:    >   (connotation)
        UNDERSIGN:    =   (undersign link)

    Node types:
        SIGNATURE: [A-Z]+    (uppercase identifier - can be construct owner)

    Structure:
        COMMENT:  (...)     (parenthesized comment)
        NEWLINE:  \\n       (line ending)
        INDENT:   -        (increased indentation)
        DEDENT:   -        (decreased indentation)
        EOF:      -        (end of file)
    """

    # Construct operators
    COUNTERSIGN = auto()   # ==
    CANONIZE = auto()      # =>
    CONNOTATE = auto()     # >
    UNDERSIGN = auto()     # =

    # Node types
    SIGNATURE = auto()     # [A-Z]+

    # Structure
    COMMENT = auto()       # (...)
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
