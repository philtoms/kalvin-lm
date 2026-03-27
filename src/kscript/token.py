"""Token types and Token dataclass for KScript lexer."""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for KScript language.

    Operators (construct types):
        COUNTERSIGN:   ==  (bidirectional link)
        CANONIZE_FWD:  =>  (forward canonization)
        CANONIZE_BWD:  <=  (backward canonization)
        CONNOTATE_FWD: >   (forward connotation)
        CONNOTATE_BWD: <   (backward connotation)
        UNDERSIGN:     =   (undersign link)

    Node types:
        SIGNATURE: [A-Z]+    (uppercase identifier - can be construct owner)
        LITERAL:   anything else (cannot be construct owner)

    Structure:
        COMMENT:  (...)     (parenthesized comment)
        NEWLINE:  \\n       (line ending)
        INDENT:   -        (increased indentation)
        DEDENT:   -        (decreased indentation)
        EOF:      -        (end of file)

    Key insight: Any token in node position that is NOT a SIGNATURE is a LITERAL.
    """

    # Construct operators
    COUNTERSIGN = auto()   # ==
    CANONIZE_FWD = auto()  # =>
    CANONIZE_BWD = auto()  # <=
    CONNOTATE_FWD = auto() # >
    CONNOTATE_BWD = auto() # <
    UNDERSIGN = auto()     # =

    # Node types
    SIGNATURE = auto()     # [A-Z]+
    LITERAL = auto()       # anything not [A-Z]+

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
