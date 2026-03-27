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

    Literals:
        SIGNATURE:      [A-Z]+    (uppercase identifier)
        STRING_LITERAL: [a-zA-Z0-9]+ (not all uppercase, unquoted)
        STRING:         "..."     (double-quoted string)
        NUMBER:         [0-9]+    (numeric literal)
        COMMENT:        (...)     (parenthesized comment)

    Structure:
        NEWLINE: \\n       (line ending)
        INDENT:   -        (increased indentation)
        DEDENT:   -        (decreased indentation)
        EOF:      -        (end of file)
    """

    # Construct operators
    COUNTERSIGN = auto()   # ==
    CANONIZE_FWD = auto()  # =>
    CANONIZE_BWD = auto()  # <=
    CONNOTATE_FWD = auto() # >
    CONNOTATE_BWD = auto() # <
    UNDERSIGN = auto()     # =

    # Literals
    SIGNATURE = auto()      # [A-Z]+
    STRING_LITERAL = auto() # [a-zA-Z0-9]+ (not all uppercase)
    STRING = auto()         # "..."
    NUMBER = auto()         # [0-9]+
    COMMENT = auto()        # (...)

    # Structure
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
