"""Token types and Token dataclass for KScript lexer."""

from dataclasses import dataclass
from enum import Enum, auto


def _build_char_bit_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-bit and bit-to-character mappings.

    Uses modulo 32 bit positions:
    - Letters A-Z: positions 0-25
    - Digits 0-9: positions 26-35 (wraps: 26-31, then 0-4)
    - Other printable ASCII: positions based on ord() % 32

    Returns:
        Tuple of (char_bit, bit_char) dictionaries
    """
    char_bit: dict[str, int] = {}
    bit_char: dict[int, str] = {}

    # Helper to add mapping, prioritizing uppercase letters for reverse mapping
    def add_mapping(char: str, bit_value: int) -> None:
        char_bit[char] = bit_value
        # Only set reverse mapping if not already set (prioritize earlier additions)
        if bit_value not in bit_char:
            bit_char[bit_value] = char

    # Uppercase letters A-Z: positions 0-25 (highest priority for reverse mapping)
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        add_mapping(c, 1 << i)

    # Digits 0-9: positions 26-35 (mod 32 wraps at position 32)
    for i, c in enumerate("0123456789"):
        add_mapping(c, 1 << ((i + 26) % 32))

    # Lowercase letters a-z: same positions as uppercase
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        add_mapping(c, 1 << i)

    # Other printable ASCII characters
    for i in range(32, 127):
        c = chr(i)
        if c not in char_bit:
            add_mapping(c, 1 << (i % 32))

    return char_bit, bit_char


CHAR_BIT, BIT_CHAR = _build_char_bit_maps()


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


def encode_mod(char: str) -> int:
    """Encode a single character to its modulo 32 bit value.

    Args:
        char: Single character to encode

    Returns:
        Bit value from CHAR_BIT mapping
    """
    upper_char = char.upper()
    return CHAR_BIT[upper_char]

def decode_mod(bit: int) -> str:
    """Decode a bit value to its character.

    Args:
        bit: Bit value to decode

    Returns:
        Character from BIT_CHAR mapping
    """
    return BIT_CHAR[bit]
