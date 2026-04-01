from kalvin.abstract import KTokenizer

# Bit 0 is reserved for PACKED flag: 1 = packed, 0 = literal
PACKED_BIT = 1

# 64-bit Signature Allocation:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Bit 0      │ PACKED_BIT: 1=packed signature, 0=literal         │
# │ Bits 1-32  │ Character tokenization (Mod32Tokenizer default)   │
# │ Bits 33-63 │ Reserved for significance encoding (future use)   │
# └─────────────────────────────────────────────────────────────────┘

# Default alphabet with alphanumeric characters first, then common punctuation
# A-Z (26) + a-z (26) + 0-9 (10) + space + backslash + common = fits in mod64 without collision
_MOD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \\\"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_="


def _build_char_bit_maps(
    alphabet: str, modulo: int
) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-bit and bit-to-character mappings.

    Args:
        alphabet: String of characters to map (order determines bit priority)
        modulo: Number of bit positions (bits 1 to modulo-1 are used)

    Bit 0 is reserved for the PACKED flag, so character bits start at bit 1.

    Returns:
        Tuple of (char_bit, bit_char) dictionaries
    """
    char_bit: dict[str, int] = {}
    bit_char: dict[int, str] = {}

    def add_mapping(char: str, bit_value: int) -> None:
        char_bit[char] = bit_value
        if bit_value not in bit_char:
            bit_char[bit_value] = char

    # Characters map to bit positions 1 to (modulo-1), wrapping
    i = 0
    for c in alphabet:
        add_mapping(c, 1 << ((i % (modulo - 1)) + 1))
        i += 1

    # Other printable ASCII characters not in alphabet
    for j in range(32, 127):
        c = chr(j)
        if c not in char_bit:
            add_mapping(c, 1 << ((i % (modulo - 1)) + 1))
            i += 1

    return char_bit, bit_char


class ModTokenizer(KTokenizer):
    """Modular tokenizer that maps characters to bit positions.

    Characters are assigned to bit positions 1 to (modulo-1), wrapping.
    Bit 0 is reserved for PACKED_BIT to distinguish packed vs literal encoding.
    """

    def __init__(self, alphabet: str | None = None, modulo: int = 32):
        """Initialize the tokenizer.

        Args:
            alphabet: Character sequence for mapping (default: _MOD_ALPHABET)
            modulo: Number of bit positions (default: 32)
        """
        self._alphabet = alphabet if alphabet is not None else _MOD_ALPHABET
        self._modulo = modulo
        self._char_bit, self._bit_char = _build_char_bit_maps(self._alphabet, modulo)

    @property
    def alphabet(self) -> str:
        """Return the character alphabet used for tokenization."""
        return self._alphabet

    @property
    def CHAR_BIT(self) -> dict[str, int]:
        """Character to bit value mapping."""
        return self._char_bit

    @property
    def BIT_CHAR(self) -> dict[int, str]:
        """Bit value to character mapping."""
        return self._bit_char

    @property
    def vocab_size(self) -> int:
        """Return the number of unique character tokens (excluding PACKED_BIT)."""
        return len(self._bit_char)

    def is_literal(self, token_id: int) -> bool:
        return not bool(token_id & PACKED_BIT)

    def encode(self, text: str, pack: bool = True, pad_ws: bool = False) -> list[int]:
        """Encode a string to token IDs.

        Args:
            text: Input string to encode
            pack: If True, multi-char strings are packed into single token (OR-ed bits)
                  and PACKED_BIT is set. If False, returns one token per character
                  without PACKED_BIT (literal encoding).
            pad_ws: If True, strip and add trailing space

        Returns:
            List of token IDs
        """
        if pad_ws:
            text = text.strip() + " "

        if not text:
            return []

        if pack:
            token_id = PACKED_BIT
            for c in text:
                token_id |= self._char_bit[c]
            return [token_id]
        else:
            return [self._char_bit[c] for c in text]

    def decode(self, ids: list[int], pack: bool | None = None) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of token IDs
            pack: If None (default), auto-detect from PACKED_BIT in token.
                  If True, treat each ID as packed (multiple bits set).
                  If False, treat each ID as a single character.

        Returns:
            Decoded string
        """
        chars = []
        for token_id in ids:
            if pack is None:
                is_packed = bool(token_id & PACKED_BIT)
            else:
                is_packed = pack

            if is_packed:
                chars.append(self._decode_packed(token_id & ~PACKED_BIT))
            else:
                chars.append(self._bit_char[token_id])
        return "".join(chars)

    def _decode_packed(self, token_id: int) -> str:
        """Decode a packed token ID to string by finding all set bits."""
        chars = []
        for bit_pos in range(1, self.vocab_size + 1):
            bit_value = 1 << bit_pos
            if token_id & bit_value:
                char = self._bit_char.get(bit_value)
                if char:
                    chars.append(char)
        return "".join(chars)

    def batch_encode(self, texts: list[str], pack: bool = True) -> list[list[int]]:
        """Encode multiple strings."""
        return [self.encode(t, pack=pack) for t in texts]


class Mod32Tokenizer(ModTokenizer):
    """Mod32 tokenizer with 31 bit positions (bits 1-31)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=32)


class Mod64Tokenizer(ModTokenizer):
    """Mod64 tokenizer with 63 bit positions (bits 1-63)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=64)


class Mod128Tokenizer(ModTokenizer):
    """Mod128 tokenizer with 127 bit positions (bits 1-127)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=128)
