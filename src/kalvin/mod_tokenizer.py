from kalvin.abstract import KTokenizer

# Bit 0 is reserved for PACKED flag: 1 = packed, 0 = literal
PACKED_BIT = 1

# 64-bit Signature Allocation:
# ┌─────────────────────────────────────────────────────────────────┐
# │ Bit 0      │ PACKED_BIT: 1=packed signature, 0=literal         │
# │ Bits 1-32  │ Character tokenization (Mod32Tokenizer default)   │
# │ Bits 33-63 │ Reserved for significance encoding (future use)   │
# └─────────────────────────────────────────────────────────────────┘

# Alphabet with alphanumeric characters first, then common punctuation including backslash
# A-Z (26) + a-z (26) + 0-9 (10) + space + backslash + common = fits in mod64 without collision
# Backslash placed early to ensure it doesn't collide with alphanumeric
mod_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \\\"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_="

def _build_char_bit_maps(modulo: int = 32) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-bit and bit-to-character mappings.

    Bit 0 is reserved for the PACKED flag, so character bits start at bit 1.

    mod 32 (with PACKED_BIT):
        Uses modulo 32 bit positions shifted by 1:
        - Letters A-Z: positions 1-26
        - Digits 0-9: positions 27-36 (wraps: 27-31, then 1-5)
        - Other printable ASCII: positions based on (ord() % 32) + 1

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

    # Uppercase letters A-Z, Digits 0-9, Lowercase letters a-z: positions 1-32 mod 32 (highest priority for reverse mapping)
    # Shift by 1 to reserve bit 0 for PACKED flag
    for i, c in enumerate(mod_alphabet):
        add_mapping(c, 1 << ((i % modulo) + 1))

    # Other printable ASCII characters
    for j in range(32, 127):
        c = chr(j)
        if c not in char_bit:
            add_mapping(c, 1 << ((i % modulo) + 1))
            i+=1

    return char_bit, bit_char


class ModTokenizer(KTokenizer):

    CHAR_BIT, BIT_CHAR = _build_char_bit_maps(32)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique character tokens (excluding PACKED_BIT)."""
        return len(self.BIT_CHAR)

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
            # Pack all characters into a single token by OR-ing their bits
            # Set PACKED_BIT to indicate packed encoding
            token_id = PACKED_BIT
            for c in text:
                token_id |= self.CHAR_BIT[c]
            return [token_id]
        else:
            # Return one token per character (literal encoding)
            # Do NOT set PACKED_BIT for literals
            return [self.CHAR_BIT[c] for c in text]

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
                # Auto-detect: check PACKED_BIT
                is_packed = bool(token_id & PACKED_BIT)
            else:
                is_packed = pack

            if is_packed:
                # Remove PACKED_BIT before decoding
                chars.append(self._decode_packed(token_id & ~PACKED_BIT))
            else:
                # Literal: single character
                chars.append(self.BIT_CHAR[token_id])
        return "".join(chars)

    def _decode_packed(self, token_id: int) -> str:
        """Decode a packed token ID to string by finding all set bits.

        Note: token_id should have PACKED_BIT already removed.
        Character bits start at position 1 (bit 0 is reserved for PACKED_BIT).
        """
        chars = []
        # Scan bits starting from position 1 (skip PACKED_BIT at position 0)
        for bit_pos in range(1, self.vocab_size + 1):
            bit_value = 1 << bit_pos
            if token_id & bit_value:
                char = self.BIT_CHAR.get(bit_value)
                if char:
                    chars.append(char)
        return "".join(chars)

    def batch_encode(self, texts: list[str], pack: bool = True) -> list[list[int]]:
        """Encode multiple strings in parallel.

        Args:
            texts: List of strings to encode
            pack: If True, pack multi-char strings into single tokens

        Returns:
            List of token ID lists
        """
        return [self.encode(t, pack=pack) for t in texts]

class Mod32Tokenizer(ModTokenizer):
    """Mod32 tokenizer"""

    CHAR_BIT, BIT_CHAR = _build_char_bit_maps(32)

class Mod64Tokenizer(ModTokenizer):
    """Mod64 tokenizer"""

    CHAR_BIT, BIT_CHAR = _build_char_bit_maps(64)

class Mod128Tokenizer(ModTokenizer):
    """Mod128 tokenizer"""

    CHAR_BIT, BIT_CHAR = _build_char_bit_maps(128)