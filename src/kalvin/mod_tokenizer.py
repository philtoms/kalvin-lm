from kalvin.abstract import KTokenizer

mod_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789\"',.;:!?\n\t/%{}[]()<>#$@£^&*+-_=abcdefghijklmnopqrstuvwxyz"

def _build_char_bit_maps(modulo: int = 32) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-bit and bit-to-character mappings.

    mod 32:
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

    # Uppercase letters A-Z, Digits 0-9, Lowercase letters a-z: positions 0-31 mod 32 (highest priority for reverse mapping)
    for i, c in enumerate(mod_alphabet):
        add_mapping(c, 1 << (i % modulo))

    # Other printable ASCII characters
    for j in range(32, 127):
        c = chr(j)
        if c not in char_bit:
            add_mapping(c, 1 << (i % modulo))
            i+=1

    return char_bit, bit_char


class ModTokenizer(KTokenizer):

    CHAR_BIT, BIT_CHAR = _build_char_bit_maps(32)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique tokens in the vocabulary."""
        return len(self.BIT_CHAR)

    def encode(self, text: str, pack: bool = True, pad_ws: bool = False) -> list[int]:
        """Encode a string to token IDs.

        Args:
            text: Input string to encode
            pack: If True, multi-char strings are packed into single token (OR-ed bits).
                  If False, returns one token per character.
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
            token_id = 0
            for c in text:
                token_id |= self.CHAR_BIT[c]
            return [token_id]
        else:
            # Return one token per character
            return [self.CHAR_BIT[c] for c in text]

    def decode(self, ids: list[int], pack: bool = True) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of token IDs
            pack: If True, treat each ID as packed (multiple bits set).
                  If False, treat each ID as a single character.

        Returns:
            Decoded string
        """
        if pack:
            # Each ID may have multiple bits set - decode all
            chars = []
            for token_id in ids:
                chars.append(self._decode_packed(token_id))
            return "".join(chars)
        else:
            # Each ID is a single character
            return "".join(self.BIT_CHAR[i] for i in ids)

    def _decode_packed(self, token_id: int) -> str:
        """Decode a packed token ID to string by finding all set bits."""
        chars = []
        for bit_pos in range(self.vocab_size):
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