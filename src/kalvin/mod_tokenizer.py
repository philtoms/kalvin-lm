"""Mod Tokenizer — modular bit-packed encoding.

Characters map to bit positions. All-uppercase-alpha strings are OR'd into
a single node with bit 0 clear (packed encoding).

See specs/tokenizer.md for the full specification.
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer

# ── Default alphabet ──────────────────────────────────────────────────────
# All printable ASCII (codes 32–126), order determines bit priority.
_MOD_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " \\\"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_="
)


def _build_char_bit_maps(alphabet: str, modulo: int) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-bit and bit-to-character mappings.

    Bit 0 is not used for packed nodes (it is clear). Character bits
    start at bit 1 and wrap at (modulo - 1).
    """
    char_bit: dict[str, int] = {}
    bit_char: dict[int, str] = {}

    def add_mapping(char: str, bit_value: int) -> None:
        char_bit[char] = bit_value
        if bit_value not in bit_char:
            bit_char[bit_value] = char

    i = 0
    for c in alphabet:
        add_mapping(c, 1 << ((i % (modulo - 1)) + 1))
        i += 1

    # Other printable ASCII not in alphabet
    for j in range(32, 127):
        c = chr(j)
        if c not in char_bit:
            add_mapping(c, 1 << ((i % (modulo - 1)) + 1))
            i += 1

    return char_bit, bit_char


class ModTokenizer(KTokenizer):
    """Modular tokenizer that maps characters to bit positions."""

    def __init__(self, alphabet: str | None = None, modulo: int = 32):
        self._alphabet = alphabet if alphabet is not None else _MOD_ALPHABET
        self._modulo = modulo
        self._char_bit, self._bit_char = _build_char_bit_maps(self._alphabet, modulo)

    @property
    def supports_mcs(self) -> bool:
        """Mod tokenizers support MCS expansion (bit-packed character decomposition)."""
        return True

    @property
    def alphabet(self) -> str:
        return self._alphabet

    @property
    def CHAR_BIT(self) -> dict[str, int]:
        return self._char_bit

    @property
    def BIT_CHAR(self) -> dict[int, str]:
        return self._bit_char

    @property
    def vocab_size(self) -> int:
        return len(self._bit_char)

    # ── Encode ────────────────────────────────────────────────────────

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to a list of nodes.

        Encoding mode is determined automatically:
        - All-uppercase-alpha strings → packed (single node, bit 0 clear)
        - Everything else → literal (one node per character, literal mask)

        Args:
            text: Input string to encode.
            pad_ws: If True, strip and add trailing space.

        Returns:
            List of uint64 node values.
        """
        if pad_ws:
            text = text.strip() + " "

        if not text:
            return []

        if text.isupper() and text.isalpha():
            # Packed encoding: OR all character bits into single node
            token_id = 0
            for c in text:
                token_id |= self._char_bit[c]
            return [token_id]
        else:
            # Non-uppercase strings: encode each character as packed
            return [(ord(c) << 32) | 0xFFFFFFFF for c in text]

    # ── Decode ────────────────────────────────────────────────────────

    def decode(self, ids: list[int]) -> str:
        """Decode node IDs back to a string."""
        chars = []
        for node in ids:
            if (node & 0xFFFFFFFF) == 0xFFFFFFFF:
                # Character-level node
                chars.append(chr(node >> 32))
            else:
                chars.append(self._decode_packed(node))
        return "".join(chars)

    def _decode_packed(self, token_id: int) -> str:
        """Decode a packed node by finding all set bits."""
        chars = []
        for bit_pos in range(1, self.vocab_size + 1):
            bit_value = 1 << bit_pos
            if token_id & bit_value:
                char = self._bit_char.get(bit_value)
                if char:
                    chars.append(char)
        return "".join(chars)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]


class Mod32Tokenizer(ModTokenizer):
    """Mod32 tokenizer — 31 character bit positions (bits 1–31)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=32)


class Mod64Tokenizer(ModTokenizer):
    """Mod64 tokenizer — 63 character bit positions (bits 1–63)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=64)
