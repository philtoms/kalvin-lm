"""Mod Tokenizer — modular bit-packed encoding.

Characters map to bit positions. Two encoding modes:
  - Packed: multi-char strings OR'd into single node, bit 0 clear.
  - Literal: one node per character, upper 32 bits = code point,
    lower 32 bits = 0xFFFFFFFF (literal mask).

See openspec/tokenizer.md for the full specification.
"""

from __future__ import annotations

from typing import Any

from kalvin.abstract import KTokenizer

# ── Literal mask ──────────────────────────────────────────────────────────
# Lower 32 bits all set — unambiguous discriminator for literal nodes.
LITERAL_MASK = 0xFFFF_FFFF

# ── Default alphabet ──────────────────────────────────────────────────────
# All printable ASCII (codes 32–126), order determines bit priority.
_MOD_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " \\\"',.;:!?/\n\t%{}[]()<>#$@£^&*+-_="
)


def _build_char_bit_maps(
    alphabet: str, modulo: int
) -> tuple[dict[str, int], dict[int, str]]:
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

    # ── Literal test ──────────────────────────────────────────────────

    def is_literal(self, node: int) -> bool:
        """Return True if node is a literal token (lower 32 bits all set)."""
        return (node & LITERAL_MASK) == LITERAL_MASK

    # ── Encode ────────────────────────────────────────────────────────

    def encode(self, text: str | int, pack: bool = True, pad_ws: bool = False) -> list[int]:
        """Encode text to a list of nodes.

        Args:
            text: Input string or integer to encode.
            pack: If True, multi-char strings are OR'd into a single packed
                  node. If False, each character becomes a literal node.
            pad_ws: If True, strip and add trailing space.

        Returns:
            List of uint64 node values.
        """
        if pad_ws and isinstance(text, str):
            text = text.strip() + " "

        if not text:
            return []

        if pack and isinstance(text, str):
            token_id = 0
            for c in text:
                token_id |= self._char_bit[c]
            return [token_id]
        elif isinstance(text, str):
            return [(ord(c) << 32) | LITERAL_MASK for c in text]
        else:
            # Raw integer literal encoding
            return [(text << 32) | LITERAL_MASK]

    # ── Decode ────────────────────────────────────────────────────────

    def decode(self, ids: list[int], pack: bool | None = None) -> str:
        """Decode node IDs back to a string.

        Auto-detects literal vs packed from the literal mask unless
        *pack* is explicitly True or False.
        """
        chars = []
        for node in ids:
            if pack is None:
                is_packed = not ((node & LITERAL_MASK) == LITERAL_MASK)
            else:
                is_packed = pack

            if is_packed:
                chars.append(self._decode_packed(node))
            else:
                chars.append(chr(node >> 32))
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

    def batch_encode(self, texts: list[str], pack: bool = True) -> list[list[int]]:
        return [self.encode(t, pack=pack) for t in texts]


class Mod32Tokenizer(ModTokenizer):
    """Mod32 tokenizer — 31 character bit positions (bits 1–31)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=32)


class Mod64Tokenizer(ModTokenizer):
    """Mod64 tokenizer — 63 character bit positions (bits 1–63)."""

    def __init__(self, alphabet: str | None = None):
        super().__init__(alphabet=alphabet, modulo=64)
