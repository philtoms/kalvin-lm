"""BPE tokenizer — a plain ``KTokenizer`` over the BPE engine.

Node layout: ``(0 << 32) | bpe_token_id`` — the upper 32 bits are zero,
so nodes are just the raw BPE token IDs. Drop-in alternative to
:class:`kalvin.nlp_tokenizer.BPETokenizer`.
"""

from __future__ import annotations

from pathlib import Path

from kalvin.abstract import KTokenizer
from kalvin.tokenizer import Tokenizer


class BPETokenizer(Tokenizer, KTokenizer):
    """A ``KTokenizer`` whose type word is always zero."""

    UNKNOWN_TYPE: int = 0

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        tokenizer_name: str = "tokenizer-32768",
    ) -> None:
        """Load the BPE engine from ``tokenizer_path`` (default: standard data dir)."""
        if tokenizer_path is None:
            from kalvin.paths import tokenizer_dir

            tokenizer_path = tokenizer_dir()
        bpe = Tokenizer._load_bpe_engine(tokenizer_path, tokenizer_name)
        super().__init__(bpe=bpe)

    @property
    def type_size(self) -> int:
        """Always 0: no type dictionary."""
        return 0

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to nodes (raw BPE token IDs; upper 32 bits zero)."""
        self._check_available()
        return self.encode_bpe(text, pad_ws)

    def decode(self, ids: list[int]) -> str:
        """Decode nodes back to text (low 32 bits hold the BPE token ID)."""
        self._check_available()
        if not ids:
            return ""
        return self.decode_bpe([node & 0xFFFFFFFF for node in ids])

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        self._check_available()
        return [self.encode(t) for t in texts]
