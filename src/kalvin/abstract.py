"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers.

KTokenizer has three real adapters (Tokenizer, ModTokenizer, NLPTokenizer) — a genuine seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kalvin.kline import KLine, KNode, KSig

__all__ = [
    "KLine", "KNode", "KSig",
    "KTokenizer",
]


# === Abstract Tokenizer ===

class KTokenizer(ABC):
    """Abstract base class for tokenizers.

    A KTokenizer converts between text and nodes. All strings go through
    the tokenizer — no branching between encoding paths.

    Three adapters: Tokenizer (BPE), ModTokenizer (modular bit-packed), NLPTokenizer (NLP-enhanced BPE).
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        ...

    @property
    def supports_mcs(self) -> bool:
        """Whether this tokenizer supports multi-character signature (MCS) expansion.

        Mod tokenizers return True — uppercase identifiers are packed as bit
        positions, so MCS expansion decomposes them into per-character entries.
        NLP-BPE and BPE tokenizers return False — identifiers are BPE tokens,
        not character compositions.

        Default: False. Override in ModTokenizer.
        """
        return False

    @abstractmethod
    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode a string to a list of node IDs."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode node IDs back to a string."""
        ...

    def lookup_grammar(self, token_id: int) -> dict | None:
        """Look up NLP grammar info for a token ID.

        Returns None by default. NLPTokenizer overrides to return
        the grammar dict entry for a BPE token ID.
        """
        return None
