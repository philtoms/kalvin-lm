"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers.

KTokenizer has two real adapters (Tokenizer, ModTokenizer) — a genuine seam.
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

    A KTokenizer converts between text and nodes. Literal testing is
    handled by is_literal_node() in the signature module — it is a
    Kalvin-level concern based on standardized bit patterns, not a
    tokenizer-specific one.

    Two adapters: Tokenizer (BPE), ModTokenizer (modular bit-packed).
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        ...

    @abstractmethod
    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode a string to a list of node IDs."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode node IDs back to a string."""
        ...
