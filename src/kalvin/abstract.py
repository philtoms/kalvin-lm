"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers, models, and agents.

KTokenizer has two real adapters (Tokenizer, ModTokenizer) — a genuine seam.
KAgent documents the interface with one adapter (Agent).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from kalvin.kline import KLine, KNode, KSig

if TYPE_CHECKING:
    from kalvin.model import Model

__all__ = [
    "KLine", "KNode", "KSig",
    "KTokenizer", "KAgent",
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


# === KAgent ===

class KAgent(ABC):
    """Abstract base class for agents.

    One adapter: Agent. The interface documents the public contract.
    """

    @property
    @abstractmethod
    def model(self) -> Model:
        """Get the knowledge graph model."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> KTokenizer:
        """Get the tokenizer."""
        ...

    @abstractmethod
    def rationalise(self, kline: KLine) -> bool:
        """Rationalise a KLine. Returns True if significant (S1, S4)."""
        ...
