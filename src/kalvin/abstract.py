"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers.

KTokenizer has one production adapter (Tokenizer); subclasses may
specialise the sig-word interpretation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kalvin.kline import KLine, KNode, KSig

__all__ = [
    "KLine",
    "KNode",
    "KSig",
    "KTokenizer",
]


# === Abstract Tokenizer ===


class KTokenizer(ABC):
    """Abstract base class for tokenizers.

    A KTokenizer converts between text and nodes. All strings go through
    the tokenizer — no branching between encoding paths.

    The production adapter is :class:`kalvin.tokenizer.Tokenizer`, which
    packs a sig word into the upper 32 bits of each node and a BPE token
    ID into the lower 32. Subclasses may specialise the sig word's
    interpretation.
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

    def lookup_type(self, token_id: int) -> int | None:
        """Return the sig word for a BPE token ID, or None if absent.

        The sig word occupies the upper 32 bits of a node. Its meaning is
        opaque to kalvin (see the NLP specialisation for one interpretation).
        """
        return None

    def lookup_type_entry(self, token_id: int) -> dict | None:
        """Return the raw type-dictionary entry for a BPE token ID, or None.

        The entry carries at least a ``sig_word`` key; any further keys are
        opaque metadata supplied by whatever generated the dictionary.
        """
        return None
