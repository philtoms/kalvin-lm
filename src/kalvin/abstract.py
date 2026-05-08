"""Abstract base classes for Kalvin.

Provides the minimum interface contracts for tokenizers, models, and agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Any

from kalvin.kline import KLine, KNode, KSig

__all__ = [
    "KLine", "KNode", "KSig",
    "KTokenizer", "KModel", "KAgent",
]


# === Abstract Tokenizer ===

class KTokenizer(ABC):
    """Abstract base class for tokenizers.

    A KTokenizer converts between text and nodes. Literal testing is
    handled by is_literal_node() in the signature module — it is a
    Kalvin-level concern based on standardized bit patterns, not a
    tokenizer-specific one.
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


# === Abstract Model ===

class KModel(ABC):
    """Abstract base class for knowledge graph models."""

    @abstractmethod
    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine already exists in any tier."""
        ...

    @abstractmethod
    def add(self, kline: KLine, dedup: bool = False) -> bool:
        """Add a KLine. Returns True if added, False if rejected."""
        ...

    @abstractmethod
    def find(self, signature: KSig) -> KLine | None:
        """Find a KLine by signature (most recently added)."""
        ...

    @abstractmethod
    def find_all(self, signature: KSig) -> list[KLine]:
        """Find all KLines with the given signature."""
        ...

    @abstractmethod
    def find_by_nodes(self, nodes_signature: KSig) -> KLine | None:
        """Find KLine by nodes signature."""
        ...

    @abstractmethod
    def remove(self, signature: KSig) -> bool:
        """Remove the most recently added KLine with given signature."""
        ...

    @abstractmethod
    def where(self, predicate: Any) -> list[KLine]:
        """Return KLines matching a predicate."""
        ...

    @abstractmethod
    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to the KLine whose signature matches."""
        ...

    @abstractmethod
    def expand(self, kline: KLine) -> list[KLine]:
        """Expand a KLine's graph context up to *depth* levels."""
        ...

    @abstractmethod
    def descendants(self, node: int) -> set[int]:
        """Recursively collect all descendant node values."""
        ...

    @abstractmethod
    def query(self, signature: KSig, depth: int = 1) -> list[KLine]:
        """Find all KLines with signature, then expand each to depth."""
        ...

    @abstractmethod
    def promote(self, kline: KLine) -> bool:
        """Promote a KLine to the base model."""
        ...

    @abstractmethod
    def promote_all(self) -> int:
        """Promote all frame KLines to the base model."""
        ...

    @abstractmethod
    def klines(self) -> list[KLine]:
        """Return all KLines in reverse insertion order."""
        ...

    @abstractmethod
    def is_s1(self, kline: KLine) -> bool:
        """Determine if a kline is structurally grounded (S1).

        A kline is S1 if:
        1. Its signature fully describes its nodes (canonical), OR
        2. It is countersigned by another kline in the model.
        """
        ...

    @abstractmethod
    def distance(self, query: KLine, candidate: KLine, level: str) -> int:
        """Packed distance (S2 and S3 components). Level is "S2" or "S3"."""
        ...

    @abstractmethod
    def is_countersigned(self, kline: KLine) -> bool:
        """Test whether a kline is countersigned by any kline in the model."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of KLines in the frame."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[KLine]:
        """Iterate over all KLines."""
        ...

    @property
    @abstractmethod
    def base(self) -> "KModel | None":
        """Return the base model."""
        ...


# === KAgent ===

class KAgent(ABC):
    """Abstract base class for agents."""

    @property
    @abstractmethod
    def model(self) -> KModel:
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
