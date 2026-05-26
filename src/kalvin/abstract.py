"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers, models, and agents.

KTokenizer has two real adapters (Tokenizer, ModTokenizer) — a real seam.
KModel and KAgent document the interface but have one adapter each (Model, Agent).
They exist to formalize the contract and enable future adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Any, Protocol, runtime_checkable

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


# === Abstract Model ===

class KModel(ABC):
    """Abstract base class for knowledge graph models.

    One adapter: Model (three-tier in-memory). The interface documents what
    callers (Agent, tests) rely on. Methods mirror Model's actual surface
    after the expand/misfit extraction.
    """

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
    def where(self, predicate: Any) -> list[KLine]:
        """Return KLines matching a predicate or signature overlap."""
        ...

    @abstractmethod
    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to the KLine whose signature matches."""
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
        """Promote a KLine from STM to the frame."""
        ...

    @abstractmethod
    def promote_all(self) -> int:
        """Promote all STM KLines to the frame."""
        ...

    @abstractmethod
    def klines(self) -> list[KLine]:
        """Return all KLines in reverse insertion order."""
        ...

    @abstractmethod
    def is_s1(self, kline: KLine) -> bool:
        """Determine if a kline is structurally grounded (S1)."""
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
    """Abstract base class for agents.

    One adapter: Agent. The interface documents the public contract.
    """

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
