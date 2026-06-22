"""Abstract base classes for Kalvin.

Provides interface contracts for tokenizers and signifiers.

KTokenizer has one production adapter (Tokenizer); subclasses may
specialise the sig-word interpretation. KSignifier has one production
implementation (NLPSignifier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from kalvin.kline import KLine, KNode, KSig

__all__ = [
    "KLine",
    "KNode",
    "KSig",
    "KSignifier",
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


# === Abstract Signifier ===


class KSignifier(ABC):
    """Abstract base class for signifiers.

    A KSignifier produces signature values from node values and relates
    signature values to each other. It is the sole authority over signature
    creation and matching; Kalvin treats node and signature values as opaque
    outside a Signifier implementation.

    The interface specifies **role and types only**. It does not require the
    operations to be deterministic, commutative, order-sensitive, or
    identity-preserving, and it does not prescribe any signature value
    (including the empty-set signature). All such characteristics are
    concrete-signifier properties; see :class:`kalvin.signifier.NLPSignifier`
    for the production implementation (the NLP masked bit-algebra).
    """

    @abstractmethod
    def make_signature(self, nodes: Sequence[int]) -> int:
        """Produce the signature value for a node sequence.

        The returned value occupies a kline's head position; the system
        indexes and retrieves klines by it.
        """
        ...

    @abstractmethod
    def signifies(self, a: int, b: int) -> bool:
        """Report whether two signature values are worth evaluating as a pair.

        A candidate-admission pre-filter used ahead of the significance
        computation in the rationalisation pipeline; not the final relevance
        determination.
        """
        ...
