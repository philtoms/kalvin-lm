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

    The interface is layout- and type-agnostic: it converts text to an
    ordered sequence of uint64 nodes and back. It does not specify a node
    layout, a type-word concept, or any bit packing — those are
    concrete-tokenizer concerns. The production concrete `KTokenizer` is
    :class:`kalvin.nlp_tokenizer.NLPTokenizer`.
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

    @abstractmethod
    def residual(self, a: int, b: int) -> int:
        """Return the residual of signature *a* over signature *b*.

        A derived value representing what *a* carries that *b* does not. Used
        to compute coverage between a kline's signature and its nodes'
        signature. The result's representation is opaque to Kalvin; it is
        passed back to the Signifier (see :meth:`classify_misfit`) and never
        inspected directly.
        """
        ...

    @abstractmethod
    def classify_misfit(self, signature: int, nodes: Sequence[int]) -> tuple[bool, bool]:
        """Classify whether a signature faithfully covers its node set.

        Returns ``(underfit, overfit)``: whether the signature over-claims its
        nodes or the nodes over-deliver. Encapsulates the residual computation
        and its emptiness test; callers receive booleans.
        """
        ...
