"""KAgent - Base class for agents that work with KScript and knowledge graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kalvin.model import KLine, Model
    from kalvin.tokenizer import Tokenizer


class KAgent(ABC):
    """
    Abstract base class for agents that can work with KScript.

    A KAgent provides the essential interface needed by the KScript compiler:
    - A tokenizer for encoding text to tokens
    - A model for storing and retrieving KLines
    - Methods for encoding/decoding text and serialization

    Subclasses can extend this interface with additional capabilities
    (e.g., NLP features, activity tracking).
    """

    @property
    @abstractmethod
    def model(self) -> Model:
        """Get the knowledge graph model."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer for encoding/decoding text."""
        ...

    # === Core operations ===

    @abstractmethod
    def encode(self, text: str, nlp_detail: str = "nlp_type32") -> KLine | None:
        """Encode text into a KLine and add it to the model.

        Args:
            text: Input string to encode
            nlp_detail: NLP detail type for encoding

        Returns:
            KLine representing the encoded text, or None if duplicate
        """
        ...

    @abstractmethod
    def decode(self, token_sig: int | None) -> str:
        """Decode a signature key back to text.

        Args:
            token_sig: signature key to decode

        Returns:
            Decoded string
        """
        ...

    @abstractmethod
    def signify(self, k1: "KLine", k2: "KLine", s: int | None = None) -> int:
        """Establish significance relationship between two KLines.

        Calculates internal significance of k1:k2 relationship.
        If requested s is higher (more significant) than internal:
        - S1: Adds bidirectional links, returns S1
        - S2: Verifies compound signature of k2.nodes == k1.signature
        - S3: Adds bidirectional links, returns S3

        Args:
            k1: First KLine
            k2: Second KLine
            s: Optional requested significance level (S1/S2/S3 bit flags)

        Returns:
            The resulting significance value
        """
        ...

    # === Serialization ===

    @abstractmethod
    def save(
        self,
        path: str | Path,
        format: Literal["binary", "json"] = "binary",
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'binary' (default, compact) or 'json' (human-readable)
        """
        ...

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str | Path,
        format: Literal["binary", "json"] | None = None,
    ) -> "KAgent":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'binary', 'json', or None (auto-detect from extension)

        Returns:
            Loaded KAgent instance
        """
        ...
