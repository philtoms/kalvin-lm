"""KAgent - Base class for agents that work with KScript and knowledge graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Literal

from kalvin.kline import KLine, KNode, KNodes, KSig, KNone

# Re-export for backwards compatibility
__all__ = ["KLine", "KNode", "KNodes", "KSig", "KNone", "KSignificance", "KTokenizer", "KModel", "KAgent"]


# === Abstract Significance ===

class KSignificance(ABC):
    """Abstract base class for significance calculations.

    A KSignificance provides the interface for encoding, decoding, and
    comparing significance values between KLines.
    """

    # === Significance level constants ===
    @property
    @abstractmethod
    def S1(self) -> KSig:
        """S1 significance level (highest - prefix match)."""
        ...

    @property
    @abstractmethod
    def S2(self) -> KSig:
        """S2 significance level (partial positional match)."""
        ...

    @property
    @abstractmethod
    def S3(self) -> KSig:
        """S3 significance level (unordered/generational match)."""
        ...

    @property
    @abstractmethod
    def S4(self) -> KSig:
        """S4 significance level (no match)."""
        ...

    # === S1 operations ===

    @abstractmethod
    def has_s1(self, sig: KSig) -> bool:
        """Check if S1 bit is set (prefix match)."""
        ...

    @abstractmethod
    def get_s1_percentage(self, sig: KSig) -> int:
        """Extract S1 percentage (0-127)."""
        ...

    @abstractmethod
    def build_s1(self, percentage: int = 100) -> KSig:
        """Build S1 significance with optional percentage."""
        ...

    # === S2 operations ===

    @abstractmethod
    def get_s2(self, sig: KSig) -> int:
        """Extract full S2 value."""
        ...

    @abstractmethod
    def get_s2_s1_percentage(self, sig: KSig) -> int:
        """Extract S2's S1 percentage."""
        ...

    @abstractmethod
    def get_s2_s2_percentage(self, sig: KSig) -> int:
        """Extract S2's S2 percentage."""
        ...

    @abstractmethod
    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance."""
        ...

    # === S3 operations ===

    @abstractmethod
    def get_s3(self, sig: KSig) -> int:
        """Extract full S3 value."""
        ...

    @abstractmethod
    def get_s3_s1_percentage(self, sig: KSig) -> int:
        """Extract S3's S1 percentage for unordered matches."""
        ...

    @abstractmethod
    def get_s3_s2_percentage(self, sig: KSig) -> int:
        """Extract S3's S2 percentage for unordered matches."""
        ...

    @abstractmethod
    def get_s3_gen_percentage(self, sig: KSig) -> int:
        """Extract S3's generational S1 percentage."""
        ...

    @abstractmethod
    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance."""
        ...

    # === Significance calculation ===

    @abstractmethod
    def calculate(self, model: "KModel", query: "KLine", target: "KLine") -> KSig:
        """Calculate significance between query and target KLines.

        Args:
            model: The Model containing the KLines (for descendant lookup)
            query: The query KLine
            target: The target KLine to compare against

        Returns:
            Significance value (higher = more significant)
        """
        ...


# === Abstract Tokenizer ===

class KTokenizer(ABC):
    """Abstract base class for tokenizers.

    A KTokenizer provides the essential interface for encoding text to tokens
    and decoding tokens back to text.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        ...

    @abstractmethod
    def encode(self, text: str, pad_ws: bool = False) -> KNodes:
        """Encode a string to token IDs.

        Args:
            text: Input string to encode
            pad_ws: If True, strip and add trailing space

        Returns:
            List of token IDs (KNodes)
        """
        ...

    @abstractmethod
    def decode(self, ids: KNodes) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string
        """
        ...


# === Abstract Model ===

class KModel(ABC):
    """Abstract base class for knowledge graph models.

    A KModel provides the essential interface for storing and querying KLines.
    """

    @abstractmethod
    def add(self, kline: KLine) -> bool:
        """Add a KLine to the model.

        Args:
            kline: KLine to add

        Returns:
            True if added, False if rejected (duplicate)
        """
        ...

    @abstractmethod
    def upgrade(self, kline: KLine, significance: KSig):
        """Upgrade the significance of a kline

        Args:
            kline: KLine to upgrade
        """
        ...

    @abstractmethod
    def find_kline(self, signature: KSig, significance: KSig | None = None) -> KLine:
        """Find a KLine by its signature.

        Args:
            signature: The signature to search for
            significance: Optional significance filter

        Returns:
            KLine if found, KNone otherwise
        """
        ...

    @abstractmethod
    def find_signed_klines(self, signature: KSig) -> list[KLine]:
        """Find all KLines matching the given signature.

        Args:
            signature: The signature to search for

        Returns:
            List of matching KLines
        """
        ...

    @abstractmethod
    def query(
        self,
        query: KSig,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Query KLines by ANDing significance with a query.

        Args:
            query: The query value to match (AND operation on signature)
            focus_limit: Number of top-level matches in fast (0 = all in fast)

        Returns:
            Tuple of (fast_generator, slow_generator) that yield KLines.
        """
        ...

    @abstractmethod
    def expand(
        self,
        focus_set: list[KLine],
        depth: int = 1,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Expand KLines and their descendants up to a given depth.

        Args:
            focus_set: List of KLines to expand (e.g., from query)
            depth: Maximum recursion depth for expanding child nodes
            focus_limit: Number of klines in fast (0 = all in fast)

        Returns:
            Tuple of (fast_generator, slow_generator) that yield expanded KLines.
        """
        ...

    @abstractmethod
    def duplicate(self) -> "KModel":
        """Create a duplicate of this model.

        Returns:
            A new KModel with copied KLines
        """
        ...

    @abstractmethod
    def get_all_descendants(self, node: KNode, visited: set[KSig] | None = None) -> set[KNode]:
        """Recursively collect all descendant nodes.

        Args:
            node: The node to start from
            visited: Set of already visited nodes (cycle detection)

        Returns:
            Set of all descendant node keys
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of KLines in the model."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[KLine]:
        """Iterate over all KLines in insertion order."""
        ...


# === KAgent ===


class KAgent(ABC):
    """
    Abstract base class for agents that can work with KScript.

    A KAgent provides the essential interface needed by the KScript compiler:
    - A tokenizer for encoding text to tokens
    - A model for storing and retrieving KLines
    - A significance instance for S1-S4 operations
    - Methods for encoding/decoding text and serialization

    Subclasses can extend this interface with additional capabilities
    (e.g., NLP features, activity tracking).
    """

    @property
    @abstractmethod
    def model(self) -> KModel:
        """Get the knowledge graph model."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> KTokenizer:
        """Get the tokenizer for encoding/decoding text."""
        ...

    @property
    @abstractmethod
    def significance(self) -> "KSignificance":
        """Get the significance instance for S1-S4 operations."""
        ...

    # === Core operations ===

    @abstractmethod
    def rationalise(self, kline: KLine, frame: KModel | None = None) -> KLine | None:
        """Rationalise a KLine.

        Args:
            kline: KLine to rationalize
            frame: existing model context    
        """
        ...

    @abstractmethod
    def encode(self, text: str, nlp_detail: str) -> KLine | None:
        """Encode text into a KLine and add it to the model.

        Args:
            text: Input string to encode
            nlp_detail: NLP detail level for type encoding

        Returns:
            KLine representing the encoded text, or None if duplicate
        """
        ...

    @abstractmethod
    def decode(self, token_sig: KNode) -> str:
        """Decode a signature key back to text.

        Args:
            token_sig: signature key to decode

        Returns:
            Decoded string
        """
        ...

    @abstractmethod
    def signify(self, k1: KLine, k2: KLine, S: KSig | None = None) -> KSig:
        """Establish significance relationship between two KLines.

        Calculates internal significance of k1:k2 relationship.
        If requested s is higher (more significant) than internal:
        - S1: Adds bidirectional links, returns S1
        - S2: Verifies compound signature of k2.nodes == k1.signature
        - S3: Adds bidirectional links, returns S3

        Args:
            k1: First KLine
            k2: Second KLine
            S: Optional requested significance level (S1/S2/S3 bit flags)

        Returns:
            The resulting significance value
        """
        ...

    # === Serialization ===

    @abstractmethod
    def save(
        self,
        path: str | Path,
        format: Literal["bin", "json"] = "bin",
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'bin' (default, compact) or 'json' (human-readable)
        """
        ...

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str | Path,
        format: Literal["bin", "json"] | None = None,
    ) -> "KAgent":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'bin', 'json', or None (auto-detect from extension)

        Returns:
            Loaded KAgent instance
        """
        ...
