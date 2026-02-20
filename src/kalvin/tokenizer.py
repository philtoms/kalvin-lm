"""BPE tokenizer wrapper using rustbpe."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import rustbpe

_rustbpe: Any = None
try:
    import rustbpe as _rustbpe_module

    _rustbpe = _rustbpe_module
except ImportError:
    pass


class TokenizerNotTrainedError(Exception):
    """Raised when encoding/decoding before training the tokenizer."""
    pass


class RustbpeNotInstalledError(Exception):
    """Raised when rustbpe operations are attempted without rustbpe installed."""
    pass


class Tokenizer:
    """BPE tokenizer wrapper using rustbpe."""

    def __init__(self, tokenizer: Any = None):
        """Initialize with optional pre-trained tokenizer.

        Args:
            tokenizer: Optional rustbpe.Tokenizer instance
        """
        self._tokenizer = tokenizer

    def _check_available(self) -> None:
        """Raise if rustbpe or tokenizer is not available."""
        if _rustbpe is None:
            raise RustbpeNotInstalledError(
                "rustbpe is not installed. Install with: pip install rustbpe"
            )
        if self._tokenizer is None:
            raise TokenizerNotTrainedError(
                "Tokenizer not trained. Call train() first."
            )

    def train(
        self,
        texts: list[str],
        vocab_size: int = 4096,
        pattern: str | None = None,
    ) -> None:
        """Train the BPE tokenizer on a corpus.

        Args:
            texts: List of training strings
            vocab_size: Target vocabulary size (default 4096)
            pattern: Optional custom regex pattern for pre-tokenization
        """
        if _rustbpe is None:
            raise RustbpeNotInstalledError(
                "rustbpe is not installed. Install with: pip install rustbpe"
            )
        self._tokenizer = _rustbpe.Tokenizer()
        self._tokenizer.train_from_iterator(
            texts,
            vocab_size=vocab_size,
            pattern=pattern,
        )

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        self._check_available()
        return self._tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode a string to token IDs.

        Args:
            text: Input string to encode

        Returns:
            List of token IDs
        """
        self._check_available()
        return self._tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string
        """
        self._check_available()
        return self._tokenizer.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple strings in parallel.

        Args:
            texts: List of strings to encode

        Returns:
            List of token ID lists
        """
        self._check_available()
        return self._tokenizer.batch_encode(texts)
