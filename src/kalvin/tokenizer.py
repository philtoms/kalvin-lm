"""BPE tokenizer wrapper using rustbpe."""

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import rustbpe
    import tiktoken

_rustbpe: Any = None
_tiktoken: Any = None
try:
    import rustbpe as _rustbpe_module

    _rustbpe = _rustbpe_module
except ImportError:
    pass
try:
    import tiktoken as _tiktoken_module

    _tiktoken = _tiktoken_module
except ImportError:
    pass


class TokenizerNotTrainedError(Exception):
    """Raised when encoding/decoding before training the tokenizer."""
    pass


class RustbpeNotInstalledError(Exception):
    """Raised when rustbpe operations are attempted without rustbpe installed."""
    pass


class TiktokenNotInstalledError(Exception):
    """Raised when loading from directory without tiktoken installed."""
    pass


class PyarrowNotInstalledError(Exception):
    """Raised when training from parquet without pyarrow installed."""
    pass


class Tokenizer:
    """BPE tokenizer wrapper supporting rustbpe (training) and tiktoken (inference)."""

    def __init__(self, tokenizer: Any = None):
        """Initialize with optional pre-trained tokenizer.

        Args:
            tokenizer: Optional rustbpe.Tokenizer or tiktoken.Encoding instance
        """
        self._tokenizer = tokenizer

    def _check_available(self) -> None:
        """Raise if tokenizer is not available."""
        if self._tokenizer is None:
            raise TokenizerNotTrainedError(
                "Tokenizer not trained. Call train() or from_directory() first."
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

    def train_from_parquet(
        self,
        paths: str | Path | list[str | Path],
        text_column: str = "text",
        vocab_size: int = 4096,
        pattern: str | None = None,
        glob_pattern: str = "*.parquet",
    ) -> None:
        """Train the BPE tokenizer from parquet files.

        Streams text data from parquet files for memory-efficient training.

        Args:
            paths: Single directory path, single file path, or list of file paths
            text_column: Name of the column containing text data
            vocab_size: Target vocabulary size (default 4096)
            pattern: Optional custom regex pattern for pre-tokenization
            glob_pattern: Glob pattern for matching files when paths is a directory
        """
        if _rustbpe is None:
            raise RustbpeNotInstalledError(
                "rustbpe is not installed. Install with: pip install rustbpe"
            )

        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise PyarrowNotInstalledError(
                "pyarrow is not installed. Install with: pip install pyarrow"
            )

        # Resolve file paths
        if isinstance(paths, (str, Path)):
            path = Path(paths)
            if path.is_dir():
                files = sorted(path.glob(glob_pattern))
            else:
                files = [path]
        else:
            files = [Path(p) for p in paths]

        if not files:
            raise ValueError(f"No parquet files found at {paths}")

        def text_iterator() -> Iterator[str]:
            for file_path in files:
                table = pq.read_table(file_path, columns=[text_column])
                for text in table[text_column].to_pylist():
                    if text is not None:
                        yield text

        self._tokenizer = _rustbpe.Tokenizer()
        self._tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=vocab_size,
            pattern=pattern,
        )

    def save_to_directory(self, path: str | Path, name: str = "tokenizer") -> None:
        """Save the trained tokenizer to a directory for later loading.

        Saves two files:
        - {name}.json: metadata (pattern, vocab_size)
        - {name}.bin: mergeable ranks (base64-encoded)

        Args:
            path: Directory path to save to
            name: Base name for saved files (default "tokenizer")
        """
        self._check_available()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Get tokenizer info
        pattern = self._tokenizer.get_pattern()
        ranks = self._tokenizer.get_mergeable_ranks()

        # Save metadata
        meta = {
            "pattern": pattern,
            "vocab_size": len(ranks),
        }
        (path / f"{name}.json").write_text(json.dumps(meta, indent=2))

        # Save mergeable ranks as base64-encoded JSON
        # Format: [{"b64": "<base64_bytes>", "rank": int}, ...]
        ranks_data = [
            {"b64": base64.b64encode(b).decode("ascii"), "rank": r}
            for b, r in ranks
        ]
        (path / f"{name}.bin").write_text(json.dumps(ranks_data))

    @classmethod
    def from_directory(cls, path: str | Path, name: str = "tokenizer") -> "Tokenizer":
        """Load a pre-trained tokenizer from a directory.

        Uses tiktoken for fast inference (recommended by rustbpe).

        Args:
            path: Directory path containing saved tokenizer
            name: Base name of saved files (default "tokenizer")

        Returns:
            Tokenizer instance with loaded tiktoken.Encoding
        """
        if _tiktoken is None:
            raise TiktokenNotInstalledError(
                "tiktoken is not installed. Install with: pip install tiktoken"
            )

        path = Path(path)

        # Load metadata
        meta = json.loads((path / f"{name}.json").read_text())
        pattern = meta["pattern"]

        # Load mergeable ranks
        ranks_data = json.loads((path / f"{name}.bin").read_text())
        mergeable_ranks = {
            base64.b64decode(item["b64"]): item["rank"] for item in ranks_data
        }

        # Create tiktoken Encoding
        encoding = _tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )

        return cls(tokenizer=encoding)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        self._check_available()
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return self._tokenizer.n_vocab

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
        if hasattr(self._tokenizer, "batch_encode"):
            return self._tokenizer.batch_encode(texts)
        # tiktoken doesn't have batch_encode, encode sequentially
        return [self._tokenizer.encode(t) for t in texts]
