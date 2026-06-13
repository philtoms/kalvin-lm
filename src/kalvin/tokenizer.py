"""BPE tokenizer wrapper using rustbpe.

Encodes text as BPE subword tokens with optional NLP type prefixes.
"""

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kalvin.abstract import KTokenizer

if TYPE_CHECKING:
    pass

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


class Tokenizer(KTokenizer):
    """BPE tokenizer wrapper supporting rustbpe (training) and tiktoken (inference)."""

    def __init__(self, tokenizer: Any = None):
        self._tokenizer = tokenizer

    def _check_available(self) -> None:
        if self._tokenizer is None:
            raise TokenizerNotTrainedError(
                "Tokenizer not trained. Call train() or from_directory() first."
            )

    def train(
        self,
        texts_iterator: Iterator,
        vocab_size: int = 32768,
        pattern: str | None = None,
    ) -> None:
        if _rustbpe is None:
            raise RustbpeNotInstalledError(
                "rustbpe is not installed. Install with: pip install rustbpe"
            )
        self._tokenizer = _rustbpe.Tokenizer()
        self._tokenizer.train_from_iterator(
            texts_iterator,
            vocab_size=vocab_size,
            pattern=pattern,
        )

    def save_to_directory(self, path: str | Path, name: str = "tokenizer") -> None:
        self._check_available()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        pattern = self._tokenizer.get_pattern()
        ranks = self._tokenizer.get_mergeable_ranks()

        meta = {"pattern": pattern, "vocab_size": len(ranks)}
        (path / f"{name}.json").write_text(json.dumps(meta, indent=2))

        ranks_data = [{"b64": base64.b64encode(b).decode("ascii"), "rank": r} for b, r in ranks]
        (path / f"{name}.bin").write_text(json.dumps(ranks_data))

    @classmethod
    def from_directory(
        cls, path: str | Path | None = None, name: str = "tokenizer-32768"
    ) -> "Tokenizer":
        if _tiktoken is None:
            raise TiktokenNotInstalledError(
                "tiktoken is not installed. Install with: pip install tiktoken"
            )
        if path is None:
            from kalvin.paths import tokenizer_dir

            path = tokenizer_dir()
        path = Path(path)
        meta = json.loads((path / f"{name}.json").read_text())
        pattern = meta["pattern"]

        ranks_data = json.loads((path / f"{name}.bin").read_text())
        mergeable_ranks = {base64.b64decode(item["b64"]): item["rank"] for item in ranks_data}

        encoding = _tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )
        return cls(tokenizer=encoding)

    @property
    def vocab_size(self) -> int:
        self._check_available()
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return self._tokenizer.n_vocab

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        self._check_available()
        if pad_ws:
            text = text.strip() + " "
        return self._tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        self._check_available()
        return self._tokenizer.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        self._check_available()
        if hasattr(self._tokenizer, "batch_encode"):
            return self._tokenizer.batch_encode(texts)
        return [self._tokenizer.encode(t) for t in texts]
