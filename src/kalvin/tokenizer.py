"""BPE engine wrapper — the foundation of the kalvin production tokenizer.

This module wraps a BPE subword engine and exposes raw BPE↔text operations.
It is **not** a `KTokenizer` (see :mod:`kalvin.abstract`): a BPE engine
produces raw vocabulary indices, not nodes. The production `KTokenizer` is
:class:`kalvin.nlp_tokenizer.NLPTokenizer`, which builds on this wrapper
and adds the node packing, type dictionary, and ``encode``/``decode``
(see the @nlp_tokenizer spec).

Engine surface:

- ``train`` / ``save_to_directory`` / ``_load_bpe_engine`` — vocabulary
  training and persistence (setup-time).
- ``encode_bpe`` / ``decode_bpe`` — raw text ↔ BPE token IDs.
- ``vocab_size`` — the BPE vocabulary size.

See specs/tokenizer.md (§BPE Engine) and specs/nlp_tokenizer.md
(§BPE Engine Foundation) for the full specification.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Optional BPE backends. rustbpe trains; tiktoken runs inference.
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
    """Raised when encoding/decoding before the BPE engine is loaded."""

    pass


class RustbpeNotInstalledError(Exception):
    """Raised when rustbpe operations are attempted without rustbpe installed."""

    pass


class TiktokenNotInstalledError(Exception):
    """Raised when loading the engine without tiktoken installed."""

    pass


class PyarrowNotInstalledError(Exception):
    """Raised when training from parquet without pyarrow installed."""

    pass


class Tokenizer:
    """BPE-engine wrapper: raw text ↔ BPE token IDs.

    Not a ``KTokenizer``. The production ``KTokenizer``
    (:class:`kalvin.nlp_tokenizer.NLPTokenizer`) builds on this wrapper,
    adding node packing and the type dictionary.
    """

    def __init__(self, bpe: Any = None) -> None:
        # ``_bpe`` is the BPE engine: a rustbpe.Tokenizer (training) or a
        # tiktoken.Encoding (inference). Tooling may access it directly.
        self._bpe = bpe

    # ── availability ─────────────────────────────────────────────────────

    def _check_available(self) -> None:
        if self._bpe is None:
            raise TokenizerNotTrainedError(
                "BPE engine not loaded. Call train() or load an engine first."
            )

    # ── training / persistence (BPE engine) ──────────────────────────────

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
        self._bpe = _rustbpe.Tokenizer()
        self._bpe.train_from_iterator(
            texts_iterator,
            vocab_size=vocab_size,
            pattern=pattern,
        )

    def save_to_directory(self, path: str | Path, name: str = "tokenizer") -> None:
        self._check_available()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        pattern = self._bpe.get_pattern()
        ranks = self._bpe.get_mergeable_ranks()

        meta = {"pattern": pattern, "vocab_size": len(ranks)}
        (path / f"{name}.json").write_text(json.dumps(meta, indent=2))

        ranks_data = [{"b64": base64.b64encode(b).decode("ascii"), "rank": r} for b, r in ranks]
        (path / f"{name}.bin").write_text(json.dumps(ranks_data))

    @classmethod
    def _load_bpe_engine(
        cls,
        path: str | Path | None,
        name: str,
    ) -> Any:
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

        return _tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )

    # ── properties ───────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        self._check_available()
        if hasattr(self._bpe, "vocab_size"):
            return self._bpe.vocab_size
        return self._bpe.n_vocab

    # ── raw BPE ↔ text ───────────────────────────────────────────────────

    def encode_bpe(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to raw BPE token IDs (no node packing).

        Engine-level accessor. The production tokenizer's ``encode`` builds
        on this (see :class:`kalvin.nlp_tokenizer.NLPTokenizer`).
        """
        self._check_available()
        if pad_ws:
            text = text.strip() + " "
        return self._bpe.encode(text)

    def decode_bpe(self, bpe_ids: list[int]) -> str:
        """Decode raw BPE token IDs back to text.

        Engine-level accessor, symmetric to :meth:`encode_bpe`. The
        production tokenizer's ``decode`` unpacks nodes then delegates here.
        """
        self._check_available()
        if not bpe_ids:
            return ""
        return self._bpe.decode(bpe_ids)
