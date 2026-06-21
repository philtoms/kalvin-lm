"""Kalvin tokenizer — BPE tokens enriched with a type word.

The kalvin tokenizer is the sole authority for how text becomes nodes.
It wraps a BPE subword engine and a type dictionary, producing 64-bit
typed nodes::

    node = (type_word << 32) | bpe_token_id

- ``type_word`` (upper 32 bits) — a significant bit pattern. Kalvin
  treats this word as opaque: it participates in signature construction
  and matching (see the @signature spec) but kalvin does not interpret
  what the bits mean. The meaning of the type word is a deployment
  concern; see the @nlp_tokenizer spec for the NLP interpretation.
- ``bpe_token_id`` (lower 32 bits) — the BPE subword vocabulary index.

The tokenizer owns the type dictionary mapping BPE token IDs to their
type word. BPE tokens without an entry fall back to ``UNKNOWN_TYPE``.

BPE training and persistence (``train``, ``save_to_directory``,
``from_directory``) are setup-time operations and work without a type
dictionary; ``encode``/``decode`` are the production operations.

See specs/tokenizer.md for the full specification.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kalvin.abstract import KTokenizer

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
    """Raised when loading from directory without tiktoken installed."""

    pass


class PyarrowNotInstalledError(Exception):
    """Raised when training from parquet without pyarrow installed."""

    pass


# The fallback type word for BPE tokens missing from the type dictionary.
# Kalvin assigns no significance to untyped tokens, so the fallback is the
# empty type word (0). A deployment may reinterpret this via a subclass
# (see kalvin.nlp_tokenizer for the NLP POS_X fallback).
UNKNOWN_TYPE = 0


class Tokenizer(KTokenizer):
    """Kalvin production tokenizer: BPE subwords combined with a type word.

    Produces 64-bit typed nodes ``(type_word << 32) | bpe_token_id`` from a
    BPE engine and a type dictionary. The type word is treated as opaque —
    kalvin operates on its bit pattern, not its meaning.
    """

    #: Fallback type word for BPE tokens absent from the type dictionary.
    #: Subclasses may override (e.g. NLP uses POS_X = 65536).
    UNKNOWN_TYPE: int = UNKNOWN_TYPE

    def __init__(
        self,
        bpe: Any = None,
        types: dict[int, dict] | None = None,
    ):
        # ``_bpe`` is the BPE engine: a rustbpe.Tokenizer (training) or a
        # tiktoken.Encoding (inference). Tooling may access it directly.
        self._bpe = bpe
        self._types: dict[int, dict] = types or {}

    # ── availability ─────────────────────────────────────────────────────

    def _check_available(self) -> None:
        if self._bpe is None:
            raise TokenizerNotTrainedError(
                "Tokenizer BPE engine not loaded. Call train() or from_directory() first."
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
    def from_directory(
        cls,
        path: str | Path | None = None,
        name: str = "tokenizer-32768",
    ) -> Tokenizer:
        """Load the BPE engine only (no type dictionary).

        Returns a tokenizer whose ``encode`` produces typed nodes with the
        fallback type word for every token. Loading a type dictionary (the
        tagged grammar) is a deployment concern handled by a subclass such
        as :class:`kalvin.nlp_tokenizer.NLPTokenizer`; use ``encode_bpe``
        for raw BPE IDs.
        """
        return cls(bpe=cls._load_bpe_engine(path, name))

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

    @property
    def type_size(self) -> int:
        """Number of entries in the type dictionary."""
        return len(self._types)

    # ── encoding (production: typed nodes) ───────────────────────────────

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to a list of typed nodes.

        Each BPE token ID is combined with its type word:
        ``(type_word << 32) | bpe_token_id``. Tokens absent from the type
        dictionary receive ``UNKNOWN_TYPE``.
        """
        self._check_available()
        bpe_ids = self.encode_bpe(text, pad_ws)
        nodes: list[int] = []
        for bpe_id in bpe_ids:
            entry = self._types.get(bpe_id)
            type_word = entry["type_word"] if entry else self.UNKNOWN_TYPE
            nodes.append((type_word << 32) | bpe_id)
        return nodes

    def encode_bpe(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to raw BPE token IDs (no type word).

        Low-level accessor for tooling that operates on the BPE vocabulary
        directly (training, tagging, decomposition).
        """
        self._check_available()
        if pad_ws:
            text = text.strip() + " "
        return self._bpe.encode(text)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode to typed nodes."""
        self._check_available()
        return [self.encode(t) for t in texts]

    # ── decoding ─────────────────────────────────────────────────────────

    def decode(self, ids: list[int]) -> str:
        """Decode typed nodes (or raw BPE IDs) back to text.

        The BPE token ID is taken from the low 32 bits of each value, so
        raw BPE IDs (which already fit in 32 bits) decode unchanged.
        """
        self._check_available()
        if not ids:
            return ""
        bpe_ids = [node & 0xFFFFFFFF for node in ids]
        return self._bpe.decode(bpe_ids)

    # ── type dictionary lookup ───────────────────────────────────────────

    def lookup_type(self, token_id: int) -> int | None:
        """Return the type word for a BPE token ID, or None if absent."""
        entry = self._types.get(token_id)
        return entry["type_word"] if entry else None

    def lookup_type_entry(self, token_id: int) -> dict | None:
        """Return the raw type-dictionary entry for a BPE token ID.

        The entry is returned as-is; its keys beyond ``type_word`` are
        opaque metadata (e.g. NLP labels when the dictionary was generated
        by NLP tooling).
        """
        return self._types.get(token_id)
