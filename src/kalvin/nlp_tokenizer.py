"""NLP tokenizer — the kalvin production tokenizer.

This is the **sole concrete ``KTokenizer``** (see :mod:`kalvin.abstract`).
It combines a BPE subword base (the :class:`kalvin.tokenizer.Tokenizer`
engine wrapper) with an **NLP type word** to form 64-bit typed nodes::

    node = (nlp_type32 << 32) | bpe_token_id

- ``nlp_type32`` (upper 32 bits) — a 32-bit NLP type bitmask (POS / DEP /
  MORPH). This is the NLP layout; kalvin operates on the bit pattern, not
  its meaning.
- ``bpe_token_id`` (lower 32 bits) — the BPE subword vocabulary index.

Everything the ``KTokenizer`` interface does not specify — node packing,
the type dictionary, ``encode``/``decode``, and the BPE-engine foundation
— **lives here**. The base ``Tokenizer`` is a BPE-engine wrapper and is
not itself a ``KTokenizer``.

BPE tokens without a type-dictionary entry fall back to ``POS_X``
(``UNKNOWN_NLP_TYPE = 65536 = 1 << 16``), the NLP "unknown" POS flag.

See specs/nlp_tokenizer.md for the full specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kalvin.abstract import KTokenizer
from kalvin.tokenizer import Tokenizer

# POS_X is the NLP fallback type word for BPE tokens missing from the type
# dictionary. Value 65536 = 0x10000 = bit 16 set (the POS_X flag), so
# untyped tokens still carry a valid NLP POS flag.
UNKNOWN_NLP_TYPE = 65536

#: The **compound marker token**. Appended by the compiler to the nodes of a
#: §11.3 compound-word kline — a single word (e.g. ``Mary``) that the
#: external tokenizer splits into multiple BPE subwords (``[M, ary]``) —
#: so the kline is built as ``Mary: [COMPOUND_TOKEN, M, ary]``. The token
#: participates in the signature algebra like any other node (its type
#: word contributes to ``make_signature``), so a compound's signature
#: *encodes* the marker without any masking: ``signature ==
#: make_signature([COMPOUND_TOKEN, M, ary])``. Detection is structural —
#: ``COMPOUND_TOKEN in kline.nodes`` — with no bit-twiddling and no
#: special-case masking in the signifier.
#:
#: The token's type word occupies type-word bit 17 (``0x0002_0000``) — the
#: sole free slot in the NLP type-word layout (every other bit is assigned
#: by ``dev/nlp`` tooling). Its BPE id is 0 (the token carries no BPE
#: component; it is a pure structural marker). The token is confined to the
#: kalvin↔NLP boundary: defined here, set only by ``ks/token_encoder.py``,
#: and read only by ``kalvin/kline.py``'s compound predicate.
COMPOUND_TOKEN_TYPE_WORD = 0x0002_0000
COMPOUND_TOKEN: int = (COMPOUND_TOKEN_TYPE_WORD << 32) | 0


def load_grammar_dict(path: str | Path) -> dict[int, dict]:
    """Load an NLP grammar dictionary from JSON.

    Reads entries keyed by BPE token ID (string keys converted to int).
    Each entry's ``nlp_type32`` field (the NLP type word produced by
    ``dev/nlp`` tooling) is normalised onto the generic ``sig_word`` key
    so the entry can feed the encoder. Other NLP fields
    (``pos``, ``pos_fine``, ``dep``, ``morph``, …) are preserved unchanged.

    Args:
        path: Path to an NLP grammar JSON file.

    Returns:
        Dictionary mapping integer BPE token IDs to grammar entries.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    normalized: dict[int, dict] = {}
    for k, entry in data.items():
        normalized[int(k)] = dict(entry)
    return normalized


class NLPTokenizer(Tokenizer, KTokenizer):
    """The production kalvin tokenizer: the sole concrete ``KTokenizer``.

    Builds on the :class:`kalvin.tokenizer.Tokenizer` BPE-engine wrapper,
    adding node packing, the NLP type dictionary, and the
    ``KTokenizer`` interface (``encode`` / ``decode`` / ``vocab_size``).

    The fallback type word is ``POS_X`` (``UNKNOWN_NLP_TYPE = 65536``), so
    untyped BPE tokens still carry a valid NLP POS flag.
    """

    #: NLP fallback type word: POS_X (bit 16).
    UNKNOWN_TYPE: int = UNKNOWN_NLP_TYPE

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        tokenizer_name: str = "tokenizer-32768",
    ) -> None:
        """Load the BPE engine and NLP type dictionary from standard paths.

        Instantiating ``NLPTokenizer()`` reads the BPE engine (via the base
        ``Tokenizer`` machinery) and the tagged-grammar type dictionary from
        ``tokenizer_path``. When ``tokenizer_path`` is *None* (the default),
        it is resolved via ``kalvin.paths.tokenizer_dir()``
        (``KALVIN_DATA_DIR`` env var or ``data/tokenizer`` relative to the
        project root).

        Args:
            tokenizer_path: Directory holding the BPE + grammar files.
                Defaults to ``kalvin.paths.tokenizer_dir()``.
            tokenizer_name: Base name of the tokenizer files (no extension).
        """
        if tokenizer_path is None:
            from kalvin.paths import tokenizer_dir

            tokenizer_path = tokenizer_dir()
        bpe = Tokenizer._load_bpe_engine(tokenizer_path, tokenizer_name)
        super().__init__(bpe=bpe)
        self._types: dict[int, dict] = NLPTokenizer._load_type_dictionary(
            tokenizer_path, tokenizer_name
        )

    @classmethod
    def _load_type_dictionary(
        cls,
        path: str | Path,
        name: str,
    ) -> dict[int, dict]:
        """Load the NLP type dictionary (tagged grammar) for ``name``.

        Reads ``{name}_tagged_grammar.json`` (generated by
        ``dev/nlp/tag_vocab.py``). Each entry maps a BPE token ID to a dict
        carrying at least a ``sig_word`` key; other keys (``pos``,
        ``pos_fine``, ``dep``, ``morph``, …) are preserved unchanged as
        opaque NLP metadata.
        """
        path = Path(path)
        tagged = path / f"{name}_tagged_grammar.json"
        if not tagged.exists():
            raise FileNotFoundError(
                f"No type dictionary found at {tagged}. "
                "Run `bash scripts/rebuild-tokenizer-data.sh` to generate it."
            )
        data = json.loads(tagged.read_text())
        types: dict[int, dict] = {}
        for k, entry in data.items():
            types[int(k)] = dict(entry)
        return types

    # ── properties ───────────────────────────────────────────────────────

    @property
    def type_size(self) -> int:
        """Number of entries in the type dictionary."""
        return len(self._types)

    @property
    def grammar_size(self) -> int:
        """Number of entries in the NLP grammar (= type dictionary)."""
        return self.type_size

    # ── KTokenizer interface: encode / decode (node packing) ─────────────

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to a list of typed nodes.

        Each BPE token ID is combined with its NLP type word:
        ``(nlp_type32 << 32) | bpe_token_id``. Tokens absent from the type
        dictionary receive ``UNKNOWN_TYPE`` (POS_X).
        """
        self._check_available()
        bpe_ids = self.encode_bpe(text, pad_ws)
        nodes: list[int] = []
        for bpe_id in bpe_ids:
            entry = self._types.get(bpe_id)
            sig_word = entry["sig_word"] if entry else self.UNKNOWN_TYPE
            nodes.append((sig_word << 32) | bpe_id)
        return nodes

    def decode(self, ids: list[int]) -> str:
        """Decode typed nodes (or raw BPE IDs) back to text.

        The BPE token ID is taken from the low 32 bits of each value, so
        raw BPE IDs (which already fit in 32 bits) decode unchanged.
        """
        self._check_available()
        if not ids:
            return ""
        bpe_ids = [node & 0xFFFFFFFF for node in ids]
        return self.decode_bpe(bpe_ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode to typed nodes."""
        self._check_available()
        return [self.encode(t) for t in texts]

    # ── type dictionary lookup ───────────────────────────────────────────

    def lookup_type(self, token_id: int) -> int | None:
        """Return the type word for a BPE token ID, or None if absent."""
        entry = self._types.get(token_id)
        return entry["sig_word"] if entry else None

    def lookup_type_entry(self, token_id: int) -> dict | None:
        """Return the raw type-dictionary entry for a BPE token ID.

        The entry is returned as-is; its keys beyond ``sig_word`` are
        opaque metadata (the NLP labels when the dictionary was generated
        by NLP tooling).
        """
        return self._types.get(token_id)

    def lookup_type_entry_for_node(self, node: int) -> dict | None:
        """Return the type-dictionary entry for a typed node.

        Unpacks the node's BPE token ID (low 32 bits) internally and
        delegates to :meth:`lookup_type_entry`. The node layout is owned by
        this tokenizer; callers pass a node, not a pre-unpacked BPE id.
        """
        return self.lookup_type_entry(node & 0xFFFFFFFF)

    def lookup_grammar(self, bpe_id: int) -> dict | None:
        """Look up the NLP grammar entry for a BPE token ID.

        Returns the raw type-dictionary entry, which (for NLP-generated
        dictionaries) carries ``pos`` / ``pos_fine`` / ``dep`` / ``morph``
        alongside the generic ``sig_word``. Returns ``None`` if absent.
        """
        return self.lookup_type_entry(bpe_id)
