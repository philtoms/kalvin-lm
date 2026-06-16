"""NLP Tokenizer — BPE tokens enriched with NLP type information.

Wraps the BPE Tokenizer with a grammar dictionary lookup to produce
64-bit NLP-BPE nodes where the high 32 bits carry NLP type information
(POS + DEP + MORPH) and the low 32 bits carry the BPE token ID.

Node format:
    (nlp_type32 << 32) | bpe_token_id

    - Upper 32 bits: NLP type flags (POS + DEP + MORPH bits)
    - Lower 32 bits: BPE token ID from the base tokenizer

Unknown BPE tokens (not in grammar dict) default to POS_X (65536).

See specs/tokenizer.md for the full specification.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalvin.abstract import KTokenizer
from kalvin.tokenizer import Tokenizer

# ── Class constant ────────────────────────────────────────────────────────
# POS_X is the fallback NLP type for BPE tokens not found in the grammar
# dictionary. Value 65536 = 0x10000 = bit 16 set (POS_X flag).
UNKNOWN_NLP_TYPE = 65536


def load_grammar_dict(path: str | Path) -> dict[int, dict]:
    """Load a grammar dictionary from JSON, converting string keys to int.

    Args:
        path: Path to the grammar JSON file.

    Returns:
        Dictionary mapping integer BPE token IDs to grammar info dicts.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.items()}


class NLPTokenizer(KTokenizer):
    """NLP-enriched BPE tokenizer producing 64-bit NLP-BPE nodes.

    Wraps a BPE Tokenizer with a grammar dictionary to produce nodes
    where the high 32 bits carry NLP type information (POS + DEP + MORPH
    flags) and the low 32 bits carry the BPE token ID.

    Unknown BPE tokens default to POS_X (UNKNOWN_NLP_TYPE = 65536).
    """

    def __init__(self, bpe_tokenizer: Tokenizer, grammar_dict: dict[int, dict]):
        self._bpe = bpe_tokenizer
        self._grammar = grammar_dict

    @property
    def vocab_size(self) -> int:
        """Return the BPE tokenizer vocabulary size."""
        return self._bpe.vocab_size

    @property
    def grammar_size(self) -> int:
        """Return the number of entries in the grammar dictionary."""
        return len(self._grammar)

    # ── Encode ────────────────────────────────────────────────────────

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        """Encode text to a list of NLP-BPE nodes.

        1. BPE-encode the text to get token IDs.
        2. For each BPE ID, look up nlp_type32 from the grammar dict.
           Unknown tokens get UNKNOWN_NLP_TYPE (POS_X).
        3. Construct 64-bit node: (nlp_type32 << 32) | bpe_id.

        Args:
            text: Input string to encode.
            pad_ws: If True, strip and add trailing space (delegated to BPE).

        Returns:
            List of 64-bit NLP-BPE node values.
        """
        bpe_ids = self._bpe.encode(text, pad_ws)
        nodes = []
        for bpe_id in bpe_ids:
            entry = self._grammar.get(bpe_id)
            nlp_type32 = entry["nlp_type32"] if entry else UNKNOWN_NLP_TYPE
            node = (nlp_type32 << 32) | bpe_id
            nodes.append(node)
        return nodes

    # ── Decode ────────────────────────────────────────────────────────

    def decode(self, ids: list[int]) -> str:
        """Decode NLP-BPE nodes back to text.

        Pure BPE: extracts the BPE token id from the low 32 bits of every
        node and decodes via the wrapped ``Tokenizer``.  There is no
        character-level decode path — legacy literal nodes are abandoned
        (Q2-A).

        Args:
            ids: List of node values (NLP-BPE).

        Returns:
            Reconstructed string.
        """
        if not ids:
            return ""

        parts: list[str] = []
        bpe_run: list[int] = []

        for node in ids:
            # NLP-BPE node: extract BPE token ID from low 32 bits
            bpe_run.append(node & 0xFFFFFFFF)

        # Flush BPE run
        if bpe_run:
            parts.append(self._bpe.decode(bpe_run))

        return "".join(parts)

    # ── Grammar lookup ──────────────────────────────────────────────────

    def lookup_grammar(self, bpe_id: int) -> dict | None:
        """Look up NLP grammar info for a BPE token ID.

        Args:
            bpe_id: BPE token ID (raw, without NLP high bits).

        Returns:
            Grammar dict entry if found, else None.
        """
        return self._grammar.get(bpe_id)

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        tokenizer_path: str | None = None,
        tokenizer_name: str = "tokenizer-32768",
        grammar_path: str | None = None,
    ) -> NLPTokenizer:
        """Load an NLPTokenizer from standard file paths.

        When ``tokenizer_path`` is *None* (the default), it is resolved via
        ``kalvin.paths.tokenizer_dir()`` (``KALVIN_DATA_DIR`` env var or
        ``data/tokenizer`` relative to the project root).

        When ``grammar_path`` is *None* (the default), the loader first looks
        for a BPE-tagged grammar (``{tokenizer_name}_tagged_grammar.json``
        alongside the tokenizer).  If that file exists it is preferred because
        every BPE token — including sub-words — carries NLP type information.
        Otherwise it falls back to a plain grammar JSON whose name matches
        the tokenizer stem.

        Args:
            tokenizer_path: Directory containing BPE tokenizer files.
            tokenizer_name: Base name of the tokenizer files.
            grammar_path: Explicit path to a grammar dictionary JSON.
                When *None*, the tagged grammar is auto-discovered.

        Returns:
            Configured NLPTokenizer instance.
        """
        if tokenizer_path is None:
            from kalvin.paths import tokenizer_dir

            tokenizer_path = str(tokenizer_dir())
        bpe_tokenizer = Tokenizer.from_directory(tokenizer_path, tokenizer_name)

        if grammar_path is None:
            # Prefer the BPE-tagged grammar (full vocab coverage)
            tagged = Path(tokenizer_path) / f"{tokenizer_name}_tagged_grammar.json"
            grammar_path = str(tagged) if tagged.exists() else None

            # Fallback: look for a grammar named after the training corpus
            if grammar_path is None:
                # Find any *_grammar.json in the tokenizer directory
                candidates = sorted(Path(tokenizer_path).glob("*_grammar.json"))
                if candidates:
                    grammar_path = str(candidates[0])

            if grammar_path is None:
                raise FileNotFoundError(
                    f"No grammar dictionary found in {tokenizer_path}. "
                    "Run `python dev/nlp/tag_vocab.py` to generate one."
                )

        grammar_dict = load_grammar_dict(grammar_path)
        return cls(bpe_tokenizer, grammar_dict)
