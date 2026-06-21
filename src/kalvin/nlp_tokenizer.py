"""NLP tokenizer — the NLP interpretation of the kalvin sig word.

The kalvin tokenizer (see kalvin.tokenizer) produces 64-bit typed nodes
``(sig_word << 32) | bpe_token_id`` and treats the sig word as opaque.
This module is the **NLP interpretation** of that word: it documents and
specialises the sig word as an NLP type encoding, so that kalvin's
generic machinery can be deployed with NLP-derived type information.

NLP sig-word layout (``nlp_type32``, 32 bits)::

    Bits 0–16:  17 coarse part-of-speech tags (POS)
    Bits 17–24: 8 simplified dependency groups (DEP)
    Bits 25–31: 7 simplified morphological features (MORPH)

So a node produced under the NLP interpretation is::

    node = (nlp_type32 << 32) | bpe_token_id

BPE tokens without a type-dictionary entry fall back to ``POS_X``
(``UNKNOWN_NLP_TYPE = 65536 = 1 << 16``), the NLP "unknown" POS flag.

Everything operational (encoding, decoding, signature construction) is
inherited unchanged from the base :class:`kalvin.tokenizer.Tokenizer`.
This subclass additionally owns loading the external NLP resources — the
BPE engine and the tagged-grammar type dictionary — from disk: a bare
``NLPTokenizer()`` reads them from the standard data paths. The base
tokenizer stores and uses the type dictionary but does not know how to
source it from disk; sourcing the NLP-tagged grammar is an NLP concern
lived here. The subclass also fixes the fallback sig word (POS_X) and
exposes NLP-named accessors over the type dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalvin.tokenizer import Tokenizer

# POS_X is the NLP fallback sig word for BPE tokens missing from the type
# dictionary. Value 65536 = 0x10000 = bit 16 set (the POS_X flag). This
# overrides the base tokenizer's generic UNKNOWN_TYPE (0) so that, under the
# NLP interpretation, untyped tokens still carry a valid NLP POS flag.
UNKNOWN_NLP_TYPE = 65536


def load_grammar_dict(path: str | Path) -> dict[int, dict]:
    """Load an NLP grammar dictionary from JSON.

    Reads entries keyed by BPE token ID (string keys converted to int).
    Each entry's ``nlp_type32`` field (the NLP sig word produced by
    ``dev/nlp`` tooling) is normalised onto the generic ``sig_word`` key
    so the entry can feed the base tokenizer's encoder. Other NLP fields
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
        entry = dict(entry)
        # Read ``sig_word`` directly, or normalise from the legacy
        # ``type_word`` / ``nlp_type32`` keys so existing data files keep
        # loading after the rename.
        if "sig_word" not in entry:
            if "type_word" in entry:
                entry["sig_word"] = entry["type_word"]
            elif "nlp_type32" in entry:
                entry["sig_word"] = entry["nlp_type32"]
        normalized[int(k)] = entry
    return normalized


class NLPTokenizer(Tokenizer):
    """Tokenizer specialised for the NLP sig-word interpretation.

    Operationally identical to :class:`kalvin.tokenizer.Tokenizer` (the
    NLP sig word is just the kalvin sig word under the NLP meaning).
    The differences are:

    - The constructor loads the external NLP resources (BPE engine +
      tagged-grammar type dictionary) from disk; a bare ``NLPTokenizer()``
      reads them from the standard data paths. The base tokenizer accepts a
      pre-built type dictionary but does not source one from disk.
    - The fallback sig word is ``POS_X`` (``UNKNOWN_NLP_TYPE = 65536``)
      instead of the base ``UNKNOWN_TYPE`` (0), so untyped BPE tokens still
      carry a valid NLP POS flag.
    - NLP-named accessors (``grammar_size``, ``lookup_grammar``) are
      provided over the type dictionary.
    """

    #: NLP fallback sig word: POS_X (bit 16).
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
        types = NLPTokenizer._load_type_dictionary(tokenizer_path, tokenizer_name)
        super().__init__(bpe=bpe, types=types)

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
            entry = dict(entry)
            # Normalise legacy ``type_word`` / ``nlp_type32`` keys onto
            # ``sig_word`` so data files written before the rename keep
            # loading. New data is written with ``sig_word`` directly.
            if "sig_word" not in entry:
                if "type_word" in entry:
                    entry["sig_word"] = entry["type_word"]
                elif "nlp_type32" in entry:
                    entry["sig_word"] = entry["nlp_type32"]
            types[int(k)] = entry
        return types

    @property
    def grammar_size(self) -> int:
        """Number of entries in the NLP grammar (= type dictionary)."""
        return self.type_size

    def lookup_grammar(self, bpe_id: int) -> dict | None:
        """Look up the NLP grammar entry for a BPE token ID.

        Returns the raw type-dictionary entry, which (for NLP-generated
        dictionaries) carries ``pos`` / ``pos_fine`` / ``dep`` / ``morph``
        alongside the generic ``sig_word``. Returns ``None`` if absent.
        """
        return self.lookup_type_entry(bpe_id)
