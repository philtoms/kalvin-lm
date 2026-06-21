"""NLP tokenizer — the NLP interpretation of the kalvin type word.

The kalvin tokenizer (see kalvin.tokenizer) produces 64-bit typed nodes
``(type_word << 32) | bpe_token_id`` and treats the type word as opaque.
This module is the **NLP interpretation** of that word: it documents and
specialises the type word as an NLP type encoding, so that kalvin's
generic machinery can be deployed with NLP-derived type information.

NLP type-word layout (``nlp_type32``, 32 bits)::

    Bits 0–16:  17 coarse part-of-speech tags (POS)
    Bits 17–24: 8 simplified dependency groups (DEP)
    Bits 25–31: 7 simplified morphological features (MORPH)

So a node produced under the NLP interpretation is::

    node = (nlp_type32 << 32) | bpe_token_id

BPE tokens without a type-dictionary entry fall back to ``POS_X``
(``UNKNOWN_NLP_TYPE = 65536 = 1 << 16``), the NLP "unknown" POS flag.

Everything operational (encoding, decoding, signature construction) is
inherited unchanged from the base :class:`kalvin.tokenizer.Tokenizer`;
this subclass only fixes the fallback type word and exposes NLP-named
accessors over the type dictionary.
"""

from __future__ import annotations

import json
from pathlib import Path

from kalvin.tokenizer import Tokenizer

# POS_X is the NLP fallback type word for BPE tokens missing from the type
# dictionary. Value 65536 = 0x10000 = bit 16 set (the POS_X flag). This
# overrides the base tokenizer's generic UNKNOWN_TYPE (0) so that, under the
# NLP interpretation, untyped tokens still carry a valid NLP POS flag.
UNKNOWN_NLP_TYPE = 65536


def load_grammar_dict(path: str | Path) -> dict[int, dict]:
    """Load an NLP grammar dictionary from JSON.

    Reads entries keyed by BPE token ID (string keys converted to int).
    Each entry's ``nlp_type32`` field (the NLP type word produced by
    ``dev/nlp`` tooling) is normalised onto the generic ``type_word`` key
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
        if "type_word" not in entry and "nlp_type32" in entry:
            entry["type_word"] = entry["nlp_type32"]
        normalized[int(k)] = entry
    return normalized


class NLPTokenizer(Tokenizer):
    """Tokenizer specialised for the NLP type-word interpretation.

    Operationally identical to :class:`kalvin.tokenizer.Tokenizer` (the
    NLP type word is just the kalvin type word under the NLP meaning).
    The differences are:

    - The fallback type word is ``POS_X`` (``UNKNOWN_NLP_TYPE = 65536``)
      instead of the base ``UNKNOWN_TYPE`` (0), so untyped BPE tokens still
      carry a valid NLP POS flag.
    - NLP-named accessors (``grammar_size``, ``lookup_grammar``) are
      provided over the type dictionary.
    """

    #: NLP fallback type word: POS_X (bit 16).
    UNKNOWN_TYPE: int = UNKNOWN_NLP_TYPE

    @property
    def grammar_size(self) -> int:
        """Number of entries in the NLP grammar (= type dictionary)."""
        return self.type_size

    def lookup_grammar(self, bpe_id: int) -> dict | None:
        """Look up the NLP grammar entry for a BPE token ID.

        Returns the raw type-dictionary entry, which (for NLP-generated
        dictionaries) carries ``pos`` / ``pos_fine`` / ``dep`` / ``morph``
        alongside the generic ``type_word``. Returns ``None`` if absent.
        """
        return self.lookup_type_entry(bpe_id)
