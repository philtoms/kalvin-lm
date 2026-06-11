"""NLP type flag definitions and description utilities.

Provides the bit-to-name mapping for NLP type flags used by the decompiler
and compiler for human-readable diagnostic output.

Layout:
  Bits 0–16:  POS tags (displayed without POS_ prefix)
  Bits 17–24: DEP groups (displayed with DEP_ prefix)
  Bits 25–31: MORPH features (displayed with MORPH_ prefix)
"""

from __future__ import annotations

# Bit-to-name mapping for NLP type description in describe_nlp_type().
# Mirrors the NLPType32 IntFlag layout from dev/nlp/nlp_analyzer.py
# (create_nlp_type32, high_bits=False).
_NLP_TYPE_FLAGS: list[tuple[int, str]] = [
    # POS tags (bits 0–16)
    (1 << 0,  "ADJ"),
    (1 << 1,  "ADP"),
    (1 << 2,  "ADV"),
    (1 << 3,  "AUX"),
    (1 << 4,  "CCONJ"),
    (1 << 5,  "DET"),
    (1 << 6,  "INTJ"),
    (1 << 7,  "NOUN"),
    (1 << 8,  "NUM"),
    (1 << 9,  "PART"),
    (1 << 10, "PRON"),
    (1 << 11, "PROPN"),
    (1 << 12, "PUNCT"),
    (1 << 13, "SCONJ"),
    (1 << 14, "SYM"),
    (1 << 15, "VERB"),
    (1 << 16, "X"),
    # DEP groups (bits 17–24)
    (1 << 17, "DEP_SUBJ"),
    (1 << 18, "DEP_OBJ"),
    (1 << 19, "DEP_OBL"),
    (1 << 20, "DEP_COMP"),
    (1 << 21, "DEP_MOD"),
    (1 << 22, "DEP_FUNC"),
    (1 << 23, "DEP_STRUCT"),
    (1 << 24, "DEP_PUNCT"),
    # MORPH features (bits 25–31)
    (1 << 25, "MORPH_PLUR"),
    (1 << 26, "MORPH_PRES"),
    (1 << 27, "MORPH_IMP"),
    (1 << 28, "MORPH_P1"),
    (1 << 29, "MORPH_P2"),
    (1 << 30, "MORPH_P3"),
    (1 << 31, "MORPH_PERF"),
]


def describe_nlp_type(sig: int) -> str:
    """Describe the NLP type bits in a signature as human-readable flag names.

    Extracts the high 32 bits (NLP type portion) and names each set flag
    using the ``_NLP_TYPE_FLAGS`` mapping. Returns a pipe-joined string
    like ``"<PROPN|VERB|DET|ADJ|NOUN>"`` or ``"<NOUN|DEP_STRUCT>"``.

    Args:
        sig: A uint64 NLP-type signature value.

    Returns:
        A descriptive string, or ``"<NLP:0>"`` if no type bits are set.
    """
    type_bits = (sig >> 32) & 0xFFFFFFFF
    if type_bits == 0:
        return "<NLP:0>"

    names = [
        display_name
        for bit_value, display_name in _NLP_TYPE_FLAGS
        if type_bits & bit_value
    ]

    if not names:
        # Unknown bits set that don't match any known flag
        return f"<NLP:{type_bits:#x}>"

    return f"<{'|'.join(names)}>"
