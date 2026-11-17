"""Derived-value labels — the word registry and MTS.

A = WDMH (what, did, Mary, have) against C = MHALL (Mary, had, a,
little, lamb): fit = Mary, underfit = WDH, overfit = HALL — decomposed
from the value's registered component words, bit-accurate (had and have
share the initial H but not the bit, so the fit is Mary alone); a single
component stands whole, a compound takes one character per word.
"""

from __future__ import annotations

from kalvin.kline import KNode, mts_initials, using_word_labels
from rationalisation import fit, overfit, underfit

W, D, M, HAVE = (1 << 32), (2 << 32), (4 << 32), (8 << 32)
HAD, A, LITTLE, LAMB = (16 << 32), (32 << 32), (64 << 32), (128 << 32)
WDMH = W | D | M | HAVE
MHALL = M | HAD | A | LITTLE | LAMB

WORDS = {
    W: "what", D: "did", M: "Mary", HAVE: "have",
    HAD: "had", A: "a", LITTLE: "little", LAMB: "lamb",
}


def test_mts_initials_decomposes_registered_words():
    with using_word_labels(WORDS):
        assert mts_initials(M) == "Mary"  # single component stands whole
        assert mts_initials(W | D | HAVE) == "WDH"
        assert mts_initials(HAD | A | LITTLE | LAMB) == "HALL"


def test_mts_initials_empty_outside_the_registry():
    assert mts_initials(WDMH) == ""
    with using_word_labels({W: "what"}):
        assert mts_initials(WDMH) == ""  # unregistered bits remain


def test_derived_labels_decompose_the_value():
    a, c = KNode(WDMH, "WDMH"), KNode(MHALL, "MHALL")
    with using_word_labels(WORDS):
        assert fit(a, c).label == "Mary"
        assert underfit(a, c).label == "WDH"
        assert overfit([KNode(MHALL, "MHALL")], a).label == "HALL"


def test_derived_labels_fall_back_to_provenance():
    a, c = KNode(WDMH, "WDMH"), KNode(MHALL, "MHALL")
    assert fit(a, c).label == "WDMH∧MHALL"
    assert underfit(a, c).label == "WDMH∖MHALL"
    assert overfit([KNode(WDMH, "WDMH")], c).label == "WDMH∖MHALL"
    assert fit(KNode(W, "what"), KNode(D, "did")).label == "⊥"
