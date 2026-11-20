"""KNode.merge — idempotent label composition.

``A.merge(A)`` keeps label ``A``; distinct labels compose in the MTS
uppercase-character notation (Mary + Subject -> MS). The composition is
associative under the fold (can / signature_of): an accumulated compound
label is all-upper, so it stands whole as the next part.
"""

from __future__ import annotations

from kalvin.kline import KNode


def test_self_merge_keeps_label():
    a = KNode(5, "A")
    assert a.merge(a) == 5
    assert a.merge(a).label == "A"


def test_distinct_labels_compose_initials():
    assert KNode(5, "Mary").merge(KNode(5, "Subject")).label == "MS"
    assert KNode(5, "had").merge(KNode(5, "Verb")).label == "HV"


def test_abbreviation_part_stands_whole():
    assert KNode(5, "SVO").merge(KNode(5, "Mary")).label == "SVOM"


def test_composition_folds_associatively():
    out = KNode(0)
    for n in (KNode(1, "Mary"), KNode(2, "Subject"), KNode(4, "Object")):
        out = out.merge(n)
    assert out.label == "MSO"


def test_plain_int_side_keeps_own_label():
    assert KNode(5, "A").merge(5).label == "A"


def test_empty_labels():
    assert KNode(5).merge(KNode(5, "A")).label == "A"
    assert KNode(5).merge(5).label == ""
