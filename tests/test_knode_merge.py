"""KNode.merge — idempotent label composition.

``A.merge(A)`` keeps label ``A``; composite labels union their ``|``-parts
with order preserved (R2 accumulation merges repeatedly, so ``A|B`` merged
with ``B`` must not re-emit ``B``).
"""

from __future__ import annotations

from kalvin.kline import KNode


def test_self_merge_keeps_label():
    a = KNode(5, "A")
    assert a.merge(a) == 5
    assert a.merge(a).label == "A"


def test_composite_parts_union():
    assert KNode(5, "A|B").merge(KNode(5, "B|C")).label == "A|B|C"


def test_plain_int_side_keeps_own_label():
    assert KNode(5, "A").merge(5).label == "A"


def test_empty_labels():
    assert KNode(5).merge(KNode(5, "A")).label == "A"
    assert KNode(5).merge(5).label == ""
