"""The 1:1 relationship split — shape vs band-true species (kalvin-algebra.md §4).

is_relationship is the band-agnostic shape; is_connotation is case 4
(uncovered, S3); is_denotation is case 6 (covered, gap-only, S2).
"""

from __future__ import annotations

from kalvin.kline import KLine, KNode, is_connotation, is_denotation, is_relationship
from kalvin.signifier import NLPSignifier


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


A, B, C = bit(0), bit(1), bit(2)
AB = A | B


def test_shape_is_band_agnostic():
    signifier = NLPSignifier()
    disjoint = KLine(A, [B])  # A:[B]
    contained = KLine(AB, [B])  # AB:[B]
    overlapping = KLine(AB, [B | C])  # AB:[BC] — covered with excess
    assert is_relationship(disjoint)
    assert is_relationship(contained)
    assert is_relationship(overlapping)


def test_connotation_is_case_four():
    signifier = NLPSignifier()
    assert is_connotation(KLine(A, [B]), signifier)  # disjoint — S3
    assert not is_connotation(KLine(AB, [B]), signifier)  # denotation — S2
    assert not is_connotation(KLine(AB, [B | C]), signifier)  # overlap — S2


def test_denotation_is_case_six():
    signifier = NLPSignifier()
    assert is_denotation(KLine(AB, [B]), signifier)  # covered, gap-only
    assert not is_denotation(KLine(A, [B]), signifier)  # uncovered
    assert not is_denotation(KLine(AB, [B | C]), signifier)  # carries excess
    assert not is_denotation(KLine(A, [A]), signifier)  # identity — terminal
    assert not is_denotation(KLine(A, []), signifier)  # unknown — terminal


def test_terminals_are_never_relationships():
    signifier = NLPSignifier()
    assert not is_relationship(KLine(A, [A]))
    assert not is_relationship(KLine(A, []))
