"""Canonical γ tests — significance.gamma_* and the expand terminal byte.

The aggregation form is fixed (ks2.md §11): atom-weighted (granularity-
invariance), union denominator (band-consistency), decay of the mean,
geometric, one knob.
"""

from __future__ import annotations

import pytest

from kalvin.expand import expand
from kalvin.kline import KLine, KNode
from kalvin.model import Model
from kalvin.significance import (
    DEFAULT_DELTA,
    SIG8_MAX,
    SIG8_MIN,
    gamma_aggregate,
    gamma_to_byte,
    geometric_decay,
    word_atom_count,
)
from kalvin.signifier import NLPSignifier


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


A, B, C = bit(0), bit(1), bit(2)
AB, AC, ABC = A | B, A | C, A | B | C


# geometric_decay


def test_decay_zero_hops_is_one():
    assert geometric_decay(0) == 1.0


def test_decay_is_geometric():
    assert geometric_decay(3, delta=0.5) == 0.125


@pytest.mark.parametrize("hops, delta", [(-1, 0.5), (1, 0.0), (1, 1.0), (1, 1.5)])
def test_decay_rejects_bad_arguments(hops, delta):
    with pytest.raises(ValueError):
        geometric_decay(hops, delta=delta)


# gamma_aggregate — band-consistency


def test_exact_match_is_one():
    assert gamma_aggregate([(1, 0), (1, 0)], AB, AB) == 1.0


def test_empty_union_is_vacuously_one():
    assert gamma_aggregate([], 0, 0) == 1.0


def test_nothing_accounted_is_zero():
    assert gamma_aggregate([(1, None)], A, C) == 0.0


def test_underfit_full_coverage_below_one():
    # A wholly inside B, every slot matched and grounded: still below 1 —
    # B's excess is weighed.
    assert gamma_aggregate([(1, 0)], A, AB) == 0.5


def test_overfit_below_one():
    # B wholly inside A, the extra slot unresolvable.
    assert gamma_aggregate([(1, 0), (1, None)], AB, A) == 0.5


# granularity-invariance


def test_slicing_is_invisible():
    assert gamma_aggregate([(2, 0)], AB, AB) == gamma_aggregate(
        [(1, 0), (1, 0)], AB, AB
    )


def test_slicing_is_invisible_at_depth():
    assert gamma_aggregate([(3, 2)], ABC, ABC) == gamma_aggregate(
        [(1, 2), (2, 2)], ABC, ABC
    )


# decay of the mean, not mean of decays


def test_decay_of_mean():
    assert gamma_aggregate([(2, 2), (2, 0)], AB, AB) == pytest.approx(
        DEFAULT_DELTA**1.0
    )
    assert gamma_aggregate([(4, 1)], AB, AB) == pytest.approx(DEFAULT_DELTA**1.0)


def test_unaccounted_excluded_from_depth():
    # The matched slot's depth is 0; the unaccounted slot costs only J.
    assert gamma_aggregate([(1, None), (1, 0)], AB, A) == pytest.approx(0.5)


def test_delta_is_the_only_knob():
    deep = gamma_aggregate([(1, 4)], A, A)
    assert deep == pytest.approx(DEFAULT_DELTA**4)
    assert gamma_aggregate([(1, 4)], A, A, delta=0.9) == pytest.approx(0.9**4)


# word_atom_count


def test_word_bits_only():
    assert word_atom_count(KNode((1 << 32) | 12345)) == 1
    assert word_atom_count(AB) == 2
    assert word_atom_count(0x1234) == 0  # BPE half never weighs


# gamma_to_byte


def test_byte_saturation():
    assert gamma_to_byte(1.0) == SIG8_MAX
    assert gamma_to_byte(0.0) == SIG8_MIN
    interior = gamma_to_byte(0.5)
    assert 0 < interior < SIG8_MAX


# expand terminal byte


def _model_with(*klines: KLine) -> Model:
    model = Model()
    for kl in klines:
        model.add_to_ltm(kl)
    return model


def test_expand_exact_grounded_pair_is_max():
    signifier = NLPSignifier()
    model = _model_with(KLine(A, [A]), KLine(B, [B]))
    query = KLine(AB, [A, B])
    (kv,) = [
        kv for kv in expand(model, query, query, signifier) if kv.kline is query
    ]
    assert kv.significance == SIG8_MAX


def test_expand_disjoint_pair_is_min():
    signifier = NLPSignifier()
    model = Model()
    query = KLine(A, [A])
    candidate = KLine(C, [C])
    (kv,) = [
        kv
        for kv in expand(model, query, candidate, signifier)
        if kv.kline is candidate
    ]
    assert kv.significance == SIG8_MIN


def test_expand_ungrounded_match_discounted():
    signifier = NLPSignifier()
    model = Model()
    model.add_to_stm(KLine(A, [A]))  # STM: not grounded — one hop of doubt
    query = KLine(A, [A])
    (kv,) = [
        kv for kv in expand(model, query, query, signifier) if kv.kline is query
    ]
    assert kv.significance == gamma_to_byte(DEFAULT_DELTA**1)
