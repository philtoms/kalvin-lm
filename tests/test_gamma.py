"""Canonical γ tests — significance.gamma_*.

The aggregation form is fixed (kalvin-algebra.md §11): atom-weighted (granularity-
invariance), union denominator (band-consistency), decay of the mean,
geometric, one knob.
"""

from __future__ import annotations

import pytest

from kalvin.kline import KLine, KNode
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


# acq_depth


def test_acq_depth_ignored_for_identity():
    assert KLine(A, [A], acq_depth=3) == KLine(A, [A])
    assert len({KLine(A, [A], acq_depth=3), KLine(A, [A])}) == 1


def test_acq_depth_round_trips_through_state_snapshot():
    from kalvin.memory import Memory

    state = Memory(NLPSignifier())
    won = KLine(A, [A], acq_depth=2)
    state.work_list.append(won)
    data = state.to_dict()
    assert data["work_list"][0][2] == 2
    rebuilt = Memory.from_dict(NLPSignifier(), data)
    assert rebuilt.work_list[0].acq_depth == 2
    # Legacy two-element snapshots still load.
    legacy = {"work_list": [[int(A), [int(A)]], ]}
    assert Memory.from_dict(NLPSignifier(), legacy).work_list[0].acq_depth == 0
