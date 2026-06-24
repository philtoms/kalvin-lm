"""Tests for KValue — specs/kvalue.md conformance (KV-1, KV-2, KV-3)."""

from dataclasses import FrozenInstanceError

import pytest

from kalvin.kline import KLine
from kalvin.kvalue import KValue


class TestKValueConstruction:
    """KV-1 — Construction requires both kline and significance (no default)."""

    def test_both_fields_stored(self):
        kl = KLine(5, [1, 2])
        kv = KValue(kl, 999)
        assert kv.kline is kl
        assert kv.significance == 999

    def test_significance_zero_allowed(self):
        # No "unset" sentinel: zero is a valid significance (S4 band).
        kl = KLine(5, [1, 2])
        kv = KValue(kl, 0)
        assert kv.significance == 0

    def test_omitting_significance_raises(self):
        kl = KLine(5, [1, 2])
        with pytest.raises(TypeError):
            KValue(kl)

    def test_omitting_kline_raises(self):
        with pytest.raises(TypeError):
            KValue(significance=1)


class TestKValueEquality:
    """KV-2 — Equality ignores significance (compares kline only)."""

    def test_same_kline_different_significance_equal(self):
        kl = KLine(5, [1, 2])
        assert KValue(kl, 1) == KValue(kl, 999)

    def test_equal_but_not_same_kline(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2])
        assert a is not b
        # Different significance, structurally-equal klines -> equal.
        assert KValue(a, 1) == KValue(b, 999)

    def test_different_klines_not_equal(self):
        a = KLine(5, [1, 2])
        b = KLine(9, [9])
        assert KValue(a, 999) != KValue(b, 1)

    def test_different_klines_not_equal_same_significance(self):
        a = KLine(5, [1, 2])
        b = KLine(9, [9])
        # Same significance, different kline -> still not equal.
        assert KValue(a, 7) != KValue(b, 7)

    def test_not_equal_to_non_kvalue(self):
        kv = KValue(KLine(5, [1, 2]), 1)
        # __eq__ returns NotImplemented -> Python falls back to identity.
        assert (kv == 42) is False
        assert (kv != 42) is True
        assert (kv == "x") is False
        assert (kv != "x") is True
        assert kv is not None

    def test_frozen_no_kline_assignment(self):
        kv = KValue(KLine(5, [1, 2]), 1)
        with pytest.raises(FrozenInstanceError):
            kv.kline = KLine(9, [9])

    def test_frozen_no_significance_assignment(self):
        kv = KValue(KLine(5, [1, 2]), 1)
        with pytest.raises(FrozenInstanceError):
            kv.significance = 999


class TestKValueHash:
    """KV-3 — Hash ignores significance (hashes over kline only)."""

    def test_same_kline_different_significance_equal_hash(self):
        kl = KLine(5, [1, 2])
        assert hash(KValue(kl, 1)) == hash(KValue(kl, 999))

    def test_collapses_in_set(self):
        kl = KLine(5, [1, 2])
        a = KValue(kl, 1)
        b = KValue(kl, 999)
        assert len({a, b}) == 1

    def test_looks_up_in_dict(self):
        kl = KLine(5, [1, 2])
        d = {KValue(kl, 1): "value"}
        assert d[KValue(kl, 999)] == "value"

    def test_unequal_klines_do_not_collide_in_set(self):
        a = KValue(KLine(5, [1, 2]), 1)
        b = KValue(KLine(9, [9]), 1)
        assert len({a, b}) == 2
