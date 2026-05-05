"""Tests for STM — bounded dual-keyed index."""

import pytest
from kalvin.kline import KLine
from kalvin.stm import STM
from kalvin.mod_tokenizer import Mod32Tokenizer


def make_stm(bound: int = 256) -> STM:
    return STM(bound=bound)


class TestSTMAdd:
    def test_add_and_get(self):
        stm = make_stm()
        k = KLine(5, [1, 2])
        assert stm.add(k) is True
        assert stm.get(5) == [k]

    def test_add_dedup(self):
        stm = make_stm()
        k1 = KLine(5, [1, 2])
        k2 = KLine(5, [1, 2])
        assert stm.add(k1) is True
        assert stm.add(k2, dedup=True) is False

    def test_add_no_dedup(self):
        stm = make_stm()
        k1 = KLine(5, [1, 2])
        k2 = KLine(5, [1, 2])
        assert stm.add(k1) is True
        assert stm.add(k2, dedup=False) is True


class TestSTMBound:
    def test_bound_enforcement(self):
        stm = make_stm(bound=3)
        for i in range(5):
            stm.add(KLine(i, [i]), dedup=False)
        assert len(stm) == 3

    def test_eviction_removes_oldest(self):
        stm = make_stm(bound=2)
        k0 = KLine(0, [0])
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        stm.add(k0, dedup=False)
        stm.add(k1, dedup=False)
        stm.add(k2, dedup=False)
        # k0 should be evicted
        assert stm.get(0) == []
        assert stm.get(1) == [k1]
        assert stm.get(2) == [k2]


class TestSTMDualKey:
    def test_indexed_by_signature_and_nodes_sig(self):
        t = Mod32Tokenizer()
        stm = STM(bound=256)
        # Create kline where sig != nodes_sig
        packed_a = t.encode("A")[0]  # e.g., 2
        packed_b = t.encode("B")[0]  # e.g., 4
        k = KLine(signature=100, nodes=[packed_a, packed_b])
        stm.add(k, dedup=False)
        # Should be found by signature
        assert stm.get(100) == [k]
        # Should be found by nodes signature (packed_a | packed_b)
        nodes_sig = packed_a | packed_b
        assert stm.get(nodes_sig) == [k]


class TestSTMQuery:
    def test_query_overlap(self):
        stm = make_stm()
        k1 = KLine(0b110, [0b10, 0b100])
        k2 = KLine(0b001, [0b001])
        stm.add(k1, dedup=False)
        stm.add(k2, dedup=False)
        results = stm.query(0b010)
        assert k1 in results
        assert k2 not in results

    def test_query_zero(self):
        stm = make_stm()
        stm.add(KLine(5, [1]), dedup=False)
        assert stm.query(0) == []
