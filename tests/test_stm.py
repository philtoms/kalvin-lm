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
        stm.add(k)
        assert stm.get(5) == [k]

    def test_add_same_pair_refreshes(self):
        """Adding same (sig, nodes) pair removes old entry and adds fresh."""
        stm = make_stm()
        k1 = KLine(5, [1, 2])
        k2 = KLine(5, [1, 2])
        stm.add(k1)
        stm.add(k2)
        assert stm.all_klines() == [k2]
        assert len(stm) == 1
        assert stm.get(5) == [k2]

    def test_add_same_pair_refreshes_fifo(self):
        """STM-5: adding same (sig, nodes) pair refreshes FIFO position."""
        stm = make_stm(bound=5)
        k1 = KLine(5, [1, 2])
        k2 = KLine(5, [1, 2])
        k_other = KLine(99, [99])
        stm.add(k1)
        stm.add(k_other)
        # k1 is oldest, k_other is newest
        assert list(stm.iter_all()) == [k1, k_other]
        # Re-add k1 (same sig+nodes) — should refresh to front
        stm.add(k2)
        assert list(stm.iter_all()) == [k_other, k2]
        assert len(stm) == 2
        # Verify _store is clean — only k2 under key 5
        assert stm.get(5) == [k2]


class TestSTMBound:
    def test_bound_enforcement(self):
        stm = make_stm(bound=3)
        for i in range(5):
            stm.add(KLine(i, [i]))
        assert len(stm) == 3

    def test_eviction_removes_oldest(self):
        stm = make_stm(bound=2)
        k0 = KLine(0, [0])
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        stm.add(k0)
        stm.add(k1)
        stm.add(k2)
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
        stm.add(k)
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
        stm.add(k1)
        stm.add(k2)
        results = stm.query(0b010)
        assert k1 in results
        assert k2 not in results

    def test_query_zero(self):
        stm = make_stm()
        stm.add(KLine(5, [1]))
        assert stm.query(0) == []


class TestSTMIterAll:
    def test_iter_all_yields_insertion_order(self):
        stm = make_stm()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        k3 = KLine(3, [3])
        stm.add(k1)
        stm.add(k2)
        stm.add(k3)
        assert list(stm.iter_all()) == [k1, k2, k3]

    def test_iter_all_returns_fresh_iterator(self):
        stm = make_stm()
        k1 = KLine(1, [1])
        stm.add(k1)
        it1 = stm.iter_all()
        it2 = stm.iter_all()
        assert list(it1) == [k1]
        assert list(it2) == [k1]
        # Iterators are independent objects
        assert it1 is not it2

    def test_iter_all_empty(self):
        stm = make_stm()
        assert list(stm.iter_all()) == []
