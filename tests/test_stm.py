"""Tests for STM — bounded dual-keyed index."""

from kalvin.kline import KLine
from kalvin.stm import STM


def t(bits: int) -> int:
    """Place type-word bits in the upper 32 bits of a uint64 (NLP layout)."""
    return bits << 32


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
        stm = STM(bound=256)
        # Create kline where sig != nodes_sig; node values are OR-reducible
        # literals (no tokenizer needed for this STM dual-key test).
        packed_a = 0b010
        packed_b = 0b100
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
        k1 = KLine(t(0b110), [t(0b10), t(0b100)])
        k2 = KLine(t(0b001), [t(0b001)])
        stm.add(k1)
        stm.add(k2)
        # query t(0b010): overlaps k1's type word (shares bit 0b010), not k2's
        results = stm.query(t(0b010))
        assert k1 in results
        assert k2 not in results

    def test_query_zero(self):
        stm = make_stm()
        stm.add(KLine(5, [1]))
        # An empty type-word signature (0) signifies nothing (vacuous for 0).
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


# ── Thread-safety: iterator snapshot semantics ──────────────────────
#
# iterator-returning methods must materialise a snapshot under the
# lock so that a caller iterating *after* the lock is released still sees a
# consistent point-in-time view (no live-list mutation mid-iteration). These
# tests assert that an iterator obtained before a mutation does not observe
# the mutation, while a fresh iterator does.


class TestSTMSnapshotSemantics:
    """Iterators snapshot state at acquisition time."""

    def test_iter_all_snapshots_before_mutation(self):
        stm = make_stm()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        stm.add(k1)
        it = stm.iter_all()  # snapshot taken now
        stm.add(k2)  # mutate after the snapshot
        # Old iterator must not see the post-snapshot entry.
        assert list(it) == [k1]
        # A fresh iterator does see it.
        assert list(stm.iter_all()) == [k1, k2]

    def test_dunder_iter_snapshots_before_mutation(self):
        stm = make_stm()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        stm.add(k1)
        it = iter(stm)
        stm.add(k2)
        assert list(it) == [k1]
        assert list(iter(stm)) == [k1, k2]

    def test_dunder_reversed_snapshots_before_mutation(self):
        stm = make_stm()
        k1 = KLine(1, [1])
        k2 = KLine(2, [2])
        stm.add(k1)
        stm.add(k2)
        rit = reversed(stm)  # snapshot is [k2, k1]
        k3 = KLine(3, [3])
        stm.add(k3)
        assert list(rit) == [k2, k1]
        assert list(reversed(stm)) == [k3, k2, k1]

