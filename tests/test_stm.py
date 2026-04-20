"""Tests for STM (Short-Term Memory) dual-keyed dictionary."""

import pytest

from kalvin.abstract import KLine, KTokenizer
from kalvin.stm import STM


class _StubTokenizer(KTokenizer):
    """Minimal tokenizer stub that exposes make_signature."""

    def __init__(self):
        self._sig_count = 0

    @property
    def vocab_size(self) -> int:
        return 0

    def is_literal(self, token_id: int) -> bool:
        return False

    def make_signature(self, nodes) -> int:
        """Derive a signature by OR-ing all node values."""
        sig = 0
        if isinstance(nodes, (list, tuple)):
            for n in nodes:
                sig |= int(n)
        elif isinstance(nodes, int):
            sig = nodes
        return sig

    def encode(self, text: str, pad_ws: bool = False) -> list[int]:
        return list(range(len(text)))

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tokenizer() -> _StubTokenizer:
    return _StubTokenizer()


@pytest.fixture
def stm(tokenizer: _StubTokenizer) -> STM:
    return STM(tokenizer)


class TestSTMAdd:
    """Tests for adding KLines to the STM."""

    def test_add_single_kline(self, stm: STM, tokenizer: _StubTokenizer):
        """A KLine is indexed by its signature."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        assert stm.add(kline) is True

        results = stm.get(0x1000)
        assert len(results) == 1
        assert results[0] is kline

    def test_add_indexes_by_nodes_signature_too(self, stm: STM, tokenizer: _StubTokenizer):
        """When nodes_sig != signature, the kline is indexed under both keys."""
        # signature=0x1000, nodes=[0x0100, 0x0200] → nodes_sig = 0x0100|0x0200 = 0x0300
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        stm.add(kline)

        nodes_sig = tokenizer.make_signature([0x0100, 0x0200])  # 0x0300
        assert stm.get(nodes_sig) == [kline]
        assert stm.get(0x1000) == [kline]

    def test_add_same_sig_and_nodes_sig_single_entry(self, stm: STM):
        """When signature == nodes_sig, only one bucket is created."""
        # nodes=[0x1000] → nodes_sig = 0x1000 == signature
        kline = KLine(signature=0x1000, nodes=[0x1000])
        stm.add(kline)

        assert len(stm) == 1
        assert stm.get(0x1000) == [kline]

    def test_add_multiple_klines_same_key(self, stm: STM):
        """Multiple different klines under the same key are all kept."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x1000, nodes=[0x0200])
        kline3 = KLine(signature=0x1000, nodes=[0x0300])

        stm.add(kline1)
        stm.add(kline2)
        stm.add(kline3)

        results = stm.get(0x1000)
        assert len(results) == 3
        assert results[0] is kline1
        assert results[1] is kline2
        assert results[2] is kline3

    def test_add_duplicate_rejected(self, stm: STM):
        """Adding an identical kline (same sig, same nodes) is rejected when dedup=True."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        assert stm.add(kline, dedup=True) is True
        assert stm.add(KLine(signature=0x1000, nodes=[0x0100, 0x0200]), dedup=True) is False

        # Only one entry stored
        results = stm.get(0x1000)
        assert len(results) == 1
        assert results[0] is kline

    def test_add_duplicate_allowed_when_dedup_false(self, stm: STM):
        """Duplicate klines are stored when dedup=False (default)."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x1000, nodes=[0x0100])

        assert stm.add(kline1) is True
        assert stm.add(kline2) is True

        results = stm.get(0x1000)
        assert len(results) == 2

    def test_add_different_nodes_same_sig_not_duplicate(self, stm: STM):
        """KLines with same signature but different nodes are not duplicates."""
        assert stm.add(KLine(signature=0x1000, nodes=[0x0100]), dedup=True) is True
        assert stm.add(KLine(signature=0x1000, nodes=[0x0200]), dedup=True) is True

        results = stm.get(0x1000)
        assert len(results) == 2

    def test_add_same_nodes_different_sig_not_duplicate(self, stm: STM):
        """KLines with same nodes but different signatures are not duplicates."""
        assert stm.add(KLine(signature=0x1000, nodes=[0x0100]), dedup=True) is True
        assert stm.add(KLine(signature=0x2000, nodes=[0x0100]), dedup=True) is True

        assert len(stm.get(0x1000)) == 1
        assert len(stm.get(0x2000)) == 1

    def test_add_multiple_klines_different_keys(self, stm: STM):
        """KLines with different signatures are stored separately."""
        # kline1: sig=0x1000, nodes_sig=0x0100 → 2 keys
        # kline2: sig=0x2000, nodes_sig=0x0200 → 2 keys
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x2000, nodes=[0x0200])

        stm.add(kline1)
        stm.add(kline2)

        assert len(stm) == 4  # 2 keys per kline (sig + nodes_sig)
        assert stm.get(0x1000) == [kline1]
        assert stm.get(0x2000) == [kline2]
        assert stm.get(0x0100) == [kline1]
        assert stm.get(0x0200) == [kline2]

    def test_add_empty_nodes(self, stm: STM):
        """KLine with empty node list still indexed by signature."""
        kline = KLine(signature=0x1000, nodes=[])
        stm.add(kline)

        assert stm.get(0x1000) == [kline]
        # nodes_sig from [] = 0, different from 0x1000
        assert stm.get(0) == [kline]

    def test_add_none_nodes(self, stm: STM):
        """KLine with None nodes indexed by signature; nodes_sig = 0."""
        kline = KLine(signature=0x1000, nodes=None)
        stm.add(kline)

        assert stm.get(0x1000) == [kline]
        assert stm.get(0) == [kline]

    def test_add_single_int_node(self, stm: STM):
        """KLine with a single int node."""
        # nodes=0x0100 → as_node_list()=[0x0100] → nodes_sig=0x0100
        kline = KLine(signature=0x1000, nodes=0x0100)
        stm.add(kline)

        assert stm.get(0x1000) == [kline]
        assert stm.get(0x0100) == [kline]


class TestSTMGet:
    """Tests for retrieving KLines from the STM."""

    def test_get_missing_key_returns_empty(self, stm: STM):
        """Getting a non-existent key returns an empty list."""
        assert stm.get(0x9999) == []

    def test_get_returns_copy(self, stm: STM):
        """get() returns a copy so mutating the result doesn't affect the store."""
        kline = KLine(signature=0x1000, nodes=[])
        stm.add(kline)
        result = stm.get(0x1000)
        result.clear()

        assert len(stm.get(0x1000)) == 1

    def test_get_kline_latest(self, stm: STM):
        """get_kline() returns the most recently added KLine."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x1000, nodes=[0x0200])

        stm.add(kline1)
        stm.add(kline2)

        assert stm.get_kline(0x1000) is kline2

    def test_get_kline_missing_returns_none(self, stm: STM):
        """get_kline() returns None for a non-existent key."""
        assert stm.get_kline(0x9999) is None


class TestSTMFindBySignature:
    """Tests for find_by_signature."""

    def test_find_by_signature(self, stm: STM):
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        stm.add(kline)

        results = stm.find_by_signature(0x1000)
        assert len(results) == 1
        assert results[0] is kline

    def test_find_by_signature_miss(self, stm: STM):
        assert stm.find_by_signature(0x9999) == []


class TestSTMFindByNodes:
    """Tests for find_by_nodes."""

    def test_find_by_nodes_list(self, stm: STM, tokenizer: _StubTokenizer):
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        stm.add(kline)

        nodes_sig = tokenizer.make_signature([0x0100, 0x0200])
        results = stm.find_by_nodes([0x0100, 0x0200])
        assert len(results) == 1
        assert results[0] is kline
        assert stm.get(nodes_sig) == results

    def test_find_by_nodes_single_int(self, stm: STM):
        kline = KLine(signature=0x1000, nodes=0x0100)
        stm.add(kline)

        results = stm.find_by_nodes(0x0100)
        assert len(results) == 1
        assert results[0] is kline

    def test_find_by_nodes_miss(self, stm: STM):
        assert stm.find_by_nodes([0x9999]) == []


class TestSTMRemove:
    """Tests for removing KLines from the STM."""

    def test_remove_kline(self, stm: STM):
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        stm.add(kline, dedup=True)
        stm.remove(kline)

        assert stm.get(0x1000) == []
        assert len(stm) == 0

    def test_remove_then_add_same_kline(self, stm: STM):
        """After removing a kline, the same kline can be re-added with dedup."""
        kline = KLine(signature=0x1000, nodes=[0x0100])
        stm.add(kline, dedup=True)
        assert stm.add(kline, dedup=True) is False  # duplicate
        stm.remove(kline)
        assert stm.add(kline, dedup=True) is True  # accepted after remove
        assert len(stm.get(0x1000)) == 1

    def test_remove_one_of_many(self, stm: STM):
        """Removing one kline from a multi-entry bucket leaves the rest."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x1000, nodes=[0x0200])
        stm.add(kline1)
        stm.add(kline2)

        stm.remove(kline1)

        assert stm.get(0x1000) == [kline2]

    def test_remove_from_both_keys(self, stm: STM, tokenizer: _StubTokenizer):
        """Removing a kline cleans up both signature and nodes_sig buckets."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        stm.add(kline)

        nodes_sig = tokenizer.make_signature([0x0100, 0x0200])
        assert len(stm) == 2  # sig + nodes_sig

        stm.remove(kline)

        assert stm.get(0x1000) == []
        assert stm.get(nodes_sig) == []
        assert len(stm) == 0

    def test_remove_nonexistent_is_noop(self, stm: STM):
        """Removing a kline not in the STM is a no-op."""
        kline = KLine(signature=0x1000, nodes=[])
        stm.remove(kline)  # should not raise


class TestSTMQuery:
    """Tests for the query method (signature overlap via AND ≠ 0)."""

    def test_query_exact_key_hit(self, stm: STM):
        """Exact key match returns the bucket in lifo order."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x1000, nodes=[0x0200])
        stm.add(kline1)
        stm.add(kline2)

        result = stm.query(0x1000)
        assert result == [kline2, kline1]

    def test_query_no_match(self, stm: STM):
        """Query with non-overlapping bits returns empty list."""
        stm.add(KLine(signature=0x1000, nodes=[0x0100]))
        assert stm.query(0x2000) == []

    def test_query_overlap_partial(self, stm: STM):
        """Returns klines whose signatures share at least one bit."""
        kline1 = KLine(signature=0x1000, nodes=[0x0100])
        kline2 = KLine(signature=0x2000, nodes=[0x0200])
        kline3 = KLine(signature=0x4000, nodes=[0x0400])
        stm.add(kline1)
        stm.add(kline2)
        stm.add(kline3)

        # 0x1000 | 0x2000 = 0x3000 overlaps kline1 and kline2 but not kline3
        result = stm.query(0x3000)
        assert len(result) == 2
        assert kline1 in result
        assert kline2 in result
        assert kline3 not in result

    def test_query_overlap_all(self, stm: STM):
        """Query with all bits set matches every kline."""
        kline1 = KLine(signature=0x1000, nodes=[])
        kline2 = KLine(signature=0x2000, nodes=[])
        stm.add(kline1)
        stm.add(kline2)

        result = stm.query(0xFFFF_FFFF_FFFF_FFFF)
        assert len(result) == 2

    def test_query_deduplicates_same_kline(self, stm: STM):
        """A kline indexed under two keys is returned only once."""
        # sig=0x1000, nodes=[0x0100] → also indexed under 0x0100
        # query with 0x1000 | 0x0100 = 0x1100 overlaps via both keys
        kline = KLine(signature=0x1000, nodes=[0x0100])
        stm.add(kline)

        result = stm.query(0x1100)
        assert len(result) == 1
        assert result[0] is kline

    def test_query_empty_stm(self, stm: STM):
        """Query on an empty STM returns empty list."""
        assert stm.query(0x1000) == []

    def test_query_zero_returns_empty(self, stm: STM):
        """Query with 0 never overlaps anything."""
        stm.add(KLine(signature=0x1000, nodes=[]))
        assert stm.query(0) == []

    def test_query_returns_lifo_order(self, stm: STM):
        """Results are returned in lifo order (most recently added first)."""
        kline1 = KLine(signature=0x0010, nodes=[])  # overlaps 0x0030
        kline2 = KLine(signature=0x0020, nodes=[])  # overlaps 0x0030
        kline3 = KLine(signature=0x0030, nodes=[])  # overlaps 0x0030
        stm.add(kline1)
        stm.add(kline2)
        stm.add(kline3)

        result = stm.query(0x0030)
        assert result == [kline3, kline2, kline1]

    def test_query_subset_bit(self, stm: STM):
        """A single set bit in the query finds any kline sharing that bit."""
        kline = KLine(signature=0x1800, nodes=[])
        stm.add(kline)

        # bit 0x1000 is set in kline → overlap
        assert len(stm.query(0x1000)) == 1
        # bit 0x0100 is NOT set in kline → no overlap
        assert stm.query(0x0100) == []


class TestSTMClear:
    """Tests for clearing the STM."""

    def test_clear(self, stm: STM):
        stm.add(KLine(signature=0x1000, nodes=[0x0100]), dedup=True)
        stm.add(KLine(signature=0x2000, nodes=[0x0200]), dedup=True)

        stm.clear()

        assert len(stm) == 0
        assert stm.get(0x1000) == []
        assert stm.get(0x2000) == []

    def test_clear_resets_dedup(self, stm: STM):
        """After clear, previously added klines can be re-added with dedup."""
        kline = KLine(signature=0x1000, nodes=[0x0100])
        stm.add(kline, dedup=True)
        assert stm.add(kline, dedup=True) is False
        stm.clear()
        assert stm.add(kline, dedup=True) is True


class TestSTMDunder:
    """Tests for dunder methods."""

    def test_len(self, stm: STM):
        assert len(stm) == 0
        # nodes=[0x0100] → nodes_sig=0x0100 != 0x1000 → 2 keys created
        stm.add(KLine(signature=0x1000, nodes=[0x0100]))
        assert len(stm) == 2

    def test_contains(self, stm: STM):
        stm.add(KLine(signature=0x1000, nodes=[0x0100]))

        assert 0x1000 in stm
        assert 0x0100 in stm  # nodes_sig
        assert 0x9999 not in stm

    def test_repr(self, stm: STM):
        r = repr(stm)
        assert "STM" in r
        assert "keys=0" in r
        assert "klines=0" in r
