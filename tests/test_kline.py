"""Tests for KLine — specs/kline.md conformance."""

import pytest
from kalvin.kline import KLine


class TestKLineConstruction:
    """KLine construction with normalized nodes."""

    def test_empty_kline(self):
        k = KLine(0)
        assert k.signature == 0
        assert k.nodes == []

    def test_nodes_none_normalized(self):
        k = KLine(0, None)
        assert k.nodes == []

    def test_nodes_int_normalized(self):
        k = KLine(5, 42)
        assert k.nodes == [42]

    def test_nodes_list_preserved(self):
        k = KLine(5, [1, 2, 3])
        assert k.nodes == [1, 2, 3]

    def test_empty_nodes_list(self):
        k = KLine(0, [])
        assert k.nodes == []
        assert len(k) == 0

    def test_single_node_kline(self):
        k = KLine(7, [3])
        assert len(k) == 1
        assert k.nodes == [3]

    def test_multi_node_kline(self):
        k = KLine(7, [1, 2, 4])
        assert len(k) == 3

    def test_dbg_text(self):
        k = KLine(0, [], dbg_text="hello")
        assert k.dbg_text == "hello"


class TestKLineEquality:
    """Equality: signature + node sequence."""

    def test_equal_klines(self):
        a = KLine(5, [1, 2, 3])
        b = KLine(5, [1, 2, 3])
        assert a == b

    def test_unequal_signature(self):
        a = KLine(5, [1, 2])
        b = KLine(6, [1, 2])
        assert a != b

    def test_unequal_nodes(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [2, 1])
        assert a != b

    def test_unequal_node_count(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2, 3])
        assert a != b

    def test_not_equal_to_other_type(self):
        k = KLine(5, [1])
        assert k != 42
        assert k != "string"
        assert k != None

    def test_empty_klines_equal(self):
        a = KLine(0, [])
        b = KLine(0, [])
        assert a == b

    def test_empty_klines_unequal_sig(self):
        a = KLine(0, [])
        b = KLine(1, [])
        assert a != b


class TestKLineHash:
    """Hashable for use in sets/dicts."""

    def test_hash_equal_klines(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2])
        assert hash(a) == hash(b)

    def test_in_set(self):
        a = KLine(5, [1, 2])
        b = KLine(5, [1, 2])
        s = {a, b}
        assert len(s) == 1

    def test_in_dict(self):
        a = KLine(5, [1, 2])
        d = {a: "value"}
        b = KLine(5, [1, 2])
        assert d[b] == "value"


class TestKLineNodeAccess:
    """Node access via .nodes and len()."""

    def test_nodes_returns_list(self):
        k = KLine(5, [1, 2, 3])
        assert isinstance(k.nodes, list)

    def test_len(self):
        assert len(KLine(0, [])) == 0
        assert len(KLine(0, [1])) == 1
        assert len(KLine(0, [1, 2, 3])) == 3

    def test_as_node_list_compat(self):
        k = KLine(5, [1, 2])
        assert k.as_node_list() == [1, 2]
