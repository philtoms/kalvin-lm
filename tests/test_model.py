"""Tests for Model — openspec/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_BOUNDARY, D_MAX
from kalvin.mod_tokenizer import Mod32Tokenizer


def make_model(stm_bound: int = 256) -> Model:
    t = Mod32Tokenizer()
    return Model(is_literal_fn=t.is_literal, stm_bound=stm_bound)


class TestModelAdd:
    def test_add_and_find(self):
        m = make_model()
        k = KLine(5, [1, 2])
        assert m.add(k) is True
        assert m.find(5) is k

    def test_add_returns_true(self):
        m = make_model()
        assert m.add(KLine(5, [1])) is True

    def test_literal_dedup(self):
        m = make_model()
        k1 = KLine(1, [42], literal=True)
        k2 = KLine(1, [42], literal=True)
        m.add(k1)
        assert m.add(k2, dedup=True) is False

    def test_non_literal_no_dedup(self):
        m = make_model()
        k1 = KLine(5, [1, 2])
        k2 = KLine(5, [1, 2])
        m.add(k1)
        # Non-literal klines are always accepted (even with dedup=True, dedup only checks literal)
        assert m.add(k2) is True


class TestModelExists:
    def test_exists_true(self):
        m = make_model()
        k = KLine(5, [1, 2])
        m.add(k)
        assert m.exists(k) is True

    def test_exists_false(self):
        m = make_model()
        assert m.exists(KLine(5, [1])) is False

    def test_exists_different_nodes(self):
        m = make_model()
        m.add(KLine(5, [1, 2]))
        assert m.exists(KLine(5, [1, 3])) is False


class TestModelFind:
    def test_find_by_signature(self):
        m = make_model()
        k = KLine(7, [1, 2, 4])
        m.add(k)
        assert m.find(7) is k

    def test_find_none(self):
        m = make_model()
        assert m.find(42) is None

    def test_find_most_recent(self):
        m = make_model()
        k1 = KLine(7, [1])
        k2 = KLine(7, [2])
        m.add(k1)
        m.add(k2)
        found = m.find(7)
        assert found is k2  # Most recently added


class TestModelFindAll:
    def test_find_all_multiple(self):
        m = make_model()
        k1 = KLine(7, [1])
        k2 = KLine(7, [2])
        m.add(k1)
        m.add(k2)
        results = m.find_all(7)
        assert len(results) == 2

    def test_find_all_empty(self):
        m = make_model()
        assert m.find_all(42) == []


class TestModelRemove:
    def test_remove(self):
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        assert m.remove(5) is True
        assert m.find(5) is None

    def test_remove_nonexistent(self):
        m = make_model()
        assert m.remove(42) is False


class TestModelLen:
    def test_len_empty(self):
        m = make_model()
        assert len(m) == 0

    def test_len_after_add(self):
        m = make_model()
        m.add(KLine(5, [1]))
        assert len(m) == 1


class TestModelWhere:
    def test_where_signature_overlap(self):
        m = make_model()
        k1 = KLine(0b110, [0b10, 0b100])
        k2 = KLine(0b001, [0b001])
        m.add(k1)
        m.add(k2)
        results = m.where(0b010)
        assert k1 in results
        assert k2 not in results


class TestModelGraphTraversal:
    def test_resolve(self):
        m = make_model()
        k = KLine(5, [10, 20])
        m.add(k)
        assert m.resolve(5) is k

    def test_expand(self):
        m = make_model()
        parent = KLine(5, [10, 20])
        child1 = KLine(10, [30])
        child2 = KLine(20, [])
        m.add(parent)
        m.add(child1)
        m.add(child2)
        expanded = m.expand(parent, depth=2)
        assert child1 in expanded
        assert child2 in expanded

    def test_expand_depth_1_returns_empty(self):
        m = make_model()
        k = KLine(5, [10])
        m.add(k)
        assert m.expand(k, depth=1) == []

    def test_descendants(self):
        m = make_model()
        root = KLine(5, [10, 20])
        child = KLine(10, [30])
        m.add(root)
        m.add(child)
        desc = m.descendants(5)
        assert 10 in desc
        assert 20 in desc
        assert 30 in desc

    def test_cycle_detection(self):
        m = make_model()
        a = KLine(1, [2])
        b = KLine(2, [1])
        m.add(a)
        m.add(b)
        desc = m.descendants(1)
        assert 1 in desc
        assert 2 in desc


class TestModelThreeTier:
    def test_base_read_through(self):
        base = make_model()
        k = KLine(5, [1])
        base.add(k)

        frame = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        assert frame.find(5) is k

    def test_add_goes_to_frame_not_base(self):
        base = make_model()
        frame = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        k = KLine(5, [1])
        frame.add(k)
        assert len(base) == 0
        assert len(frame) == 1


class TestModelPromote:
    def test_promote_to_base(self):
        base = make_model()
        frame = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        k = KLine(5, [1])
        frame.add(k)
        assert frame.promote(k) is True
        assert base.find(5) is k

    def test_promote_all(self):
        base = make_model()
        frame = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        frame.add(KLine(5, [1]))
        frame.add(KLine(6, [2]))
        count = frame.promote_all()
        assert count == 2
        assert len(base) == 2


class TestModelSignificanceAPI:
    def test_is_s1_exact_match(self):
        m = make_model()
        k = KLine(5, [1, 2])
        m.add(k)
        assert m.is_s1(5, k) is True

    def test_is_s1_no_match(self):
        m = make_model()
        k = KLine(5, [1, 2])
        assert m.is_s1(42, k) is False

    def test_s2_distance(self):
        m = make_model()
        k = KLine(10, [10, 20, 30])
        d = m.s2_distance(k, k)
        assert 1 <= d < D_BOUNDARY

    def test_s3_distance(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        d = m.s3_distance(q, c)
        assert D_BOUNDARY <= d < D_MAX

    def test_is_countersigned(self):
        m = make_model()
        a = KLine(5, [10, 20])
        b = KLine(10, [5, 30])
        assert m.is_countersigned(a, b) is True

    def test_is_countersigned_one_way(self):
        m = make_model()
        a = KLine(5, [10, 20])
        b = KLine(10, [30, 40])
        assert m.is_countersigned(a, b) is False

    def test_is_countersigned_with_literal_nodes(self):
        """Literal nodes (with 0xFFFFFFFF mask) can't match a signature by value."""
        m = make_model()
        lit_node = (65 << 32) | 0xFFFF_FFFF
        a = KLine(5, [lit_node, 10])
        b = KLine(10, [5])
        # b.sig (10) IS in a.nodes [lit_node, 10] → True
        # a.sig (5) IS in b.nodes [5] → True
        # So they ARE countersigned
        assert m.is_countersigned(a, b) is True
