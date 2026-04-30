"""Tests for Model — openspec/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_BOUNDARY, D_MAX, MAX_HOP
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


class TestIsS1:
    def test_is_s1_resolves(self):
        """Node that matches a kline signature in the model → True."""
        m = make_model()
        k = KLine(5, [1, 2])
        m.add(k)
        assert m.is_s1(5) is True

    def test_is_s1_no_resolve(self):
        """Node with no matching kline in the model → False."""
        m = make_model()
        k = KLine(5, [1, 2])
        assert m.is_s1(42) is False

    def test_is_s1_node_not_signature(self):
        """Node that exists in kline.nodes but not as a signature → False."""
        m = make_model()
        k = KLine(5, [10, 20])
        m.add(k)
        # Node 10 is in k.nodes but no kline with sig 10 exists
        assert m.is_s1(10) is False


class TestIsCanon:
    def test_canon_match(self):
        """sig == make_signature(nodes) → canonical."""
        m = make_model()
        k = KLine(10, [10])  # make_signature([10]) = 10 (non-literal)
        assert m._is_canon(k) is True

    def test_canon_mismatch(self):
        """sig != make_signature(nodes) → non-canonical."""
        m = make_model()
        k = KLine(5, [10])  # make_signature([10]) = 10 ≠ 5
        assert m._is_canon(k) is False


class TestEdgeHops:
    def test_edge_hops_unresolvable(self):
        """Node that doesn't resolve → empty generator."""
        m = make_model()
        assert list(m._edge_hops(99)) == []

    def test_edge_hops_canonical(self):
        """Node that resolves to canonical → empty generator."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        assert list(m._edge_hops(10)) == []

    def test_edge_hops_chain(self):
        """Non-canonical chain: 5→(1,10)→(2,20)→(3,30) where 30 is canonical."""
        m = make_model()
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon: sig=20, make_sig([30])=30
        m.add(KLine(10, [20]))  # non-canon: sig=10, make_sig([20])=20
        m.add(KLine(5, [10]))   # non-canon: sig=5,  make_sig([10])=10
        assert list(m._edge_hops(5))  == [(1, 10), (2, 20), (3, 30)]
        assert list(m._edge_hops(10)) == [(1, 20), (2, 30)]
        assert list(m._edge_hops(20)) == [(1, 30)]
        assert list(m._edge_hops(30)) == []  # canonical
        assert list(m._edge_hops(99)) == []  # unresolvable


class TestS2Distance:
    def test_s2_distance_self_no_model(self):
        """Self-comparison: all nodes match, grounding credit only."""
        m = make_model()
        k = KLine(10, [10, 20, 30])
        d = m.s2_distance(k, k)
        # All nodes match, no resolution → distance = -3 (no grounding), clamped to 1
        assert d == 1

    def test_s2_distance_empty_query(self):
        """Empty query → minimum distance 1."""
        m = make_model()
        q = KLine(0, [])
        c = KLine(10, [1, 2])
        assert m.s2_distance(q, c) == 1

    def test_s2_distance_no_resolution(self):
        """Mismatched nodes with no model entries → MAX_HOP each."""
        m = make_model()
        q = KLine(5, [1, 2, 3])
        c = KLine(6, [1, 4, 5])
        # matched: {1}, mismatched_q: {2,3}, mismatched_c: {4,5}
        # No chains → all MAX_HOP
        # grounded: find(1) → None → no credit
        expected = 4 * 100  # 4 mismatched × MAX_HOP
        assert m.s2_distance(q, c) == expected

    def test_s2_distance_with_grounding_credit(self):
        """Matched node that resolves → grounding credit."""
        m = make_model()
        m.add(KLine(1, [10]))  # node 1 resolves
        q = KLine(5, [1, 2])
        c = KLine(6, [1, 3])
        # matched: {1}, mismatched_q: {2}, mismatched_c: {3}
        # No chains reach opposing mismatch → 2 × MAX_HOP = 200
        # grounded: find(1) → kline → -1
        expected = 200 - 1
        assert m.s2_distance(q, c) == expected

    def test_s2_distance_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        # Chain: 5→10→20→30(canonical)
        # make_sig chain: find(5)→nodes[10]→make_sig=10, find(10)→nodes[20]→make_sig=20, etc.
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon
        m.add(KLine(10, [20]))  # non-canon
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])    # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])   # mismatched_c: {10, 3}
        # matched: {}, no grounding credit
        #
        # mismatched_q node 5: edge_hops yields (1,10),(2,20),(3,30)
        #   hop 1 → sig 10 ∈ mismatched_c → hop_distance = 1
        # mismatched_q node 2: no resolution → MAX_HOP
        # mismatched_c node 10: edge_hops yields (1,20),(2,30)
        #   neither 20 nor 30 ∈ mismatched_q → MAX_HOP
        # mismatched_c node 3: no resolution → MAX_HOP
        expected = 1 + 100 + 100 + 100  # = 301
        assert m.s2_distance(q, c) == expected

    def test_s2_distance_bidirectional_hop_match(self):
        """Both query and candidate mismatched nodes reach opposing sets."""
        m = make_model()
        # Chain: 5→(1,10), 10 is canonical
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon

        # Chain: 20→(1,30), 30 is canonical
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon

        q = KLine(100, [5, 20])     # mismatched_q: {5, 20}
        c = KLine(200, [10, 30])    # mismatched_c: {10, 30}
        # matched: {}
        #
        # q-node 5: edge_hops yields (1,10). 10 ∈ mismatched_c → hop_distance=1
        # q-node 20: edge_hops yields (1,30). 30 ∈ mismatched_c → hop_distance=1
        # c-node 10: canonical → no hops → MAX_HOP
        # c-node 30: canonical → no hops → MAX_HOP
        expected = 1 + 1 + 100 + 100  # = 202
        assert m.s2_distance(q, c) == expected

    def test_s2_distance_all_matched_grounded(self):
        """All nodes match and all resolve → distance clamped to 1."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical, node 10 resolves
        m.add(KLine(20, [20]))  # canonical, node 20 resolves
        q = KLine(5, [10, 20])
        c = KLine(6, [10, 20])
        # matched: {10, 20}, mismatched: none
        # grounded: find(10) and find(20) → grounded → -2
        # distance = -2, clamped to 1
        assert m.s2_distance(q, c) == 1

    def test_s2_distance_clamped_to_boundary(self):
        """Result is always < D_BOUNDARY."""
        m = make_model()
        # Many mismatched nodes, none resolve
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))  # many mismatched candidate nodes
        d = m.s2_distance(q, c)
        assert d < D_BOUNDARY

    def test_s2_distance_range(self):
        """S2 distance is always in [1, D_BOUNDARY)."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        d = m.s2_distance(q, c)
        assert 1 <= d < D_BOUNDARY


class TestModelSignificanceAPI:
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
