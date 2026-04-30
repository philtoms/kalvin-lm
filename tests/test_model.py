"""Tests for Model — specs/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_PACK_SHIFT, D_MAX, MAX_HOP
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


class TestDistance:
    def test_distance_self_no_model(self):
        """Self-comparison: all nodes match, ungrounded penalty only."""
        m = make_model()
        k = KLine(10, [10, 20, 30])
        d = m.distance(k, k, "S2")
        # All nodes match, none resolve → s2 = 3 (ungrounded), s3 = 0
        # packed: (0 << 32) + 3 = 3
        assert d == 3

    def test_distance_no_resolution(self):
        """Mismatched nodes with no model entries → MAX_HOP each."""
        m = make_model()
        q = KLine(5, [1, 2, 3])
        c = KLine(6, [1, 4, 5])
        # matched: {1}, mismatched_q: {2,3}, mismatched_c: {4,5}
        # No chains → all MAX_HOP
        # ungrounded: find(1) → None → s2 += 1
        # level_distance = 4 * MAX_HOP = 400
        # s2 = 400 + 1 = 401, s3 = 0
        expected = 401
        assert m.distance(q, c, "S2") == expected

    def test_distance_with_grounding(self):
        """Matched node that resolves → no ungrounded penalty."""
        m = make_model()
        m.add(KLine(1, [10]))  # node 1 resolves
        q = KLine(5, [1, 2])
        c = KLine(6, [1, 3])
        # matched: {1}, mismatched_q: {2}, mismatched_c: {3}
        # No chains reach opposing mismatch → 2 × MAX_HOP = 200
        # grounded: find(1) → kline → no ungrounded penalty
        # s2 = 200, s3 = 0
        expected = 200
        assert m.distance(q, c, "S2") == expected

    def test_distance_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon
        m.add(KLine(10, [20]))  # non-canon
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])    # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])   # mismatched_c: {10, 3}
        # matched: {}, no ungrounded penalty
        #
        # mismatched_q node 5: edge_hops yields (1,10),(2,20),(3,30)
        #   hop 1 → sig 10 ∈ mismatched_c → hop_distance = 1
        # mismatched_q node 2: no resolution → MAX_HOP
        # mismatched_c node 10: edge_hops yields (1,20),(2,30)
        #   neither 20 nor 30 ∈ mismatched_q → MAX_HOP
        # mismatched_c node 3: no resolution → MAX_HOP
        # level_distance = 1 + 100 + 100 + 100 = 301
        # s2 = 301, s3 = 0
        expected = 301
        assert m.distance(q, c, "S2") == expected

    def test_distance_bidirectional_hop_match(self):
        """Both query and candidate mismatched nodes reach opposing sets."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon
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
        # level_distance = 1 + 1 + 100 + 100 = 202
        expected = 202
        assert m.distance(q, c, "S2") == expected

    def test_distance_all_matched_grounded(self):
        """All nodes match and all resolve → no ungrounded penalty."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical, node 10 resolves
        m.add(KLine(20, [20]))  # canonical, node 20 resolves
        q = KLine(5, [10, 20])
        c = KLine(6, [10, 20])
        # matched: {10, 20}, both grounded → no ungrounded penalty
        # level_distance = 0
        # s2 = 0
        assert m.distance(q, c, "S2") == 0

    def test_distance_clamped_to_max(self):
        """Result is clamped to D_MAX - 1."""
        m = make_model()
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))
        d = m.distance(q, c, "S2")
        assert d < D_MAX

    def test_distance_range_s2(self):
        """S2 distance returns a valid packed value."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        d = m.distance(q, c, "S2")
        assert d >= 0

    def test_distance_s3_route(self):
        """S3 route puts level_distance in upper bits."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        d = m.distance(q, c, "S3")
        # 4 mismatched nodes × MAX_HOP = 400 in s3 component
        # packed: (400 << 32) + 0
        assert d >= (1 << D_PACK_SHIFT)

    def test_distance_packed_encoding(self):
        """Verify packed encoding: s3 in upper bits, s2 in lower bits."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])     # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])    # mismatched_c: {10, 3}
        d = m.distance(q, c, "S2")
        # q-node 5: hops → (1,10). 10 ∈ mismatched_c → hop_distance = 1
        # q-node 2: no resolution → MAX_HOP
        # c-node 10: canonical → MAX_HOP
        # c-node 3: no resolution → MAX_HOP
        # level_distance = 301 → s2 = 301
        s2_component = d & ((1 << D_PACK_SHIFT) - 1)
        s3_component = d >> D_PACK_SHIFT
        assert s2_component == 301
        assert s3_component == 0

    def test_distance_connotation_bridging(self):
        """Connotation bridging: indirect path through intermediate signature."""
        m = make_model()
        # Chain: 50 → (1, 60) where 60 is canonical
        m.add(KLine(60, [60]))  # canonical
        m.add(KLine(50, [60]))  # non-canon: edge_hops(50) = [(1, 60)]

        # Chain: 70 → (1, 60) where 60 is canonical
        m.add(KLine(70, [60]))  # non-canon: edge_hops(70) = [(1, 60)]

        q = KLine(100, [50])     # mismatched_q: {50}
        c = KLine(200, [70])     # mismatched_c: {70}
        # q-node 50: edge_hops yields (1, 60). 60 ∉ mismatched_c → connotation[60] = 1
        #   No more hops → hop_distance = MAX_HOP. level_distance += 100
        # c-node 70: edge_hops yields (1, 60). 60 ∉ mismatched_q, but 60 ∈ connotation →
        #   s3_distance += 1 + 1 = 2, hop_distance = 0. level_distance += 0
        # For S2: s2 = 100 (level_distance), s3 = 2 (connotation)
        d_s2 = m.distance(q, c, "S2")
        assert d_s2 == (2 << D_PACK_SHIFT) + 100

        # For S3: s3 = 2 (connotation) + 100 (level_distance)
        d_s3 = m.distance(q, c, "S3")
        assert d_s3 == (102 << D_PACK_SHIFT) + 0

    def test_distance_component_clamp(self):
        """Each component is clamped to its bit budget."""
        m = make_model()
        # Create massive mismatch to overflow component budgets
        q = KLine(5, list(range(1000)))
        c = KLine(6, list(range(1000, 2000)))
        d = m.distance(q, c, "S2")
        s2_component = d & ((1 << D_PACK_SHIFT) - 1)
        s3_component = d >> D_PACK_SHIFT
        s2_max = (1 << D_PACK_SHIFT) - 1
        s3_max = (1 << (64 - D_PACK_SHIFT)) - 1
        assert s2_component <= s2_max
        assert s3_component <= s3_max


class TestModelSignificanceAPI:
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
