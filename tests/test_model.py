"""Tests for Model — specs/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_MAX, MASK64, MAX_HOP
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
    def test_remove_from_stm(self):
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        assert m.remove(5) is True
        assert m.find(5) is None

    def test_remove_from_frame(self):
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        m.promote(k)
        assert m.remove(5) is True
        assert m.find(5) is None

    def test_remove_nonexistent(self):
        m = make_model()
        assert m.remove(42) is False


class TestModelLen:
    def test_len_empty(self):
        m = make_model()
        assert len(m) == 0

    def test_len_counts_frame_only(self):
        """add() goes to STM, not frame. len counts frame entries."""
        m = make_model()
        m.add(KLine(5, [1]))
        assert len(m) == 0  # STM only, not promoted to frame

    def test_len_after_promote(self):
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        m.promote(k)
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

    def test_query_expand(self):
        m = make_model()
        parent = KLine(5, [10, 20])
        child1 = KLine(10, [30])
        child2 = KLine(20, [])
        m.add(parent)
        m.add(child1)
        m.add(child2)
        expanded = m.query_expand(parent, depth=2)
        assert child1 in expanded
        assert child2 in expanded

    def test_query_expand_depth_1_returns_empty(self):
        m = make_model()
        k = KLine(5, [10])
        m.add(k)
        assert m.query_expand(k, depth=1) == []

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
        base.promote(k)

        frame = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        assert frame.find(5) is k

    def test_add_goes_to_stm_not_frame(self):
        base = make_model()
        session = Model(base=base, is_literal_fn=Mod32Tokenizer().is_literal)
        k = KLine(5, [1])
        session.add(k)
        assert len(base) == 0
        assert len(session) == 0  # frame is empty — kline is in STM only
        assert session.find(5) is k  # but still discoverable via STM


class TestModelPromote:
    def test_promote_to_frame(self):
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        assert len(m) == 0  # STM only
        assert m.promote(k) is True
        assert len(m) == 1  # now in frame

    def test_promote_requires_stm(self):
        """promote() rejects klines not in STM."""
        m = make_model()
        k = KLine(5, [1])
        assert m.promote(k) is False  # never added to STM

    def test_promote_idempotent(self):
        """Promoting same kline twice is a no-op."""
        m = make_model()
        k = KLine(5, [1])
        m.add(k)
        assert m.promote(k) is True
        assert m.promote(k) is False  # already in frame

    def test_promote_all(self):
        m = make_model()
        m.add(KLine(5, [1]))
        m.add(KLine(6, [2]))
        assert len(m) == 0  # STM only
        count = m.promote_all()
        assert count == 2
        assert len(m) == 2  # now in frame


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


class TestExpand:
    def test_expand_self_no_model(self):
        """Self-comparison: all nodes match, ungrounded penalty only."""
        m = make_model()
        k = KLine(10, [10, 20, 30])
        results = list(m.expand(k, k, "S2"))
        # All nodes match, none resolve → distance=3 → significance=~3
        assert len(results) == 1
        assert results[-1].significance == (~3) & MASK64

    def test_expand_no_resolution(self):
        """Mismatched nodes with no model entries → MAX_HOP each."""
        m = make_model()
        q = KLine(5, [1, 2, 3])
        c = KLine(6, [1, 4, 5])
        # matched: {1}, mismatched_q: {2,3}, mismatched_c: {4,5}
        # No chains → all MAX_HOP, ungrounded +1 → distance=401
        expected_distance = 401
        results = list(m.expand(q, c, "S2"))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_with_grounding(self):
        """Matched node that resolves → no ungrounded penalty."""
        m = make_model()
        m.add(KLine(1, [10]))  # node 1 resolves
        q = KLine(5, [1, 2])
        c = KLine(6, [1, 3])
        # distance=200 (2 × MAX_HOP, grounded match)
        expected_distance = 200
        results = list(m.expand(q, c, "S2"))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon
        m.add(KLine(10, [20]))  # non-canon
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])    # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])   # mismatched_c: {10, 3}
        # distance=301 (1 hop + 3 × MAX_HOP), 3 connotations + terminal
        results = list(m.expand(q, c, "S2"))
        assert len(results) == 4
        assert results[-1].significance == (~301) & MASK64

    def test_expand_bidirectional_hop_match(self):
        """Both query and candidate mismatched nodes reach opposing sets."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon

        q = KLine(100, [5, 20])     # mismatched_q: {5, 20}
        c = KLine(200, [10, 30])    # mismatched_c: {10, 30}
        # distance=202, 2 connotations + terminal
        results = list(m.expand(q, c, "S2"))
        assert len(results) == 3
        assert results[-1].significance == (~202) & MASK64

    def test_expand_all_matched_grounded(self):
        """All nodes match and all resolve → no ungrounded penalty → max significance."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical, node 10 resolves
        m.add(KLine(20, [20]))  # canonical, node 20 resolves
        q = KLine(5, [10, 20])
        c = KLine(6, [10, 20])
        # distance=0 → significance=D_MAX (all bits set)
        results = list(m.expand(q, c, "S2"))
        assert results[-1].significance == D_MAX

    def test_expand_clamped_to_valid(self):
        """Significance is always in valid range [1, D_MAX]."""
        m = make_model()
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))
        results = list(m.expand(q, c, "S2"))
        assert 0 < results[-1].significance <= D_MAX

    def test_expand_range_s2(self):
        """S2 significance is a valid uint64."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        results = list(m.expand(q, c, "S2"))
        assert 0 < results[-1].significance <= D_MAX

    def test_expand_s3_route(self):
        """S3 route yields lower significance than equivalent S2 route."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        results = list(m.expand(q, c, "S3"))
        # S3 routes level_distance to the s3 (upper bits) component.
        # distance=400 (packed in upper bits) → lower significance than if in S2.
        # Significance should be less than the S2 equivalent with same mismatch.
        sig_s3 = results[-1].significance
        # Also run as S2 for comparison — same mismatched nodes, S2 distance is lower
        results_s2 = list(m.expand(q, c, "S2"))
        sig_s2 = results_s2[-1].significance
        assert sig_s3 < sig_s2  # S3 is less significant than S2 for same mismatch

    def test_expand_significance_ordering(self):
        """Verify significance ordering: closer match → higher significance."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])     # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])    # mismatched_c: {10, 3}
        # distance=301 (q-node 5 reaches c-node 10 at hop 1, rest MAX_HOP)
        results = list(m.expand(q, c, "S2"))
        assert results[-1].significance == (~301) & MASK64

    def test_expand_connotation_bridging(self):
        """Connotation bridging: indirect path through intermediate signature."""
        m = make_model()
        m.add(KLine(60, [60]))  # canonical
        m.add(KLine(50, [60]))  # non-canon: edge_hops(50) = [(1, 60)]
        m.add(KLine(70, [60]))  # non-canon: edge_hops(70) = [(1, 60)]

        q = KLine(100, [50])     # mismatched_q: {50}
        c = KLine(200, [70])     # mismatched_c: {70}
        # Terminal distance: s2=100 (level_distance), s3=0 → packed=100
        # Connotation: packed=(102<<32)=4398046511104
        results = list(m.expand(q, c, "S2"))
        assert len(results) == 2

        # S3 connotation yield: packed=(102<<32)
        connotation = results[0]
        assert connotation.query.signature == 70
        assert connotation.candidate.signature == 60
        # Connotation has higher distance → lower significance than terminal
        connotation_distance = (102 << 32)  # s3 component only
        assert connotation.significance == (~connotation_distance) & MASK64

        # Terminal yield: packed=100
        terminal = results[1]
        assert terminal.query is q
        assert terminal.candidate is c
        terminal_distance = 100  # s2 component only
        assert terminal.significance == (~terminal_distance) & MASK64
        # Terminal has lower distance → higher significance than connotation
        assert terminal.significance > connotation.significance

        # S3 route: level_distance goes to s3 instead
        results_s3 = list(m.expand(q, c, "S3"))
        assert len(results_s3) == 2
        terminal_s3 = results_s3[-1]
        # S3 terminal: s3=100, s2=0 → packed=(100<<32)=429496729600
        s3_route_distance = (100 << 32)
        assert terminal_s3.significance == (~s3_route_distance) & MASK64
        # S3 route has higher distance (in upper bits) → lower significance than S2 route
        assert terminal_s3.significance < terminal.significance

    def test_expand_significance_in_range(self):
        """Significance is always in valid uint64 range."""
        m = make_model()
        q = KLine(5, list(range(1000)))
        c = KLine(6, list(range(1000, 2000)))
        results = list(m.expand(q, c, "S2"))
        sig = results[-1].significance
        assert 0 < sig <= D_MAX


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
