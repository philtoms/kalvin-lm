"""Tests for Model — specs/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_MAX, MASK64, MAX_HOP, _pack, _S3_BIAS
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
        results = list(m.expand(k, k))
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
        results = list(m.expand(q, c))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_with_grounding(self):
        """Matched node that resolves → no ungrounded penalty."""
        m = make_model()
        m.add(KLine(1, [10]))  # node 1 resolves
        q = KLine(5, [1, 2])
        c = KLine(6, [1, 3])
        # distance=200 (2 × MAX_HOP, grounded match)
        expected_distance = 200
        results = list(m.expand(q, c))
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
        # distance=301 (1 hop + 3 × MAX_HOP)
        # Results include:
        #   - S2 exact-match chain yields (expand(20,30) → expand(10,20) → expand(5,10))
        #   - S2 signifies yields from c-nodes: 10→30 in expand(5,10) and top level
        #   - Terminal with significance (~301) & MASK64
        results = list(m.expand(q, c))
        assert len(results) == 6
        assert results[-1].significance == (~301) & MASK64

        # S2 signifies candidates: c-node 10 reaches sig 30 (10 & 30 ≠ 0)
        # at distance 2 in both expand(5,10) and top-level
        signifies_results = [r for r in results[:-1]
                            if r.significance == (~2) & MASK64]
        assert len(signifies_results) == 2

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
        results = list(m.expand(q, c))
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
        results = list(m.expand(q, c))
        assert results[-1].significance == D_MAX

    def test_expand_clamped_to_valid(self):
        """Significance is always in valid range [1, D_MAX]."""
        m = make_model()
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))
        results = list(m.expand(q, c))
        assert 0 < results[-1].significance <= D_MAX

    def test_expand_range_s2(self):
        """S2 significance is a valid uint64."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        results = list(m.expand(q, c))
        assert 0 < results[-1].significance <= D_MAX

    def test_expand_level_independent(self):
        """Distance is topology-driven, not level-driven.

        With the simplified distance model, the level parameter does not
        affect distance computation. Same query/candidate → same significance
        regardless of level.
        """
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        sig = list(m.expand(q, c))[-1].significance
        assert sig > 0  # topology drives distance

    def test_expand_significance_ordering(self):
        """Verify significance ordering: closer match → higher significance."""
        m = make_model()
        m.add(KLine(10, [10]))  # canonical
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])     # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])    # mismatched_c: {10, 3}
        # distance=301 (q-node 5 reaches c-node 10 at hop 1, rest MAX_HOP)
        results = list(m.expand(q, c))
        assert results[-1].significance == (~301) & MASK64

    def test_expand_connotation_bridging(self):
        """Connotation bridging: indirect path through intermediate signature.

        S3 connotation hops are biased by _pack(hop_count + _S3_BIAS),
        ensuring S3 distances moderately exceed S2 distances while remaining
        close enough for temperature to bridge the gap.

        Uses powers-of-2 nodes (4, 2, 8) so that signifies() returns False,
        ensuring the S3 connotation path is exercised (signifies short-circuits
        before S3 when signatures share bits).
        """
        m = make_model()
        m.add(KLine(8, [8]))    # canonical
        m.add(KLine(4, [8]))    # non-canon: edge_hops(4) = [(1, 8)]
        m.add(KLine(2, [8]))    # non-canon: edge_hops(2) = [(1, 8)]

        q = KLine(100, [4])     # mismatched_q: {4}
        c = KLine(200, [2])     # mismatched_c: {2}

        # signifies(4, 8) = False, signifies(2, 8) = False → S3 path exercised
        # s3_connotations[8] = 1 (from q-node 4)
        # c-node 2 resolves via s3_connotation: s3_hop = 1 + 1 = 2
        # Connotation distance = _pack(2 + _S3_BIAS) = _pack(11) = 121
        # Terminal distance = MAX_HOP = 100 (q-node 4 unresolved at terminal level)

        results = list(m.expand(q, c))
        assert len(results) == 2

        # S3 connotation yield: distance = _pack(2 + _S3_BIAS)
        connotation = results[0]
        assert connotation.query.signature == 2
        assert connotation.candidate.signature == 8
        connotation_distance = _pack(2 + _S3_BIAS)
        assert connotation.significance == (~connotation_distance) & MASK64

        # Terminal yield: distance = 100 (MAX_HOP for unresolved q-node)
        terminal = results[1]
        assert terminal.query is q
        assert terminal.candidate is c
        assert terminal.significance == (~100) & MASK64

        # Terminal has higher significance than connotation (closer match)
        assert terminal.significance > connotation.significance

        # Verify S3 distance exceeds S2: connotation (121) > terminal (100)
        assert connotation_distance > 100  # S3 packed > S2 raw

    def test_expand_signifies_cogitation(self):
        """S2 signifies loose match yields additional QueryCandidates.

        When a mismatched node's edge hop reaches a signature that shares bits
        (signifies) but isn't an exact match, a QueryCandidate is yielded for
        cogitation. The mismatched node still contributes MAX_HOP to the
        terminal distance (signifies doesn't resolve the mismatch).
        """
        m = make_model()
        m.add(KLine(30, [30]))  # canonical
        m.add(KLine(20, [30]))  # non-canon
        m.add(KLine(10, [20]))  # non-canon
        m.add(KLine(5, [10]))   # non-canon

        q = KLine(100, [5])      # mismatched_q: {5}
        c = KLine(200, [10])     # mismatched_c: {10}

        # q-node 5: edge_hops(5) = [(1,10), (2,20), (3,30)]
        #   hop 1: match_sig=10 IS in mismatched_c → exact match, expand
        #   → expand(5, 10, 1) produces its own results
        #
        # c-node 10: edge_hops(10) = [(1,20), (2,30)]
        #   hop 1: match_sig=20 not in mismatched_q, signifies(10,20)=False
        #   hop 2: match_sig=30 not in mismatched_q, signifies(10,30)=True
        #   → yields QueryCandidate(find(10), find(30), (~2) & MASK64)
        #   hop_distance stays MAX_HOP
        #
        # Terminal: distance = 1 (exact match hop) + MAX_HOP (c-node unresolved)

        results = list(m.expand(q, c))

        # Find the signifies candidate from c-node 10→30
        signifies_candidates = [r for r in results
                                if r.query.signature == 10
                                and r.candidate.signature == 30]
        assert len(signifies_candidates) == 1
        sig_cand = signifies_candidates[0]
        assert sig_cand.significance == (~2) & MASK64

        # Terminal: c-node 10 contributes MAX_HOP (signifies doesn't resolve)
        assert results[-1].significance == (~101) & MASK64

    def test_expand_signifies_before_s3(self):
        """Signifies (S2) takes precedence over s3_connotations (S3).

        When signifies matches, s3_connotations is not populated for that
        signature, preventing S3 connotation bridging for the same hop.
        """
        m = make_model()
        m.add(KLine(0b11100, [0b11100]))  # canonical (28)
        m.add(KLine(0b10100, [0b11100]))  # non-canon: sig=20, make_sig=28
        m.add(KLine(0b01100, [0b11100]))  # non-canon: sig=12, make_sig=28

        q = KLine(100, [0b10100])  # mismatched_q: {20}
        c = KLine(200, [0b01100])  # mismatched_c: {12}

        # q-node 20: edge_hops(20) = [(1, 28)]
        #   28 not in mismatched_c, signifies(20, 28) = True (20 & 28 = 20)
        #   → S2 signifies candidate, break (s3_connotations NOT populated)
        #
        # c-node 12: edge_hops(12) = [(1, 28)]
        #   28 not in mismatched_q, signifies(12, 28) = True (12 & 28 = 12)
        #   → S2 signifies candidate, break

        results = list(m.expand(q, c))
        assert len(results) == 3  # 2 signifies + terminal

        # Both signifies candidates reach sig 28 at distance 1
        assert results[0].candidate.signature == 0b11100
        assert results[0].significance == (~1) & MASK64
        assert results[1].candidate.signature == 0b11100
        assert results[1].significance == (~1) & MASK64

        # Terminal: both mismatched nodes unresolved (2 × MAX_HOP)
        assert results[-1].significance == (~(2 * MAX_HOP)) & MASK64

    def test_expand_significance_in_range(self):
        """Significance is always in valid uint64 range."""
        m = make_model()
        q = KLine(5, list(range(1000)))
        c = KLine(6, list(range(1000, 2000)))
        results = list(m.expand(q, c))
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
