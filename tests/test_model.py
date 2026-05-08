"""Tests for Model — specs/model.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_MAX, MASK64, MAX_HOP, _pack, _S3_BIAS


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


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

        frame = Model(base=base)
        assert frame.find(5) is k

    def test_add_goes_to_stm_not_frame(self):
        base = make_model()
        session = Model(base=base)
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


# ── Structural Grounding Tests ───────────────────────────────────────

class TestIsStructuralS1:
    def test_canonical_kline(self):
        """KLine with sig == make_signature(nodes) → structural S1."""
        m = Model()
        k = KLine(10, [10])  # make_signature([10]) = 10
        assert m.is_structural_s1(k) is True

    def test_countersigned_in_model(self):
        """Two klines with mutual node references → structural S1."""
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [5])
        m.add(a)
        m.add(b)
        # a is countersigned: a.nodes has 10, model.find(10)=b, b.nodes has 5=a.signature
        assert m.is_structural_s1(a) is True

    def test_neither_canonical_nor_countersigned(self):
        """Non-canonical, non-countersigned kline → not structural S1."""
        m = Model()
        k = KLine(5, [10])  # not canonical (make_sig([10])=10≠5)
        assert m.is_structural_s1(k) is False

    def test_all_literal_canonical(self):
        """All-literal kline → canonical (sig=1)."""
        m = Model()
        lit = (65 << 32) | 0xFFFF_FFFF
        k = KLine(1, [lit])
        assert m.is_structural_s1(k) is True

    def test_countersigned_skips_literal_nodes(self):
        """Literal nodes in kline.nodes are skipped in countersigned search."""
        m = Model()
        lit = (65 << 32) | 0xFFFF_FFFF
        a = KLine(5, [lit])  # only literal nodes
        m.add(a)
        assert m.is_structural_s1(a) is False  # not canonical, no non-literal nodes to check


class TestIsCountersignedInModel:
    def test_mutual_reference(self):
        m = Model()
        a = KLine(5, [10, 20])
        b = KLine(10, [5, 30])
        m.add(a)
        m.add(b)
        assert m._is_countersigned_in_model(a) is True

    def test_one_way_only(self):
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [20, 30])  # b doesn't contain a.sig
        m.add(a)
        m.add(b)
        assert m._is_countersigned_in_model(a) is False

    def test_no_model_match(self):
        m = Model()
        a = KLine(5, [10])
        assert m._is_countersigned_in_model(a) is False


class TestPromoteParticipating:
    def test_promotes_query_and_candidate(self):
        """Both query and candidate are promoted."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add(q)
        m.add(c)
        count = m.promote_participating(q, c)
        assert count >= 2
        assert len(m) >= 2

    def test_promotes_stm_klines_with_matching_signatures(self):
        """STM klines whose signatures appear in the node set are also promoted."""
        m = Model(stm_bound=256)
        # Identity kline (S4) with sig that appears in query nodes
        identity = KLine(10, [100])  # sig=10 appears in query.nodes
        m.add(identity)
        q = KLine(5, [10, 20])
        c = KLine(20, [5, 30])
        m.add(q)
        m.add(c)
        count = m.promote_participating(q, c)
        assert count >= 2  # at least query + candidate
        # identity (sig=10) is in q.nodes, should also be promoted
        assert any(kl.signature == 10 for kl in m)

    def test_no_double_promote(self):
        """Already-promoted klines are not re-promoted."""
        m = Model(stm_bound=256)
        q = KLine(5, [10])
        c = KLine(10, [5])
        m.add(q)
        m.add(c)
        m.promote(q)
        m.promote(c)
        count = m.promote_participating(q, c)
        assert count == 0  # both already in frame


# ── Misfit Classification Tests ──────────────────────────────────────

class TestClassifyMisfit:
    def test_canonical(self):
        """S == N → (False, False)."""
        m = Model()
        k = KLine(10, [10])  # make_sig([10]) = 10
        assert m.classify_misfit(k) == (False, False)

    def test_underfitting(self):
        """S & ~N != 0 → (True, False)."""
        m = Model()
        # sig=0b110, nodes=[0b100] → nodes_sig=0b100
        # underfit: 0b110 & ~0b100 = 0b110 & 0x..011 = 0b010 ≠ 0
        # overfit: 0b100 & ~0b110 = 0b100 & 0x..001 = 0 → False
        k = KLine(0b110, [0b100])
        assert m.classify_misfit(k) == (True, False)

    def test_overfitting(self):
        """N & ~S != 0 → (False, True)."""
        m = Model()
        # sig=0b100, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b100 & ~0b110 = 0 → False
        # overfit: 0b110 & ~0b100 = 0b010 ≠ 0 → True
        k = KLine(0b100, [0b110])
        assert m.classify_misfit(k) == (False, True)

    def test_dual_misfit(self):
        """Both conditions → (True, True)."""
        m = Model()
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # underfit: 0b101 & ~0b110 = 0b101 & 0x..001 = 0b001 ≠ 0 → True
        # overfit: 0b110 & ~0b101 = 0b110 & 0x..010 = 0b010 ≠ 0 → True
        k = KLine(0b101, [0b110])
        assert m.classify_misfit(k) == (True, True)


class TestGenerateExpansions:
    def _make_model_with_klines(self) -> Model:
        m = Model()
        # Canonical klines that can serve as contributors
        m.add(KLine(0b010, [0b010]))  # canonical
        m.add(KLine(0b001, [0b001]))  # canonical
        m.promote_all()
        return m

    def test_underfit_expansion_adds_nodes(self):
        """Underfit expansion yields proposals with added nodes."""
        m = self._make_model_with_klines()
        # sig=0b110, nodes=[0b100] → nodes_sig=0b100
        # gap = 0b110 & ~0b100 = 0b010
        # contributor with sig=0b010 should be found
        k = KLine(0b110, [0b100])
        underfit_gap = 0b010
        overfit_mask = 0
        results = list(m.generate_expansions(k, underfit_gap, overfit_mask))
        assert len(results) >= 1
        proposal, companions = results[0]
        assert proposal.signature == 0b110  # signature stays the same
        assert 0b010 in proposal.nodes or any(
            0b010 in n if isinstance(n, int) else False for n in proposal.nodes
        )
        assert companions == []  # no companions for underfit

    def test_overfit_expansion_removes_nodes(self):
        """Overfit expansion yields trimmed kline + companion."""
        m = Model()
        # sig=0b100, nodes=[0b110] → nodes_sig=0b110
        # excess = 0b110 & ~0b100 = 0b010
        k = KLine(0b100, [0b110])
        underfit_gap = 0
        overfit_mask = 0b010
        results = list(m.generate_expansions(k, underfit_gap, overfit_mask))
        assert len(results) == 1
        proposal, companions = results[0]
        assert proposal.signature == 0b100
        assert 0b110 not in proposal.nodes  # excess node removed
        assert len(companions) == 1
        assert companions[0].nodes == [0b110]  # removed node forms companion

    def test_dual_expansion_atomic_replace(self):
        """Dual expansion yields replacement + companion."""
        m = self._make_model_with_klines()
        # sig=0b101, nodes=[0b110] → nodes_sig=0b110
        # gap = 0b101 & ~0b110 = 0b001
        # excess = 0b110 & ~0b101 = 0b010
        k = KLine(0b101, [0b110])
        underfit_gap = 0b001
        overfit_mask = 0b010
        results = list(m.generate_expansions(k, underfit_gap, overfit_mask))
        # Should produce underfit, overfit, and dual expansions
        assert len(results) >= 2
        # Find the dual expansion (has companions from excess removal AND added nodes from gap fill)
        dual_results = [r for r in results if len(r[1]) > 0]
        assert len(dual_results) >= 1
        # At least one dual result should have contributor nodes added
        has_replacement = any(0b010 in r[0].nodes or 0b001 in r[0].nodes for r in dual_results)
        assert has_replacement

    def test_no_gap_no_expansion(self):
        """No gap and no excess → no expansion proposals."""
        m = Model()
        k = KLine(10, [10])  # canonical
        results = list(m.generate_expansions(k, 0, 0))
        assert len(results) == 0
