"""Tests for expand module — direct function calls, not Model forwarding."""

import pytest

from kalvin.expand import (
    _S3_BIAS,
    D_MAX,
    MASK64,
    MAX_HOP,
    S2_S3_DISTANCE,
    boundaries,
    classify,
    edge_hops,
    expand,
    is_canon,
    is_countersigned,
    is_s1,
    promote_participating,
    propose_expansions,
)
from kalvin.kline import KLine
from kalvin.model import Model


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


def T(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits of a uint64.

    signifies() masks off the lower (BPE) 32 bits, so node/signature values
    that must participate in significance matching are shifted up here.
    """
    return bits << 32


class TestIsCanon:
    def test_canon_match(self):
        """sig == make_signature(nodes), non-self-referential → canonical."""
        # sig = 0b110 = OR(0b100, 0b010); neither node equals the signature.
        k = KLine(0b110, [0b100, 0b010])
        assert is_canon(k) is True

    def test_canon_mismatch(self):
        """sig != make_signature(nodes) → non-canonical."""
        k = KLine(5, [10])  # make_signature([10]) = 10 ≠ 5
        assert is_canon(k) is False

    def test_self_referential_is_not_canon(self):
        """{S: [S]} is identity, not canon — overrules canon classification."""
        k = KLine(10, [10])  # make_signature([10]) = 10, but self-referential
        assert is_canon(k) is False


class TestEdgeHops:
    def test_edge_hops_unresolvable(self):
        """Node that doesn't resolve → empty generator."""
        m = make_model()
        assert list(edge_hops(m, 99)) == []

    def test_edge_hops_canonical(self):
        """Node that resolves to canonical → empty generator."""
        m = make_model()
        # Genuine canon: sig 0b110 = OR(0b100, 0b010).
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))
        assert list(edge_hops(m, 0b110)) == []

    def test_edge_hops_identity(self):
        """Node that resolves to identity (incl. self-referential) → empty."""
        m = make_model()
        m.add_to_frame(KLine(10, []))  # empty-nodes identity
        assert list(edge_hops(m, 10)) == []
        m.add_to_frame(KLine(30, [30]))  # self-referential identity
        assert list(edge_hops(m, 30)) == []

    def test_edge_hops_chain(self):
        """Non-canonical chain terminates at a genuine canon."""
        m = make_model()
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))  # canonical terminator
        m.add_to_frame(KLine(20, [0b110]))  # non-canon: sig=20, make_sig=[0b110]=0b110
        m.add_to_frame(KLine(10, [20]))  # non-canon: sig=10, make_sig([20])=20
        m.add_to_frame(KLine(5, [10]))  # non-canon: sig=5,  make_sig([10])=10
        assert list(edge_hops(m, 5)) == [(1, 10), (2, 20), (3, 0b110)]
        assert list(edge_hops(m, 10)) == [(1, 20), (2, 0b110)]
        assert list(edge_hops(m, 20)) == [(1, 0b110)]
        assert list(edge_hops(m, 0b110)) == []  # canonical
        assert list(edge_hops(m, 99)) == []  # unresolvable

    def test_edge_hops_cycle_detection_er1(self):
        """ER-1: Countersigned pair produces bounded hops, not MAX_HOP."""
        m = make_model()
        # {A: [B]} ↔ {B: [A]} — mutual non-canonical resolution
        m.add_to_frame(KLine(5, [10]))  # sig=5, make_sig([10])=10
        m.add_to_frame(KLine(10, [5]))  # sig=10, make_sig([5])=5
        hops = list(edge_hops(m, 5))
        # Without cycle detection: 100 hops alternating 10,5,10,5...
        # With cycle detection: at most 2 hops before revisiting sig
        assert len(hops) <= 3
        assert hops == [(1, 10), (2, 5)]

    def test_edge_hops_identity_kline_er2(self):
        """ER-2: Identity kline {A: []} yields zero hops."""
        m = make_model()
        # Identity kline: sig > 0, nodes = []
        # make_signature([]) = 0, so it's not canonical (sig ≠ 0)
        m.add_to_frame(KLine(42, []))  # identity, not canonical
        hops = list(edge_hops(m, 42))
        # Without guard: yields (1, 0) which is a dead end
        # With guard: yields nothing (breaks on sig == 0)
        assert hops == []


class TestExpand:
    def test_expand_self_no_model(self):
        """Self-comparison: all nodes match, ungrounded penalty only."""
        m = make_model()
        k = KLine(10, [10, 20, 30])
        results = list(expand(m, k, k))
        # All nodes match, none resolve → distance=3 → significance=~3
        assert len(results) == 1
        assert results[-1].significance == (~3) & MASK64

    def test_expand_no_resolution(self):
        """Mismatched nodes with no model entries → MAX_HOP each."""
        m = make_model()
        q = KLine(5, [1, 2, 3])
        c = KLine(6, [1, 4, 5])
        # matched: {1}, mismatched_q: {2,3}, mismatched_c: {4,5}
        # No chains → all MAX_HOP, ungrounded matched +1 → distance=401
        expected_distance = 401
        results = list(expand(m, q, c))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_with_grounding(self):
        """Matched node that resolves to structural S1 → no ungrounded penalty."""
        m = make_model()
        # Genuine canon (S1): sig 0b110 = OR(0b100, 0b010).
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))
        q = KLine(5, [0b110, 2])
        c = KLine(6, [0b110, 3])
        # distance = 2 * MAX_HOP (grounded match, 2 unresolved nodes)
        expected_distance = 200
        results = list(expand(m, q, c))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        m.add_to_frame(KLine(T(0b110), [T(0b100), T(0b010)]))  # genuine canon — chain terminator
        m.add_to_frame(KLine(T(20), [T(0b110)]))  # non-canon
        m.add_to_frame(KLine(T(10), [T(20)]))  # non-canon
        m.add_to_frame(KLine(T(5), [T(10)]))  # non-canon

        q = KLine(100, [T(5), T(2)])  # mismatched_q: {5, 2}
        c = KLine(200, [T(10), T(3)])  # mismatched_c: {10, 3}
        # distance=301 (1 hop + 3 × MAX_HOP)
        results = list(expand(m, q, c))
        assert len(results) == 6
        assert results[-1].significance == (~301) & MASK64

        # S2 signifies candidates: c-node 10 reaches sig 0b110 (10 & 0b110 ≠ 0)
        # at distance 2 in both expand(5,10) and top-level
        signifies_results = [r for r in results[:-1] if r.significance == (~2) & MASK64]
        assert len(signifies_results) == 2

    def test_expand_bidirectional_hop_match(self):
        """Both query and candidate mismatched nodes reach opposing sets."""
        m = make_model()
        m.add_to_frame(KLine(10, [10]))  # identity
        m.add_to_frame(KLine(5, [10]))  # non-canon
        m.add_to_frame(KLine(30, [30]))  # identity
        m.add_to_frame(KLine(20, [30]))  # non-canon

        q = KLine(100, [5, 20])  # mismatched_q: {5, 20}
        c = KLine(200, [10, 30])  # mismatched_c: {10, 30}
        # distance=202, 2 connotations + terminal
        results = list(expand(m, q, c))
        assert len(results) == 3
        assert results[-1].significance == (~202) & MASK64

    def test_expand_all_matched_grounded(self):
        """All nodes match and all resolve to S1 → no penalty → max significance."""
        m = make_model()
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))  # genuine canon, node 0b110 is S1
        m.add_to_frame(KLine(0b1100, [0b1000, 0b0100]))  # genuine canon, node 0b1100 is S1
        q = KLine(5, [0b110, 0b1100])
        c = KLine(6, [0b110, 0b1100])
        # distance=0 → significance=D_MAX (all bits set)
        results = list(expand(m, q, c))
        assert results[-1].significance == D_MAX

    def test_expand_clamped_to_valid(self):
        """Significance is always in valid range [1, D_MAX]."""
        m = make_model()
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))
        results = list(expand(m, q, c))
        assert 0 < results[-1].significance <= D_MAX

    def test_expand_range_s2(self):
        """S2 significance is a valid uint64."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        results = list(expand(m, q, c))
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
        sig = list(expand(m, q, c))[-1].significance
        assert sig > 0  # topology drives distance

    def test_expand_significance_ordering(self):
        """Verify significance ordering: closer match → higher significance."""
        m = make_model()
        m.add_to_frame(KLine(10, [10]))  # identity — chain terminator
        m.add_to_frame(KLine(5, [10]))  # non-canon

        q = KLine(100, [5, 2])  # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])  # mismatched_c: {10, 3}
        # distance=301 (q-node 5 reaches c-node 10 at hop 1, rest MAX_HOP)
        results = list(expand(m, q, c))
        assert results[-1].significance == (~301) & MASK64

    def test_expand_connotation_bridging(self):
        """Connotation bridging: indirect path through intermediate signature.

        S3 connotation hops use linear distance S2_S3_DISTANCE + hop_count,
        ensuring S3 distances exceed S2 distances with linear (not quadratic)
        growth.

        Uses powers-of-2 nodes (4, 2, 8) so that signifies() returns False,
        ensuring the S3 connotation path is exercised (signifies short-circuits
        before S3 when signatures share bits).
        """
        m = make_model()
        m.add_to_frame(KLine(8, [8]))  # identity (self-referential) — dead end
        m.add_to_frame(KLine(4, [8]))  # non-canon: edge_hops(4) = [(1, 8)]
        m.add_to_frame(KLine(2, [8]))  # non-canon: edge_hops(2) = [(1, 8)]

        q = KLine(100, [4])  # mismatched_q: {4}
        c = KLine(200, [2])  # mismatched_c: {2}

        # signifies(4, 8) = False, signifies(2, 8) = False → S3 path exercised
        # s3_connotations[8] = 1 (from q-node 4)
        # c-node 2 resolves via s3_connotation: s3_hop = 1 + 1 = 2
        # Connotation linear distance = S2_S3_DISTANCE + 2 = 102. The recursive
        # expand({2:[8]}, {8:[8]}) has a matched node 8 that is identity (not
        # S1), adding +1 ungrounded penalty → 103.
        # Terminal distance = MAX_HOP = 100 (q-node 4 unresolved at terminal level)

        results = list(expand(m, q, c))
        assert len(results) == 2

        # S3 connotation yield: distance = 103 (102 linear + 1 ungrounded)
        connotation = results[0]
        assert connotation.query.signature == 2
        assert connotation.candidate.signature == 8
        connotation_distance = S2_S3_DISTANCE + 2 + 1  # 102 linear + 1 ungrounded
        assert connotation.significance == (~connotation_distance) & MASK64

        # Terminal yield: distance = 100 (MAX_HOP for unresolved q-node)
        terminal = results[1]
        assert terminal.query is q
        assert terminal.candidate is c
        assert terminal.significance == (~100) & MASK64

        # Terminal has higher significance than connotation (closer match)
        assert terminal.significance > connotation.significance

        # Verify S3 distance exceeds S2: connotation (102) > terminal (100)
        assert connotation_distance > 100  # S3 linear > S2 raw

    def test_expand_signifies_cogitation(self):
        """S2 signifies loose match yields additional QueryCandidates.

        When a mismatched node's edge hop reaches a signature that shares bits
        (signifies) but isn't an exact match, a QueryCandidate is yielded for
        cogitation. The mismatched node still contributes MAX_HOP to the
        terminal distance (signifies doesn't resolve the mismatch).
        """
        m = make_model()
        m.add_to_frame(KLine(T(30), [T(30)]))  # identity — chain terminator
        m.add_to_frame(KLine(T(20), [T(30)]))  # non-canon
        m.add_to_frame(KLine(T(10), [T(20)]))  # non-canon
        m.add_to_frame(KLine(T(5), [T(10)]))  # non-canon

        q = KLine(100, [T(5)])  # mismatched_q: {5}
        c = KLine(200, [T(10)])  # mismatched_c: {10}

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

        results = list(expand(m, q, c))

        # Find the signifies candidate from c-node 10→30
        signifies_candidates = [
            r for r in results if r.query.signature == T(10) and r.candidate.signature == T(30)
        ]
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
        m.add_to_frame(KLine(T(0b11100), [T(0b11100)]))  # identity (self-referential) (28)
        m.add_to_frame(KLine(T(0b10100), [T(0b11100)]))  # non-canon: sig=20, make_sig=28
        m.add_to_frame(KLine(T(0b01100), [T(0b11100)]))  # non-canon: sig=12, make_sig=28

        q = KLine(100, [T(0b10100)])  # mismatched_q: {20}
        c = KLine(200, [T(0b01100)])  # mismatched_c: {12}

        # q-node 20: edge_hops(20) = [(1, 28)]
        #   28 not in mismatched_c, signifies(20, 28) = True (20 & 28 = 20)
        #   → S2 signifies candidate, break (s3_connotations NOT populated)
        #
        # c-node 12: edge_hops(12) = [(1, 28)]
        #   28 not in mismatched_q, signifies(12, 28) = True (12 & 28 = 12)
        #   → S2 signifies candidate, break

        results = list(expand(m, q, c))
        assert len(results) == 3  # 2 signifies + terminal

        # Both signifies candidates reach sig 28 at distance 1
        assert results[0].candidate.signature == T(0b11100)
        assert results[0].significance == (~1) & MASK64
        assert results[1].candidate.signature == T(0b11100)
        assert results[1].significance == (~1) & MASK64

        # Terminal: both mismatched nodes unresolved (2 × MAX_HOP)
        assert results[-1].significance == (~(2 * MAX_HOP)) & MASK64

    def test_expand_significance_in_range(self):
        """Significance is always in valid uint64 range."""
        m = make_model()
        q = KLine(5, list(range(1000)))
        c = KLine(6, list(range(1000, 2000)))
        results = list(expand(m, q, c))
        sig = results[-1].significance
        assert 0 < sig <= D_MAX

    def test_expand_no_crash_on_unresolvable_match_sig_er6(self):
        """ER-6: expand() does not crash when edge_hops yields an unresolvable sig."""
        m = make_model()
        # Build a scenario where edge_hops produces match_sig=0 from identity kline
        # Identity kline {42: []} → make_sig([]) = 0, which doesn't resolve
        m.add_to_ltm(KLine(42, []))  # identity
        # Query and candidate that trigger the mismatched-node path
        q = KLine(42 | 10, [42, 10])  # sig includes 42, nodes include 42
        c = KLine(42 | 10, [10])  # partial overlap on 10, mismatched_c = {}
        m.add_to_ltm(c)
        # expand should complete without ValueError
        results = list(expand(m, q, c))
        assert len(results) >= 1  # at least terminal yield

    def test_expand_countersign_cycle_no_crash_er7(self):
        """ER-7: S2 scenario with countersigned klines completes without exception."""
        m = make_model()
        # Uppercase to mirror KLine protocol signatures; noqa to avoid clash with model `m`.
        M = 0x2000  # noqa: N806
        H = 0x100  # noqa: N806
        A = 0x2  # noqa: N806
        MH = M | H  # noqa: N806
        # Identities
        m.add_to_ltm(KLine(M, []))
        m.add_to_ltm(KLine(H, []))
        m.add_to_ltm(KLine(A, []))
        # Countersign pair
        m.add_to_ltm(KLine(M, [H]))
        m.add_to_ltm(KLine(H, [M]))
        # MTS canonical
        m.add_to_ltm(KLine(MH, [M, H]))
        # The S2 query
        query = KLine(MH, [H, A])
        candidate = KLine(M, [H])  # misfit, routes S2
        # Must not raise ValueError
        results = list(expand(m, query, candidate))
        assert len(results) >= 1


# ── Structural Grounding Tests ───────────────────────────────────────


class TestIsS1:
    def test_canonical_kline(self):
        """Genuine canon (sig == make_signature(nodes), non-self-referential) → S1."""
        m = Model()
        # sig 0b110 = OR(0b100, 0b010); a genuine canon.
        k = KLine(0b110, [0b100, 0b010])
        assert is_s1(m, k) is True

    def test_self_referential_is_not_s1(self):
        """{S: [S]} is identity, not canon → not S1 by canon."""
        m = Model()
        k = KLine(10, [10])
        assert is_s1(m, k) is False

    def test_countersigned_in_model(self):
        """Two klines with mutual node references → S1."""
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [5])
        m.add_to_frame(a)
        m.add_to_frame(b)
        # a is countersigned: a.nodes has 10, model.find(10)=b, b.nodes has 5=a.signature
        assert is_s1(m, a) is True

    def test_neither_canonical_nor_countersigned(self):
        """Non-canonical, non-countersigned kline → not S1."""
        m = Model()
        k = KLine(5, [10])  # not canonical (make_sig([10])=10≠5)
        assert is_s1(m, k) is False

    def test_countersigned_skips_unresolved_nodes(self):
        """Unresolved nodes in kline.nodes are skipped in countersigned search."""
        m = Model()
        a = KLine(5, [99])  # node 99 not in model
        m.add_to_frame(a)
        assert is_s1(m, a) is False  # not canonical, no resolved nodes to check


class TestIsCountersigned:
    def test_countersigned_in_model(self):
        """Query = {5: [10, 20]}, Countersigner = {make_sig([10,20]): [5]}"""
        m = Model()
        query = KLine(5, [10, 20])
        # make_sig([10, 20]) = 30 (XOR)
        countersigner = KLine(30, [5])
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query) is True

    def test_one_way_only(self):
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [20, 30])  # sig doesn't match make_sig(a.nodes)
        m.add_to_frame(a)
        m.add_to_frame(b)
        assert is_countersigned(m, a) is False

    def test_no_model_match(self):
        m = Model()
        a = KLine(5, [10])
        assert is_countersigned(m, a) is False

    def test_countersigner_wrong_node(self):
        """Countersigner has matching sig but wrong node → not countersigned."""
        m = Model()
        query = KLine(5, [10, 20])
        countersigner = KLine(30, [99])  # make_sig([10,20])=30, but node != query.sig
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query) is False

    def test_countersigner_multiple_nodes(self):
        """Countersigner has matching sig but more than one node → not countersigned."""
        m = Model()
        query = KLine(5, [10, 20])
        countersigner = KLine(30, [5, 99])  # make_sig([10,20])=30, node has 5 but len>1
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query) is False


class TestPromoteParticipating:
    def test_promotes_query_and_candidate(self):
        """Both query and candidate are promoted to LTM."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        promote_participating(m, q, c)
        assert m.find(q.signature) is not None
        assert m.find(c.signature) is not None

    def test_promotes_stm_klines_with_matching_signatures(self):
        """STM klines whose signatures appear in the node set are also promoted."""
        m = Model(stm_bound=256)
        # Identity kline (S4) with sig that appears in query nodes
        identity = KLine(10, [100])  # sig=10 appears in query.nodes
        m.add_to_frame(identity)
        q = KLine(5, [10, 20])
        c = KLine(20, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        promote_participating(m, q, c)
        # identity (sig=10) is in q.nodes, should also be promoted via LTM cascade
        assert m.find(10) is not None

    def test_no_double_promote(self):
        """Calling promote_participating on already-LTM klines is safe (idempotent)."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        m.add_to_ltm(q)  # promote to LTM first
        m.add_to_ltm(c)
        promote_participating(m, q, c)
        # Klines still exist in the model after double promotion
        assert m.find(q.signature) is not None
        assert m.find(c.signature) is not None

    def test_promote_participating_returns_none(self):
        """promote_participating returns None (void)."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        result = promote_participating(m, q, c)
        assert result is None


# ── Significance Boundary Tests ───────────────────────────────────────


class TestBoundaries:
    """Verify boundaries() returns correct (S1|S2, S2|S3, S3|S4) thresholds."""

    def test_boundaries_values(self):
        """Boundaries are D_MAX, ~S2_S3_DISTANCE masked, and 0."""
        s12, s23, s34 = boundaries()
        assert s12 == D_MAX
        assert s23 == (~S2_S3_DISTANCE) & MASK64
        assert s34 == 0

    def test_all_values_valid_uint64(self):
        """All boundary values are non-negative and within uint64 range."""
        for val in boundaries():
            assert 0 <= val <= MASK64

    def test_s23_sits_between_max_hop_and_min_s3(self):
        """S2|S3 boundary sits between MAX_HOP and linear S3 min distance.

        S2_S3_DISTANCE (100) < S2_S3_DISTANCE + 1 (101), ensuring S2 and S3
        significance tiers are cleanly separated with linear S3 distance.
        """
        min_s3_distance = S2_S3_DISTANCE + _S3_BIAS  # 100 + 1 = 101
        assert S2_S3_DISTANCE < min_s3_distance


class TestClassify:
    """Verify classify() returns correct significance bands."""

    @pytest.fixture()
    def bounds(self):
        return boundaries()

    def test_classify_at_s1_boundary(self, bounds):
        """sig = D_MAX → S1 (maximum significance)."""
        s12, s23, s34 = bounds
        assert classify(D_MAX, s12, s23, s34) == "S1"

    def test_classify_at_s12_exact(self, bounds):
        """sig = D_MAX - 1 (distance 1) → S2 (top of S2, not S1)."""
        s12, s23, s34 = bounds
        assert classify(D_MAX - 1, s12, s23, s34) == "S2"

    def test_classify_just_below_s12(self, bounds):
        """sig = D_MAX - 1 and D_MAX - 2 → S2 (top of S2 and just below S1|S2)."""
        s12, s23, s34 = bounds
        assert classify(D_MAX - 1, s12, s23, s34) == "S2"
        assert classify(D_MAX - 2, s12, s23, s34) == "S2"

    def test_classify_at_s23_boundary(self, bounds):
        """sig = S2|S3 boundary value → S2."""
        s12, s23, s34 = bounds
        assert classify(s23, s12, s23, s34) == "S2"

    def test_classify_in_s3_range(self, bounds):
        """sig = 1 → S3 (above S3|S4)."""
        s12, s23, s34 = bounds
        assert classify(1, s12, s23, s34) == "S3"

    def test_classify_at_s34_boundary(self, bounds):
        """sig = 0 → S3 (0 >= 0 is True, S4 is unreachable for uint64).

        Note: S4 is never produced by classify() since significance values
        are always non-negative uint64 and s34 = 0. S4 only comes from
        the routing path in KAgent._route().
        """
        s12, s23, s34 = bounds
        assert classify(0, s12, s23, s34) == "S3"


class TestProposeExpansions:
    """Verify propose_expansions() yields (KLine, int) tuples for misfits."""

    def test_canonical_yields_nothing(self):
        """Genuine canon (non-self-referential) → no proposals."""
        m = Model()
        # sig 0b110 = OR(0b100, 0b010); a genuine canon.
        k = KLine(0b110, [0b100, 0b010])
        result = list(propose_expansions(m, k, 42))
        assert result == []

    def test_identity_yields_nothing(self):
        """Identity (empty or self-referential) → no proposals.

        An expansion proposal must carry decomposition information; identity
        klines do not.
        """
        m = Model()
        assert list(propose_expansions(m, KLine(10, []), 42)) == []
        assert list(propose_expansions(m, KLine(10, [10]), 42)) == []

    def test_underfit_yields_proposals(self):
        """Underfit candidate (sig promises more than nodes deliver) → proposals.

        KLine(sig=0b110, nodes=[0b100]) has gap 0b010. With a genuine canon
        contributor headed by 0b010 in the model, propose_expansions yields
        proposal klines with the passed significance.
        """
        m = Model()
        # Genuine canon contributor whose signature overlaps the gap 0b010.
        contributor = KLine(0b110, [0b100, 0b010])
        m.add_to_frame(contributor)
        m.add_to_ltm(contributor)

        # Underfit kline: sig=0b110 promises bits that nodes=[0b100] don't deliver
        candidate = KLine(0b110, [0b100])
        significance = 0xDEAD

        results = list(propose_expansions(m, candidate, significance))
        assert len(results) >= 1
        for proposal, sig in results:
            assert isinstance(proposal, KLine)
            assert sig == significance

    def test_overfit_yields_trimmed_and_companion(self):
        """Overfit candidate → trimmed proposal; identity companion is dropped.

        KLine(sig=0b110, nodes=[0b100, 0b010, 0b001]) has excess 0b001.
        Trimming yields the genuine canon `{0b110: [0b100, 0b010]}`, which is
        emitted. The companion from the single excess node would be
        `{0b001: [0b001]}` — identity — and is dropped.
        """
        m = Model()
        candidate = KLine(0b110, [0b100, 0b010, 0b001])
        significance = 0xBEEF

        results = list(propose_expansions(m, candidate, significance))
        assert len(results) == 1
        proposal, sig = results[0]
        assert proposal.signature == 0b110
        assert proposal.nodes == [0b100, 0b010]
        assert sig == significance

    def test_yields_are_kline_int_tuples(self):
        """Every yield is a (KLine, int) tuple."""
        m = Model()
        # Genuine canon contributor whose signature overlaps the gap.
        contributor = KLine(0b011, [0b001, 0b010])
        m.add_to_frame(contributor)
        candidate = KLine(0b110, [0b100])

        for item in propose_expansions(m, candidate, 42):
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], KLine)
            assert isinstance(item[1], int)

    def test_no_self_referential_proposals(self):
        """Self-referential proposals (`{S: [S]}`) are never yielded.

        Regression: a single excess node produced a companion `{n: [n]}`
        and a self-contributor produced a proposal whose nodes included its
        own signature. `{S: [S]}` is identity (not a valid decomposition),
        so it is never emitted as an expansion proposal; under Recency
        Precedence such klines would otherwise displace genuine canons and
        collapse `Model.unpack()` to identity.
        """
        m = Model()
        # Genuine canon contributor headed by 0b100 (overlaps the candidate sig).
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))
        # Overfit candidate whose single excess node (0b010) would yield a
        # self-loop companion {0b010: [0b010]}.
        candidate = KLine(0b100, [0b110])
        for proposal, _ in propose_expansions(m, candidate, 0):
            assert proposal.signature not in proposal.nodes
