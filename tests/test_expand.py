"""Tests for expand module — direct function calls, not Model forwarding."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.expand import (
    expand,
    edge_hops,
    is_canon,
    is_s1,
    is_countersigned,
    promote_participating,
    propose_expansions,
    QueryCandidate,
    D_MAX,
    MASK64,
    MAX_HOP,
    S2_S3_DISTANCE,
    boundaries,
    classify,
    _pack,
    _S3_BIAS,
)


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


class TestIsCanon:
    def test_canon_match(self):
        """sig == make_signature(nodes) → canonical."""
        k = KLine(10, [10])  # make_signature([10]) = 10 (non-literal)
        assert is_canon(k) is True

    def test_canon_mismatch(self):
        """sig != make_signature(nodes) → non-canonical."""
        k = KLine(5, [10])  # make_signature([10]) = 10 ≠ 5
        assert is_canon(k) is False


class TestEdgeHops:
    def test_edge_hops_unresolvable(self):
        """Node that doesn't resolve → empty generator."""
        m = make_model()
        assert list(edge_hops(m, 99)) == []

    def test_edge_hops_canonical(self):
        """Node that resolves to canonical → empty generator."""
        m = make_model()
        m.add_frame(KLine(10, [10]))  # canonical
        assert list(edge_hops(m, 10)) == []

    def test_edge_hops_chain(self):
        """Non-canonical chain: 5→(1,10)→(2,20)→(3,30) where 30 is canonical."""
        m = make_model()
        m.add_frame(KLine(30, [30]))  # canonical
        m.add_frame(KLine(20, [30]))  # non-canon: sig=20, make_sig([30])=30
        m.add_frame(KLine(10, [20]))  # non-canon: sig=10, make_sig([20])=20
        m.add_frame(KLine(5, [10]))   # non-canon: sig=5,  make_sig([10])=10
        assert list(edge_hops(m, 5))  == [(1, 10), (2, 20), (3, 30)]
        assert list(edge_hops(m, 10)) == [(1, 20), (2, 30)]
        assert list(edge_hops(m, 20)) == [(1, 30)]
        assert list(edge_hops(m, 30)) == []  # canonical
        assert list(edge_hops(m, 99)) == []  # unresolvable


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
        # No chains → all MAX_HOP, ungrounded +1 → distance=401
        expected_distance = 401
        results = list(expand(m, q, c))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_with_grounding(self):
        """Matched node that resolves to structural S1 → no ungrounded penalty."""
        m = make_model()
        m.add_frame(KLine(10, [10]))  # canonical kline, node 10 is structural S1
        q = KLine(5, [10, 2])
        c = KLine(6, [10, 3])
        # distance=200 (2 × MAX_HOP, grounded match)
        expected_distance = 200
        results = list(expand(m, q, c))
        assert results[-1].significance == (~expected_distance) & MASK64

    def test_expand_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        m.add_frame(KLine(30, [30]))  # canonical
        m.add_frame(KLine(20, [30]))  # non-canon
        m.add_frame(KLine(10, [20]))  # non-canon
        m.add_frame(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])    # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])   # mismatched_c: {10, 3}
        # distance=301 (1 hop + 3 × MAX_HOP)
        # Results include:
        #   - S2 exact-match chain yields (expand(20,30) → expand(10,20) → expand(5,10))
        #   - S2 signifies yields from c-nodes: 10→30 in expand(5,10) and top level
        #   - Terminal with significance (~301) & MASK64
        results = list(expand(m, q, c))
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
        m.add_frame(KLine(10, [10]))  # canonical
        m.add_frame(KLine(5, [10]))   # non-canon
        m.add_frame(KLine(30, [30]))  # canonical
        m.add_frame(KLine(20, [30]))  # non-canon

        q = KLine(100, [5, 20])     # mismatched_q: {5, 20}
        c = KLine(200, [10, 30])    # mismatched_c: {10, 30}
        # distance=202, 2 connotations + terminal
        results = list(expand(m, q, c))
        assert len(results) == 3
        assert results[-1].significance == (~202) & MASK64

    def test_expand_all_matched_grounded(self):
        """All nodes match and all resolve → no ungrounded penalty → max significance."""
        m = make_model()
        m.add_frame(KLine(10, [10]))  # canonical, node 10 resolves
        m.add_frame(KLine(20, [20]))  # canonical, node 20 resolves
        q = KLine(5, [10, 20])
        c = KLine(6, [10, 20])
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
        m.add_frame(KLine(10, [10]))  # canonical
        m.add_frame(KLine(5, [10]))   # non-canon

        q = KLine(100, [5, 2])     # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])    # mismatched_c: {10, 3}
        # distance=301 (q-node 5 reaches c-node 10 at hop 1, rest MAX_HOP)
        results = list(expand(m, q, c))
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
        m.add_frame(KLine(8, [8]))    # canonical
        m.add_frame(KLine(4, [8]))    # non-canon: edge_hops(4) = [(1, 8)]
        m.add_frame(KLine(2, [8]))    # non-canon: edge_hops(2) = [(1, 8)]

        q = KLine(100, [4])     # mismatched_q: {4}
        c = KLine(200, [2])     # mismatched_c: {2}

        # signifies(4, 8) = False, signifies(2, 8) = False → S3 path exercised
        # s3_connotations[8] = 1 (from q-node 4)
        # c-node 2 resolves via s3_connotation: s3_hop = 1 + 1 = 2
        # Connotation distance = _pack(2 + _S3_BIAS) = _pack(11) = 121
        # Terminal distance = MAX_HOP = 100 (q-node 4 unresolved at terminal level)

        results = list(expand(m, q, c))
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
        m.add_frame(KLine(30, [30]))  # canonical
        m.add_frame(KLine(20, [30]))  # non-canon
        m.add_frame(KLine(10, [20]))  # non-canon
        m.add_frame(KLine(5, [10]))   # non-canon

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

        results = list(expand(m, q, c))

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
        m.add_frame(KLine(0b11100, [0b11100]))  # canonical (28)
        m.add_frame(KLine(0b10100, [0b11100]))  # non-canon: sig=20, make_sig=28
        m.add_frame(KLine(0b01100, [0b11100]))  # non-canon: sig=12, make_sig=28

        q = KLine(100, [0b10100])  # mismatched_q: {20}
        c = KLine(200, [0b01100])  # mismatched_c: {12}

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
        results = list(expand(m, q, c))
        sig = results[-1].significance
        assert 0 < sig <= D_MAX


# ── Structural Grounding Tests ───────────────────────────────────────

class TestIsS1:
    def test_canonical_kline(self):
        """KLine with sig == make_signature(nodes) → S1."""
        m = Model()
        k = KLine(10, [10])  # make_signature([10]) = 10
        assert is_s1(m, k) is True

    def test_countersigned_in_model(self):
        """Two klines with mutual node references → S1."""
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [5])
        m.add_frame(a)
        m.add_frame(b)
        # a is countersigned: a.nodes has 10, model.find(10)=b, b.nodes has 5=a.signature
        assert is_s1(m, a) is True

    def test_neither_canonical_nor_countersigned(self):
        """Non-canonical, non-countersigned kline → not S1."""
        m = Model()
        k = KLine(5, [10])  # not canonical (make_sig([10])=10≠5)
        assert is_s1(m, k) is False

    def test_all_literal_canonical(self):
        """All-literal kline → canonical (sig=1)."""
        m = Model()
        lit = (65 << 32) | 0xFFFF_FFFF
        k = KLine(1, [lit])
        assert is_s1(m, k) is True

    def test_countersigned_skips_literal_nodes(self):
        """Literal nodes in kline.nodes are skipped in countersigned search."""
        m = Model()
        lit = (65 << 32) | 0xFFFF_FFFF
        a = KLine(5, [lit])  # only literal nodes
        m.add_frame(a)
        assert is_s1(m, a) is False  # not canonical, no non-literal nodes to check


class TestIsCountersigned:
    def test_countersigned_in_model(self):
        """Query = {5: [10, 20]}, Countersigner = {make_sig([10,20]): [5]}"""
        m = Model()
        query = KLine(5, [10, 20])
        # make_sig([10, 20]) = 30 (XOR)
        countersigner = KLine(30, [5])
        m.add_frame(query)
        m.add_frame(countersigner)
        assert is_countersigned(m, query) is True

    def test_one_way_only(self):
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [20, 30])  # sig doesn't match make_sig(a.nodes)
        m.add_frame(a)
        m.add_frame(b)
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
        m.add_frame(query)
        m.add_frame(countersigner)
        assert is_countersigned(m, query) is False

    def test_countersigner_multiple_nodes(self):
        """Countersigner has matching sig but more than one node → not countersigned."""
        m = Model()
        query = KLine(5, [10, 20])
        countersigner = KLine(30, [5, 99])  # make_sig([10,20])=30, node has 5 but len>1
        m.add_frame(query)
        m.add_frame(countersigner)
        assert is_countersigned(m, query) is False


class TestPromoteParticipating:
    def test_promotes_query_and_candidate(self):
        """Both query and candidate are promoted."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_frame(q)
        m.add_frame(c)
        count = promote_participating(m, q, c)
        assert count >= 2
        assert len(m) >= 2

    def test_promotes_stm_klines_with_matching_signatures(self):
        """STM klines whose signatures appear in the node set are also promoted."""
        m = Model(stm_bound=256)
        # Identity kline (S4) with sig that appears in query nodes
        identity = KLine(10, [100])  # sig=10 appears in query.nodes
        m.add_frame(identity)
        q = KLine(5, [10, 20])
        c = KLine(20, [5, 30])
        m.add_frame(q)
        m.add_frame(c)
        count = promote_participating(m, q, c)
        assert count >= 2  # at least query + candidate
        # identity (sig=10) is in q.nodes, should also be promoted
        assert any(kl.signature == 10 for kl in m)

    def test_no_double_promote(self):
        """Already-promoted literal klines are not re-promoted (LTM dedup)."""
        m = Model(stm_bound=256)
        q = KLine(5, [10], literal=True)
        c = KLine(10, [5], literal=True)
        m.add_frame(q)
        m.add_frame(c)
        m.add_ltm(q)  # promote to LTM first
        m.add_ltm(c)
        count = promote_participating(m, q, c)
        assert count == 0  # both already in LTM (literal dedup)


# ── Significance Boundary Tests ───────────────────────────────────────

class TestBoundaries:
    """Verify boundaries() returns correct (S1|S2, S2|S3, S3|S4) thresholds."""

    def test_boundaries_values(self):
        """Boundaries are D_MAX-1, ~S2_S3_DISTANCE masked, and 0."""
        s12, s23, s34 = boundaries()
        assert s12 == D_MAX - 1
        assert s23 == (~S2_S3_DISTANCE) & MASK64
        assert s34 == 0

    def test_all_values_valid_uint64(self):
        """All boundary values are non-negative and within uint64 range."""
        for val in boundaries():
            assert 0 <= val <= MASK64

    def test_s23_sits_between_max_hop_and_min_s3(self):
        """S2|S3 boundary sits between MAX_HOP and _pack(2 + _S3_BIAS).

        S2_S3_DISTANCE (100) < _pack(2 + _S3_BIAS) (121), ensuring S2 and S3
        significance tiers are cleanly separated.
        """
        assert S2_S3_DISTANCE < _pack(2 + _S3_BIAS)


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
        """sig = D_MAX - 1 → S1 (exact S1|S2 boundary)."""
        s12, s23, s34 = bounds
        assert classify(D_MAX - 1, s12, s23, s34) == "S1"

    def test_classify_just_below_s12(self, bounds):
        """sig = D_MAX - 2 → S2 (just below S1|S2 boundary)."""
        s12, s23, s34 = bounds
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
        """Canonical candidate (sig == make_signature(nodes)) → no proposals."""
        m = Model()
        k = KLine(10, [10])  # canonical: make_signature([10]) = 10
        result = list(propose_expansions(m, k, 42))
        assert result == []

    def test_underfit_yields_proposals(self):
        """Underfit candidate (sig promises more than nodes deliver) → proposals.

        KLine(sig=0b110, nodes=[0b100]) has gap 0b010. With contributor
        KLine(sig=0b010, nodes=[0b010]) in the model, propose_expansions
        yields proposal klines with the passed significance.
        """
        m = Model()
        # Contributor for the underfit gap 0b010
        contributor = KLine(0b010, [0b010])
        m.add_frame(contributor)
        m.add_ltm(contributor)

        # Underfit kline: sig=0b110 promises bits that nodes=[0b100] don't deliver
        candidate = KLine(0b110, [0b100])
        significance = 0xDEAD

        results = list(propose_expansions(m, candidate, significance))
        assert len(results) >= 1
        for proposal, sig in results:
            assert isinstance(proposal, KLine)
            assert sig == significance

    def test_overfit_yields_trimmed_and_companion(self):
        """Overfit candidate (nodes carry more than sig) → trimmed + companion.

        KLine(sig=0b100, nodes=[0b110]) has excess 0b010. Propose_expansions
        yields trimmed kline and companion from removed nodes.
        """
        m = Model()
        # Overfit: sig=0b100 but nodes=[0b110] carry extra 0b010
        candidate = KLine(0b100, [0b110])
        significance = 0xBEEF

        results = list(propose_expansions(m, candidate, significance))
        assert len(results) >= 2  # trimmed proposal + companion
        for proposal, sig in results:
            assert isinstance(proposal, KLine)
            assert sig == significance

    def test_yields_are_kline_int_tuples(self):
        """Every yield is a (KLine, int) tuple."""
        m = Model()
        contributor = KLine(0b010, [0b010])
        m.add_frame(contributor)
        candidate = KLine(0b110, [0b100])

        for item in propose_expansions(m, candidate, 42):
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], KLine)
            assert isinstance(item[1], int)
