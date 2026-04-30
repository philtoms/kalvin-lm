"""Tests for Significance Pipeline — specs/significance.md conformance.

Routing uses node-membership testing: does the node value exist in the
candidate's node sequence? No model function is called during routing.
"""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_PACK_SHIFT, D_MAX
from kalvin.significance import (
    compute_significance,
    significance_pipeline,
    no_candidates_result,
    D_MAX as SIG_D_MAX,
)
from kalvin.mod_tokenizer import Mod32Tokenizer

MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def make_model() -> Model:
    t = Mod32Tokenizer()
    return Model(is_literal_fn=t.is_literal)


class TestRouting:
    """Routing is self-contained node-membership testing."""

    def test_all_nodes_match_s1(self):
        """All query nodes exist in candidate's nodes → S1."""
        m = make_model()
        q = KLine(5, [10, 20])
        c = KLine(99, [10, 20, 30])  # both 10 and 20 present
        result = compute_significance(q, c, m)
        assert result.match_count == 2
        assert result.level == "S1"
        assert result.distance == 0
        assert result.significance == MASK64

    def test_some_nodes_match_s2(self):
        """Some query nodes exist in candidate → S2."""
        m = make_model()
        q = KLine(5, [10, 20, 99])
        c = KLine(99, [10, 20, 30])  # 10 and 20 match, 99 does not
        result = compute_significance(q, c, m)
        assert result.match_count == 2
        assert result.level == "S2"
        assert result.distance > 0

    def test_no_nodes_match_s3(self):
        """No query nodes exist in candidate → S3."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        result = compute_significance(q, c, m)
        assert result.match_count == 0
        assert result.level == "S3"
        assert result.distance > 0

    def test_single_node_match_s1(self):
        """Single query node exists in candidate → S1."""
        m = make_model()
        q = KLine(5, [10])
        c = KLine(99, [10, 20])
        result = compute_significance(q, c, m)
        assert result.match_count == 1
        assert result.level == "S1"
        assert result.distance == 0
        assert result.significance == MASK64

    def test_empty_query_s4(self):
        """Empty query → S4."""
        m = make_model()
        q = KLine(0, [])
        c = KLine(10, [1])
        result = compute_significance(q, c, m)
        assert result.level == "S4"


class TestS4NoCandidates:
    def test_no_candidates_result(self):
        result = no_candidates_result()
        assert result.significance == 0
        assert result.distance == SIG_D_MAX
        assert result.level == "S4"
        assert result.match_count == 0
        assert result.total_nodes == 0

    def test_pipeline_empty_candidates(self):
        m = make_model()
        q = KLine(5, [1, 2])
        results = significance_pipeline(q, [], m)
        assert len(results) == 1
        candidate, result = results[0]
        assert candidate is None
        assert result.significance == 0
        assert result.level == "S4"
        assert result.distance == SIG_D_MAX

    def test_pipeline_s4_result_is_no_candidates_result(self):
        m = make_model()
        q = KLine(5, [1, 2])
        results = significance_pipeline(q, [], m)
        _, pipeline_result = results[0]
        assert pipeline_result == no_candidates_result()


class TestSignificanceOrdering:
    """S1 > S2 > S3 > S4 by significance value."""

    def test_s1_greater_than_s2(self):
        m = make_model()
        c = KLine(99, [10])
        # S1: all nodes match
        q_s1 = KLine(10, [10])
        r_s1 = compute_significance(q_s1, c, m)
        # S2: only some nodes match
        q_s2 = KLine(10, [10, 20])
        r_s2 = compute_significance(q_s2, c, m)
        assert r_s1.significance > r_s2.significance

    def test_s2_greater_than_s3(self):
        m = make_model()
        c = KLine(99, [5])
        # S2: some nodes match
        q_s2 = KLine(5, [5, 100])
        r_s2 = compute_significance(q_s2, c, m)
        # S3: no nodes match
        q_s3 = KLine(5, [200, 300])
        r_s3 = compute_significance(q_s3, c, m)
        assert r_s2.significance > r_s3.significance

    def test_s3_greater_than_s4(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        r_s3 = compute_significance(q, c, m)
        r_s4 = no_candidates_result()
        assert r_s3.significance > r_s4.significance


class TestDistancePacking:
    def test_s2_distance_in_range(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])  # node 1 matches → S2
        result = compute_significance(q, c, m)
        assert result.distance > 0
        assert result.distance < D_MAX

    def test_s3_distance_in_range(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])  # no nodes match → S3
        result = compute_significance(q, c, m)
        assert result.distance > 0
        assert result.distance < D_MAX


class TestPipelineMultipleCandidates:
    def test_pipeline_returns_all_results(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c1 = KLine(1, [1, 3])
        c2 = KLine(2, [2, 4])
        results = significance_pipeline(q, [c1, c2], m)
        assert len(results) == 2
        # c1: node 1 matches → S2; c2: node 2 matches → S2
        assert all(r.level == "S2" for _, r in results)


class TestRoutingSelfContained:
    """Verify routing does not call model.is_s1."""

    def test_routing_uses_membership_not_is_s1(self):
        """Node that is_s1 would reject but membership accepts still routes correctly.

        model.is_s1(node, candidate) checks node == candidate.signature.
        Routing checks node in candidate.nodes.
        These are fundamentally different tests.
        """
        m = make_model()
        # Node 10 is in candidate.nodes but node 10 != candidate.signature (99)
        # Old behaviour (is_s1): 10 != 99 → no match → S3
        # New behaviour (membership): 10 is in [10, 20] → match → S2
        q = KLine(5, [10])
        c = KLine(99, [10, 20])
        result = compute_significance(q, c, m)
        assert result.match_count == 1
        assert result.level == "S1"
        assert result.distance == 0

    def test_routing_independent_of_candidate_signature(self):
        """Routing only cares about candidate's node sequence, not signature."""
        m = make_model()
        q = KLine(5, [42])
        c = KLine(999, [42, 100])  # 42 is in nodes regardless of sig
        result = compute_significance(q, c, m)
        assert result.match_count == 1
        assert result.level == "S1"
