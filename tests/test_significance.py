"""Tests for Significance Pipeline — openspec/significance.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.model import Model, D_BOUNDARY, D_MAX
from kalvin.significance import (
    compute_significance,
    significance_pipeline,
    no_candidates_result,
    D_BOUNDARY as SIG_D_BOUNDARY,
    D_MAX as SIG_D_MAX,
)
from kalvin.mod_tokenizer import Mod32Tokenizer

MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def make_model() -> Model:
    t = Mod32Tokenizer()
    return Model(is_literal_fn=t.is_literal)


class TestComputeSignificance:
    def test_s1_all_nodes_match(self):
        m = make_model()
        # Query nodes that equal candidate signature
        q = KLine(5, [10, 20])
        c = KLine(10, [1, 2])
        # Node 10 matches candidate.signature (10) → S1 for that node
        # Node 20 does not match candidate.signature (10)
        # So s1_count = 1 out of 2 → S2
        result = compute_significance(q, c, m)
        assert result.s1_count == 1
        assert result.level == "S2"

    def test_s1_all_match(self):
        m = make_model()
        # Both nodes equal candidate sig
        q = KLine(10, [10, 10])
        c = KLine(10, [1, 2])
        result = compute_significance(q, c, m)
        assert result.s1_count == 2
        assert result.level == "S1"
        assert result.distance == 0
        assert result.significance == MASK64

    def test_s3_no_match(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        result = compute_significance(q, c, m)
        assert result.s1_count == 0
        assert result.level == "S3"
        assert D_BOUNDARY <= result.distance < D_MAX

    def test_empty_query(self):
        m = make_model()
        q = KLine(0, [])
        c = KLine(10, [1])
        result = compute_significance(q, c, m)
        assert result.level == "S4"

    def test_s1_significance_is_max(self):
        m = make_model()
        q = KLine(10, [10])
        c = KLine(10, [])
        result = compute_significance(q, c, m)
        assert result.significance == MASK64

    def test_s4_significance_is_zero(self):
        result = no_candidates_result()
        assert result.significance == 0
        assert result.level == "S4"

    def test_distance_clamped_s2(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [3, 4])
        result = compute_significance(q, c, m)
        assert 1 <= result.distance < D_BOUNDARY

    def test_distance_clamped_s3(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        result = compute_significance(q, c, m)
        assert D_BOUNDARY <= result.distance < D_MAX


class TestSignificanceOrdering:
    """S1 > S2 > S3 > S4 by significance value."""

    def test_s1_greater_than_s2(self):
        m = make_model()
        q_s1 = KLine(10, [10])
        c = KLine(10, [])
        r_s1 = compute_significance(q_s1, c, m)

        q_s2 = KLine(10, [10, 20])
        r_s2 = compute_significance(q_s2, c, m)

        assert r_s1.significance > r_s2.significance

    def test_s2_greater_than_s3(self):
        m = make_model()
        q_s2 = KLine(5, [5, 100])  # One S1 match
        q_s3 = KLine(5, [200, 300])  # No S1 match
        c = KLine(5, [])

        r_s2 = compute_significance(q_s2, c, m)
        r_s3 = compute_significance(q_s3, c, m)

        assert r_s2.significance > r_s3.significance

    def test_s3_greater_than_s4(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        r_s3 = compute_significance(q, c, m)
        r_s4 = no_candidates_result()
        assert r_s3.significance > r_s4.significance


class TestSignificancePipeline:
    def test_pipeline_returns_results(self):
        m = make_model()
        q = KLine(5, [1, 2])
        c1 = KLine(1, [3])
        c2 = KLine(2, [4])
        results = significance_pipeline(q, [c1, c2], m)
        assert len(results) == 2

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
        """Pipeline S4 result matches no_candidates_result()."""
        m = make_model()
        q = KLine(5, [1, 2])
        results = significance_pipeline(q, [], m)
        _, pipeline_result = results[0]
        assert pipeline_result == no_candidates_result()
