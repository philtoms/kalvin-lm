"""Tests for the signify method."""

import pytest

from kalvin import Kalvin
from kalvin.model import KLine
from kalvin.significance import (
    S1,S2, S4,
    build_s2,
    build_s3,
    calculate_significance,
    get_s1_percentage,
    has_s1,
)


class TestSignifyReturnsInternal:
    """Tests for returning internal significance."""

    def test_signify_returns_internal_when_none(self):
        """Returns internal significance when s=None."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x30, 0x40])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        internal = calculate_significance(kalvin.model, k1, k2)
        result = kalvin.signify(k1, k2, None)
        assert result == internal

    def test_signify_returns_internal_when_sufficient(self):
        """Returns internal when s <= internal (internal is more significant)."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        # Create matching nodes for S1 match
        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x10, 0x20])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        internal = calculate_significance(kalvin.model, k1, k2)
        # Internal should be S1 (high), so requesting S3 (low) returns internal
        s3_request = build_s3(50, 0, 0)
        result = kalvin.signify(k1, k2, s3_request)
        assert result == internal


class TestSignifyS1:
    """Tests for S1 significance handling."""

    def test_signify_s1_creates_countersigned_link(self):
        """S1 request creates countersigned link in the model."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        initial_count = len(kalvin.model)
        result = kalvin.signify(k1, k2, S1)

        # Should return S1
        assert has_s1(result)

        # Should have added 2 new KLines (countersigned link)
        assert len(kalvin.model) == initial_count + 2

    def test_signify_s1_link_content(self):
        """Verify the content of S1 countersigned link."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20, 0x30])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        kalvin.signify(k1, k2, S1)

        # Check that k1 now has k2's signature
        link1 = kalvin.model.find_kline(k1.signature)
        assert link1 is not None
        # Should find the most recently added one with k2's signature
        assert link1.nodes[-1] == k2.signature


class TestSignifyS2:
    """Tests for S2 significance handling."""

    def test_signify_s2_verifies_compound_match(self):
        """S2 verification succeeds when compound of k2.nodes == k1.signature."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        # Set up so k2.nodes OR'd together equals k1.signature
        # Use non-overlapping node values so internal significance is low (S4)
        compound_sig = 0x1000 | 0x2000  # = 0x3000
        k1 = KLine(signature=compound_sig, nodes=[0x100])
        k2 = KLine(signature=0x4000, nodes=[0x1000, 0x2000])  # nodes OR to compound_sig
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        # Internal significance will be S4 (no match), but we request S2
        s2_request = build_s2(50, 50)
        result = kalvin.signify(k1, k2, s2_request)

        # Should return S2 since compound matches
        from kalvin.significance import get_s2
        assert get_s2(result) > 0

    def test_signify_s2_falls_through_on_mismatch(self):
        """S2 verification failure continues to S3 check."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        # Compound won't match k1.signature, and nodes don't overlap
        k1 = KLine(signature=0x9999, nodes=[0x100])  # Different signature
        k2 = KLine(signature=0x2000, nodes=[0x1000, 0x2000])  # nodes OR to 0x3000
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        # Request S2 + S3 (S2 will fail verification, should fall through to S3)
        s2_request = build_s2(50, 50)
        s3_request = build_s3(50, 0, 0)
        combined_request = s2_request | s3_request

        initial_count = len(kalvin.model)
        result = kalvin.signify(k1, k2, combined_request)

        # Should have fallen through to S3 and created links
        assert len(kalvin.model) > initial_count


class TestSignifyS3:
    """Tests for S3 significance handling."""

    def test_signify_s3_creates_countersigned_link(self):
        """S3 request creates countersigned link."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        initial_count = len(kalvin.model)
        s3_request = build_s3(100, 0, 0)
        result = kalvin.signify(k1, k2, s3_request)

        # Should return S3
        from kalvin.significance import get_s3
        assert get_s3(result) > 0

        # Should have added 1 new KLines
        assert len(kalvin.model) == initial_count + 1


class TestSignifyEdgeCases:
    """Tests for edge cases."""

    def test_signify_with_empty_nodes(self):
        """Test signify with KLines that have empty nodes."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[])
        k2 = KLine(signature=0x200, nodes=[])

        result = kalvin.signify(k1, k2, None)
        # Empty nodes get S1 (perfect match) per calculate_significance
        assert has_s1(result)

    def test_signify_adds_links_each_call(self):
        """Verify that signify adds links (model allows duplicates with train=False)."""
        kalvin = Kalvin.__new__(Kalvin)
        kalvin._model = Kalvin()._model

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20])
        kalvin.model.add(k1)
        kalvin.model.add(k2)

        # First signify
        kalvin.signify(k1, k2, S1)
        count_after_first = len(kalvin.model)

        # Second signify with same params
        # Note: model.add with train=False doesn't dedupe, so links are added again
        kalvin.signify(k1, k2, S1)
        assert len(kalvin.model) == count_after_first + 2  # 2 more linkw added
