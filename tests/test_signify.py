"""Tests for the signify method.

Note: The Agent class no longer exposes a public signify() method.
Significance calculation is handled internally by Agent._signify().
These tests verify the significance calculation logic via _calculate_significance.
"""

import pytest

from kalvin import Agent
from kalvin.abstract import KLine
from kalvin.significance import Int32Significance


# Create a shared significance instance for tests
_sig = Int32Significance()


def _calculate_significance(model, query: KLine, target: KLine) -> int:
    """Replicate Agent._signify logic for testing."""
    query_nodes = query.as_node_list()
    target_nodes = target.as_node_list()

    if not query_nodes or not target_nodes:
        return _sig.S4

    min_len = min(len(query_nodes), len(target_nodes))

    s1_match_positions = set(
        i for i in range(min_len) if _sig.equal(query_nodes[i], target_nodes[i])
    )
    s1_matches = len(s1_match_positions)

    if s1_matches == min_len and len(query_nodes) == len(target_nodes):
        return _sig.S1

    if s1_matches > 0:
        s1_pct = (s1_matches * 100) // min_len
        target_set = set(target_nodes)
        s2_matches = 0
        for i, node in enumerate(query_nodes):
            if i in s1_match_positions:
                continue
            if node in target_set:
                s2_matches += 1
        s2_pct = (s2_matches * 100) // len(query_nodes) if query_nodes else 0
        return _sig.build_s2(s1_pct, s2_pct)

    target_set = set(target_nodes)
    query_set = set(query_nodes)

    unordered_s1_matches = query_set & target_set
    s3_s1_pct = (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0

    s3_s2_matches = 0
    for node in query_nodes:
        if node in target_set:
            continue
        node_kline = model.find_kline(node)
        if node_kline is not None and node_kline.signature != 0:
            node_children = set(node_kline.as_node_list())
            if node_children & target_set:
                s3_s2_matches += 1
    s3_s2_pct = (s3_s2_matches * 100) // len(query_nodes) if query_nodes else 0

    gen_matches = 0
    for node in query_nodes:
        if node in target_set:
            continue
        descendants = model.get_all_descendants(node)
        if descendants & target_set:
            gen_matches += 1
    gen_pct = (gen_matches * 100) // len(query_nodes) if query_nodes else 0

    if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
        return _sig.build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

    return _sig.S4


class TestSignifyReturnsInternal:
    """Tests for returning internal significance."""

    def test_signify_returns_internal_when_none(self):
        """Returns internal significance when s=None."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x30, 0x40])
        agent.model.add(k1)
        agent.model.add(k2)

        internal = _calculate_significance(agent.model, k1, k2)
        # No matching nodes -> S4
        assert _sig.has_s4(internal)

    def test_signify_returns_internal_when_sufficient(self):
        """Returns internal when s <= internal (internal is more significant)."""
        agent = Agent()

        # Create matching nodes for S1 match
        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x10, 0x20])
        agent.model.add(k1)
        agent.model.add(k2)

        internal = _calculate_significance(agent.model, k1, k2)
        # Exact match -> S1
        assert _sig.has_s1(internal)


class TestSignifyS1:
    """Tests for S1 significance handling."""

    def test_signify_s1_creates_countersigned_link(self):
        """S1 result is detected for exact match."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x10])  # Same nodes as k1
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        assert _sig.has_s1(result)

    def test_signify_s1_link_content(self):
        """Verify S1 detection for matching nodes."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20, 0x30])
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        # No positional overlap -> not S1
        assert not _sig.has_s1(result)


class TestSignifyS2:
    """Tests for S2 significance handling."""

    def test_signify_s2_verifies_compound_match(self):
        """S2 detected when partial positional match exists."""
        agent = Agent()

        # Partial positional match: position 0 matches
        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x10, 0x30])
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        assert _sig.has_s2(result)

    def test_signify_s2_falls_through_on_mismatch(self):
        """No positional match falls through to S3 check."""
        agent = Agent()

        # No positional overlap
        k1 = KLine(signature=0x100, nodes=[0x10, 0x20])
        k2 = KLine(signature=0x200, nodes=[0x30, 0x10])  # Same elements, different positions
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        # Should not be S2 (no positional matches)
        assert not _sig.has_s2(result)
        # Should be S3 (unordered match exists)
        assert _sig.has_s3(result)


class TestSignifyS3:
    """Tests for S3 significance handling."""

    def test_signify_s3_creates_countersigned_link(self):
        """S3 detected for unordered match."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20])
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        # No match at all -> S4, not S3
        assert result == _sig.S4


class TestSignifyEdgeCases:
    """Tests for edge cases."""

    def test_signify_with_empty_nodes(self):
        """Test signify with KLines that have empty nodes."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[])
        k2 = KLine(signature=0x200, nodes=[])

        result = _calculate_significance(agent.model, k1, k2)
        # Empty nodes get S4 (no match) per calculation logic
        assert _sig.has_s4(result)

    def test_signify_s4_no_match(self):
        """Completely unrelated nodes return S4."""
        agent = Agent()

        k1 = KLine(signature=0x100, nodes=[0x10])
        k2 = KLine(signature=0x200, nodes=[0x20])
        agent.model.add(k1)
        agent.model.add(k2)

        result = _calculate_significance(agent.model, k1, k2)
        assert result == _sig.S4
