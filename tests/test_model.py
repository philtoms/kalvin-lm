import pytest
from kalvin.model import (
    KLine,
    KLineType,
    HIGH_BIT_MASK,
    get_node_type,
    create_node_key,
    create_embedding_key,
    nodes_equal,
    add_kline,
    query_significance,
)


class TestKLineType:
    def test_kline_type_values(self):
        """Test KLineType enum values."""
        assert KLineType.NODE == 1
        assert KLineType.EMBEDDING == 0

    def test_high_bit_mask(self):
        """Test that HIGH_BIT_MASK is the correct bit 63 mask."""
        assert HIGH_BIT_MASK == 0x8000_0000_0000_0000
        assert HIGH_BIT_MASK == (1 << 63)


class TestKNode:
    def test_get_node_type_node(self):
        """Test get_node_type returns NODE for high bit = 1."""
        node = 0x8000_0000_0000_0000
        assert get_node_type(node) == KLineType.NODE

    def test_get_node_type_embedding(self):
        """Test get_node_type returns EMBEDDING for high bit = 0."""
        node = 0x7FFF_FFFF_FFFF_FFFF
        assert get_node_type(node) == KLineType.EMBEDDING

    def test_create_node_key(self):
        """Test create_node_key sets high bit."""
        key = create_node_key(0x7FFF_0000_0000_0000)
        assert key == 0xFFFF_0000_0000_0000
        assert get_node_type(key) == KLineType.NODE

    def test_create_embedding_key(self):
        """Test create_embedding_key clears high bit."""
        key = create_embedding_key(0x7FFF_0000_0000_0000)
        assert key == 0x7FFF_0000_0000_0000
        assert get_node_type(key) == KLineType.EMBEDDING

    def test_create_node_key_rejects_high_bit(self):
        """Test create_node_key raises AssertionError when high bit already set."""
        with pytest.raises(AssertionError, match="Key value must not use high bit"):
            create_node_key(0x8000_0000_0000_0001)

    def test_create_embedding_key_rejects_high_bit(self):
        """Test create_embedding_key raises AssertionError when high bit already set."""
        with pytest.raises(AssertionError, match="Key value must not use high bit"):
            create_embedding_key(0x8000_0000_0000_0001)


class TestKLine:
    def test_create_kline(self):
        """Test creating a KLine with int s_key and list of KNode ints."""
        s_key = 0x123456789ABCDEF0
        nodes = [0x1000, 0x2000]

        kl = KLine(s_key=s_key, nodes=nodes)

        assert kl.s_key == s_key
        assert kl.nodes == [0x1000, 0x2000]

    def test_type_property_node(self):
        """Test that type property returns NODE when high bit is 1."""
        kl = KLine(s_key=0x8000_0000_0000_0000, nodes=[])
        assert kl.type == KLineType.NODE

    def test_type_property_embedding(self):
        """Test that type property returns EMBEDDING when high bit is 0."""
        kl = KLine(s_key=0x7FFF_FFFF_FFFF_FFFF, nodes=[])
        assert kl.type == KLineType.EMBEDDING

    def test_create_node_factory(self):
        """Test create_node factory sets high bit."""
        kl = KLine.create_node(s_key=0x7FFF_0000_0000_0000, nodes=[0x100, 0x200])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0xFFFF_0000_0000_0000
        assert kl.nodes == [0x100, 0x200]

    def test_create_node_factory_preserves_high_bit(self):
        """Test create_node preserves key when high bit already set."""
        kl = KLine.create_node(s_key=0x9234_5678_9ABC_DEF0, nodes=[])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0x9234_5678_9ABC_DEF0

    def test_create_embedding_factory(self):
        """Test create_embedding factory clears high bit."""
        kl = KLine.create_embedding(s_key=0x9234_5678_9ABC_DEF0, nodes=[0x100])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x1234_5678_9ABC_DEF0
        assert kl.nodes == [0x100]

    def test_create_embedding_factory_idempotent(self):
        """Test create_embedding is idempotent when high bit already cleared."""
        kl = KLine.create_embedding(s_key=0x7FFF_0000_0000_0001, nodes=[])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x7FFF_0000_0000_0001

    def test_store_in_list(self):
        """Test storing KLine objects in a list."""
        kl_list = []

        kl1 = KLine(s_key=0x1000000000000000, nodes=[])
        kl2 = KLine(s_key=0x1000000000000001, nodes=[])
        kl3 = KLine(s_key=0x2000000000000000, nodes=[])

        kl_list.append(kl1)
        kl_list.append(kl2)
        kl_list.append(kl3)

        assert len(kl_list) == 3
        assert kl_list[0].s_key == 0x1000000000000000
        assert kl_list[1].s_key == 0x1000000000000001
        assert kl_list[2].s_key == 0x2000000000000000

    def test_nested_klines_structure(self):
        """Test nested KLine structure with node references."""
        leaf1 = KLine(s_key=0x0100, nodes=[])
        leaf2 = KLine(s_key=0x0200, nodes=[])
        leaf3 = KLine(s_key=0x0300, nodes=[])

        intermediate = KLine(s_key=0x0010, nodes=[0x0100, 0x0200])
        root = KLine(s_key=0x0001, nodes=[0x0010, 0x0300])

        assert len(root.nodes) == 2
        assert root.nodes[0] == 0x0010
        assert root.nodes[1] == 0x0300

        kl_list = [root, intermediate, leaf1, leaf2, leaf3]
        assert len(kl_list) == 5


class TestNodesEqual:
    def test_empty_lists_equal(self):
        """Empty node lists are equal."""
        assert nodes_equal([], []) is True

    def test_same_lists_equal(self):
        """Identical node lists are equal."""
        assert nodes_equal([0x1000, 0x2000], [0x1000, 0x2000]) is True

    def test_different_lengths_not_equal(self):
        """Lists of different lengths are not equal."""
        assert nodes_equal([0x1000], [0x1000, 0x2000]) is False

    def test_different_values_not_equal(self):
        """Lists with different values are not equal."""
        assert nodes_equal([0x1000, 0x2000], [0x1000, 0x3000]) is False


class TestAddKLine:
    def test_add_new_key(self):
        """Adding a kline with new key succeeds."""
        kv_list = []
        kl = KLine(s_key=0x1000, nodes=[])

        result = add_kline(kv_list, kl)

        assert result is True
        assert len(kv_list) == 1
        assert kv_list[0] == kl

    def test_add_duplicate_key_different_nodes(self):
        """Adding kline with same key but different nodes succeeds."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100])
        kl2 = KLine(s_key=0x1000, nodes=[0x0200])
        kv_list = [kl1]

        result = add_kline(kv_list, kl2)

        assert result is True
        assert len(kv_list) == 2

    def test_reject_exact_duplicate(self):
        """Adding exact duplicate (same key and nodes) is rejected."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        kl2 = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        kv_list = [kl1]

        result = add_kline(kv_list, kl2)

        assert result is False
        assert len(kv_list) == 1

    def test_reject_exact_duplicate_empty_nodes(self):
        """Adding exact duplicate with empty nodes is rejected."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x1000, nodes=[])
        kv_list = [kl1]

        result = add_kline(kv_list, kl2)

        assert result is False
        assert len(kv_list) == 1

    def test_multiple_keys_all_added(self):
        """Multiple klines with different keys are all added."""
        kv_list = []
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        kl3 = KLine(s_key=0x3000, nodes=[])

        assert add_kline(kv_list, kl1) is True
        assert add_kline(kv_list, kl2) is True
        assert add_kline(kv_list, kl3) is True
        assert len(kv_list) == 3


class TestQuerySignificance:
    def test_no_match_returns_empty(self):
        """If no kline matches, return empty list."""
        kl1 = KLine(s_key=0x0001, nodes=[])
        kl2 = KLine(s_key=0x0002, nodes=[])

        results = query_significance([kl1, kl2], query=0xFF00)
        assert len(results) == 0

    def test_match_with_only_embeddings_returns_kline(self):
        """If match has only embeddings, return just the matching kline."""
        # EMBEDDING keys have high bit cleared
        embedding1 = create_embedding_key(0x1000)
        embedding2 = create_embedding_key(0x2000)
        matching = KLine(s_key=0xFF00, nodes=[embedding1, embedding2])
        non_matching = KLine(s_key=0x0001, nodes=[])

        results = query_significance([non_matching, matching], query=0xFF00)

        assert len(results) == 1
        assert results[0] == matching

    def test_match_with_no_nodes_returns_kline(self):
        """If match has no nodes, return just the matching kline."""
        matching = KLine(s_key=0xFF00, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])

        results = query_significance([non_matching, matching], query=0xFF00)

        assert len(results) == 1
        assert results[0] == matching

    def test_match_with_child_klines_returns_all(self):
        """If match has child klines, return matching kline + children."""
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)

        child1 = KLine(s_key=key_child1, nodes=[])
        child2 = KLine(s_key=key_child2, nodes=[])
        parent = KLine(s_key=0xFF00, nodes=[key_child1, key_child2])  # references children

        results = query_significance([parent, child1, child2], query=0xFF00, depth=2)

        assert len(results) == 3
        assert results[0] == parent
        assert child1 in results
        assert child2 in results

    def test_depth_limits_expansion(self):
        """Depth parameter limits how many levels of children are expanded."""
        key_grandchild = create_node_key(0x0100)
        key_child = create_node_key(0x0010)
        key_parent = create_node_key(0xF000)  # Use 0xF000 so only parent matches

        grandchild = KLine(s_key=key_grandchild, nodes=[])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        parent = KLine(s_key=key_parent, nodes=[key_child])

        # Query 0xF000 only matches parent (0xF000 in low bits)
        # Child has 0x0010, grandchild has 0x0100 - neither overlaps with 0xF000

        # depth=1: only parent, no child expansion
        results = query_significance([parent, child, grandchild], query=0xF000, depth=1)
        assert len(results) == 1
        assert results[0] == parent

        # depth=2: parent + child, no grandchild
        results = query_significance([parent, child, grandchild], query=0xF000, depth=2)
        assert len(results) == 2
        assert results[0] == parent
        assert results[1] == child

        # depth=3: parent + child + grandchild
        results = query_significance([parent, child, grandchild], query=0xF000, depth=3)
        assert len(results) == 3
        assert results[0] == parent
        assert results[1] == child
        assert results[2] == grandchild

    def test_depth_zero_returns_empty(self):
        """depth=0 returns empty list."""
        matching = KLine(s_key=0xFF00, nodes=[])
        results = query_significance([matching], query=0xFF00, depth=0)
        assert len(results) == 0

    def test_cycle_detection_stops_expansion(self):
        """Circular references stop expansion and return list so far."""
        key_a = create_node_key(0x0001)
        key_b = create_node_key(0x0002)

        kl_a = KLine(s_key=key_a, nodes=[key_b])
        kl_b = KLine(s_key=key_b, nodes=[key_a])

        # kl_a matches, expands to kl_b, which references kl_a (already in result)
        results = query_significance([kl_a, kl_b], query=key_a, depth=100)

        # Should have kl_a and kl_b, then stop when it sees kl_a again
        assert len(results) == 2
        assert kl_a in results
        assert kl_b in results

    def test_self_reference_stops_expansion(self):
        """Self-referencing KLine stops expansion."""
        key = create_node_key(0xFF00)
        kl = KLine(s_key=key, nodes=[key])

        results = query_significance([kl], query=key, depth=100)

        # kl added, then tries to expand self but it's already in result
        assert len(results) == 1
        assert results[0] == kl

    def test_embedding_keys_not_expanded(self):
        """EMBEDDING keys in nodes list are not expanded."""
        embedding_key = create_embedding_key(0x1000)
        node_key = create_node_key(0x2000)

        child = KLine(s_key=node_key, nodes=[])
        parent = KLine(s_key=0xFF00, nodes=[embedding_key, node_key])

        results = query_significance([parent, child], query=0xFF00, depth=2)

        # Only parent and child (from node_key), not any lookup for embedding_key
        assert len(results) == 2
        assert results[0] == parent
        assert child in results

    def test_all_matches_are_returned(self):
        """All matching klines in list order are returned."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])

        # Both match1 and match2 match the query
        results = query_significance([non_matching, match1, match2], query=0xFF00)

        assert len(results) == 2
        assert match1 in results
        assert match2 in results

    def test_cap_limits_top_level_matches(self):
        """Cap limits number of top-level matches."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])

        results = query_significance([match1, match2, match3], query=0xFF00, cap=2)
        assert len(results) == 2

    def test_cap_zero_means_no_limit(self):
        """Cap=0 means no limit on results."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])

        results = query_significance([match1, match2, match3], query=0xFF00, cap=0)
        assert len(results) == 3

    def test_cap_limits_children_per_parent(self):
        """Cap limits children per parent at each level."""
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)
        key_child3 = create_node_key(0x0030)

        child1 = KLine(s_key=key_child1, nodes=[])
        child2 = KLine(s_key=key_child2, nodes=[])
        child3 = KLine(s_key=key_child3, nodes=[])
        parent1 = KLine(s_key=0xFF00, nodes=[key_child1, key_child2, key_child3])

        key_child4 = create_node_key(0x0040)
        key_child5 = create_node_key(0x0050)
        key_child6 = create_node_key(0x0060)

        child4 = KLine(s_key=key_child4, nodes=[])
        child5 = KLine(s_key=key_child5, nodes=[])
        child6 = KLine(s_key=key_child6, nodes=[])
        parent2= KLine(s_key=0xFF00, nodes=[key_child4, key_child5, key_child6])

        # cap=2: (parent + max 2 children) * 2
        results = query_significance([parent1, parent2,child1, child2, child3, child4, child5, child6], query=0xF000, depth=2, cap=2)
        assert len(results) == 6  # 2 parent + 4 children
        assert parent1 in results
        assert child1 in results
        assert child2 in results
        assert child3 not in results  # cap limits to 2 children
        assert parent2 in results
        assert child4 in results
        assert child5 in results
        assert child6 not in results  # cap limits to 2 children

    def test_nested_hierarchy_expansion(self):
        """Test deeply nested hierarchy is expanded correctly."""
        key_leaf1 = create_node_key(0x1000)
        key_leaf2 = create_node_key(0x2000)
        key_leaf3 = create_node_key(0x3000)
        key_intermediate = create_node_key(0x0010)

        leaf1 = KLine(s_key=key_leaf1, nodes=[])
        leaf2 = KLine(s_key=key_leaf2, nodes=[])
        leaf3 = KLine(s_key=key_leaf3, nodes=[])
        intermediate = KLine(s_key=key_intermediate, nodes=[key_leaf1, key_leaf2])
        root = KLine(s_key=0xFF00, nodes=[key_intermediate, key_leaf3])

        results = query_significance(
            [root, intermediate, leaf1, leaf2, leaf3],
            query=0xFF00,
            depth=3
        )

        # root -> intermediate, leaf3
        # intermediate -> leaf1, leaf2
        assert len(results) == 5
        assert root in results
        assert intermediate in results
        assert leaf1 in results
        assert leaf2 in results
        assert leaf3 in results

    def test_cyclic_children_stops_expansion(self):
        """Cyclic children (child references ancestor) stop expansion."""
        # Structure: root -> child -> grandchild -> root (cycle back to top)
        key_root = create_node_key(0xFF00)
        key_child = create_node_key(0x0010)
        key_grandchild = create_node_key(0x0100)

        grandchild = KLine(s_key=key_grandchild, nodes=[key_root])  # cycles back to root
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        root = KLine(s_key=key_root, nodes=[key_child])

        results = query_significance(
            [root, child, grandchild],
            query=0xFF00,
            depth=10
        )

        # root added, then child, then grandchild
        # grandchild references root which is already in results -> stop
        assert len(results) == 3
        assert results[0] == root
        assert results[1] == child
        assert results[2] == grandchild

    def test_cyclic_grandchildren_stops_expansion(self):
        """Cyclic children (child references ancestor) stop expansion."""
        # Structure: root -> child -> grandchild -> child (cycle back to parent)
        key_root = create_node_key(0xFF00)
        key_child = create_node_key(0x0010)
        key_grandchild = create_node_key(0x0100)

        grandchild = KLine(s_key=key_grandchild, nodes=[key_child])  # cycles back to root
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        root = KLine(s_key=key_root, nodes=[key_child])

        results = query_significance(
            [root, child, grandchild],
            query=0xFF00,
            depth=10
        )

        # root added, then child, then grandchild
        # grandchild references child which is already in results -> stop
        assert len(results) == 3
        assert results[0] == root
        assert results[1] == child
        assert results[2] == grandchild
