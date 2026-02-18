import pytest
from kalvin.model import (
    KLine,
    KLineType,
    HIGH_BIT_MASK,
    get_node_type,
    create_node_key,
    create_embedding_key,
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
        key = create_embedding_key(0xFFFF_0000_0000_0000)
        assert key == 0x7FFF_0000_0000_0000
        assert get_node_type(key) == KLineType.EMBEDDING


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

        grandchild = KLine(s_key=key_grandchild, nodes=[])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        parent = KLine(s_key=0xFF00, nodes=[key_child])

        # depth=1: only parent, no child expansion
        results = query_significance([parent, child, grandchild], query=0xFF00, depth=1)
        assert len(results) == 1
        assert results[0] == parent

        # depth=2: parent + child, no grandchild
        results = query_significance([parent, child, grandchild], query=0xFF00, depth=2)
        assert len(results) == 2
        assert results[0] == parent
        assert results[1] == child

        # depth=3: parent + child + grandchild
        results = query_significance([parent, child, grandchild], query=0xFF00, depth=3)
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

    def test_first_match_in_list_is_returned(self):
        """First matching kline in list order is returned."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])

        # match1 comes before match2
        results = query_significance([non_matching, match1, match2], query=0xFF00)

        assert len(results) == 1
        assert results[0] == match1  # match1 found first, match2 never checked

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
        # Structure: root -> child -> grandchild -> child (cycle back to top)
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
