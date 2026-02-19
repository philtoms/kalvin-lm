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
    expand_significance,
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
        """If no kline matches, both streams are empty."""
        kl1 = KLine(s_key=0x0001, nodes=[])
        kl2 = KLine(s_key=0x0002, nodes=[])

        fast, slow = query_significance([kl1, kl2], query=0xFF00)
        assert list(fast) == []
        assert list(slow) == []

    def test_single_match_in_fast(self):
        """If match found, it's in the fast stream."""
        matching = KLine(s_key=0xFF00, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])

        fast, slow = query_significance([non_matching, matching], query=0xFF00)

        assert list(fast) == [matching]
        assert list(slow) == []

    def test_all_matches_in_fast_when_no_limit(self):
        """All matching klines in fast when focus_limit=0."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])

        fast, slow = query_significance([non_matching, match1, match2], query=0xFF00)

        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_focus_limit_splits_streams(self):
        """focus_limit splits matches into fast and slow streams."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])

        fast, slow = query_significance([match1, match2, match3], query=0xFF00, focus_limit=2)

        assert list(fast) == [match1, match2]
        assert list(slow) == [match3]

    def test_streams_are_independent(self):
        """Fast and slow streams can be consumed independently."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])

        fast, slow = query_significance([match1, match2, match3], query=0xFF00, focus_limit=1)

        # Consume fast first
        fast_list = list(fast)
        assert fast_list == [match1]

        # Slow is still available
        slow_list = list(slow)
        assert slow_list == [match2, match3]


class TestExpandSignificance:
    def test_depth_one_returns_klines_only(self):
        """depth=1 returns klines without expansion."""
        key_child = create_node_key(0x0010)
        parent = KLine(s_key=0xFF00, nodes=[key_child])
        child = KLine(s_key=key_child, nodes=[])

        fast_q, _ = query_significance([parent, child], query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance([parent, child], klines, depth=1)
        results = list(fast)

        assert len(results) == 1
        assert results[0] == parent

    def test_depth_expands_children(self):
        """depth=2 expands direct children."""
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)

        child1 = KLine(s_key=key_child1, nodes=[])
        child2 = KLine(s_key=key_child2, nodes=[])
        parent = KLine(s_key=0xFF00, nodes=[key_child1, key_child2])

        fast_q, _ = query_significance([parent, child1, child2], query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance([parent, child1, child2], klines, depth=2)
        results = list(fast)

        assert len(results) == 3
        assert results[0] == parent
        assert child1 in results
        assert child2 in results

    def test_depth_limits_expansion(self):
        """Depth parameter limits how many levels of children are expanded."""
        key_grandchild = create_node_key(0x0100)
        key_child = create_node_key(0x0010)
        key_parent = create_node_key(0xF000)

        grandchild = KLine(s_key=key_grandchild, nodes=[])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        parent = KLine(s_key=key_parent, nodes=[key_child])

        kv_list = [parent, child, grandchild]
        fast_q, _ = query_significance(kv_list, query=0xF000)
        klines = list(fast_q)

        # depth=1: only parent, no child expansion
        fast, _ = expand_significance(kv_list, klines, depth=1)
        results = list(fast)
        assert len(results) == 1
        assert results[0] == parent

        # depth=2: parent + child, no grandchild
        fast, _ = expand_significance(kv_list, klines, depth=2)
        results = list(fast)
        assert len(results) == 2
        assert results[0] == parent
        assert results[1] == child

        # depth=3: parent + child + grandchild
        fast, _ = expand_significance(kv_list, klines, depth=3)
        results = list(fast)
        assert len(results) == 3
        assert results[0] == parent
        assert results[1] == child
        assert results[2] == grandchild

    def test_depth_zero_returns_empty(self):
        """depth=0 returns empty streams."""
        matching = KLine(s_key=0xFF00, nodes=[])
        fast_q, _ = query_significance([matching], query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance([matching], klines, depth=0)
        assert list(fast) == []
        assert list(slow) == []

    def test_cycle_detection_stops_expansion(self):
        """Circular references stop expansion."""
        key_a = create_node_key(0x0001)
        key_b = create_node_key(0x0002)

        kl_a = KLine(s_key=key_a, nodes=[key_b])
        kl_b = KLine(s_key=key_b, nodes=[key_a])

        kv_list = [kl_a, kl_b]
        fast_q, _ = query_significance(kv_list, query=key_a)
        klines = list(fast_q)

        fast, _ = expand_significance(kv_list, klines, depth=100)
        results = list(fast)

        assert len(results) == 2
        assert kl_a in results
        assert kl_b in results

    def test_self_reference_stops_expansion(self):
        """Self-referencing KLine stops expansion."""
        key = create_node_key(0xFF00)
        kl = KLine(s_key=key, nodes=[key])

        fast_q, _ = query_significance([kl], query=key)
        klines = list(fast_q)

        fast, _ = expand_significance([kl], klines, depth=100)
        results = list(fast)

        assert len(results) == 1
        assert results[0] == kl

    def test_embedding_keys_not_expanded(self):
        """EMBEDDING keys in nodes list are not expanded."""
        embedding_key = create_embedding_key(0x1000)
        node_key = create_node_key(0x2000)

        child = KLine(s_key=node_key, nodes=[])
        parent = KLine(s_key=0xFF00, nodes=[embedding_key, node_key])

        kv_list = [parent, child]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, _ = expand_significance(kv_list, klines, depth=2)
        results = list(fast)

        assert len(results) == 2
        assert results[0] == parent
        assert child in results

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

        kv_list = [root, intermediate, leaf1, leaf2, leaf3]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, _ = expand_significance(kv_list, klines, depth=3)
        results = list(fast)

        assert len(results) == 5
        assert root in results
        assert intermediate in results
        assert leaf1 in results
        assert leaf2 in results
        assert leaf3 in results

    def test_cyclic_children_stops_expansion(self):
        """Cyclic children (child references ancestor) stop expansion."""
        key_root = create_node_key(0xFF00)
        key_child = create_node_key(0x0010)
        key_grandchild = create_node_key(0x0100)

        grandchild = KLine(s_key=key_grandchild, nodes=[key_root])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        root = KLine(s_key=key_root, nodes=[key_child])

        kv_list = [root, child, grandchild]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, _ = expand_significance(kv_list, klines, depth=10)
        results = list(fast)

        assert len(results) == 3
        assert results[0] == root
        assert results[1] == child
        assert results[2] == grandchild

    def test_cyclic_grandchildren_stops_expansion(self):
        """Cyclic grandchildren (grandchild references parent) stop expansion."""
        key_root = create_node_key(0xFF00)
        key_child = create_node_key(0x0010)
        key_grandchild = create_node_key(0x0100)

        grandchild = KLine(s_key=key_grandchild, nodes=[key_child])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        root = KLine(s_key=key_root, nodes=[key_child])

        kv_list = [root, child, grandchild]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, _ = expand_significance(kv_list, klines, depth=10)
        results = list(fast)

        assert len(results) == 3
        assert results[0] == root
        assert results[1] == child
        assert results[2] == grandchild

    def test_focus_limit_splits_streams(self):
        """focus_limit in expand_significance splits into fast and slow."""
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)
        key_child3 = create_node_key(0x0030)

        child1 = KLine(s_key=key_child1, nodes=[])
        child2 = KLine(s_key=key_child2, nodes=[])
        child3 = KLine(s_key=key_child3, nodes=[])
        parent1 = KLine(s_key=0xF000, nodes=[key_child1, key_child2, key_child3])

        key_child4 = create_node_key(0x0040)
        key_child5 = create_node_key(0x0050)
        key_child6 = create_node_key(0x0060)

        child4 = KLine(s_key=key_child4, nodes=[])
        child5 = KLine(s_key=key_child5, nodes=[])
        child6 = KLine(s_key=key_child6, nodes=[])
        parent2 = KLine(s_key=0xF001, nodes=[key_child4, key_child5, key_child6])

        kv_list = [parent1, parent2, child1, child2, child3, child4, child5, child6]
        fast_q, _ = query_significance(kv_list, query=0xF000)
        klines = list(fast_q)

        # focus_limit=1: parent1 + children in fast, parent2 + children in slow
        fast, slow = expand_significance(kv_list, klines, depth=2, focus_limit=1)

        fast_results = list(fast)
        assert len(fast_results) == 4  # parent1 + 3 children
        assert parent1 in fast_results
        assert child1 in fast_results
        assert child2 in fast_results
        assert child3 in fast_results

        slow_results = list(slow)
        assert len(slow_results) == 4  # parent2 + 3 children
        assert parent2 in slow_results
        assert child4 in slow_results
        assert child5 in slow_results
        assert child6 in slow_results

    def test_expand_without_focus_limit(self):
        """Without focus_limit, all klines in fast stream."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])

        kv_list = [match1, match2]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=1)
        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_slow_empty_when_focus_limit_zero(self):
        """When focus_limit=0, slow is empty (all in fast)."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])

        kv_list = [match1, match2]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=1, focus_limit=0)
        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_slow_stream_consumed_independently(self):
        """Slow stream can be consumed before fast stream."""
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)

        child1 = KLine(s_key=key_child1, nodes=[])
        child2 = KLine(s_key=key_child2, nodes=[])
        parent1 = KLine(s_key=0xF000, nodes=[key_child1])
        parent2 = KLine(s_key=0xF001, nodes=[key_child2])

        kv_list = [parent1, parent2, child1, child2]
        fast_q, _ = query_significance(kv_list, query=0xF000)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=2, focus_limit=1)

        # Consume slow first
        slow_results = list(slow)
        assert len(slow_results) == 2  # parent2 + child2
        assert parent2 in slow_results
        assert child2 in slow_results

        # Fast still works
        fast_results = list(fast)
        assert len(fast_results) == 2  # parent1 + child1
        assert parent1 in fast_results
        assert child1 in fast_results

    def test_slow_with_larger_focus_limit(self):
        """focus_limit larger than klines puts all in fast, empty slow."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])

        kv_list = [match1, match2]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=1, focus_limit=10)
        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_slow_with_nested_hierarchy(self):
        """Slow stream expands nested hierarchy correctly."""
        key_grandchild1 = create_node_key(0x0100)
        key_grandchild2 = create_node_key(0x0200)
        key_child1 = create_node_key(0x0010)
        key_child2 = create_node_key(0x0020)

        grandchild1 = KLine(s_key=key_grandchild1, nodes=[])
        grandchild2 = KLine(s_key=key_grandchild2, nodes=[])
        child1 = KLine(s_key=key_child1, nodes=[key_grandchild1])
        child2 = KLine(s_key=key_child2, nodes=[key_grandchild2])
        parent1 = KLine(s_key=0xF000, nodes=[key_child1])
        parent2 = KLine(s_key=0xF001, nodes=[key_child2])

        kv_list = [parent1, parent2, child1, child2, grandchild1, grandchild2]
        fast_q, _ = query_significance(kv_list, query=0xF000)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=3, focus_limit=1)

        # Fast: parent1 + child1 + grandchild1
        fast_results = list(fast)
        assert len(fast_results) == 3
        assert parent1 in fast_results
        assert child1 in fast_results
        assert grandchild1 in fast_results

        # Slow: parent2 + child2 + grandchild2
        slow_results = list(slow)
        assert len(slow_results) == 3
        assert parent2 in slow_results
        assert child2 in slow_results
        assert grandchild2 in slow_results

    def test_slow_depth_limits_expansion(self):
        """Depth parameter limits expansion in slow stream too."""
        key_grandchild = create_node_key(0x0100)
        key_child = create_node_key(0x0010)

        grandchild = KLine(s_key=key_grandchild, nodes=[])
        child = KLine(s_key=key_child, nodes=[key_grandchild])
        parent1 = KLine(s_key=0xF000, nodes=[key_child])
        parent2 = KLine(s_key=0xF001, nodes=[key_child])

        kv_list = [parent1, parent2, child, grandchild]
        fast_q, _ = query_significance(kv_list, query=0xF000)
        klines = list(fast_q)

        # depth=1: no child expansion
        fast, slow = expand_significance(kv_list, klines, depth=1, focus_limit=1)
        assert list(fast) == [parent1]
        assert list(slow) == [parent2]

        # depth=2: expand to children, not grandchildren
        fast, slow = expand_significance(kv_list, klines, depth=2, focus_limit=1)
        fast_results = list(fast)
        assert len(fast_results) == 2  # parent1 + child
        slow_results = list(slow)
        assert len(slow_results) == 2  # parent2 + child

    def test_multiple_klines_in_slow(self):
        """Multiple klines go to slow when focus_limit is small."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])
        match4 = KLine(s_key=0xFF03, nodes=[])

        kv_list = [match1, match2, match3, match4]
        fast_q, _ = query_significance(kv_list, query=0xFF00)
        klines = list(fast_q)

        fast, slow = expand_significance(kv_list, klines, depth=1, focus_limit=1)

        assert list(fast) == [match1]
        slow_results = list(slow)
        assert len(slow_results) == 3
        assert match2 in slow_results
        assert match3 in slow_results
        assert match4 in slow_results
