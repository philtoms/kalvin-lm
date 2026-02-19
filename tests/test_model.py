import pytest
from kalvin.model import (
    KLine,
    KLineType,
    Model,
    HIGH_BIT_MASK,
    get_node_type,
    create_node_key,
    create_embedding_key,
    nodes_equal,
    # Significance
    Significance,
    S1_BIT,
    S1_PCT_SHIFT,
    S2_SHIFT,
    S3_SHIFT,
    S4_VALUE,
    has_s1,
    get_s1_percentage,
    get_s2,
    get_s2_s1_percentage,
    get_s2_s2_percentage,
    build_s1,
    build_s2,
    get_s3,
    get_s3_s1_percentage,
    get_s3_s2_percentage,
    get_s3_gen_percentage,
    build_s3,
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


class TestModelAdd:
    def test_add_new_key(self):
        """Adding a kline with new key succeeds."""
        model = Model()
        kl = KLine(s_key=0x1000, nodes=[])

        result = model.add(kl)

        assert result is True
        assert len(model) == 1
        assert model[0] == kl

    def test_add_duplicate_key_different_nodes(self):
        """Adding kline with same key but different nodes succeeds."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100])
        kl2 = KLine(s_key=0x1000, nodes=[0x0200])
        model = Model([kl1])

        result = model.add(kl2)

        assert result is True
        assert len(model) == 2

    def test_reject_exact_duplicate(self):
        """Adding exact duplicate (same key and nodes) is rejected."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        kl2 = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        model = Model([kl1])

        result = model.add(kl2)

        assert result is False
        assert len(model) == 1

    def test_reject_exact_duplicate_empty_nodes(self):
        """Adding exact duplicate with empty nodes is rejected."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x1000, nodes=[])
        model = Model([kl1])

        result = model.add(kl2)

        assert result is False
        assert len(model) == 1

    def test_multiple_keys_all_added(self):
        """Multiple klines with different keys are all added."""
        model = Model()
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        kl3 = KLine(s_key=0x3000, nodes=[])

        assert model.add(kl1) is True
        assert model.add(kl2) is True
        assert model.add(kl3) is True
        assert len(model) == 3


class TestModelQuery:
    def test_no_match_returns_empty(self):
        """If no kline matches, both streams are empty."""
        kl1 = KLine(s_key=0x0001, nodes=[])
        kl2 = KLine(s_key=0x0002, nodes=[])
        model = Model([kl1, kl2])

        fast, slow = model.query(query=0xFF00)
        assert list(fast) == []
        assert list(slow) == []

    def test_single_match_in_fast(self):
        """If match found, it's in the fast stream."""
        matching = KLine(s_key=0xFF00, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])
        model = Model([non_matching, matching])

        fast, slow = model.query(query=0xFF00)

        assert list(fast) == [matching]
        assert list(slow) == []

    def test_all_matches_in_fast_when_no_limit(self):
        """All matching klines in fast when focus_limit=0."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        non_matching = KLine(s_key=0x0001, nodes=[])
        model = Model([non_matching, match1, match2])

        fast, slow = model.query(query=0xFF00)

        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_focus_limit_splits_streams(self):
        """focus_limit splits matches into fast and slow streams."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])
        model = Model([match1, match2, match3])

        fast, slow = model.query(query=0xFF00, focus_limit=2)

        assert list(fast) == [match1, match2]
        assert list(slow) == [match3]

    def test_streams_are_independent(self):
        """Fast and slow streams can be consumed independently."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        match3 = KLine(s_key=0xFF02, nodes=[])
        model = Model([match1, match2, match3])

        fast, slow = model.query(query=0xFF00, focus_limit=1)

        # Consume fast first
        fast_list = list(fast)
        assert fast_list == [match1]

        # Slow is still available
        slow_list = list(slow)
        assert slow_list == [match2, match3]


class TestModelExpand:
    def test_depth_one_returns_klines_only(self):
        """depth=1 returns focus_set without expansion."""
        key_child = create_node_key(0x0010)
        parent = KLine(s_key=0xFF00, nodes=[key_child])
        child = KLine(s_key=key_child, nodes=[])
        model = Model([parent, child])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=1)
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
        model = Model([parent, child1, child2])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=2)
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
        model = Model([parent, child, grandchild])

        fast_q, _ = model.query(query=0xF000)
        focus_set = list(fast_q)

        # depth=1: only parent, no child expansion
        fast, _ = model.expand(focus_set, depth=1)
        results = list(fast)
        assert len(results) == 1
        assert results[0] == parent

        # depth=2: parent + child, no grandchild
        fast, _ = model.expand(focus_set, depth=2)
        results = list(fast)
        assert len(results) == 2
        assert results[0] == parent
        assert results[1] == child

        # depth=3: parent + child + grandchild
        fast, _ = model.expand(focus_set, depth=3)
        results = list(fast)
        assert len(results) == 3
        assert results[0] == parent
        assert results[1] == child
        assert results[2] == grandchild

    def test_depth_zero_returns_empty(self):
        """depth=0 returns empty streams."""
        matching = KLine(s_key=0xFF00, nodes=[])
        model = Model([matching])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=0)
        assert list(fast) == []
        assert list(slow) == []

    def test_cycle_detection_stops_expansion(self):
        """Circular references stop expansion."""
        key_a = create_node_key(0x0001)
        key_b = create_node_key(0x0002)

        kl_a = KLine(s_key=key_a, nodes=[key_b])
        kl_b = KLine(s_key=key_b, nodes=[key_a])
        model = Model([kl_a, kl_b])

        fast_q, _ = model.query(query=key_a)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=100)
        results = list(fast)

        assert len(results) == 2
        assert kl_a in results
        assert kl_b in results

    def test_self_reference_stops_expansion(self):
        """Self-referencing KLine stops expansion."""
        key = create_node_key(0xFF00)
        kl = KLine(s_key=key, nodes=[key])
        model = Model([kl])

        fast_q, _ = model.query(query=key)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=100)
        results = list(fast)

        assert len(results) == 1
        assert results[0] == kl

    def test_embedding_keys_not_expanded(self):
        """EMBEDDING keys in nodes list are not expanded."""
        embedding_key = create_embedding_key(0x1000)
        node_key = create_node_key(0x2000)

        child = KLine(s_key=node_key, nodes=[])
        parent = KLine(s_key=0xFF00, nodes=[embedding_key, node_key])
        model = Model([parent, child])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=2)
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
        model = Model([root, intermediate, leaf1, leaf2, leaf3])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=3)
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
        model = Model([root, child, grandchild])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=10)
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
        model = Model([root, child, grandchild])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, _ = model.expand(focus_set, depth=10)
        results = list(fast)

        assert len(results) == 3
        assert results[0] == root
        assert results[1] == child
        assert results[2] == grandchild

    def test_focus_limit_splits_streams(self):
        """focus_limit in expand splits into fast and slow."""
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

        model = Model([parent1, parent2, child1, child2, child3, child4, child5, child6])
        fast_q, _ = model.query(query=0xF000)
        focus_set = list(fast_q)

        # focus_limit=1: parent1 + children in fast, parent2 + children in slow
        fast, slow = model.expand(focus_set, depth=2, focus_limit=1)

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
        model = Model([match1, match2])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=1)
        assert list(fast) == [match1, match2]
        assert list(slow) == []

    def test_slow_empty_when_focus_limit_zero(self):
        """When focus_limit=0, slow is empty (all in fast)."""
        match1 = KLine(s_key=0xFF00, nodes=[])
        match2 = KLine(s_key=0xFF01, nodes=[])
        model = Model([match1, match2])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=1, focus_limit=0)
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
        model = Model([parent1, parent2, child1, child2])

        fast_q, _ = model.query(query=0xF000)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=2, focus_limit=1)

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
        model = Model([match1, match2])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=1, focus_limit=10)
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
        model = Model([parent1, parent2, child1, child2, grandchild1, grandchild2])

        fast_q, _ = model.query(query=0xF000)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=3, focus_limit=1)

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
        model = Model([parent1, parent2, child, grandchild])

        fast_q, _ = model.query(query=0xF000)
        focus_set = list(fast_q)

        # depth=1: no child expansion
        fast, slow = model.expand(focus_set, depth=1, focus_limit=1)
        assert list(fast) == [parent1]
        assert list(slow) == [parent2]

        # depth=2: expand to children, not grandchildren
        fast, slow = model.expand(focus_set, depth=2, focus_limit=1)
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
        model = Model([match1, match2, match3, match4])

        fast_q, _ = model.query(query=0xFF00)
        focus_set = list(fast_q)

        fast, slow = model.expand(focus_set, depth=1, focus_limit=1)

        assert list(fast) == [match1]
        slow_results = list(slow)
        assert len(slow_results) == 3
        assert match2 in slow_results
        assert match3 in slow_results
        assert match4 in slow_results


class TestModelIterators:
    def test_iterate_over_model(self):
        """Can iterate over all KLines in model."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        model = Model([kl1, kl2])

        klines = list(model)
        assert len(klines) == 2
        assert kl1 in klines
        assert kl2 in klines

    def test_getitem_access(self):
        """Can access KLines by index."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        model = Model([kl1, kl2])

        assert model[0] == kl1
        assert model[1] == kl2

    def test_find_by_key(self):
        """Can find KLine by its s_key."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100])
        kl2 = KLine(s_key=0x2000, nodes=[0x0200])
        model = Model([kl1, kl2])

        found = model.find_by_key(0x1000)
        assert found == kl1

        not_found = model.find_by_key(0x3000)
        assert not_found is None


class TestSignificanceHelpers:
    def test_build_s1_100_percent(self):
        """S1 at 100% sets S1 bit with max percentage (127)."""
        sig = build_s1(100)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_build_s1_50_percent(self):
        """S1 at 50% sets S1 bit with 63 in percentage bits."""
        sig = build_s1(50)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 63

    def test_build_s1_0_percent(self):
        """S1 at 0% still sets S1 bit with 0 percentage."""
        sig = build_s1(0)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 0

    def test_build_s1_clamps_negative(self):
        """Negative percentage is clamped to 0."""
        sig = build_s1(-10)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 0

    def test_build_s1_clamps_over_100(self):
        """Percentage over 100 is clamped to 100."""
        sig = build_s1(150)
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_build_s1_default(self):
        """S1 with no percentage defaults to 100%."""
        sig = build_s1()
        assert has_s1(sig) is True
        assert get_s1_percentage(sig) == 127

    def test_s1_bit_value(self):
        """S1 bit is at position 56."""
        sig = build_s1()
        assert sig == S1_BIT | (127 << S1_PCT_SHIFT)

    def test_build_s2_full(self):
        """S2 with both percentages gives correct values."""
        sig = build_s2(50, 50)
        assert get_s2_s1_percentage(sig) == 127
        assert get_s2_s2_percentage(sig) == 127

    def test_build_s2_zero_s1(self):
        """S2 with zero S1%."""
        sig = build_s2(0, 100)
        assert get_s2_s1_percentage(sig) == 0
        assert get_s2_s2_percentage(sig) == 255

    def test_build_s2_zero_s2(self):
        """S2 with zero S2%."""
        sig = build_s2(100, 0)
        assert get_s2_s1_percentage(sig) == 255
        assert get_s2_s2_percentage(sig) == 0

    def test_get_s2_returns_combined(self):
        """get_s2 returns the full 16-bit S2 value."""
        sig = build_s2(100, 100)
        assert get_s2(sig) == 0xFFFF

    def test_s4_value_is_zero(self):
        """S4 (no significance) is 0."""
        assert S4_VALUE == 0


class TestCalculateSignificance:
    def test_s1_exact_match(self):
        """Exact node match returns S1."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is True

    def test_s1_prefix_match_query_shorter(self):
        """Query prefix matches model (query shorter)."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is True  # All prefix nodes match

    def test_s1_prefix_match_query_longer(self):
        """Query prefix matches model (query longer)."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is True  # All prefix nodes match

    def test_s1_empty_both(self):
        """Both empty nodes returns S1."""
        query = KLine(s_key=0x1000, nodes=[])
        model_kline = KLine(s_key=0x2000, nodes=[])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is True

    def test_s4_query_empty_model_not(self):
        """Query empty, model not empty returns S4."""
        query = KLine(s_key=0x1000, nodes=[])
        model_kline = KLine(s_key=0x2000, nodes=[0x100])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert sig == S4_VALUE

    def test_s4_model_empty_query_not(self):
        """Model empty, query not empty returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert sig == S4_VALUE

    def test_s2_partial_match(self):
        """Partial positional match returns S2."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x300])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2_s1_percentage(sig) == 127  # 50% positional match

    def test_s2_with_non_positional_match(self):
        """S2 includes non-positional matches."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x100, 0x300, 0x200])  # 0x200 at pos 2
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2_s1_percentage(sig) == 127  # 50% positional
        assert get_s2_s2_percentage(sig) == 127  # 50% non-positional

    def test_s4_no_match(self):
        """No matching nodes returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x200])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert sig == S4_VALUE


class TestSignificanceComparison:
    def test_s1_greater_than_s2(self):
        """S1 is more significant than S2."""
        s1_sig = build_s1(50)
        s2_sig = build_s2(100, 100)
        assert s1_sig > s2_sig

    def test_s1_100_greater_than_s1_50(self):
        """Higher S1% is more significant."""
        sig_high = build_s1(100)
        sig_low = build_s1(50)
        assert sig_high > sig_low

    def test_s2_greater_than_s4(self):
        """S2 is more significant than S4."""
        s2_sig = build_s2(1, 1)
        assert s2_sig > S4_VALUE

    def test_s2_higher_s1_pct_more_significant(self):
        """S2 with higher S1% is more significant."""
        sig_high = build_s2(100, 0)
        sig_low = build_s2(50, 0)
        assert sig_high > sig_low

    def test_s2_higher_s2_pct_more_significant(self):
        """S2 with higher S2% is more significant (same S1%)."""
        sig_high = build_s2(50, 100)
        sig_low = build_s2(50, 50)
        assert sig_high > sig_low

    def test_s1_s2_s4_ordering(self):
        """Full ordering: S1 > S2 > S4."""
        s1 = build_s1(100)
        s2 = build_s2(100, 100)
        s4 = S4_VALUE
        assert s1 > s2 > s4

    def test_calculated_significance_ordering(self):
        """Real calculated significances maintain ordering."""
        m = Model()

        # S1: exact match
        q = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        t1 = KLine(s_key=0x2000, nodes=[0x100, 0x200])
        m.add(q)
        m.add(t1)
        sig_s1 = m.calculate_significance(q, t1)

        # S2: partial match
        t2 = KLine(s_key=0x3000, nodes=[0x100, 0x300])
        m.add(t2)
        sig_s2 = m.calculate_significance(q, t2)

        # S4: no match
        t3 = KLine(s_key=0x4000, nodes=[0x999])
        m.add(t3)
        sig_s4 = m.calculate_significance(q, t3)

        assert sig_s1 > sig_s2 > sig_s4


class TestSignificanceHelpersS3:
    def test_build_s3_full(self):
        """S3 with all percentages gives correct values."""
        sig = build_s3(100, 100, 100)
        assert get_s3_s1_percentage(sig) == 255
        assert get_s3_s2_percentage(sig) == 255
        assert get_s3_gen_percentage(sig) == 255

    def test_build_s3_partial(self):
        """S3 with partial percentages."""
        sig = build_s3(50, 50, 50)
        assert get_s3_s1_percentage(sig) == 127
        assert get_s3_s2_percentage(sig) == 127
        assert get_s3_gen_percentage(sig) == 127

    def test_build_s3_zero(self):
        """S3 with zero percentages."""
        sig = build_s3(0, 0, 0)
        assert get_s3(sig) == 0

    def test_get_s3_returns_combined(self):
        """get_s3 returns the full 24-bit S3 value."""
        sig = build_s3(100, 100, 100)
        assert get_s3(sig) == 0xFFFFFF


class TestCalculateSignificanceS3:
    def test_s3_unordered_match(self):
        """S3 when nodes match but at different positions (no positional overlap)."""
        # Query: [a, b], Model: [b, c, a] - a and b exist but not at same positions
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x300, 0x100])  # 0x100 at different position
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert has_s1(sig) is False  # Not S1
        assert get_s2(sig) == 0  # Not S2 (no positional matches)
        assert get_s3_s1_percentage(sig) > 0  # Has unordered S1 matches

    def test_s3_reversed_nodes(self):
        """S3 when nodes are in reverse order."""
        query = KLine(s_key=0x1000, nodes=[0x100, 0x200])
        model_kline = KLine(s_key=0x2000, nodes=[0x200, 0x100])  # Reversed
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert get_s3_s1_percentage(sig) == 255  # 100% unordered match

    def test_s3_generational_match(self):
        """S3 generational match through child nodes."""
        # K1 -> N1 -> N3, K2 has N3 directly
        n3 = KLine(s_key=0x0030, nodes=[])
        n1 = KLine(s_key=0x0010, nodes=[0x0030])  # N1 has child N3
        k1 = KLine(s_key=0x1000, nodes=[0x0010])  # K1 has child N1
        k2 = KLine(s_key=0x2000, nodes=[0x0020, 0x0030])  # K2 has N2 and N3
        m = Model([n3, n1, k1, k2])

        sig = m.calculate_significance(k1, k2)
        # K1's node N1 has child N3 which matches K2's node N3
        assert get_s3_s2_percentage(sig) > 0  # Child match

    def test_s3_no_match_returns_s4(self):
        """No unordered or generational match returns S4."""
        query = KLine(s_key=0x1000, nodes=[0x100])
        model_kline = KLine(s_key=0x2000, nodes=[0x200])
        m = Model([query, model_kline])

        sig = m.calculate_significance(query, model_kline)
        assert sig == S4_VALUE


class TestSignificanceComparisonS3:
    def test_s2_greater_than_s3(self):
        """S2 is more significant than S3."""
        s2_sig = build_s2(1, 1)
        s3_sig = build_s3(100, 100, 100)
        assert s2_sig > s3_sig

    def test_s3_greater_than_s4(self):
        """S3 is more significant than S4."""
        s3_sig = build_s3(1, 0, 0)
        assert s3_sig > S4_VALUE

    def test_s3_higher_s1_pct_more_significant(self):
        """S3 with higher S1% is more significant."""
        sig_high = build_s3(100, 0, 0)
        sig_low = build_s3(50, 0, 0)
        assert sig_high > sig_low

    def test_s3_higher_gen_pct_more_significant(self):
        """S3 with higher gen% is more significant."""
        sig_high = build_s3(0, 0, 100)
        sig_low = build_s3(0, 0, 50)
        assert sig_high > sig_low

    def test_full_ordering(self):
        """Full ordering: S1 > S2 > S3 > S4."""
        s1 = build_s1(100)
        s2 = build_s2(100, 100)
        s3 = build_s3(100, 100, 100)
        s4 = S4_VALUE
        assert s1 > s2 > s3 > s4

    def test_calculated_full_ordering(self):
        """Real calculated significances maintain full ordering."""
        # Build a model with various KLines for testing
        n3 = KLine(s_key=0x0030, nodes=[])

        q = KLine(s_key=0x1000, nodes=[0x100, 0x200])  # Query

        # S1: exact match
        t1 = KLine(s_key=0x2000, nodes=[0x100, 0x200])

        # S2: partial positional match
        t2 = KLine(s_key=0x3000, nodes=[0x100, 0x300])

        # S3: only unordered match (reversed)
        t3 = KLine(s_key=0x4000, nodes=[0x200, 0x100])

        # S4: no match
        t4 = KLine(s_key=0x5000, nodes=[0x999])

        m = Model([n3, q, t1, t2, t3, t4])

        sig_s1 = m.calculate_significance(q, t1)
        sig_s2 = m.calculate_significance(q, t2)
        sig_s3 = m.calculate_significance(q, t3)
        sig_s4 = m.calculate_significance(q, t4)

        assert sig_s1 > sig_s2 > sig_s3 > sig_s4
