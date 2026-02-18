import pytest
from kalvin.model import KLine, KLineType, HIGH_BIT_MASK, query_significance


class TestKLineType:
    def test_kline_type_values(self):
        """Test KLineType enum values."""
        assert KLineType.NODE == 0
        assert KLineType.EMBEDDING == 1

    def test_high_bit_mask(self):
        """Test that HIGH_BIT_MASK is the correct bit 63 mask."""
        assert HIGH_BIT_MASK == 0x8000_0000_0000_0000
        assert HIGH_BIT_MASK == (1 << 63)


class TestKLine:
    def test_create_kline(self):
        """Test creating a KLine with int s_key and list of child KLines."""
        s_key = 0x123456789ABCDEF0
        child1 = KLine(s_key=0x1000, nodes=[])
        child2 = KLine(s_key=0x2000, nodes=[])
        nodes = [child1, child2]

        kl = KLine(s_key=s_key, nodes=nodes)

        assert kl.s_key == s_key
        assert kl.nodes == [child1, child2]
        assert kl.nodes[0].s_key == 0x1000

    def test_type_property_node(self):
        """Test that type property returns NODE when high bit is 0."""
        kl = KLine(s_key=0x7FFF_FFFF_FFFF_FFFF, nodes=[])
        assert kl.type == KLineType.NODE

    def test_type_property_embedding(self):
        """Test that type property returns EMBEDDING when high bit is 1."""
        kl = KLine(s_key=0x8000_0000_0000_0000, nodes=[])
        assert kl.type == KLineType.EMBEDDING

    def test_create_node_factory(self):
        """Test create_node factory clears high bit."""
        child1 = KLine(s_key=0x0100, nodes=[])
        child2 = KLine(s_key=0x0200, nodes=[])

        kl = KLine.create_node(s_key=0xFFFF_0000_0000_0000, nodes=[child1, child2])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0x7FFF_0000_0000_0000  # high bit cleared
        assert kl.nodes == [child1, child2]

    def test_create_node_factory_preserves_zero_high_bit(self):
        """Test create_node preserves key when high bit already 0."""
        kl = KLine.create_node(s_key=0x1234_5678_9ABC_DEF0, nodes=[])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0x1234_5678_9ABC_DEF0  # unchanged

    def test_create_embedding_factory(self):
        """Test create_embedding factory sets high bit."""
        child = KLine(s_key=0x0100, nodes=[])
        kl = KLine.create_embedding(s_key=0x1234_5678_9ABC_DEF0, nodes=[child])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x9234_5678_9ABC_DEF0  # high bit set
        assert kl.nodes == [child]

    def test_create_embedding_factory_idempotent(self):
        """Test create_embedding is idempotent when high bit already set."""
        kl = KLine.create_embedding(s_key=0x8000_0000_0000_0001, nodes=[])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x8000_0000_0000_0001  # unchanged

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

    def test_nested_klines(self):
        """Test nested KLine structure."""
        # Leaf nodes
        leaf1 = KLine(s_key=0x0100, nodes=[])
        leaf2 = KLine(s_key=0x0200, nodes=[])
        leaf3 = KLine(s_key=0x0300, nodes=[])

        # Intermediate node
        intermediate = KLine(s_key=0x0010, nodes=[leaf1, leaf2])

        # Root node
        root = KLine(s_key=0x0001, nodes=[intermediate, leaf3])

        assert len(root.nodes) == 2
        assert root.nodes[0] == intermediate
        assert root.nodes[1] == leaf3
        assert len(root.nodes[0].nodes) == 2
        assert root.nodes[0].nodes[0] == leaf1
        assert root.nodes[0].nodes[1] == leaf2

    def test_circular_dependency(self):
        """Test that circular dependencies are supported."""
        # Create two KLines that reference each other
        kl_a = KLine(s_key=0x0001, nodes=[])
        kl_b = KLine(s_key=0x0002, nodes=[kl_a])

        # Now create the circular reference: A -> B -> A
        kl_a.nodes.append(kl_b)

        # Verify the circular structure
        assert kl_a.nodes[0] == kl_b
        assert kl_b.nodes[0] == kl_a

        # Follow the circle multiple times to prove it works
        assert kl_a.nodes[0].nodes[0] == kl_a  # A -> B -> A
        assert kl_b.nodes[0].nodes[0] == kl_b  # B -> A -> B

        # Can traverse infinitely without error
        current = kl_a
        for _ in range(100):
            current = current.nodes[0]
        assert current == kl_a  # After 100 hops (even), back at kl_a

    def test_self_reference(self):
        """Test that a KLine can reference itself."""
        kl = KLine(s_key=0x1234, nodes=[])
        kl.nodes.append(kl)  # Self-reference

        assert kl.nodes[0] == kl
        assert kl.nodes[0].nodes[0] == kl  # Still itself

        # Can traverse infinitely
        current = kl
        for _ in range(1000):
            current = current.nodes[0]
        assert current == kl  # Still itself after 1000 hops


class TestQueryByMask:
    def test_query_by_mask_basic(self):
        """Test querying KLines by ANDing s_key with a mask."""
        kl_list = [
            KLine(s_key=0b1111000000000000, nodes=[]),
            KLine(s_key=0b1110000000000000, nodes=[]),
            KLine(s_key=0b1111000000000001, nodes=[]),
            KLine(s_key=0b0000111100000000, nodes=[]),
        ]

        # Query for upper 4 bits = 1111
        query = 0b1111000000000000

        results = query_significance(kl_list, query)

        assert len(results) == 3
        assert results[0].s_key == 0b1111000000000000
        assert results[1].s_key == 0b1110000000000000
        assert results[2].s_key == 0b1111000000000001

    def test_query_by_mask_full_64bit(self):
        """Test querying with full 64-bit s_keys and masks."""
        kl_list = [
            KLine(s_key=0x7F00_0000_0000_0000, nodes=[]),
            KLine(s_key=0x7F00_0000_0000_0001, nodes=[]),
            KLine(s_key=0x00FF_0000_0000_0000, nodes=[]),
            KLine(s_key=0x00A0_0000_0000_0000, nodes=[]),
        ]

        # Query for s_keys where bits 56-63 = 0x7F (excluding high bit)
        query = 0x7F00_0000_0000_0000

        results = query_significance(kl_list, query)

        assert len(results) == 2
        assert results[0].s_key == 0x7F00_0000_0000_0000
        assert results[1].s_key == 0x7F00_0000_0000_0001

    def test_query_by_mask_no_match(self):
        """Test query that returns no matches."""
        kl_list = [
            KLine(s_key=0x1000, nodes=[]),
            KLine(s_key=0x2000, nodes=[]),
        ]

        query = 0x4000

        results = query_significance(kl_list, query)

        assert len(results) == 0

    def test_query_by_mask_all_match(self):
        """Test query that matches all items."""
        kl_list = [
            KLine(s_key=0x1234, nodes=[]),
            KLine(s_key=0x1256, nodes=[]),
            KLine(s_key=0x1278, nodes=[]),
        ]

        # Match all by using query with all bits set except high bit
        results = query_significance(kl_list, query=0x7FFF_FFFF_FFFF_FFFF)

        assert len(results) == 3

    def test_signifies_method(self):
        """Test the signifies method on KLine."""
        kl = KLine(s_key=0x7F00_0000_0000_0000, nodes=[])

        # Should match - high 16 bits (excluding type bit) = 0x7F00
        assert kl.signifies(query=0x7F00_0000_0000_0000) is True

        # Should not match - low bits don't overlap
        assert kl.signifies(query=0x0000_0000_ABCD_0000) is False

        # Match only bits that overlap
        assert kl.signifies(query=0x7F00_0000_0000_0000) is True

    def test_query_by_type(self):
        """Test querying to filter by NODE vs EMBEDDING type."""
        child1 = KLine(s_key=0x0001, nodes=[])
        child2 = KLine(s_key=0x0002, nodes=[])
        child3 = KLine(s_key=0x0003, nodes=[])
        child4 = KLine(s_key=0x0004, nodes=[])

        kl_list = [
            KLine.create_node(s_key=0x1000, nodes=[child1]),
            KLine.create_embedding(s_key=0x1000, nodes=[child2]),
            KLine.create_node(s_key=0x2000, nodes=[child3]),
            KLine.create_embedding(s_key=0x2000, nodes=[child4]),
        ]

        # Query only NODEs (high bit = 0)
        node_results = [kl for kl in kl_list if kl.type == KLineType.NODE]
        assert len(node_results) == 2
        assert node_results[0].nodes == [child1]
        assert node_results[1].nodes == [child3]

        # Query only EMBEDDINGs (high bit = 1)
        embedding_results = [kl for kl in kl_list if kl.type == KLineType.EMBEDDING]
        assert len(embedding_results) == 2
        assert embedding_results[0].nodes == [child2]
        assert embedding_results[1].nodes == [child4]


class TestQueryDepth:
    def test_query_depth_default_is_one(self):
        """Test that default depth=1 only searches top level."""
        # Child with matching key
        child = KLine(s_key=0xFF00, nodes=[])

        # Parent with non-matching key but matching child
        parent = KLine(s_key=0x0001, nodes=[child])

        # Default depth=1 should only find parent (which doesn't match)
        results = query_significance([parent], query=0xFF00, depth=1)
        assert len(results) == 0

    def test_query_depth_two_finds_child(self):
        """Test that depth=2 finds matching children."""
        child = KLine(s_key=0xFF00, nodes=[])
        parent = KLine(s_key=0x0001, nodes=[child])

        # depth=2 should find the matching child
        results = query_significance([parent], query=0xFF00, depth=2)
        assert len(results) == 1
        assert results[0] == child

    def test_query_depth_three_finds_grandchild(self):
        """Test that depth=3 finds matching grandchildren."""
        grandchild = KLine(s_key=0xFF00, nodes=[])
        child = KLine(s_key=0x0001, nodes=[grandchild])
        parent = KLine(s_key=0x0002, nodes=[child])

        # depth=2 should not find grandchild
        results = query_significance([parent], query=0xFF00, depth=2)
        assert len(results) == 0

        # depth=3 should find grandchild
        results = query_significance([parent], query=0xFF00, depth=3)
        assert len(results) == 1
        assert results[0] == grandchild

    def test_query_circular_halts(self):
        """Test that circular dependencies are detected and halted."""
        # Create circular reference: A -> B -> A
        kl_a = KLine(s_key=0xFF00, nodes=[])  # matches query
        kl_b = KLine(s_key=0xFF00, nodes=[kl_a])  # also matches
        kl_a.nodes.append(kl_b)  # close the circle

        # Even with high depth, should not infinite loop
        results = query_significance([kl_a], query=0xFF00, depth=100)

        # Should find both kl_a and kl_b, but only once each
        # Use id() for comparison to avoid infinite recursion in dataclass eq
        result_ids = {id(r) for r in results}
        assert len(results) == 2
        assert id(kl_a) in result_ids
        assert id(kl_b) in result_ids

    def test_query_self_reference_halts(self):
        """Test that self-referencing KLines are handled."""
        kl = KLine(s_key=0xFF00, nodes=[])
        kl.nodes.append(kl)  # self-reference

        # Should find kl only once, not infinite times
        results = query_significance([kl], query=0xFF00, depth=100)
        assert len(results) == 1
        assert results[0] == kl

    def test_query_shared_child_counted_once(self):
        """Test that shared children are only counted once."""
        # Two parents share the same child
        shared_child = KLine(s_key=0xFF00, nodes=[])
        parent1 = KLine(s_key=0x0001, nodes=[shared_child])
        parent2 = KLine(s_key=0x0002, nodes=[shared_child])

        # Query both parents - shared_child should appear only once in results
        results = query_significance([parent1, parent2], query=0xFF00, depth=2)
        assert len(results) == 1
        assert results[0] == shared_child

    def test_query_depth_zero_finds_nothing(self):
        """Test that depth=0 finds nothing."""
        kl = KLine(s_key=0xFF00, nodes=[])
        results = query_significance([kl], query=0xFF00, depth=0)
        assert len(results) == 0
