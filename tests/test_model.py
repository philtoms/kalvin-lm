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
        """Test creating a KLine with int s_key and list nodes."""
        s_key = 0x123456789ABCDEF0
        nodes = [42, 43, 44]

        kl = KLine(s_key=s_key, nodes=nodes)

        assert kl.s_key == s_key
        assert kl.nodes == [42, 43, 44]

    def test_type_property_node(self):
        """Test that type property returns NODE when high bit is 0."""
        kl = KLine(s_key=0x7FFF_FFFF_FFFF_FFFF, nodes=[1])
        assert kl.type == KLineType.NODE

    def test_type_property_embedding(self):
        """Test that type property returns EMBEDDING when high bit is 1."""
        kl = KLine(s_key=0x8000_0000_0000_0000, nodes=[1])
        assert kl.type == KLineType.EMBEDDING

    def test_create_node_factory(self):
        """Test create_node factory clears high bit."""
        # Pass a key with high bit set - should be cleared
        kl = KLine.create_node(s_key=0xFFFF_0000_0000_0000, nodes=[1, 2])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0x7FFF_0000_0000_0000  # high bit cleared
        assert kl.nodes == [1, 2]

    def test_create_node_factory_preserves_zero_high_bit(self):
        """Test create_node preserves key when high bit already 0."""
        kl = KLine.create_node(s_key=0x1234_5678_9ABC_DEF0, nodes=[1])

        assert kl.type == KLineType.NODE
        assert kl.s_key == 0x1234_5678_9ABC_DEF0  # unchanged

    def test_create_embedding_factory(self):
        """Test create_embedding factory sets high bit."""
        kl = KLine.create_embedding(s_key=0x1234_5678_9ABC_DEF0, nodes=[3, 4])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x9234_5678_9ABC_DEF0  # high bit set
        assert kl.nodes == [3, 4]

    def test_create_embedding_factory_idempotent(self):
        """Test create_embedding is idempotent when high bit already set."""
        kl = KLine.create_embedding(s_key=0x8000_0000_0000_0001, nodes=[1])

        assert kl.type == KLineType.EMBEDDING
        assert kl.s_key == 0x8000_0000_0000_0001  # unchanged

    def test_store_in_list(self):
        """Test storing KLine objects in a list."""
        kl_list = []

        kl1 = KLine(s_key=0x1000000000000000, nodes=[1])
        kl2 = KLine(s_key=0x1000000000000001, nodes=[2, 3])
        kl3 = KLine(s_key=0x2000000000000000, nodes=[4, 5, 6])

        kl_list.append(kl1)
        kl_list.append(kl2)
        kl_list.append(kl3)

        assert len(kl_list) == 3
        assert kl_list[0].nodes == [1]
        assert kl_list[1].nodes == [2, 3]
        assert kl_list[2].nodes == [4, 5, 6]


class TestQueryByMask:
    def test_query_by_mask_basic(self):
        """Test querying KLines by ANDing s_key with a mask."""
        kl_list = [
            KLine(s_key=0b1111000000000000, nodes=[1]),       # upper 4 bits = 1111
            KLine(s_key=0b1110000000000000, nodes=[2, 3]),    # upper 4 bits = 1110
            KLine(s_key=0b1111000000000001, nodes=[4, 5, 6]), # upper 4 bits = 1111
            KLine(s_key=0b0000111100000000, nodes=[7]),       # upper 4 bits = 0000
        ]

        # Query for upper 4 bits = 1111
        query = 0b1111000000000000

        results = query_significance(kl_list, query)

        assert len(results) == 3
        assert results[0].nodes == [1]
        assert results[1].nodes == [2, 3]
        assert results[2].nodes == [4, 5, 6]

    def test_query_by_mask_full_64bit(self):
        """Test querying with full 64-bit s_keys and masks."""
        kl_list = [
            KLine(s_key=0x7F00_0000_0000_0000, nodes=[100, 101]),  # NODE type
            KLine(s_key=0x7F00_0000_0000_0001, nodes=[102]),       # NODE type
            KLine(s_key=0x00FF_0000_0000_0000, nodes=[200]),       # no overlap with query
            KLine(s_key=0x00A0_0000_0000_0000, nodes=[300]),       # no overlap with query
        ]

        # Query for s_keys where bits 56-63 = 0x7F (excluding high bit)
        query = 0x7F00_0000_0000_0000

        results = query_significance(kl_list, query)

        assert len(results) == 2
        assert results[0].nodes == [100, 101]
        assert results[1].nodes == [102]

    def test_query_by_mask_no_match(self):
        """Test query that returns no matches."""
        kl_list = [
            KLine(s_key=0x1000, nodes=[1]),
            KLine(s_key=0x2000, nodes=[2]),
        ]

        query = 0x4000

        results = query_significance(kl_list, query)

        assert len(results) == 0

    def test_query_by_mask_all_match(self):
        """Test query that matches all items."""
        kl_list = [
            KLine(s_key=0x1234, nodes=[1]),
            KLine(s_key=0x1256, nodes=[2, 3]),
            KLine(s_key=0x1278, nodes=[4, 5, 6]),
        ]

        # Match all by using query with all bits set except high bit
        results = query_significance(kl_list, query=0x7FFF_FFFF_FFFF_FFFF)

        assert len(results) == 3

    def test_signifies_method(self):
        """Test the signifies method on KLine."""
        kl = KLine(s_key=0x7F00_0000_0000_0000, nodes=[42, 43])

        # Should match - high 16 bits (excluding type bit) = 0x7F00
        assert kl.signifies(query=0x7F00_0000_0000_0000) is True

        # Should not match - low bits don't overlap
        assert kl.signifies(query=0x0000_0000_ABCD_0000) is False

        # Match only bits that overlap
        assert kl.signifies(query=0x7F00_0000_0000_0000) is True

    def test_query_by_type(self):
        """Test querying to filter by NODE vs EMBEDDING type."""
        kl_list = [
            KLine.create_node(s_key=0x1000, nodes=[1]),
            KLine.create_embedding(s_key=0x1000, nodes=[2]),
            KLine.create_node(s_key=0x2000, nodes=[3]),
            KLine.create_embedding(s_key=0x2000, nodes=[4]),
        ]

        # Query only NODEs (high bit = 0)
        node_results = [kl for kl in kl_list if kl.type == KLineType.NODE]
        assert len(node_results) == 2
        assert node_results[0].nodes == [1]
        assert node_results[1].nodes == [3]

        # Query only EMBEDDINGs (high bit = 1)
        embedding_results = [kl for kl in kl_list if kl.type == KLineType.EMBEDDING]
        assert len(embedding_results) == 2
        assert embedding_results[0].nodes == [2]
        assert embedding_results[1].nodes == [4]
