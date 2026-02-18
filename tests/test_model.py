import pytest
from kalvin.model import KeyValue, query_by_mask


class TestKeyValue:
    def test_create_keyvalue(self):
        """Test creating a KeyValue with int key and list value."""
        key = 0x123456789ABCDEF0
        value = [42, 43, 44]

        kv = KeyValue(key=key, value=value)

        assert kv.key == key
        assert kv.value == [42, 43, 44]

    def test_store_in_list(self):
        """Test storing KeyValue objects in a list."""
        kv_list = []

        kv1 = KeyValue(key=0x1000000000000000, value=[1])
        kv2 = KeyValue(key=0x1000000000000001, value=[2, 3])
        kv3 = KeyValue(key=0x2000000000000000, value=[4, 5, 6])

        kv_list.append(kv1)
        kv_list.append(kv2)
        kv_list.append(kv3)

        assert len(kv_list) == 3
        assert kv_list[0].value == [1]
        assert kv_list[1].value == [2, 3]
        assert kv_list[2].value == [4, 5, 6]


class TestQueryByMask:
    def test_query_by_mask_basic(self):
        """Test querying KeyValues by ANDing key with a mask."""
        kv_list = [
            KeyValue(key=0b1111000000000000, value=[1]),       # upper 4 bits = 1111
            KeyValue(key=0b1110000000000000, value=[2, 3]),    # upper 4 bits = 1110
            KeyValue(key=0b1111000000000001, value=[4, 5, 6]), # upper 4 bits = 1111
            KeyValue(key=0b0000111100000000, value=[7]),       # upper 4 bits = 0000
        ]

        # Query for upper 4 bits = 1111
        query = 0b1111000000000000

        results = query_by_mask(kv_list, query)

        assert len(results) == 3
        assert results[0].value == [1]
        assert results[1].value == [2, 3]
        assert results[2].value == [4, 5, 6]

    def test_query_by_mask_full_64bit(self):
        """Test querying with full 64-bit keys and masks."""
        kv_list = [
            KeyValue(key=0xFF00_0000_0000_0000, value=[100, 101]),
            KeyValue(key=0xFF00_0000_0000_0001, value=[102]),
            KeyValue(key=0x00FF_0000_0000_0000, value=[200]),
            KeyValue(key=0xAB00_0000_0000_0000, value=[300]),
        ]

        # Query for keys where high byte = 0xFF
        query = 0xFF00_0000_0000_0000

        results = query_by_mask(kv_list, query)

        assert len(results) == 3
        assert results[0].value == [100, 101]
        assert results[1].value == [102]
        assert results[2].value == [300]

    def test_query_by_mask_no_match(self):
        """Test query that returns no matches."""
        kv_list = [
            KeyValue(key=0x1000, value=[1]),
            KeyValue(key=0x2000, value=[2]),
        ]

        query = 0x4000

        results = query_by_mask(kv_list, query)

        assert len(results) == 0

    def test_query_by_mask_all_match(self):
        """Test query that matches all items."""
        kv_list = [
            KeyValue(key=0x1234, value=[1]),
            KeyValue(key=0x1256, value=[2, 3]),
            KeyValue(key=0x1278, value=[4, 5, 6]),
        ]

        # Match all by using query = -1
        results = query_by_mask(kv_list, query=-1)

        assert len(results) == 3

    def test_matches_method(self):
        """Test the matches method on KeyValue."""
        kv = KeyValue(key=0xFF00_0000_0000_0000, value=[42, 43])

        # Should match - high 16 bits = 0xABCD
        assert kv.matches(query=0xABCD_0000_0000_0000) is True

        # Should not match - low 16 bits != 0x1234
        assert kv.matches(query=0x0000_0000_ABCD_0000) is False

        # Match only high byte
        assert kv.matches(query=0xAB00_0000_0000_0000) is True
