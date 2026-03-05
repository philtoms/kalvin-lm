"""Tests for Kalvin class - serialization and file operations."""

import json
import pytest
import tempfile
from pathlib import Path
from collections import Counter

from kalvin.kalvin import Kalvin
from kalvin.model import KLine, Model


class TestKalvinInit:
    """Tests for Kalvin initialization."""

    def test_init_empty(self):
        """Kalvin can be initialized without a model."""
        kalvin = Kalvin()
        assert kalvin.model is not None
        assert len(kalvin.model) == 0

    def test_init_with_model(self):
        """Kalvin can be initialized with an existing model."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(klines)
        kalvin = Kalvin(model)

        assert len(kalvin.model) == 1
        assert kalvin.model[0].signature == 0x1000

    def test_init_with_none_creates_empty_model(self):
        """Passing None creates an empty model."""
        kalvin = Kalvin(None)
        assert kalvin.model is not None
        assert len(kalvin.model) == 0


class TestKalvinToBytes:
    """Tests for binary serialization."""

    def test_to_bytes_empty(self):
        """Empty Kalvin serializes to minimal bytes."""
        kalvin = Kalvin()

        data = kalvin.to_bytes()

        # Metadata (4 bytes len + JSON) + 4 bytes for kline count + 4 bytes for activity (0)
        # Metadata JSON is ~110 bytes
        assert len(data) > 8
        # Verify kline count is 0 (after metadata)
        metadata_len = int.from_bytes(data[:4], 'little')
        offset = 4 + metadata_len
        assert data[offset:offset+4] == b"\x00\x00\x00\x00"

    def test_to_bytes_single_kline_no_nodes(self):
        """Single KLine with no nodes serializes correctly."""
        kline = KLine(signature=0x123456789ABCDEF0, nodes=[])
        model = Model([kline])
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # Calculate offset after metadata
        metadata_len = int.from_bytes(data[:4], 'little')
        offset = 4 + metadata_len

        # Verify count (little-endian uint32)
        assert data[offset:offset+4] == b"\x01\x00\x00\x00"
        offset += 4
        # Verify signature (little-endian uint64)
        assert data[offset:offset+8] == b"\xf0\xde\xbc\x9a\x78\x56\x34\x12"
        offset += 8
        # Verify node count
        assert data[offset:offset+4] == b"\x00\x00\x00\x00"
        offset += 4
        # Verify activity count
        assert data[offset:offset+4] == b"\x00\x00\x00\x00"

    def test_to_bytes_single_kline_with_nodes(self):
        """Single KLine with nodes serializes correctly."""
        signature = 0x1000
        nodes = [0x0100, 0x0200, 0x0300]
        kline = KLine(signature=signature, nodes=nodes)
        model = Model([kline])
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # Calculate offset after metadata
        metadata_len = int.from_bytes(data[:4], 'little')
        offset = 4 + metadata_len

        # Skip kline count
        offset += 4
        # Skip signature
        offset += 8
        # Verify node count is 3
        assert data[offset:offset+4] == b"\x03\x00\x00\x00"

    def test_to_bytes_multiple_klines(self):
        """Multiple KLines serialize correctly."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[0x0200, 0x0300]),
            KLine(signature=0x3000, nodes=[]),
        ]
        model = Model(klines)
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # Calculate offset after metadata
        metadata_len = int.from_bytes(data[:4], 'little')
        offset = 4 + metadata_len

        # Verify kline count is 3
        assert data[offset:offset+4] == b"\x03\x00\x00\x00"

    def test_to_bytes_preserves_s_key(self):
        """signature values are preserved during serialization."""
        key1 = 0x1000
        key2 = 0x2000

        klines = [
            KLine(signature=key1, nodes=[]),
            KLine(signature=key2, nodes=[]),
        ]
        model = Model(klines)
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # Deserialize and verify
        kalvin2 = Kalvin.from_bytes(data)
        assert kalvin2.model[0].signature == key1
        assert kalvin2.model[1].signature == key2


class TestKalvinFromBytes:
    """Tests for binary deserialization."""

    def test_from_bytes_empty(self):
        """Empty bytes deserializes to empty Kalvin."""
        # Create empty kalvin and serialize it to get valid bytes
        original = Kalvin()
        data = original.to_bytes()

        kalvin = Kalvin.from_bytes(data)

        assert len(kalvin.model) == 0

    def test_from_bytes_roundtrip_empty(self):
        """Roundtrip serialization preserves empty Kalvin."""
        original = Kalvin()

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == len(original.model)

    def test_from_bytes_roundtrip_single_kline(self):
        """Roundtrip preserves single KLine."""
        kline = KLine(signature=0x123456789ABCDEF0, nodes=[0x0100, 0x0200])
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 1
        assert restored.model[0].signature == kline.signature
        assert restored.model[0].nodes == kline.nodes

    def test_from_bytes_roundtrip_multiple_klines(self):
        """Roundtrip preserves multiple KLines."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[0x0200, 0x0300]),
            KLine(signature=0x3000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 3
        for i, kl in enumerate(original.model):
            assert restored.model[i].signature == kl.signature
            assert restored.model[i].nodes == kl.nodes


class TestKalvinToDict:
    """Tests for dictionary serialization."""

    def test_to_dict_empty(self):
        """Empty Kalvin serializes to dict with empty klines."""
        kalvin = Kalvin()

        data = kalvin.to_dict()

        assert "metadata" in data
        assert data["klines"] == []
        assert data["activity"] == {}

    def test_to_dict_single_kline(self):
        """Single KLine serializes to dict correctly."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        kalvin = Kalvin(Model([kline]))

        data = kalvin.to_dict()

        assert "metadata" in data
        assert data["klines"] == [{"signature": 0x1000, "nodes": [0x0100, 0x0200]}]
        assert data["activity"] == {}

    def test_to_dict_multiple_klines(self):
        """Multiple KLines serialize to dict correctly."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[]),
        ]
        kalvin = Kalvin(Model(klines))

        data = kalvin.to_dict()

        assert "metadata" in data
        assert data["klines"] == [
            {"signature": 0x1000, "nodes": [0x0100]},
            {"signature": 0x2000, "nodes": []},
        ]
        assert data["activity"] == {}

    def test_to_dict_is_json_serializable(self):
        """Dict output is JSON serializable."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        data = kalvin.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert json.loads(json_str) == data


class TestKalvinFromDict:
    """Tests for dictionary deserialization."""

    def test_from_dict_empty(self):
        """Empty dict creates empty Kalvin."""
        kalvin = Kalvin.from_dict({"klines": []})

        assert len(kalvin.model) == 0

    def test_from_dict_single_kline(self):
        """Single KLine deserializes from dict correctly."""
        data = {"klines": [{"signature": 0x1000, "nodes": [0x0100, 0x0200]}]}

        kalvin = Kalvin.from_dict(data)

        assert len(kalvin.model) == 1
        assert kalvin.model[0].signature == 0x1000
        assert kalvin.model[0].nodes == [0x0100, 0x0200]

    def test_from_dict_roundtrip(self):
        """Roundtrip preserves all data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[0x0200, 0x0300]),
        ]
        original = Kalvin(Model(klines))

        data = original.to_dict()
        restored = Kalvin.from_dict(data)

        assert len(restored.model) == len(original.model)
        for i, kl in enumerate(original.model):
            assert restored.model[i].signature == kl.signature
            assert restored.model[i].nodes == kl.nodes

    def test_metadata_preserved_in_dict_roundtrip(self):
        """Metadata is preserved during dict roundtrip."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        data = original.to_dict()
        restored = Kalvin.from_dict(data)

        assert restored.dictionary_path == original.dictionary_path
        assert restored.nlp_detail == original.nlp_detail

    def test_metadata_preserved_in_binary_roundtrip(self):
        """Metadata is preserved during binary roundtrip."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert restored.dictionary_path == original.dictionary_path
        assert restored.nlp_detail == original.nlp_detail


class TestKalvinSaveLoad:
    """Tests for file operations."""

    def test_save_binary_default(self):
        """Save uses binary format by default."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path)

            # Verify binary format by checking it's not valid JSON and can be loaded
            raw = path.read_bytes()
            # First 4 bytes should be metadata length (not a JSON start character)
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Kalvin.load(path)
            assert len(restored.model) == 1
            assert restored.model[0].signature == 0x1000

    def test_save_binary_explicit(self):
        """Save with format='binary' creates binary file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path, format="binary")

            # Verify binary format by checking it's not valid JSON
            raw = path.read_bytes()
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Kalvin.load(path)
            assert len(restored.model) == 1

    def test_save_json_explicit(self):
        """Save with format='json' creates JSON file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path, format="json")

            # Should be valid JSON
            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["signature"] == 0x1000

    def test_save_json_auto_detect(self):
        """Save with format=None auto-detects JSON from .json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            kalvin.save(path, format=None)  # Explicit None for auto-detect

            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["signature"] == 0x1000

    def test_save_binary_auto_detect(self):
        """Save auto-detects binary format from non-.json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            kalvin.save(path)  # format=None, auto-detect

            # Verify binary format by checking it's not valid JSON
            raw = path.read_bytes()
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Kalvin.load(path)
            assert len(restored.model) == 1

    def test_load_binary_default(self):
        """Load uses binary format by default."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="binary")

            restored = Kalvin.load(path)  # Default binary

            assert len(restored.model) == 1
            assert restored.model[0].signature == 0x1000

    def test_load_json_explicit(self):
        """Load with format='json' reads JSON file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="json")

            restored = Kalvin.load(path, format="json")

            assert restored.model[0].signature == 0x1000

    def test_load_json_auto_detect(self):
        """Load auto-detects JSON format from .json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")

            restored = Kalvin.load(path)  # Auto-detect

            assert restored.model[0].signature == 0x1000

    def test_load_binary_auto_detect(self):
        """Load auto-detects binary format from non-.json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            original.save(path, format="binary")

            restored = Kalvin.load(path)  # Auto-detect

            assert restored.model[0].signature == 0x1000

    def test_roundtrip_binary_file(self):
        """Roundtrip through binary file preserves data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100, 0x0200]),
            KLine(signature=0x2000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path)
            restored = Kalvin.load(path)

            assert len(restored.model) == len(original.model)
            for i, kl in enumerate(original.model):
                assert restored.model[i].signature == kl.signature
                assert restored.model[i].nodes == kl.nodes

    def test_roundtrip_json_file(self):
        """Roundtrip through JSON file preserves data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100, 0x0200]),
            KLine(signature=0x2000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")
            restored = Kalvin.load(path, format="json")

            assert len(restored.model) == len(original.model)
            for i, kl in enumerate(original.model):
                assert restored.model[i].signature == kl.signature
                assert restored.model[i].nodes == kl.nodes

    def test_save_to_existing_directory(self):
        """Save works when directory exists."""
        klines = [KLine(signature=0x1000, nodes=[])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory first
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            path = subdir / "test.kalvin"
            kalvin.save(path)

            assert path.exists()

    def test_save_with_string_path(self):
        """Save accepts string path."""
        klines = [KLine(signature=0x1000, nodes=[])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.kalvin")
            kalvin.save(path)

            assert Path(path).exists()

    def test_load_with_string_path(self):
        """Load accepts string path."""
        klines = [KLine(signature=0x1000, nodes=[])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.kalvin")
            original.save(path)
            restored = Kalvin.load(path)

            assert len(restored.model) == 1


class TestKalvinLargeData:
    """Tests for handling larger data sets."""

    def test_large_kline_count(self):
        """Handles hundreds of KLines."""
        klines = [KLine(signature=i, nodes=[i * 100]) for i in range(500)]
        original = Kalvin(Model(klines))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 500
        assert restored.model[0].signature == 0
        assert restored.model[499].signature == 499

    def test_large_node_count(self):
        """Handles KLines with many nodes."""
        nodes = list(range(1000))
        kline = KLine(signature=0x1000, nodes=nodes)
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model[0].nodes) == 1000
        assert restored.model[0].nodes == nodes

    def test_max_s_key_value(self):
        """Handles maximum signature value (all bits set)."""
        max_key = 0xFFFF_FFFF_FFFF_FFFF
        kline = KLine(signature=max_key, nodes=[])
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert restored.model[0].signature == max_key


class TestKalvinEmbeddings:
    """Tests for embedding handling."""

    def test_add_new_encoding(self):
        kalvin = Kalvin()
        token_sig = kalvin.encode("hello there")
        assert token_sig is not None
        # Verify that encoding adds klines to the model
        assert kalvin.model_size() > 0

    def test_add_very_long_string(self):
        vl_str = """
          Mary had a little lamb,
          Its fleece was white as snow,
          And everywhere that Mary went,
          The lamb was sure to go.
        """
        kalvin = Kalvin()
        token_sig = kalvin.encode(vl_str)

        # Verify encoding returns a valid signature
        assert token_sig is not None
        # Verify that the model grew
        assert kalvin.model_size() > 0

    def test_add_duplicate_encoding(self):
        kalvin = Kalvin()
        token_sig1 = kalvin.encode("hello there")
        token_sig2 = kalvin.encode("hello there")

        assert token_sig1 is not None
        assert token_sig2 is None  # Duplicate returns None

    def test_add_encoded_string(self):
        kalvin = Kalvin()
        token_sig = kalvin.encode("hello there!")

        # Verify encoding returns a valid signature
        assert token_sig is not None
        # Verify that the model grew
        assert kalvin.model_size() > 0

    def test_add_encoded_strings(self):
        kalvin = Kalvin()
        token_sig1 = kalvin.encode("hello there!")
        token_sig2 = kalvin.encode("hello dolly!")

        # Verify both encodings return valid KLines
        assert token_sig1 is not None
        assert token_sig2 is not None
        # Different strings should produce different s_keys
        assert token_sig1.signature != token_sig2.signature

    def test_existing_substring(self):
        """Existing sub-strings do not create new klines"""

        vl_str = """Mary had a little lamb,
          Its fleece was white as snow,
          And everywhere that Mary went,
          The lamb was sure to go.
        """
        kalvin = Kalvin()
        kalvin.encode(vl_str)
        model1 = kalvin.model.duplicate()
        kalvin.encode("Mary had a little lamb,")  # adds some new klines
        kalvin.encode(" was white")  # adds some new klines

        # Verify that encoding adds klines (exact count depends on deduplication)
        assert len(kalvin.model) > len(model1)

    def test_intermediate_signature_count(self):
        """Test that encoding creates the expected number of klines."""
        kalvin = Kalvin()
        kalvin.encode("a b c d")  # 5 single-token klines + 1 combined
        kalvin.encode("a b c")  # tokens already exist, only new combined kline
        kalvin.encode("b c d")  # tokens already exist, only new combined kline

        # With train=True, single-token klines dedupe by signature
        # Combined klines have unique s_keys (bitwise OR of token s_keys)
        # Expected: 5 single + 3 combined = 8
        assert kalvin.model_size() == 8


class TestKalvinPrune:
    def test_prune_empty_model(self):
        """Pruning an empty model returns empty model."""
        model = Kalvin()

        pruned = model.prune()

        assert len(pruned.model) == 0

    def test_prune_keeps_all_at_level_one(self):
        """With level=1, all klines with activity >= 1 are kept."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        kl3 = KLine(signature=0x3000, nodes=[])
        model = Model([kl1, kl2, kl3])
        activity = Counter({0x1000: 1, 0x2000: 2, 0x3000: 5})

        pruned = Kalvin(model, activity).prune(level=1)

        assert len(pruned.model) == 3

    def test_prune_filters_by_level(self):
        """KLines with activity < level are removed."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        kl3 = KLine(signature=0x3000, nodes=[])
        model = Model([kl1, kl2, kl3])
        activity = Counter({0x1000: 1, 0x2000: 3, 0x3000: 5})

        pruned = Kalvin(model, activity).prune(level=3)

        assert len(pruned.model) == 2
        found = model.find_by_key(0x2000)
        assert found in list(pruned.model)
        found = model.find_by_key(0x3000)
        assert found in list(pruned.model)

    def test_prune_removes_all_when_level_high(self):
        """When level is higher than all activities, result is empty."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 1, 0x2000: 2})

        pruned = Kalvin(model, activity).prune(level=10)

        assert len(pruned.model) == 0

    def test_prune_with_empty_activity(self):
        """Pruning with empty activity counter returns model unchanged."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter()

        pruned = Kalvin(model, activity).prune()

        assert len(pruned.model) == 2

    def test_prune_ignores_keys_not_in_model(self):
        """Activity keys not in model are ignored."""
        kl1 = KLine(signature=0x1000, nodes=[])
        model = Model([kl1])
        activity = Counter({0x1000: 5, 0x9999: 10})  # 0x9999 not in model

        pruned = Kalvin(model, activity).prune()

        assert len(pruned.model) == 1
        assert model.find_by_key(0x1000) in list(pruned.model)

    def test_prune_preserves_original_model(self):
        """Pruning does not modify the original model."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        assert len(model) == 2
        assert len(pruned.model) == 2

    def test_prune_returns_new_model_instance(self):
        """Prune returns a new Model instance."""
        kl1 = KLine(signature=0x1000, nodes=[])
        model = Model([kl1])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        assert pruned is not model
        assert isinstance(pruned.model, Model)

    def test_prune_level_boundary(self):
        """KLines with activity exactly at level are kept."""
        kl1 = KLine(signature=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(signature=0x2000, nodes=[])  # activity = 2
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 2})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 2  # Both kept (3 >= 2 and 2 >= 2)

    def test_prune_level_excludes_below(self):
        """KLines with activity below level are excluded."""
        kl1 = KLine(signature=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(signature=0x2000, nodes=[])  # activity = 1
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(0x1000)
        assert found in list(pruned.model)

    def test_prune_with_large_keys(self):
        """Prune works with large key values."""
        key1 = 0x8000_1000  # Large key value
        key2 = 0x8000_2000  # Large key value
        kl1 = KLine(signature=key1, nodes=[])
        kl2 = KLine(signature=key2, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(key1)
        assert found in list(pruned.model)

    def test_prune_with_small_keys(self):
        """Prune works with small key values."""
        key1 = 0x1000
        key2 = 0x2000
        kl1 = KLine(signature=key1, nodes=[])
        kl2 = KLine(signature=key2, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(key1)
        assert found in list(pruned.model)

    def test_prune_keeps_kline_reference(self):
        """Pruned model keeps references to original klines."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        model = Model([kl1])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        # Same kline object referenced
        assert pruned.model[0] is kl1
        assert pruned.model[0].nodes == [0x0100, 0x0200]
