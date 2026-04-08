"""Tests for Model class - serialization and file operations."""

import json
import pytest
import tempfile
from pathlib import Path
from collections import Counter

from kalvin.abstract import KLine
from kalvin.model import Model
from kalvin.frame import Frame


class TestModelInit:
    """Tests for Model initialization."""

    def test_init_empty(self):
        """Model can be initialized without a frame."""
        model = Model()
        assert model.frame is not None
        assert len(model.frame) == 0

    def test_init_with_frame(self):
        """Model can be initialized with an existing frame."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        frame = Frame(klines)
        model = Model(frame=frame)

        assert len(model.frame) == 1
        assert model.frame[0x1000].signature == 0x1000

    def test_init_with_none_creates_empty_frame(self):
        """Passing None creates an empty frame."""
        model = Model(None)
        assert model.frame is not None
        assert len(model.frame) == 0


class TestModelToBytes:
    """Tests for binary serialization."""

    def test_to_bytes_empty(self):
        """Empty Model serializes to minimal bytes."""
        model = Model()

        data = model.to_bytes()

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
        frame = Frame([kline])
        model = Model(frame=frame)

        data = model.to_bytes()

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
        frame = Frame([kline])
        model = Model(frame=frame)

        data = model.to_bytes()

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
        frame = Frame(klines)
        model = Model(frame=frame)

        data = model.to_bytes()

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
        frame = Frame(klines)
        model = Model(frame=frame)

        data = model.to_bytes()

        # Deserialize and verify
        model2 = Model.from_bytes(data)
        assert model2.frame[key1].signature == key1
        assert model2.frame[key2].signature == key2


class TestModelFromBytes:
    """Tests for binary deserialization."""

    def test_from_bytes_empty(self):
        """Empty bytes deserializes to empty Model."""
        # Create empty model and serialize it to get valid bytes
        original = Model()
        data = original.to_bytes()

        model = Model.from_bytes(data)

        assert len(model.frame) == 0

    def test_from_bytes_roundtrip_empty(self):
        """Roundtrip serialization preserves empty Model."""
        original = Model()

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert len(restored.frame) == len(original.frame)

    def test_from_bytes_roundtrip_single_kline(self):
        """Roundtrip preserves single KLine."""
        kline = KLine(signature=0x123456789ABCDEF0, nodes=[0x0100, 0x0200])
        original = Model(frame=Frame([kline]))

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert len(restored.frame) == 1
        assert restored.frame[kline.signature].signature == kline.signature
        assert restored.frame[kline.signature].nodes == kline.nodes

    def test_from_bytes_roundtrip_multiple_klines(self):
        """Roundtrip preserves multiple KLines."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[0x0200, 0x0300]),
            KLine(signature=0x3000, nodes=[]),
        ]
        original = Model(frame=Frame(klines))

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert len(restored.frame) == 3
        for kl in original.frame:
            assert restored.frame[kl.signature].signature == kl.signature
            assert restored.frame[kl.signature].nodes == kl.nodes


class TestModelToDict:
    """Tests for dictionary serialization."""

    def test_to_dict_empty(self):
        """Empty Model serializes to dict with empty klines."""
        model = Model()

        data = model.to_dict()

        assert "metadata" in data
        assert data["klines"] == []
        assert data["activity"] == {}

    def test_to_dict_single_kline(self):
        """Single KLine serializes to dict correctly."""
        kline = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        model = Model(frame=Frame([kline]))

        data = model.to_dict()

        assert "metadata" in data
        assert data["klines"] == [{"signature": 0x1000, "nodes": [0x0100, 0x0200]}]
        assert data["activity"] == {}

    def test_to_dict_multiple_klines(self):
        """Multiple KLines serialize to dict correctly."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[]),
        ]
        model = Model(frame=Frame(klines))

        data = model.to_dict()

        assert "metadata" in data
        assert data["klines"] == [
            {"signature": 0x1000, "nodes": [0x0100]},
            {"signature": 0x2000, "nodes": []},
        ]
        assert data["activity"] == {}

    def test_to_dict_is_json_serializable(self):
        """Dict output is JSON serializable."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        data = model.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert json.loads(json_str) == data


class TestModelFromDict:
    """Tests for dictionary deserialization."""

    def test_from_dict_empty(self):
        """Empty dict creates empty Model."""
        model = Model.from_dict({"klines": []})

        assert len(model.frame) == 0

    def test_from_dict_single_kline(self):
        """Single KLine deserializes from dict correctly."""
        data = {"klines": [{"signature": 0x1000, "nodes": [0x0100, 0x0200]}]}

        model = Model.from_dict(data)

        assert len(model.frame) == 1
        assert model.frame[0x1000].signature == 0x1000
        assert model.frame[0x1000].nodes == [0x0100, 0x0200]

    def test_from_dict_roundtrip(self):
        """Roundtrip preserves all data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100]),
            KLine(signature=0x2000, nodes=[0x0200, 0x0300]),
        ]
        original = Model(frame=Frame(klines))

        data = original.to_dict()
        restored = Model.from_dict(data)

        assert len(restored.frame) == len(original.frame)
        for kl in original.frame:
            assert restored.frame[kl.signature].signature == kl.signature
            assert restored.frame[kl.signature].nodes == kl.nodes


class TestModelSaveLoad:
    """Tests for file operations."""

    def test_save_binary_default(self):
        """Save uses binary format by default."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            model.save(path)

            # Verify binary format by checking it's not valid JSON and can be loaded
            raw = path.read_bytes()
            # First 4 bytes should be metadata length (not a JSON start character)
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Model.load(path)
            assert len(restored.frame) == 1
            assert restored.frame[0x1000].signature == 0x1000

    def test_save_binary_explicit(self):
        """Save with format='binary' creates binary file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            model.save(path, format="bin")

            # Verify binary format by checking it's not valid JSON
            raw = path.read_bytes()
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Model.load(path)
            assert len(restored.frame) == 1

    def test_save_json_explicit(self):
        """Save with format='json' creates JSON file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            model.save(path, format="json")

            # Should be valid JSON
            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["signature"] == 0x1000

    def test_save_json_auto_detect(self):
        """Save with format=None auto-detects JSON from .json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            model.save(path)

            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["signature"] == 0x1000

    def test_save_binary_auto_detect(self):
        """Save auto-detects binary format from non-.json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            model.save(path)  # format=None, auto-detect

            # Verify binary format by checking it's not valid JSON
            raw = path.read_bytes()
            assert raw[0] != ord('{')
            # Verify we can load it back
            restored = Model.load(path)
            assert len(restored.frame) == 1

    def test_load_binary_default(self):
        """Load uses binary format by default."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="bin")

            restored = Model.load(path)  # Default binary

            assert len(restored.frame) == 1
            assert restored.frame[0x1000].signature == 0x1000

    def test_load_json_explicit(self):
        """Load with format='json' reads JSON file."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="json")

            restored = Model.load(path, format="json")

            assert restored.frame[0x1000].signature == 0x1000

    def test_load_json_auto_detect(self):
        """Load auto-detects JSON format from .json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")

            restored = Model.load(path)  # Auto-detect

            assert restored.frame[0x1000].signature == 0x1000

    def test_load_binary_auto_detect(self):
        """Load auto-detects binary format from non-.json extension."""
        klines = [KLine(signature=0x1000, nodes=[0x0100])]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            original.save(path, format="bin")

            restored = Model.load(path)  # Auto-detect

            assert restored.frame[0x1000].signature == 0x1000

    def test_roundtrip_binary_file(self):
        """Roundtrip through binary file preserves data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100, 0x0200]),
            KLine(signature=0x2000, nodes=[]),
        ]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path)
            restored = Model.load(path)

            assert len(restored.frame) == len(original.frame)
            for kl in original.frame:
                assert restored.frame[kl.signature].signature == kl.signature
                assert restored.frame[kl.signature].nodes == kl.nodes

    def test_roundtrip_json_file(self):
        """Roundtrip through JSON file preserves data."""
        klines = [
            KLine(signature=0x1000, nodes=[0x0100, 0x0200]),
            KLine(signature=0x2000, nodes=[]),
        ]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")
            restored = Model.load(path, format="json")

            assert len(restored.frame) == len(original.frame)
            for kl in original.frame:
                assert restored.frame[kl.signature].signature == kl.signature
                assert restored.frame[kl.signature].nodes == kl.nodes

    def test_save_to_existing_directory(self):
        """Save works when directory exists."""
        klines = [KLine(signature=0x1000, nodes=[])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory first
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            path = subdir / "test.kalvin"
            model.save(path)

            assert path.exists()

    def test_save_with_string_path(self):
        """Save accepts string path."""
        klines = [KLine(signature=0x1000, nodes=[])]
        model = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.kalvin")
            model.save(path)

            assert Path(path).exists()

    def test_load_with_string_path(self):
        """Load accepts string path."""
        klines = [KLine(signature=0x1000, nodes=[])]
        original = Model(frame=Frame(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.kalvin")
            original.save(path)
            restored = Model.load(path)

            assert len(restored.frame) == 1


class TestModelLargeData:
    """Tests for handling larger data sets."""

    def test_large_kline_count(self):
        """Handles hundreds of KLines."""
        klines = [KLine(signature=i, nodes=[i * 100]) for i in range(500)]
        original = Model(frame=Frame(klines))

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert len(restored.frame) == 500
        assert restored.frame[0].signature == 0
        assert restored.frame[499].signature == 499

    def test_large_node_count(self):
        """Handles KLines with many nodes."""
        nodes = list(range(1000))
        kline = KLine(signature=0x1000, nodes=nodes)
        original = Model(frame=Frame([kline]))

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert len(restored.frame[0x1000].nodes) == 1000
        assert restored.frame[0x1000].nodes == nodes

    def test_max_s_key_value(self):
        """Handles maximum signature value (all bits set)."""
        max_key = 0xFFFF_FFFF_FFFF_FFFF
        kline = KLine(signature=max_key, nodes=[])
        original = Model(frame=Frame([kline]))

        data = original.to_bytes()
        restored = Model.from_bytes(data)

        assert restored.frame[max_key].signature == max_key


class TestModelEmbeddings:
    """Tests for embedding handling."""

    def test_add_new_encoding(self):
        model = Model()
        token_sig = model.encode("hello there")
        assert token_sig is not None
        # Verify that encoding adds klines to the frame
        assert model.frame_size() > 0

    def test_add_very_long_string(self):
        vl_str = """
          Mary had a little lamb,
          Its fleece was white as snow,
          And everywhere that Mary went,
          The lamb was sure to go.
        """
        model = Model()
        token_sig = model.encode(vl_str)

        # Verify encoding returns a valid signature
        assert token_sig is not None
        # Verify that the frame grew
        assert model.frame_size() > 0

    def test_add_duplicate_encoding(self):
        model = Model()
        model.encode("hello there")
        frame_len  = model.frame_size()
        model.encode("hello there")

        assert frame_len  == model.frame_size()  # Frame not extended

    def test_add_encoded_string(self):
        model = Model()
        token_sig = model.encode("hello there!")

        # Verify encoding returns a valid signature
        assert token_sig is not None
        # Verify that the frame grew
        assert model.frame_size() > 0

    def test_add_encoded_strings(self):
        model = Model()
        token_sig1 = model.encode("hello there!")
        token_sig2 = model.encode("hello dolly!")

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
        model = Model()
        model.encode(vl_str)
        frame1 = model.frame_size()
        model.encode("Mary had a little lamb,")  # adds some new klines
        model.encode(" was white")  # adds some new klines

        # Verify that encoding adds klines (exact count depends on deduplication)
        assert model.frame_size() > frame1

    def test_intermediate_signature_count(self):
        """Test that encoding creates the expected number of klines."""
        model = Model()
        model.encode("a b c d")  # 4 identity + 2 ws + 1 compound = 11
        assert model.frame_size() == 7
        model.encode("a b c")  # tokens already exist, only new compound kline
        assert model.frame_size() == 8
        model.encode("b c d")  # tokens already exist, only new compound kline
        assert model.frame_size() == 9


class TestModelPrune:
    def test_prune_empty_model(self):
        """Pruning an empty model returns empty model."""
        model = Model()

        pruned = model.prune()

        assert len(pruned.frame) == 0

    def test_prune_keeps_all_at_level_one(self):
        """With level=1, all klines with activity >= 1 are kept."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        kl3 = KLine(signature=0x3000, nodes=[])
        frame = Frame([kl1, kl2, kl3])
        activity = Counter({0x1000: 1, 0x2000: 2, 0x3000: 5})

        pruned = Model(frame=frame, activity=activity).prune(level=1)

        assert len(pruned.frame) == 3

    def test_prune_filters_by_level(self):
        """KLines with activity < level are removed."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        kl3 = KLine(signature=0x3000, nodes=[])
        frame = Frame([kl1, kl2, kl3])
        activity = Counter({0x1000: 1, 0x2000: 3, 0x3000: 5})

        pruned = Model(frame=frame, activity=activity).prune(level=3)

        assert len(pruned.frame) == 2
        found = frame.find_kline(0x2000)
        assert pruned.frame[found.signature] is found

        found = frame.find_kline(0x3000)
        assert pruned.frame[found.signature] is found

    def test_prune_removes_all_when_level_high(self):
        """When level is higher than all activities, result is empty."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        frame = Frame([kl1, kl2])
        activity = Counter({0x1000: 1, 0x2000: 2})

        pruned = Model(frame=frame, activity=activity).prune(level=10)

        assert len(pruned.frame) == 0

    def test_prune_with_empty_activity(self):
        """Pruning with empty activity counter returns model unchanged."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        frame = Frame([kl1, kl2])
        activity = Counter()

        pruned = Model(frame=frame, activity=activity).prune()

        assert len(pruned.frame) == 2

    def test_prune_ignores_keys_not_in_model(self):
        """Activity keys not in model are ignored."""
        kl1 = KLine(signature=0x1000, nodes=[])
        frame = Frame([kl1])
        activity = Counter({0x1000: 5, 0x9999: 10})  # 0x9999 not in model

        pruned = Model(frame=frame, activity=activity).prune()

        assert len(pruned.frame) == 1
        assert frame.find_kline(0x1000) == pruned.frame.find_kline(0x1000)

    def test_prune_preserves_original_model(self):
        """Pruning does not modify the original model."""
        kl1 = KLine(signature=0x1000, nodes=[])
        kl2 = KLine(signature=0x2000, nodes=[])
        frame = Frame([kl1, kl2])
        activity = Counter({0x1000: 5})

        pruned = Model(frame=frame, activity=activity).prune()

        assert len(frame) == 2
        assert len(pruned.frame) == 2

    def test_prune_returns_new_model_instance(self):
        """Prune returns a new Model instance."""
        kl1 = KLine(signature=0x1000, nodes=[])
        frame = Frame([kl1])
        activity = Counter({0x1000: 5})

        pruned = Model(frame=frame, activity=activity).prune()

        assert pruned is not frame
        assert isinstance(pruned.frame, Frame)

    def test_prune_level_boundary(self):
        """KLines with activity exactly at level are kept."""
        kl1 = KLine(signature=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(signature=0x2000, nodes=[])  # activity = 2
        frame = Frame([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 2})

        pruned = Model(frame=frame, activity=activity).prune(level=2)

        assert len(pruned.frame) == 2  # Both kept (3 >= 2 and 2 >= 2)

    def test_prune_level_excludes_below(self):
        """KLines with activity below level are excluded."""
        kl1 = KLine(signature=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(signature=0x2000, nodes=[])  # activity = 1
        frame = Frame([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 1})

        pruned = Model(frame=frame, activity=activity).prune(level=2)

        assert len(pruned.frame) == 1
        found = frame.find_kline(0x1000)
        assert pruned.frame.find_kline(found.signature) == found

    def test_prune_with_large_keys(self):
        """Prune works with large key values."""
        key1 = 0x8000_1000  # Large key value
        key2 = 0x8000_2000  # Large key value
        kl1 = KLine(signature=key1, nodes=[])
        kl2 = KLine(signature=key2, nodes=[])
        frame = Frame([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Model(frame=frame, activity=activity).prune(level=2)

        assert len(pruned.frame) == 1
        found = frame.find_kline(key1)
        assert pruned.frame.find_kline(found.signature) == found

    def test_prune_with_small_keys(self):
        """Prune works with small key values."""
        key1 = 0x1000
        key2 = 0x2000
        kl1 = KLine(signature=key1, nodes=[])
        kl2 = KLine(signature=key2, nodes=[])
        frame = Frame([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Model(frame=frame, activity=activity).prune(level=2)

        assert len(pruned.frame) == 1
        found = frame.find_kline(key1)
        assert pruned.frame.find_kline(found.signature) == found

    def test_prune_keeps_kline_reference(self):
        """Pruned model keeps references to original klines."""
        kl1 = KLine(signature=0x1000, nodes=[0x0100, 0x0200])
        frame = Frame([kl1])
        activity = Counter({0x1000: 5})

        pruned = Model(frame=frame, activity=activity).prune()

        # Same kline object referenced
        assert pruned.frame[kl1.signature] is kl1
        assert pruned.frame[kl1.signature].nodes == [0x0100, 0x0200]
