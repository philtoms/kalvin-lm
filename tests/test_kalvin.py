"""Tests for Kalvin class - serialization and file operations."""

import json
import pytest
import tempfile
from pathlib import Path
from collections import Counter

from kalvin.kalvin import Kalvin
from kalvin.model import KLine, KLineType, Model, create_signature_key, create_embedding_key


class TestKalvinInit:
    """Tests for Kalvin initialization."""

    def test_init_empty(self):
        """Kalvin can be initialized without a model."""
        kalvin = Kalvin()
        assert kalvin.model is not None
        assert len(kalvin.model) == 0

    def test_init_with_model(self):
        """Kalvin can be initialized with an existing model."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        model = Model(klines)
        kalvin = Kalvin(model)

        assert len(kalvin.model) == 1
        assert kalvin.model[0].s_key == 0x1000

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

        # 4 bytes for kline count + 4 bytes for activity (0)
        assert data == b"\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_to_bytes_single_kline_no_nodes(self):
        """Single KLine with no nodes serializes correctly."""
        kline = KLine(s_key=0x123456789ABCDEF0, nodes=[])
        model = Model([kline])
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # 4 bytes count + 8 bytes s_key + 4 bytes node count + 4 bytes for activity  (0)
        assert len(data) == 20
        # Verify count (little-endian uint32)
        assert data[:4] == b"\x01\x00\x00\x00"
        # Verify s_key (little-endian uint64)
        assert data[4:12] == b"\xF0\xDE\xBC\x9A\x78\x56\x34\x12"
        # Verify node count
        assert data[12:16] == b"\x00\x00\x00\x00"
        # Verify node count
        assert data[16:20] == b"\x00\x00\x00\x00"

    def test_to_bytes_single_kline_with_nodes(self):
        """Single KLine with nodes serializes correctly."""
        s_key = 0x1000
        nodes = [0x0100, 0x0200, 0x0300]
        kline = KLine(s_key=s_key, nodes=nodes)
        model = Model([kline])
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # 4 bytes count + 8 bytes s_key + 4 bytes node count + 3*8 bytes nodes + 4 bytes activity
        assert len(data) == 4 + 8 + 4 + 24 + 4
        # Verify node count is 3
        assert data[12:16] == b"\x03\x00\x00\x00"

    def test_to_bytes_multiple_klines(self):
        """Multiple KLines serialize correctly."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100]),
            KLine(s_key=0x2000, nodes=[0x0200, 0x0300]),
            KLine(s_key=0x3000, nodes=[]),
        ]
        model = Model(klines)
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # 4 bytes count + (8+4+8) + (8+4+16) + (8+4)
        assert len(data) == 4 + 20 + 28 + 12 + 4
        # Verify kline count is 3
        assert data[:4] == b"\x03\x00\x00\x00"

    def test_to_bytes_preserves_high_bit(self):
        """High bit in s_key is preserved during serialization."""
        node_key = create_signature_key(0x1000)
        embedding_key = create_embedding_key(0x2000)

        klines = [
            KLine(s_key=node_key, nodes=[]),
            KLine(s_key=embedding_key, nodes=[]),
        ]
        model = Model(klines)
        kalvin = Kalvin(model)

        data = kalvin.to_bytes()

        # Deserialize and verify
        kalvin2 = Kalvin.from_bytes(data)
        assert kalvin2.model[0].s_key == node_key
        assert kalvin2.model[1].s_key == embedding_key


class TestKalvinFromBytes:
    """Tests for binary deserialization."""

    def test_from_bytes_empty(self):
        """Empty bytes deserializes to empty Kalvin."""
        data = b"\x00\x00\x00\x00\x00\x00\x00\x00"

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
        kline = KLine(s_key=0x123456789ABCDEF0, nodes=[0x0100, 0x0200])
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 1
        assert restored.model[0].s_key == kline.s_key
        assert restored.model[0].nodes == kline.nodes

    def test_from_bytes_roundtrip_multiple_klines(self):
        """Roundtrip preserves multiple KLines."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100]),
            KLine(s_key=0x2000, nodes=[0x0200, 0x0300]),
            KLine(s_key=0x3000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 3
        for i, kl in enumerate(original.model):
            assert restored.model[i].s_key == kl.s_key
            assert restored.model[i].nodes == kl.nodes


class TestKalvinToDict:
    """Tests for dictionary serialization."""

    def test_to_dict_empty(self):
        """Empty Kalvin serializes to dict with empty klines."""
        kalvin = Kalvin()

        data = kalvin.to_dict()

        assert data == {"klines": [], "activity": {}}

    def test_to_dict_single_kline(self):
        """Single KLine serializes to dict correctly."""
        kline = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        kalvin = Kalvin(Model([kline]))

        data = kalvin.to_dict()

        assert data == {
            "klines": [
                {"s_key": 0x1000, "nodes": [0x0100, 0x0200]}
            ],
            "activity": {}
        }

    def test_to_dict_multiple_klines(self):
        """Multiple KLines serialize to dict correctly."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100]),
            KLine(s_key=0x2000, nodes=[]),
        ]
        kalvin = Kalvin(Model(klines))

        data = kalvin.to_dict()

        assert data == {
            "klines": [
                {"s_key": 0x1000, "nodes": [0x0100]},
                {"s_key": 0x2000, "nodes": []},
            ],
            "activity": {}
        }

    def test_to_dict_is_json_serializable(self):
        """Dict output is JSON serializable."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
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
        data = {"klines": [{"s_key": 0x1000, "nodes": [0x0100, 0x0200]}]}

        kalvin = Kalvin.from_dict(data)

        assert len(kalvin.model) == 1
        assert kalvin.model[0].s_key == 0x1000
        assert kalvin.model[0].nodes == [0x0100, 0x0200]

    def test_from_dict_roundtrip(self):
        """Roundtrip preserves all data."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100]),
            KLine(s_key=0x2000, nodes=[0x0200, 0x0300]),
        ]
        original = Kalvin(Model(klines))

        data = original.to_dict()
        restored = Kalvin.from_dict(data)

        assert len(restored.model) == len(original.model)
        for i, kl in enumerate(original.model):
            assert restored.model[i].s_key == kl.s_key
            assert restored.model[i].nodes == kl.nodes


class TestKalvinSaveLoad:
    """Tests for file operations."""

    def test_save_binary_default(self):
        """Save uses binary format by default."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path)

            # Read raw bytes to verify binary format
            raw = path.read_bytes()
            assert raw[:4] == b"\x01\x00\x00\x00"  # Binary kline count

    def test_save_binary_explicit(self):
        """Save with format='binary' creates binary file."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path, format="binary")

            raw = path.read_bytes()
            assert raw[:4] == b"\x01\x00\x00\x00"

    def test_save_json_explicit(self):
        """Save with format='json' creates JSON file."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            kalvin.save(path, format="json")

            # Should be valid JSON
            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["s_key"] == 0x1000

    def test_save_json_auto_detect(self):
        """Save with format=None auto-detects JSON from .json extension."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            kalvin.save(path, format=None)  # Explicit None for auto-detect

            content = path.read_text()
            data = json.loads(content)
            assert data["klines"][0]["s_key"] == 0x1000

    def test_save_binary_auto_detect(self):
        """Save auto-detects binary format from non-.json extension."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            kalvin.save(path)  # format=None, auto-detect

            raw = path.read_bytes()
            assert raw[:4] == b"\x01\x00\x00\x00"  # Binary format

    def test_load_binary_default(self):
        """Load uses binary format by default."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="binary")

            restored = Kalvin.load(path)  # Default binary

            assert len(restored.model) == 1
            assert restored.model[0].s_key == 0x1000

    def test_load_json_explicit(self):
        """Load with format='json' reads JSON file."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path, format="json")

            restored = Kalvin.load(path, format="json")

            assert restored.model[0].s_key == 0x1000

    def test_load_json_auto_detect(self):
        """Load auto-detects JSON format from .json extension."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")

            restored = Kalvin.load(path)  # Auto-detect

            assert restored.model[0].s_key == 0x1000

    def test_load_binary_auto_detect(self):
        """Load auto-detects binary format from non-.json extension."""
        klines = [KLine(s_key=0x1000, nodes=[0x0100])]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            original.save(path, format="binary")

            restored = Kalvin.load(path)  # Auto-detect

            assert restored.model[0].s_key == 0x1000

    def test_roundtrip_binary_file(self):
        """Roundtrip through binary file preserves data."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100, 0x0200]),
            KLine(s_key=0x2000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.kalvin"
            original.save(path)
            restored = Kalvin.load(path)

            assert len(restored.model) == len(original.model)
            for i, kl in enumerate(original.model):
                assert restored.model[i].s_key == kl.s_key
                assert restored.model[i].nodes == kl.nodes

    def test_roundtrip_json_file(self):
        """Roundtrip through JSON file preserves data."""
        klines = [
            KLine(s_key=0x1000, nodes=[0x0100, 0x0200]),
            KLine(s_key=0x2000, nodes=[]),
        ]
        original = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original.save(path, format="json")
            restored = Kalvin.load(path, format="json")

            assert len(restored.model) == len(original.model)
            for i, kl in enumerate(original.model):
                assert restored.model[i].s_key == kl.s_key
                assert restored.model[i].nodes == kl.nodes

    def test_save_to_existing_directory(self):
        """Save works when directory exists."""
        klines = [KLine(s_key=0x1000, nodes=[])]
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
        klines = [KLine(s_key=0x1000, nodes=[])]
        kalvin = Kalvin(Model(klines))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.kalvin")
            kalvin.save(path)

            assert Path(path).exists()

    def test_load_with_string_path(self):
        """Load accepts string path."""
        klines = [KLine(s_key=0x1000, nodes=[])]
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
        klines = [KLine(s_key=i, nodes=[i * 100]) for i in range(500)]
        original = Kalvin(Model(klines))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model) == 500
        assert restored.model[0].s_key == 0
        assert restored.model[499].s_key == 499

    def test_large_node_count(self):
        """Handles KLines with many nodes."""
        nodes = list(range(1000))
        kline = KLine(s_key=0x1000, nodes=nodes)
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert len(restored.model[0].nodes) == 1000
        assert restored.model[0].nodes == nodes

    def test_max_s_key_value(self):
        """Handles maximum s_key value (all bits set)."""
        max_key = 0xFFFF_FFFF_FFFF_FFFF
        kline = KLine(s_key=max_key, nodes=[])
        original = Kalvin(Model([kline]))

        data = original.to_bytes()
        restored = Kalvin.from_bytes(data)

        assert restored.model[0].s_key == max_key

class TestKalvinEmbeddings:
    """Tests for embedding handling."""

    def test_add_new_encoding(self):
        kalvin = Kalvin()
        token_sig = kalvin.encode("hello there")
        assert token_sig is not None

        decoded = kalvin.decode(token_sig)
        assert decoded == "hello there"


    def test_add_very_long_string(self):
        vl_str = """
          Mary had a little lamb, 
          Its fleece was white as snow, 
          And everywhere that Mary went, 
          The lamb was sure to go. 
        """
        kalvin = Kalvin()
        token_sig = kalvin.encode(vl_str)

        assert kalvin.decode(token_sig) == vl_str

    def test_add_duplicate_encoding(self):
        kalvin = Kalvin()
        token_sig1 = kalvin.encode("hello there")
        token_sig2 = kalvin.encode("hello there")

        assert token_sig1 != token_sig2

    def test_add_encoded_string(self):
        kalvin = Kalvin()
        token_sig = kalvin.encode("hello there!")

        assert kalvin.decode(token_sig) == "hello there!"


    def test_add_encoded_strings(self):
        kalvin = Kalvin()
        token_sig1 = kalvin.encode("hello there!")
        token_sig2 = kalvin.encode("hello dolly!")

        assert kalvin.decode(token_sig1) == "hello there!"
        assert kalvin.decode(token_sig2) == "hello dolly!"

  
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
        kalvin.encode("Mary had a little lamb,") # + 2
        kalvin.encode(" was white") # + 2

        assert len(kalvin.model) == len(model1) + 4


    def test_intermediate_signature_count(self):
        
        kalvin = Kalvin()
        kalvin.encode("a b c d")    # 4 + 2 + 1
        kalvin.encode("a b c")      # _ + _ + 1
        kalvin.encode("b c d")      # 1 + 1 + 1

        assert kalvin.model_size() == 11

    
class TestKalvinPrune:
    def test_prune_empty_model(self):
        """Pruning an empty model returns empty model."""
        model = Kalvin()

        pruned = model.prune()

        assert len(pruned.model) == 0

    def test_prune_keeps_all_at_level_one(self):
        """With level=1, all klines with activity >= 1 are kept."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        kl3 = KLine(s_key=0x3000, nodes=[])
        model = Model([kl1, kl2, kl3])
        activity = Counter({0x1000: 1, 0x2000: 2, 0x3000: 5})

        pruned = Kalvin(model, activity).prune(level=1)

        assert len(pruned.model) == 3

    def test_prune_filters_by_level(self):
        """KLines with activity < level are removed."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        kl3 = KLine(s_key=0x3000, nodes=[])
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
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 1, 0x2000: 2})

        pruned = Kalvin(model, activity).prune(level=10)

        assert len(pruned.model) == 0

    def test_prune_with_empty_activity(self):
        """Pruning with empty activity counter returns empty model."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter()

        pruned = Kalvin(model, activity).prune()

        assert len(pruned.model) == 0

    def test_prune_ignores_keys_not_in_model(self):
        """Activity keys not in model are ignored."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        model = Model([kl1])
        activity = Counter({0x1000: 5, 0x9999: 10})  # 0x9999 not in model

        pruned = Kalvin(model, activity).prune()

        assert len(pruned.model) == 1
        assert model.find_by_key(0x1000) in list(pruned.model)

    def test_prune_preserves_original_model(self):
        """Pruning does not modify the original model."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        kl2 = KLine(s_key=0x2000, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        assert len(model) == 2
        assert len(pruned.model) == 1

    def test_prune_returns_new_model_instance(self):
        """Prune returns a new Model instance."""
        kl1 = KLine(s_key=0x1000, nodes=[])
        model = Model([kl1])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        assert pruned is not model
        assert isinstance(pruned.model, Model)

    def test_prune_level_boundary(self):
        """KLines with activity exactly at level are kept."""
        kl1 = KLine(s_key=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(s_key=0x2000, nodes=[])  # activity = 2
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 2})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 2  # Both kept (3 >= 2 and 2 >= 2)

    def test_prune_level_excludes_below(self):
        """KLines with activity below level are excluded."""
        kl1 = KLine(s_key=0x1000, nodes=[])  # activity = 3
        kl2 = KLine(s_key=0x2000, nodes=[])  # activity = 1
        model = Model([kl1, kl2])
        activity = Counter({0x1000: 3, 0x2000: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(0x1000)
        assert found in list(pruned.model)

    def test_prune_with_signature_keys(self):
        """Prune works with signature (high bit) keys."""
        key1 = create_signature_key(0x1000)
        key2 = create_signature_key(0x2000)
        kl1 = KLine(s_key=key1, nodes=[])
        kl2 = KLine(s_key=key2, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(key1)
        assert found in list(pruned.model)

    def test_prune_with_embedding_keys(self):
        """Prune works with embedding (no high bit) keys."""
        key1 = create_embedding_key(0x1000)
        key2 = create_embedding_key(0x2000)
        kl1 = KLine(s_key=key1, nodes=[])
        kl2 = KLine(s_key=key2, nodes=[])
        model = Model([kl1, kl2])
        activity = Counter({key1: 5, key2: 1})

        pruned = Kalvin(model, activity).prune(level=2)

        assert len(pruned.model) == 1
        found = model.find_by_key(key1)
        assert found in list(pruned.model)

    def test_prune_keeps_kline_reference(self):
        """Pruned model keeps references to original klines."""
        kl1 = KLine(s_key=0x1000, nodes=[0x0100, 0x0200])
        model = Model([kl1])
        activity = Counter({0x1000: 5})

        pruned = Kalvin(model, activity).prune()

        # Same kline object referenced
        assert pruned.model[0] is kl1
        assert pruned.model[0].nodes == [0x0100, 0x0200]
