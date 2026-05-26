"""Tests for AgentCodec — direct codec tests independent of Agent."""

import tempfile
from collections import Counter
from pathlib import Path

from kalvin.agent_codec import AgentCodec, BinaryAdapter, JsonAdapter
from kalvin.kline import KLine
from kalvin.model import Model

# ── Helpers ────────────────────────────────────────────────────────────

def _make_model_with_klines() -> tuple[Model, Counter]:
    """Build a model with a few klines and a non-empty activity counter."""
    model = Model()
    model.add(KLine(5, [1, 2]))
    model.promote(KLine(5, [1, 2]))
    model.add(KLine(10, [3, 4]))
    model.promote(KLine(10, [3, 4]))
    model.add(KLine(0, []))
    model.promote(KLine(0, []))
    activity = Counter({5: 3, 10: 1})
    return model, activity


def _models_equal(a: Model, b: Model) -> bool:
    """Compare two models by checking all klines."""
    a_klines = {(kl.signature, tuple(kl.nodes)) for kl in a}
    b_klines = {(kl.signature, tuple(kl.nodes)) for kl in b}
    return a_klines == b_klines


# ── Binary Roundtrip ──────────────────────────────────────────────────

class TestAgentCodec:
    def test_binary_roundtrip(self):
        """Codec to_bytes → from_bytes preserves model and activity."""
        model, activity = _make_model_with_klines()
        codec = AgentCodec(model, activity)
        data = codec.to_bytes()
        loaded_model, loaded_activity = AgentCodec.from_bytes(data)
        assert _models_equal(model, loaded_model)
        assert activity == loaded_activity

    def test_json_roundtrip(self):
        """Codec to_dict → from_dict preserves model and activity."""
        model, activity = _make_model_with_klines()
        codec = AgentCodec(model, activity)
        d = codec.to_dict()
        loaded_model, loaded_activity = AgentCodec.from_dict(d)
        assert _models_equal(model, loaded_model)
        assert activity == loaded_activity

    def test_binary_adapter_direct(self):
        """BinaryAdapter encode/decode in isolation."""
        model, activity = _make_model_with_klines()
        data = BinaryAdapter.encode(model, activity)
        assert isinstance(data, bytes)
        assert len(data) > 0
        loaded_model, loaded_activity = BinaryAdapter.decode(data)
        assert _models_equal(model, loaded_model)
        assert activity == loaded_activity

    def test_json_adapter_direct(self):
        """JsonAdapter encode/decode in isolation."""
        model, activity = _make_model_with_klines()
        d = JsonAdapter.encode(model, activity)
        assert isinstance(d, dict)
        assert "klines" in d
        assert "activity" in d
        loaded_model, loaded_activity = JsonAdapter.decode(d)
        assert _models_equal(model, loaded_model)
        assert activity == loaded_activity

    def test_save_load_json(self):
        """File roundtrip via codec (JSON format)."""
        model, activity = _make_model_with_klines()
        codec = AgentCodec(model, activity)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            codec.save(path)
            loaded_model, loaded_activity = AgentCodec.load(path)
            assert _models_equal(model, loaded_model)
            assert activity == loaded_activity
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_bin(self):
        """File roundtrip via codec (binary format)."""
        model, activity = _make_model_with_klines()
        codec = AgentCodec(model, activity)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            codec.save(path, format="bin")
            loaded_model, loaded_activity = AgentCodec.load(path, format="bin")
            assert _models_equal(model, loaded_model)
            assert activity == loaded_activity
        finally:
            path.unlink(missing_ok=True)

    def test_empty_codec(self):
        """Empty model/activity roundtrips correctly."""
        model = Model()
        activity: Counter = Counter()
        codec = AgentCodec(model, activity)

        # Binary
        data = codec.to_bytes()
        loaded_model, loaded_activity = AgentCodec.from_bytes(data)
        assert len(loaded_model) == 0
        assert loaded_activity == Counter()

        # JSON
        d = codec.to_dict()
        loaded_model, loaded_activity = AgentCodec.from_dict(d)
        assert len(loaded_model) == 0
        assert loaded_activity == Counter()

    def test_activity_preserved_binary(self):
        """Counter activity data survives binary roundtrip."""
        model = Model()
        activity = Counter({42: 100, 99: 1, 0: 50})
        codec = AgentCodec(model, activity)
        data = codec.to_bytes()
        _, loaded_activity = AgentCodec.from_bytes(data)
        assert loaded_activity == activity

    def test_activity_preserved_json(self):
        """Counter activity data survives JSON roundtrip."""
        model = Model()
        activity = Counter({42: 100, 99: 1, 0: 50})
        codec = AgentCodec(model, activity)
        d = codec.to_dict()
        _, loaded_activity = AgentCodec.from_dict(d)
        assert loaded_activity == activity
