"""Tests for AgentCodec — direct codec tests independent of Agent."""

import tempfile
from collections import Counter
from pathlib import Path
from struct import pack

from kalvin.agent_codec import AgentCodec, BinaryAdapter, JsonAdapter
from kalvin.kline import KLine
from kalvin.model import Model

# ── Helpers ────────────────────────────────────────────────────────────

def _make_model_with_klines() -> tuple[Model, Counter]:
    """Build a model with a few klines and a non-empty activity counter."""
    model = Model()
    model.add_ltm(KLine(5, [1, 2]))
    model.add_ltm(KLine(10, [3, 4]))
    model.add_ltm(KLine(0, []))
    activity = Counter({5: 3, 10: 1})
    return model, activity


def _make_model_with_all_tiers() -> tuple[Model, Counter]:
    """Build a model with distinct entries in STM, Frame, and LTM.

    Strategy:
      1. add_ltm for entries that should be in LTM + Frame + STM.
      2. add_frame for entries that should be in Frame + STM only.
    """
    model = Model()

    # Step 1: add_ltm — writes LTM + Frame + STM
    kl_a = KLine(5, [1, 2])
    model.add_ltm(kl_a)

    # Step 2: add_frame — writes Frame + STM (not LTM)
    kl_b = KLine(10, [3, 4])
    kl_c = KLine(15, [5, 6])
    kl_d = KLine(20, [7, 8])
    model.add_frame(kl_b)
    model.add_frame(kl_c)
    model.add_frame(kl_d)

    # Now:
    #   STM = [kl_a, kl_b, kl_c, kl_d] (all four)
    #   Frame = [kl_a, kl_b, kl_c, kl_d] (all four)
    #   LTM = [kl_a] (only add_ltm one)

    activity = Counter({5: 2, 15: 1})
    return model, activity


def _models_equal(a: Model, b: Model) -> bool:
    """Compare two models by checking Frame klines (via __iter__)."""
    a_klines = {(kl.signature, tuple(kl.nodes)) for kl in a}
    b_klines = {(kl.signature, tuple(kl.nodes)) for kl in b}
    return a_klines == b_klines


def _ltm_signatures(model: Model) -> set[tuple[int, tuple[int, ...]]]:
    """Extract (signature, nodes-tuple) pairs from LTM."""
    return {(kl.signature, tuple(kl.nodes)) for kl in model._ltm}


def _stm_signatures(model: Model) -> set[tuple[int, tuple[int, ...]]]:
    """Extract (signature, nodes-tuple) pairs from STM."""
    return {(kl.signature, tuple(kl.nodes)) for kl in model.iter_stm()}


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


# ── Three-Tier Roundtrip Tests ─────────────────────────────────────────

class TestThreeTierRoundtrip:
    """Verify STM, Frame, and LTM survive roundtrip independently."""

    def test_binary_roundtrip_three_tiers(self):
        """Binary roundtrip preserves STM, Frame, and LTM contents."""
        model, activity = _make_model_with_all_tiers()

        # Capture pre-serialization state
        orig_frame = {(kl.signature, tuple(kl.nodes)) for kl in model}
        orig_ltm = _ltm_signatures(model)
        orig_stm = _stm_signatures(model)

        data = BinaryAdapter.encode(model, activity)
        loaded_model, loaded_activity = BinaryAdapter.decode(data)

        # Verify Frame
        loaded_frame = {(kl.signature, tuple(kl.nodes)) for kl in loaded_model}
        assert loaded_frame == orig_frame

        # Verify LTM
        loaded_ltm = _ltm_signatures(loaded_model)
        assert loaded_ltm == orig_ltm

        # Verify STM (stm_contains for each original STM entry)
        for sig, nodes in orig_stm:
            assert loaded_model.stm_contains(KLine(sig, list(nodes)))

        assert loaded_activity == activity

    def test_json_roundtrip_three_tiers(self):
        """JSON roundtrip preserves STM, Frame, and LTM contents."""
        model, activity = _make_model_with_all_tiers()

        orig_frame = {(kl.signature, tuple(kl.nodes)) for kl in model}
        orig_ltm = _ltm_signatures(model)
        orig_stm = _stm_signatures(model)

        d = JsonAdapter.encode(model, activity)
        loaded_model, loaded_activity = JsonAdapter.decode(d)

        loaded_frame = {(kl.signature, tuple(kl.nodes)) for kl in loaded_model}
        assert loaded_frame == orig_frame

        loaded_ltm = _ltm_signatures(loaded_model)
        assert loaded_ltm == orig_ltm

        for sig, nodes in orig_stm:
            assert loaded_model.stm_contains(KLine(sig, list(nodes)))

        assert loaded_activity == activity

    def test_json_dict_structure(self):
        """to_dict() returns dict with stm, klines, ltm, activity keys."""
        model, activity = _make_model_with_all_tiers()
        d = JsonAdapter.encode(model, activity)

        assert isinstance(d["stm"], list)
        assert isinstance(d["klines"], list)
        assert isinstance(d["ltm"], list)
        assert isinstance(d["activity"], dict)

        # Each kline entry has the expected fields
        for key in ("stm", "klines", "ltm"):
            for entry in d[key]:
                assert "signature" in entry
                assert "nodes" in entry
                assert "literal" in entry
                assert isinstance(entry["literal"], bool)


# ── Backward Compatibility Tests ───────────────────────────────────────

class TestBackwardCompat:
    """Verify legacy formats decode correctly."""

    def test_binary_backward_compat(self):
        """Legacy binary blob (no magic) decodes as Frame-only."""
        # Manually encode a legacy blob: uint32 count + klines (no literal) + activity
        kl1 = KLine(5, [1, 2])
        kl2 = KLine(10, [3, 4])
        parts: list[bytes] = []
        parts.append(pack("<I", 2))  # kline_count
        for kl in [kl1, kl2]:
            parts.append(pack("<Q", kl.signature))
            parts.append(pack("<I", len(kl.nodes)))
            for n in kl.nodes:
                parts.append(pack("<Q", n))
        parts.append(pack("<I", 1))  # activity_count
        parts.append(pack("<Q", 5))
        parts.append(pack("<I", 3))
        legacy_data = b"".join(parts)

        model, activity = BinaryAdapter.decode(legacy_data)

        # Frame has the klines
        frame_sigs = {(kl.signature, tuple(kl.nodes)) for kl in model}
        assert frame_sigs == {(5, (1, 2)), (10, (3, 4))}

        # LTM is empty (no promote was done for legacy data)
        assert list(model._ltm) == []

        # Activity preserved
        assert activity == Counter({5: 3})

    def test_json_backward_compat(self):
        """Legacy JSON dict (no stm/ltm keys) decodes with empty LTM."""
        legacy_dict = {
            "klines": [
                {"signature": 5, "nodes": [1, 2]},
                {"signature": 10, "nodes": [3, 4]},
            ],
            "activity": {"5": 3, "10": 1},
        }

        model, activity = JsonAdapter.decode(legacy_dict)

        # Frame has the klines
        frame_sigs = {(kl.signature, tuple(kl.nodes)) for kl in model}
        assert frame_sigs == {(5, (1, 2)), (10, (3, 4))}

        # LTM is empty (no promote was done for legacy data)
        assert list(model._ltm) == []

        # Activity preserved
        assert activity == Counter({5: 3, 10: 1})


# ── Literal Preservation Tests ─────────────────────────────────────────

class TestLiteralPreservation:
    """Verify the literal flag survives roundtrip."""

    def test_literal_preserved_binary(self):
        """KLine with literal=True survives binary roundtrip."""
        model = Model()
        kl_lit = KLine(100, [200], literal=True)
        kl_norm = KLine(200, [300], literal=False)
        model.add_frame(kl_lit)
        model.add_frame(kl_norm)
        activity = Counter()

        data = BinaryAdapter.encode(model, activity)
        loaded_model, _ = BinaryAdapter.decode(data)

        # Find the literal kline by signature
        loaded_lit = loaded_model.find(100)
        assert loaded_lit is not None
        assert loaded_lit.literal is True

        loaded_norm = loaded_model.find(200)
        assert loaded_norm is not None
        assert loaded_norm.literal is False

    def test_literal_preserved_json(self):
        """KLine with literal=True survives JSON roundtrip."""
        model = Model()
        kl_lit = KLine(100, [200], literal=True)
        kl_norm = KLine(200, [300], literal=False)
        model.add_frame(kl_lit)
        model.add_frame(kl_norm)
        activity = Counter()

        d = JsonAdapter.encode(model, activity)
        loaded_model, _ = JsonAdapter.decode(d)

        loaded_lit = loaded_model.find(100)
        assert loaded_lit is not None
        assert loaded_lit.literal is True

        loaded_norm = loaded_model.find(200)
        assert loaded_norm is not None
        assert loaded_norm.literal is False


# ── Empty Tier Tests ──────────────────────────────────────────────────

class TestEmptyTiers:
    """Model with only Frame entries roundtrips correctly."""

    def test_empty_stm_ltm(self):
        """Model with only Frame entries (no add_ltm) roundtrips."""
        model = Model()
        model.add_frame(KLine(5, [1, 2]))
        model.add_frame(KLine(10, [3, 4]))
        activity = Counter({5: 1})

        # STM has klines (from add_frame), Frame has klines, LTM is empty
        orig_frame = {(kl.signature, tuple(kl.nodes)) for kl in model}

        data = BinaryAdapter.encode(model, activity)
        loaded_model, loaded_activity = BinaryAdapter.decode(data)

        loaded_frame = {(kl.signature, tuple(kl.nodes)) for kl in loaded_model}
        assert loaded_frame == orig_frame
        assert loaded_activity == activity

        # LTM should be empty (nothing was promoted)
        assert list(loaded_model._ltm) == []

        # JSON roundtrip
        d = JsonAdapter.encode(model, activity)
        loaded_model2, loaded_activity2 = JsonAdapter.decode(d)
        loaded_frame2 = {(kl.signature, tuple(kl.nodes)) for kl in loaded_model2}
        assert loaded_frame2 == orig_frame
        assert loaded_activity2 == activity
        assert list(loaded_model2._ltm) == []
