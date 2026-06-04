"""AgentCodec — serialization layer for Agent persistence.

Separates the binary and JSON serialization concerns from the Agent's
rationalisation pipeline. The codec owns the persistence format; Agent
owns the reasoning pipeline.

Binary format v2 (byte-for-byte stable):
    uint32  magic       = 0x4B4C564E ("KLVN")
    uint32  version     = 2
    uint32  stm_count
    per stm entry:   uint64 signature, uint32 node_count, node_count × uint64 nodes, uint8 literal
    uint32  frame_count
    per frame entry: uint64 signature, uint32 node_count, node_count × uint64 nodes, uint8 literal
    uint32  ltm_count
    per ltm entry:   uint64 signature, uint32 node_count, node_count × uint64 nodes, uint8 literal
    uint32  activity_count
    per activity:    uint64 key, uint32 count

Legacy binary (no magic prefix) is decoded as Frame-only with empty STM/LTM.

JSON format:
    {
        "stm":      [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
        "klines":   [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
        "ltm":      [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
        "activity": {"str_key": int, ...}
    }

Legacy JSON (only "klines" + "activity") decodes with empty STM/LTM.

Adapter classes:
    BinaryAdapter — encode/decode for the binary format.
    JsonAdapter   — encode/decode for the JSON dict format.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from struct import pack, unpack
from typing import Literal

from kalvin.kline import KLine
from kalvin.model import Model


# ── LTM iteration helper ─────────────────────────────────────────────
# TODO(KB-050): Remove this helper once Model.iter_ltm() is available.
# Uses the public API when present, otherwise falls back to the private
# _ltm KLineStore which supports the same iteration protocol.

def _iter_ltm(model: Model):
    """Yield LTM KLines in insertion order.

    Prefers the public ``model.iter_ltm()`` accessor (delivered by KB-050).
    Falls back to iterating ``model._ltm`` directly when the accessor
    is not yet available.
    """
    if hasattr(model, "iter_ltm"):
        return model.iter_ltm()
    return iter(model._ltm)


# ── Binary constants ──────────────────────────────────────────────────

MAGIC = 0x4B4C564E  # "KLVN"
FORMAT_VERSION = 2


# ── Binary Adapter ────────────────────────────────────────────────────

class BinaryAdapter:
    """Encode/decode Model + Counter to/from a stable binary format.

    v2 format writes magic + version + three kline sections (STM, Frame,
    LTM) + activity.  Each kline now includes a uint8 literal flag.

    Legacy format (no magic prefix) is decoded as Frame-only with empty
    STM and LTM.
    """

    @staticmethod
    def _encode_kline_section(klines: list[KLine], parts: list[bytes]) -> None:
        """Append a counted kline section to *parts*."""
        parts.append(pack("<I", len(klines)))
        for kl in klines:
            parts.append(pack("<Q", kl.signature))
            parts.append(pack("<I", len(kl.nodes)))
            for n in kl.nodes:
                parts.append(pack("<Q", n))
            parts.append(pack("<B", 1 if kl.literal else 0))

    @staticmethod
    def encode(model: Model, activity: Counter) -> bytes:
        """Serialize model STM, Frame, LTM and activity counter to bytes."""
        stm_klines = [kl for kl in model.iter_stm() if kl is not None]
        frame_klines = [kl for kl in model if kl is not None]
        ltm_klines = [kl for kl in _iter_ltm(model) if kl is not None]

        parts: list[bytes] = []
        parts.append(pack("<I", MAGIC))
        parts.append(pack("<I", FORMAT_VERSION))

        BinaryAdapter._encode_kline_section(stm_klines, parts)
        BinaryAdapter._encode_kline_section(frame_klines, parts)
        BinaryAdapter._encode_kline_section(ltm_klines, parts)

        # Activity
        parts.append(pack("<I", len(activity)))
        for key, count in activity.items():
            parts.append(pack("<Q", key))
            parts.append(pack("<I", count))

        return b"".join(parts)

    # ── decode helpers ────────────────────────────────────────────────

    @staticmethod
    def _read_kline_section(data: bytes, offset: int) -> tuple[list[KLine], int]:
        """Read a counted kline section (with uint8 literal) from *offset*.

        Returns (klines, new_offset).
        """
        count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        klines: list[KLine] = []
        for _ in range(count):
            sig = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            n_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes: list[int] = []
            for _ in range(n_count):
                nodes.append(unpack("<Q", data[offset:offset + 8])[0])
                offset += 8
            literal_byte = unpack("<B", data[offset:offset + 1])[0]
            offset += 1
            klines.append(KLine(sig, nodes, literal=bool(literal_byte)))
        return klines, offset

    @staticmethod
    def _read_kline_section_legacy(data: bytes, offset: int) -> tuple[list[KLine], int]:
        """Read a legacy kline section (no literal field) from *offset*.

        Returns (klines, new_offset).
        """
        count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        klines: list[KLine] = []
        for _ in range(count):
            sig = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            n_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes: list[int] = []
            for _ in range(n_count):
                nodes.append(unpack("<Q", data[offset:offset + 8])[0])
                offset += 8
            klines.append(KLine(sig, nodes))  # literal defaults to False
        return klines, offset

    @staticmethod
    def _read_activity(data: bytes, offset: int) -> tuple[Counter, int]:
        """Read activity counter from *offset*. Returns (Counter, new_offset)."""
        count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        activity: Counter = Counter()
        for _ in range(count):
            key = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            val = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            activity[key] = val
        return activity, offset

    @staticmethod
    def _reconstruct_model(frame_klines: list[KLine], ltm_klines: list[KLine],
                           stm_klines: list[KLine]) -> Model:
        """Reconstruct a Model from three tiers in the correct order.

        Order: add_ltm → add_frame → add_stm. Since serialized tier lists
        overlap (add_ltm writes LTM + Frame + STM), we check existence
        before add_frame/add_stm to avoid Frame duplication for non-literals.
        Literal dedup guards handle literal dedup automatically.
        """
        model = Model()
        for kl in ltm_klines:
            model.add_ltm(kl)
        for kl in frame_klines:
            if model.exists(kl):
                # Already written by add_ltm — just refresh STM position
                model.add_stm(kl)
            else:
                model.add_frame(kl)
        for kl in stm_klines:
            model.add_stm(kl)
        return model

    @staticmethod
    def decode(data: bytes) -> tuple[Model, Counter]:
        """Deserialize bytes into (Model, Counter).

        Detects v2 format by checking for the magic prefix 0x4B4C564E.
        Falls back to legacy single-section decode (Frame-only) otherwise.
        """
        if len(data) < 4:
            return Model(), Counter()

        first_uint32 = unpack("<I", data[0:4])[0]

        if first_uint32 == MAGIC:
            # ── v2 format ─────────────────────────────────────────────
            offset = 4
            _version = unpack("<I", data[offset:offset + 4])[0]
            offset += 4

            stm_klines, offset = BinaryAdapter._read_kline_section(data, offset)
            frame_klines, offset = BinaryAdapter._read_kline_section(data, offset)
            ltm_klines, offset = BinaryAdapter._read_kline_section(data, offset)
            activity, offset = BinaryAdapter._read_activity(data, offset)

            model = BinaryAdapter._reconstruct_model(frame_klines, ltm_klines, stm_klines)
            return model, activity

        # ── Legacy format (no magic) ──────────────────────────────────
        frame_klines, offset = BinaryAdapter._read_kline_section_legacy(data, 0)
        activity, offset = BinaryAdapter._read_activity(data, offset)

        model = BinaryAdapter._reconstruct_model(frame_klines, [], [])
        return model, activity


# ── JSON Adapter ──────────────────────────────────────────────────────

class JsonAdapter:
    """Encode/decode Model + Counter to/from a JSON-compatible dict.

    Three-section format:
        {
            "stm":     [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
            "klines":  [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
            "ltm":     [{"signature": int, "nodes": [int, ...], "literal": bool}, ...],
            "activity": {"str_key": int, ...}
        }

    Backward compatible: dicts with only "klines" and "activity" decode
    with empty STM and LTM.
    """

    @staticmethod
    def _kline_to_dict(kl: KLine) -> dict:
        """Convert a KLine to a JSON-serializable dict."""
        return {"signature": kl.signature, "nodes": list(kl.nodes), "literal": kl.literal}

    @staticmethod
    def _kline_from_dict(item: dict) -> KLine:
        """Convert a dict back to a KLine, defaulting literal to False."""
        return KLine(item["signature"], item["nodes"], literal=item.get("literal", False))

    @staticmethod
    def encode(model: Model, activity: Counter) -> dict:
        """Serialize model STM, Frame, LTM and activity counter to a dict."""
        return {
            "stm": [JsonAdapter._kline_to_dict(kl) for kl in model.iter_stm()],
            "klines": [JsonAdapter._kline_to_dict(kl) for kl in model],
            "ltm": [JsonAdapter._kline_to_dict(kl) for kl in _iter_ltm(model)],
            "activity": {str(k): v for k, v in activity.items()},
        }

    @staticmethod
    def decode(data: dict) -> tuple[Model, Counter]:
        """Deserialize a dict into (Model, Counter).

        Reconstruction order: add_ltm, add_frame, add_stm. Since serialized
        tier lists overlap (add_ltm writes LTM + Frame + STM), we check
        existence before add_frame to avoid Frame duplication for non-literals.
        Backward compatible with single-section JSON (only "klines" + "activity").
        """
        model = Model()

        # (a) Add LTM klines (writes LTM + Frame + STM)
        ltm_klines = [
            JsonAdapter._kline_from_dict(item) for item in data.get("ltm", [])
        ]
        for kl in ltm_klines:
            model.add_ltm(kl)

        # (b) Add Frame klines — check existence first to avoid Frame duplicates
        frame_klines = [
            JsonAdapter._kline_from_dict(item) for item in data.get("klines", [])
        ]
        for kl in frame_klines:
            if model.exists(kl):
                # Already written by add_ltm — just refresh STM position
                model.add_stm(kl)
            else:
                model.add_frame(kl)

        # (c) Refresh STM entries (writes STM only; literal dedup skips those already present)
        stm_klines = [
            JsonAdapter._kline_from_dict(item) for item in data.get("stm", [])
        ]
        for kl in stm_klines:
            model.add_stm(kl)

        activity: Counter = Counter()
        if "activity" in data:
            activity = Counter({int(k): v for k, v in data["activity"].items()})

        return model, activity


# ── AgentCodec ────────────────────────────────────────────────────────

class AgentCodec:
    """Serialization codec for Agent persistence.

    Wraps BinaryAdapter and JsonAdapter, providing to_bytes/from_bytes,
    to_dict/from_dict, and save/load convenience methods.

    Parameters
    ----------
    model:
        Model instance to serialize.
    activity:
        Counter tracking activity data.
    """

    def __init__(self, model: Model, activity: Counter):
        self._model = model
        self._activity = activity

    def to_bytes(self) -> bytes:
        """Serialize to binary via BinaryAdapter."""
        return BinaryAdapter.encode(self._model, self._activity)

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple[Model, Counter]:
        """Deserialize from binary via BinaryAdapter. Returns (Model, Counter)."""
        return BinaryAdapter.decode(data)

    def to_dict(self) -> dict:
        """Serialize to dict via JsonAdapter."""
        return JsonAdapter.encode(self._model, self._activity)

    @classmethod
    def from_dict(cls, data: dict) -> tuple[Model, Counter]:
        """Deserialize from dict via JsonAdapter. Returns (Model, Counter)."""
        return JsonAdapter.decode(data)

    def save(self, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
        """Persist to file. Format auto-detected from extension if not specified."""
        path = Path(path)
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"
        if format == "bin":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path = "data/agent.bin",
        format: Literal["bin", "json"] | None = None,
    ) -> tuple[Model, Counter]:
        """Load from file. Returns (Model, Counter). Format auto-detected from extension."""
        path = Path(path)
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "bin"
        if format == "bin":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
