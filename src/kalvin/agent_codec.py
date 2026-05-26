"""AgentCodec — serialization layer for Agent persistence.

Separates the binary and JSON serialization concerns from the Agent's
rationalisation pipeline. The codec owns the persistence format; Agent
owns the reasoning pipeline.

Binary format (byte-for-byte stable):
    - uint32  kline_count
    - per kline:  uint64 signature, uint32 node_count, node_count × uint64 nodes
    - uint32  activity_count
    - per entry: uint64 key, uint32 count

JSON format:
    {"klines": [{"signature": int, "nodes": [int, ...]}], "activity": {"str_key": int, ...}}

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


# ── Binary Adapter ────────────────────────────────────────────────────

class BinaryAdapter:
    """Encode/decode Model + Counter to/from a stable binary format."""

    @staticmethod
    def encode(model: Model, activity: Counter) -> bytes:
        """Serialize model klines and activity counter to bytes."""
        klines = [kl for kl in model if kl is not None]
        parts: list[bytes] = []
        parts.append(pack("<I", len(klines)))
        for kl in klines:
            parts.append(pack("<Q", kl.signature))
            parts.append(pack("<I", len(kl.nodes)))
            for n in kl.nodes:
                parts.append(pack("<Q", n))

        parts.append(pack("<I", len(activity)))
        for key, count in activity.items():
            parts.append(pack("<Q", key))
            parts.append(pack("<I", count))

        return b"".join(parts)

    @staticmethod
    def decode(data: bytes) -> tuple[Model, Counter]:
        """Deserialize bytes into (Model, Counter)."""
        offset = 0
        kline_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            sig = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            n_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes = []
            for _ in range(n_count):
                nodes.append(unpack("<Q", data[offset:offset + 8])[0])
                offset += 8
            klines.append(KLine(sig, nodes))

        activity_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        activity: Counter = Counter()
        for _ in range(activity_count):
            key = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            activity[key] = count

        model = Model()
        for kl in klines:
            model.add(kl)
            model.promote(kl)

        return model, activity


# ── JSON Adapter ──────────────────────────────────────────────────────

class JsonAdapter:
    """Encode/decode Model + Counter to/from a JSON-compatible dict."""

    @staticmethod
    def encode(model: Model, activity: Counter) -> dict:
        """Serialize model klines and activity counter to a dict."""
        return {
            "klines": [
                {"signature": kl.signature, "nodes": kl.nodes}
                for kl in model
            ],
            "activity": {str(k): v for k, v in activity.items()},
        }

    @staticmethod
    def decode(data: dict) -> tuple[Model, Counter]:
        """Deserialize a dict into (Model, Counter)."""
        klines = [
            KLine(item["signature"], item["nodes"])
            for item in data.get("klines", [])
        ]
        model = Model()
        for kl in klines:
            model.add(kl)
            model.promote(kl)

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
