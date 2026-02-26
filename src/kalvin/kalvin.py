"""Kalvin - Knowledge graph with tokenization support."""

from dataclasses import dataclass
from pathlib import Path
from struct import pack, unpack
from typing import Literal
from collections import Counter
import json

from kalvin.model import KLine, KNode, Model
from kalvin.tokenizer import Tokenizer


@dataclass
class Kalvin:
    def __init__(
        self,
        model: Model | None = None,
        activity: Counter | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        """Initialize Kalvin with optional model and tokenizer.

        Args:
            model: Optional Model instance
            tokenizer: Optional Tokenizer instance
        """
        self.model = model if model else Model()
        self.tokenizer = tokenizer if tokenizer else Tokenizer.from_directory()
        self.activity = activity if activity else Counter()

    # === Tokenization ===

    def encode(self, text: str) -> KNode | None:
        """Encode a string to a list of KNodes (token IDs).

        Args:
            text: Input string to encode

        Returns:
            List of KNode integers (token IDs)
        """
        tokens = self.tokenizer.encode(text)
        t_end = len(tokens) - 1
        build_token = 0
        prefix = None
        suffix = 0
        for idx, token in enumerate(tokens):
            self.model.add(KLine(s_key=token, nodes=[token]))
            if prefix and idx < t_end:
                suffix = tokens[idx + 1]
                self.model.add(KLine(s_key=token, nodes=[prefix, token, suffix]))
            prefix = token
            build_token |= token

        self.activity.update(tokens)
        return self.model.add(KLine(s_key=build_token, nodes=tokens))

    def decode(self, token_sig: int | None) -> str:
        """Decode a list of KNodes (token IDs) back to a string.

        Args:
            nodes: List of KNode integers (token IDs)

        Returns:
            Decoded string
        """
        kline = self.model.find_by_key(token_sig)
        if kline is None:
            return ""
        return self.tokenizer.decode(kline.nodes)

    def prune(self, level: int = 1) -> "Kalvin":
        """Prune model to activity level + untracked activity.

        Args:
            level: activity level to keep
        """
        pruned_model: list[KLine] = []
        pruned_activity = Counter()

        for kline in self.model:  # retain untracked activity
            if kline.s_key not in self.activity:
                pruned_model.append(kline)

        for key, count in self.activity.items():
            if count >= level:
                kline = self.model.find_by_key(key)
                if kline:
                    pruned_model.append(kline)
                    pruned_activity[key] = count

        model = Model(pruned_model)
        return Kalvin(model, pruned_activity)

    def model_size(self) -> int:
        """Return the number of KLines in the model.

        Returns:
            Number of KLines stored in the model
        """
        return len(self.model)

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize model to binary format.

        Binary layout:
        - 4 bytes: number of klines (uint32)
        - For each kline:
          - 8 bytes: s_key (uint64)
          - 4 bytes: node count (uint32)
          - N * 8 bytes: nodes (uint64 each)
        - 4 bytes: number of activity entries (uint32)
        - For each activity entry:
          - 8 bytes: key (uint64)
          - 4 bytes: count (uint32)
        """
        parts = [pack("<I", len(self.model))]
        for kline in self.model:
            parts.append(pack("<Q", kline.s_key))
            parts.append(pack("<I", len(kline.nodes)))
            for node in kline.nodes:
                parts.append(pack("<Q", node))

        # Serialize activity Counter
        activity = self.activity
        parts.append(pack("<I", len(activity)))
        for key, count in activity.items():
            parts.append(pack("<Q", key))
            parts.append(pack("<I", count))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Kalvin":
        """Deserialize model from binary format."""
        from collections import Counter

        offset = 0
        kline_count = unpack("<I", data[offset : offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            s_key = unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            node_count = unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            nodes: list[KNode] = []
            for _ in range(node_count):
                node = unpack("<Q", data[offset : offset + 8])[0]
                offset += 8
                nodes.append(node)
            klines.append(KLine(s_key=s_key, nodes=nodes))

        # Deserialize activity Counter
        activity_count = unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        activity: Counter = Counter()
        for _ in range(activity_count):
            key = unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
            count = unpack("<I", data[offset : offset + 4])[0]
            offset += 4
            activity[key] = count

        model = Model(klines=klines)
        return cls(model, activity)

    def to_dict(self) -> dict:
        """Serialize model to dict (for JSON)."""
        return {
            "klines": [{"s_key": kline.s_key, "nodes": kline.nodes} for kline in self.model],
            "activity": {str(k): v for k, v in self.activity.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Kalvin":
        """Deserialize model from dict."""
        from collections import Counter

        klines = [KLine(s_key=item["s_key"], nodes=item["nodes"]) for item in data["klines"]]
        model = Model(klines=klines)
        activity = Counter()
        if "activity" in data:
            activity.update({int(k): v for k, v in data["activity"].items()})
        return cls(model, activity)

    def save(
        self,
        path: str | Path,
        format: Literal["binary", "json"] = "binary",
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'binary' (default, compact) or 'json' (human-readable)
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path = "data/kalvin.bin",
        format: Literal["binary", "json"] | None = None,
    ) -> "Kalvin":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'binary', 'json', or None (auto-detect from extension)

        Returns:
            Loaded Model instance
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
