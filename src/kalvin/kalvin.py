"""Kalvin - Knowledge graph with tokenization support."""

from dataclasses import dataclass
from pathlib import Path
from struct import pack, unpack
from typing import Literal
import json

from kalvin.model import KLine, KNode, Model
from kalvin.tokenizer import Tokenizer


@dataclass
class Kalvin:

    def __init__(
        self,
        model: Model | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        """Initialize Kalvin with optional model and tokenizer.

        Args:
            model: Optional Model instance
            tokenizer: Optional Tokenizer instance
        """
        self.model = model if model else Model()
        self.tokenizer = tokenizer if tokenizer else Tokenizer()

    def train_tokenizer(
        self,
        texts: list[str],
        vocab_size: int = 4096,
        pattern: str | None = None,
    ) -> None:
        """Train the BPE tokenizer on a corpus.

        Args:
            texts: List of training strings
            vocab_size: Target vocabulary size (default 4096)
            pattern: Optional custom regex pattern for pre-tokenization
        """
        self.tokenizer.train(texts, vocab_size=vocab_size, pattern=pattern)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.tokenizer.vocab_size

    # === Tokenization ===

    def encode(self, text: str) -> list[KNode]:
        """Encode a string to a list of KNodes (token IDs).

        Args:
            text: Input string to encode

        Returns:
            List of KNode integers (token IDs)
        """
        return self.tokenizer.encode(text)

    def decode(self, nodes: list[KNode]) -> str:
        """Decode a list of KNodes (token IDs) back to a string.

        Args:
            nodes: List of KNode integers (token IDs)

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(nodes)

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize model to binary format.

        Binary layout:
        - 4 bytes: number of klines (uint32)
        - For each kline:
          - 8 bytes: s_key (uint64)
          - 4 bytes: node count (uint32)
          - N * 8 bytes: nodes (uint64 each)
        """
        klines = self.model._klines
        parts = [pack("<I", len(klines))]
        for kline in klines:
            parts.append(pack("<Q", kline.s_key))
            parts.append(pack("<I", len(kline.nodes)))
            for node in kline.nodes:
                parts.append(pack("<Q", node))
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Kalvin":
        """Deserialize model from binary format."""
        offset = 0
        kline_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            s_key = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            node_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes: list[KNode] = []
            for _ in range(node_count):
                node = unpack("<Q", data[offset:offset + 8])[0]
                offset += 8
                nodes.append(node)
            klines.append(KLine(s_key=s_key, nodes=nodes))

        return cls(Model(klines=klines))

    def to_dict(self) -> dict:
        """Serialize model to dict (for JSON)."""
        klines = self.model._klines
        return {
            "klines": [
                {"s_key": kline.s_key, "nodes": kline.nodes}
                for kline in klines
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Kalvin":
        """Deserialize model from dict."""
        klines = [
            KLine(s_key=item["s_key"], nodes=item["nodes"])
            for item in data["klines"]
        ]
        return cls(Model(klines=klines))

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
        path: str | Path,
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
